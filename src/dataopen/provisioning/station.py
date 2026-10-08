"""The factory side: the manufacturer's signing service, the serial allocator, the test jig, and the provisioning procedure that ties them
together (docs/PROVISIONING.md section 2).

The jig's electrical checks run on the REAL bridge core through `ctl.sim.World` (mouse through the bridge and direct, soft and long Panic,
watchdog, power cut); the checks for video, BLE, rails and LEDs are parametric models with injectable faults, because nothing here can
measure a signal. SIMULATION: the HSM is a key in memory, the serial database is a counter.
"""
from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..ctl import firmware as FW
from ..ctl.identity import _lib
from ..hw import failsafe as F
from . import records as R
from .device import DeviceAgent, DeviceError

DEFAULT_HW = b"DOHW0001"
STEPS = (
    ("flash", "Прошить MCU моста (оба банка), SoM (оба банка) и золотой образ; каждый образ проверяется подписью производителя"),
    ("selftest", "Стенд проверяет питание, реле, мышь, Panic, watchdog, пропадание питания, видео, BLE, светодиоды; всё прошло → DEV_TESTED"),
    ("keygen", "Устройство само рождает DAK внутри secure element и ключи владельца; наружу идут только открытые ключи и доказательство владения"),
    ("attest", "HSM проверяет доказательство, выдаёт серийный номер из базы и подписывает аттестацию (серийник, hw_id, DAK, плата)"),
    ("record", "Устройство проверяет аттестацию и пишет запись один раз в OTP / заблокированный слот; → PROVISIONED"),
    ("verify", "Станция читает запись обратно, проверяет цепочку производитель → DAK → карточка владельца и повторяет быстрый тест bypass"),
    ("lock", "Хэш отчёта в OTP; OTP, слот ключей, отладочный порт и secure boot блокируются (необратимо); → SHIPPED"),
    ("label", "Печать наклейки: серийный номер и QR (карточка рождения с цепочкой)"),
    ("release", "HSM отмечает серийник выпущенным; отчёт сохраняется в базе станции"),
)


# ----------------------------------------------------------------------------------------------------------------------------- the HSM
class HsmError(RuntimeError):
    """`key`: bad_pop | bad_serial | duplicate_serial | duplicate_dak | duplicate_board | revoked | unknown_serial | action"""

    def __init__(self, key: str, msg: str = "") -> None:
        super().__init__(msg or key)
        self.key = key


class SerialDb:
    """Allocates serial numbers: a counter per (SKU, year, week). Real life: a database with a unique key."""

    def __init__(self) -> None:
        self.count: dict[tuple[int, int, int], int] = {}

    def next(self, sku_n: int, year: int, week: int) -> str:
        k = (sku_n, year, week)
        self.count[k] = self.count.get(k, 0) + 1
        return R.make_serial(sku_n, year, week, self.count[k])


class VendorHsm:
    def __init__(self, signer, hw_id: bytes = DEFAULT_HW) -> None:
        ser, _, _ = _lib()
        self._signer = signer
        self.pub = signer.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)
        self.hw_id = hw_id.ljust(R.HW_ID_LEN, b"\0")[:R.HW_ID_LEN]
        self.serial_db = SerialDb()                                # the manufacturer's allocator: one authority, so two boards never get the same number
        self.serials: dict[str, bytes] = {}                        # serial -> DAK
        self.boards: dict[bytes, str] = {}
        self.revoked: set[str] = set()
        self.shipped: set[str] = set()
        self.tokens: set[tuple[str, bytes]] = set()
        self.log: list[tuple[str, str, bool]] = []

    def _verify_pop(self, req: dict) -> tuple[bytes, bytes, bytes]:
        _, ed25519, _ = _lib()
        try:
            hw, dak, board = bytes.fromhex(req["hw"]), base64.b64decode(req["dak"], validate=True), base64.b64decode(req["board"], validate=True)
            nonce, pop = bytes.fromhex(req["nonce"]), base64.b64decode(req["pop"], validate=True)
            ed25519.Ed25519PublicKey.from_public_bytes(dak).verify(pop, R.POP_DOMAIN + nonce + dak + board + hw)
        except Exception:
            raise HsmError("bad_pop", "the device did not prove that it holds the key") from None
        if hw != self.hw_id or len(dak) != 32 or len(board) != 32:
            raise HsmError("bad_pop", "hardware id or key sizes")
        return hw, dak, board

    def attest(self, req: dict, serial: str, date: str) -> R.Attestation:
        try:
            R.parse_serial(serial)
        except R.RecordError:
            self.log.append(("attest", serial, False))
            raise HsmError("bad_serial") from None
        try:
            hw, dak, board = self._verify_pop(req)
            if serial in self.revoked:
                raise HsmError("revoked", "this serial was revoked")
            if serial in self.serials:
                raise HsmError("duplicate_serial", "this serial is already attested to another key")
            if dak in self.serials.values():
                raise HsmError("duplicate_dak", "this device key already has a serial")
            if board in self.boards:
                raise HsmError("duplicate_board", "this board already has a serial")
        except HsmError:
            self.log.append(("attest", serial, False))
            raise
        att = R.Attestation(serial, hw, dak, board, date, R.vendor_id(self.pub))
        att = R.Attestation(att.serial, att.hw_id, att.dak, att.board, att.date, att.vendor, self._signer.sign(att.body()))
        self.serials[serial], self.boards[board] = dak, serial
        self.log.append(("attest", serial, True))
        return att

    def revoke(self, serial: str) -> None:
        self.revoked.add(serial)
        self.log.append(("revoke", serial, True))

    def mark_shipped(self, serial: str) -> None:
        if serial not in self.serials or serial in self.revoked:
            raise HsmError("unknown_serial")
        self.shipped.add(serial)
        self.log.append(("ship", serial, True))

    def issue_service_token(self, serial: str, action: str, nonce: bytes) -> R.ServiceToken:
        """Support's decision, signed. Only for a serial this manufacturer made and shipped; one token per challenge."""
        if action not in R.SERVICE_ACTIONS:
            raise HsmError("action")
        if serial not in self.shipped or serial in self.revoked:
            raise HsmError("unknown_serial", "not a shipped device of this manufacturer")
        if (serial, nonce) in self.tokens:
            raise HsmError("action", "a token for this challenge was already issued")
        self.tokens.add((serial, nonce))
        tok = R.ServiceToken(serial, action, nonce)
        self.log.append(("token:" + action, serial, True))
        return R.ServiceToken(serial, action, nonce, self._signer.sign(tok.body()))


# -------------------------------------------------------------------------------------------------------------------------------- jig
@dataclass
class BoardFaults:
    """Things that can be wrong with a board, for testing the tester."""
    relay_stuck: bool = False              # the contacts never release (welded / stuck)
    relay_slow_ms: float = 0.0             # extra release time
    watchdog_dead: bool = False
    panic_stuck: bool = False              # the panic loop never opens
    mouse_dead: bool = False
    ble_silent: bool = False
    ble_weak: bool = False
    video_dead: bool = False
    led_dead: bool = False
    rail_low: bool = False
    se_dead: bool = False


@dataclass
class Check:
    name: str
    ok: bool
    measured: object
    limit: str
    detail: str = ""


LIMITS = {"power_cut_ms": 20.0, "mode_ms": 20.0, "watchdog_ms": F.WDG_MS + 30.0, "panic_long_ms": (F.PANIC_RC_MS - 100, F.PANIC_RC_MS + 400),
          "video_lock_ms": F.MONITOR_RELOCK_MS[1], "ble_rssi_dbm": -75, "rail_tol": 0.05}
RAILS = {"VEXT": 5.0, "3V3_AON": 3.3, "3V3": 3.3, "1V8": 1.8, "1V2": 1.2}


class Jig:
    """Runs the checks on one board. `world_factory` builds the bridge rig with the real C core (a fresh one per check)."""

    def __init__(self, faults: Optional[BoardFaults] = None, world_factory: Optional[Callable] = None) -> None:
        self.faults = faults or BoardFaults()
        self._wf = world_factory

    def world(self):
        if self._wf is not None:
            return self._wf()
        from ..ctl.sim import World
        w = World(tempfile.mkdtemp(prefix="dataopen-jig-"))
        f = self.faults
        if f.watchdog_dead:
            w.rig.hw.wdg_ok = lambda t: True
        if f.panic_stuck:
            w.rig.hw.panic_loop_ok = lambda t: True
        if f.mouse_dead:
            w.rig.unplug()
        return w

    # -- timings: logic time from the real rig; the contacts add their own release time (and a stuck relay never releases)
    def _contacts_ms(self, logic_ms: Optional[float]) -> Optional[float]:
        if logic_ms is None or self.faults.relay_stuck:
            return None
        return logic_ms + F.RELEASE_MS + self.faults.relay_slow_ms

    @staticmethod
    def _until_bypass(w, trigger: Callable[[], None], max_ms: int) -> Optional[float]:
        t0 = w.rig.t
        trigger()
        ok = w.rig.run_until(lambda r: r.route == "bypass", max_ms)
        return (w.rig.t - t0) / 1000.0 if ok else None

    def run(self) -> list[Check]:
        f, L = self.faults, LIMITS
        out: list[Check] = []

        def add(name, ok, measured, limit, detail=""):
            out.append(Check(name, bool(ok), measured, limit, detail))

        # rails
        low = {k: round(v * (0.85 if (f.rail_low and k == "3V3") else 1.0), 3) for k, v in RAILS.items()}
        bad = [k for k, v in low.items() if abs(v - RAILS[k]) / RAILS[k] > L["rail_tol"]]
        add("rails", not bad, low, f"±{int(L['rail_tol'] * 100)} %", ", ".join(bad))
        add("secure_element", not f.se_dead, "answers" if not f.se_dead else "silent", "answers")
        # the mouse goes through the bridge
        w = self.world()
        if f.mouse_dead:
            add("mouse_through_bridge", False, "no mouse", "20 of 20 reports")
        else:
            w.rig.run_until(lambda r: r.pc_conn == "bridge", 4000)             # the PC has finished enumerating the bridge
            n0 = len(w.rig.pc_reports)
            for i in range(20):
                w.rig.move(3 + i % 3, -2)
                w.rig.run(5)
            w.rig.run(50)
            got = [r for r in w.rig.pc_reports[n0:] if r[1] == "bridge"]
            add("mouse_through_bridge", len(got) == 20, f"{len(got)} of 20 reports", "20 of 20 reports")
        # soft Panic: help off, no re-enumeration
        w = self.world()
        w.rig.panic(True)
        w.rig.run(150)
        w.rig.panic(False)
        w.rig.run(200)
        add("panic_soft", w.bridge().state == 2 and w.rig.route == "bridge" and w.reason() == "PANIC", f"{w.reason()} / route {w.rig.route}",
            "PASSTHRU, reason PANIC, route bridge (the pointer does not drop)")
        # long Panic: the contacts release at about 3 s, with no firmware involved
        w = self.world()
        if f.mouse_dead:
            ms = None
        else:
            ms = self._contacts_ms(self._until_bypass(w, lambda: w.rig.panic(True), 6000))
        lo, hi = L["panic_long_ms"]
        add("panic_long", ms is not None and lo <= ms <= hi, ms, f"{lo:.0f} .. {hi:.0f} ms")
        # the same press, but the firmware never hears about it: only the hardware RC loop can release the contacts (the second line)
        w = self.world()
        if f.mouse_dead:
            ms = None
        else:
            ms = self._contacts_ms(self._until_bypass(w, lambda: setattr(w.rig.hw, "panic_since", w.rig.t), 6000))
        add("panic_loop_hardware", ms is not None and lo <= ms <= hi, ms, f"{lo:.0f} .. {hi:.0f} ms, firmware blind")
        # a hung firmware
        w = self.world()
        ms = None if f.mouse_dead else self._contacts_ms(self._until_bypass(w, w.rig.kill_firmware, 1000))
        add("watchdog_hang", ms is not None and ms <= L["watchdog_ms"], ms, f"<= {L['watchdog_ms']:.0f} ms")
        # the MODE slider
        w = self.world()
        ms = None if f.mouse_dead else self._contacts_ms(self._until_bypass(w, lambda: setattr(w.rig.hw, "switch_on", False), 500))
        add("mode_switch", ms is not None and ms <= L["mode_ms"], ms, f"<= {L['mode_ms']:.0f} ms")
        # the power cut, then the mouse really works on the direct path
        w = self.world()
        ms = None if f.mouse_dead else self._contacts_ms(self._until_bypass(w, w.rig.power_off, 500))
        direct = False
        if ms is not None:
            n0 = len(w.rig.pc_reports)
            w.rig.run(500)
            w.rig.move(5, 5)
            w.rig.run(100)
            direct = any(r[1] == "direct" for r in w.rig.pc_reports[n0:])
        add("power_cut_bypass", ms is not None and ms <= L["power_cut_ms"] and direct, ms, f"<= {L['power_cut_ms']:.0f} ms, then the mouse works directly")
        # video: the logic says direct without power; a dead path shows as no lock
        eng = F.evaluate(F.Inputs(vext_ok=False, mcu_engage=True, video_en=True))
        lock = None if f.video_dead else 900.0
        add("video_direct", (not eng.video_engaged) and lock is not None and lock <= L["video_lock_ms"], lock, f"locks within {L['video_lock_ms']:.0f} ms, EDID unchanged")
        # BLE
        rssi = None if f.ble_silent else (-82 if f.ble_weak else -58)
        add("ble_visible", rssi is not None and rssi >= L["ble_rssi_dbm"], rssi, f">= {L['ble_rssi_dbm']} dBm in the shielded box")
        add("indication", not f.led_dead, "all patterns seen" if not f.led_dead else "RGB dark", "RGB, slot LEDs, BYPASS LED, buzzer respond")
        return out


class SkipJig(Jig):
    """No measurements at all: for the development server, which wants a provisioned device quickly. NEVER a factory jig."""

    def run(self) -> list[Check]:
        return []


# ------------------------------------------------------------------------------------------------------------------------- the procedure
@dataclass
class Step:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    sku: str
    steps: list[Step] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    serial: str = ""
    quarantined: bool = False
    failed_step: str = ""
    label: dict = field(default_factory=dict)
    digest: str = ""

    def to_json(self) -> dict:
        return {"sku": self.sku, "serial": self.serial, "quarantined": self.quarantined, "failed_step": self.failed_step,
                "steps": [vars(s) for s in self.steps], "checks": [{**vars(c), "measured": c.measured} for c in self.checks]}

    def seal(self) -> str:
        self.digest = hashlib.sha256(R.canon(self.to_json())).hexdigest()
        return self.digest


@dataclass
class Images:
    """What goes into the board: signed images for the bridge MCU (both banks), the SoM (both banks) and the golden recovery image."""
    som: bytes
    mcu: bytes
    golden: bytes

    @staticmethod
    def dev(version: int = 1, golden_version: Optional[int] = None, signer=None, hw_id: bytes = DEFAULT_HW) -> "Images":
        """Images signed by the simulation's TEST manufacturer (or by `signer`, for tests of a second manufacturer)."""
        from ..ctl.sim import dev_image
        gv = golden_version or version

        def make(v, tag, n):
            payload = b"%s-%d-" % (tag, v) * n
            return dev_image(v, 1, payload) if signer is None else FW.build_image(payload, hw_id, v, 1, signer)
        return Images(make(version, b"som-firmware", 300), make(version, b"mcu-firmware", 200), make(gv, b"golden-recovery", 300))


def flash(directory: Path, images: Images, vendor_pub: bytes, hw_id: bytes) -> dict:
    """Write the images the way the board would hold them: SoM banks via SlotManager, the golden image and both MCU banks as verified files."""
    d = Path(directory)
    for _name, raw in (("som", images.som), ("mcu", images.mcu), ("golden", images.golden)):
        FW.verify_image(raw, vendor_pub, hw_id)                       # an unsigned or foreign image never reaches a board
    mgr = FW.SlotManager(vendor_pub, hw_id)
    mgr.install_factory(images.som)
    mgr.slots["B"] = FW.Slot(mgr.slots["A"].image, raw=images.som)    # the second bank holds the same factory image until the first update
    mgr.save(d / "fw")
    (d / "golden").mkdir(parents=True, exist_ok=True)
    (d / "golden" / "golden.img").write_bytes(images.golden)
    (d / "golden" / "golden-mcu.img").write_bytes(images.mcu)         # the protected copy the SoM re-flashes the bridge MCU from
    (d / "mcu").mkdir(parents=True, exist_ok=True)
    for bank in "AB":
        (d / "mcu" / f"bank-{bank}.img").write_bytes(images.mcu)
    return {"som": FW.verify_image(images.som, vendor_pub, hw_id).version, "golden": FW.verify_image(images.golden, vendor_pub, hw_id).version}


CHECK_DOCS = {
    "rails": ("питание D0..D4: VEXT, 3V3_AON, 3V3, 1V8, 1V2", "±5 %"),
    "secure_element": ("secure element отвечает", "отвечает"),
    "mouse_through_bridge": ("мышь (эмулятор) проходит через мост: 20 отчётов", "20 из 20"),
    "panic_soft": ("короткое Panic: помощь выключена, указатель не пропал", "PASSTHRU, причина PANIC, маршрут через мост"),
    "panic_long": ("удержание Panic: контакты отпускаются", "≈ 2,9-3,4 с"),
    "panic_loop_hardware": ("то же, но прошивка о нажатии не знает: работает только аппаратная RC-петля", "≈ 2,9-3,4 с"),
    "watchdog_hang": ("стенд останавливает кики: контакты отпускаются", "≤ 130 мс"),
    "mode_switch": ("ползунок MODE на BYPASS", "≤ 20 мс"),
    "power_cut_bypass": ("стенд отключает VEXT: мышь работает напрямую", "≤ 20 мс и мышь работает"),
    "video_direct": ("без питания видео напрямую, захват сигнала монитором", "≤ 3 с, EDID не изменён"),
    "ble_visible": ("BLE-реклама видна в экранированном боксе", "≥ -75 дБм"),
    "indication": ("RGB, светодиоды слотов, BYPASS и зуммер отвечают", "все рисунки видны"),
}


def qr_payload(card_json: dict) -> str:
    return json.dumps({"serial": card_json.get("device", {}).get("att", {}).get("serial"), "card": card_json}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def provision(directory: str | Path, hsm: VendorHsm, *, sku: str = "DO-1", serial_db: Optional[SerialDb] = None, jig: Optional[Jig] = None,
              images: Optional[Images] = None, date: str = "2026-10-08", year_week: tuple[int, int] = (26, 41), chip_uid: Optional[bytes] = None) -> Report:
    """One board, start to finish. Any failure quarantines the board: no label, no release, and an attestation already issued is revoked."""
    from ..hw import spec as S
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    rep = Report(sku)
    serial_db = serial_db or hsm.serial_db
    jig = jig or Jig()
    images = images or Images.dev()
    sku_n = int(S.sku(sku).code.split("-")[1])
    agent = DeviceAgent(d, hw_id=hsm.hw_id, vendor_pub=hsm.pub, chip_uid=chip_uid)
    attested = ""

    def fail(name: str, detail: str):
        rep.steps.append(Step(name, False, detail))
        rep.quarantined, rep.failed_step = True, name
        if attested:
            hsm.revoke(attested)
        return rep

    if agent.lifecycle != R.BLANK:                      # never write images onto a board that already has an identity
        return fail("flash", f"the board is not blank (lifecycle {agent.lifecycle}); a service return goes through `recover full-return`")
    try:
        v = flash(d, images, hsm.pub, hsm.hw_id)
    except FW.FirmwareError as e:
        return fail("flash", f"image refused: {e.key}")
    rep.steps.append(Step("flash", True, f"SoM v{v['som']}, golden v{v['golden']}"))
    rep.checks = jig.run()
    bad = [c.name for c in rep.checks if not c.ok]
    if bad:
        return fail("selftest", "failed: " + ", ".join(bad))
    agent.mark_tested()
    rep.steps.append(Step("selftest", True, f"{len(rep.checks)} checks"))
    stage = "keygen"
    try:
        req = agent.begin()
        rep.steps.append(Step("keygen", True, "DAK and owner keys born on the device"))
        stage = "attest"
        serial = serial_db.next(sku_n, *year_week)
        att = hsm.attest(req, serial, date)
        attested = serial
        rep.steps.append(Step("attest", True, serial))
        stage = "record"
        rec = agent.accept(att.to_json(), sku=sku, factory_fw=v["som"])
    except (DeviceError, HsmError) as e:
        return fail(stage, f"{type(e).__name__}: {e.key}")
    rep.serial = rec.serial
    rep.steps.append(Step("record", True, "written once"))
    # read back and check the chain from the outside, the way a receiver would
    from ..ctl.identity import Card
    owner = agent.owner()
    cj = owner.card().to_json()
    cj["device"] = agent.cert().to_json()
    try:
        chain_serial = R.verify_chain(Card.from_json(cj), hsm.pub, hsm.hw_id)
    except R.RecordError as e:
        return fail("verify", f"chain: {e.key}")
    if chain_serial != rec.serial:
        return fail("verify", "serial mismatch")
    rep.steps.append(Step("verify", True, "manufacturer → DAK → owner card"))
    rep.label = {"serial": rec.serial, "birth_id": owner.id, "qr": qr_payload(cj)}
    digest = rep.seal()
    agent.finalize(digest)
    rep.steps.append(Step("lock", True, "OTP and key slot locked"))
    rep.steps.append(Step("label", True, rec.serial))
    hsm.mark_shipped(rec.serial)
    rep.steps.append(Step("release", True, ""))
    return rep
