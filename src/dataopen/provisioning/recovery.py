"""Recovery and unbrick: the boot chain as a model, the ways back, and a table of scenarios computed from them (docs/PROVISIONING.md section 4).

Layers, from the one that never fails to the one that needs a computer:

  0  the hardware fail-safe (docs/HARDWARE.md): whatever is dead, the mouse and the monitor are wired straight through
  1  the bridge MCU: two banks and a boot counter; if it is dead, the SoM re-flashes it over a BOOT / RESET / UART line (no computer)
  2  the SoM: banks A / B with a probation; if both are bad a signed read-only RECOVERY system from the write-protected golden image starts
     by itself and restores them on the person's button (no computer); the data partition (the profiles) is not touched
  3  the SoM boot ROM ('maskrom', cannot be erased): through the recessed RECOVERY button and the service data lines of the POWER port,
     a computer with the support tool writes the loader and the golden image again
  4  nothing helps (secure element or eMMC chip dead, supply dead): return to the manufacturer; the mouse still works directly

SIMULATION of the rules; there is no bootloader.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..ctl.firmware import MAX_BOOTS


@dataclass
class Unit:
    # what hardware gives and the field can not repair
    supply_ok: bool = True
    relays_ok: bool = True
    se_ok: bool = True
    emmc_ok: bool = True
    rom_som_ok: bool = True
    rom_mcu_ok: bool = True
    # software state
    loader_ok: bool = True
    golden_ok: bool = True
    bank_ok: dict = field(default_factory=lambda: {"A": True, "B": True})
    active: str = "A"
    mcu_ok: dict = field(default_factory=lambda: {"A": True, "B": True})
    data_ok: bool = True
    golden_version: int = 1
    floor: int = 1
    profiles: bool = True                  # the profiles are there and readable
    bonds: int = 2
    # what the person or the technician does
    chord: bool = False                    # SLOT + CONFIRM held at power-up
    service_port: bool = False             # a computer on the POWER port's data lines
    recovery_button: bool = False


@dataclass
class Boot:
    stage: str                             # normal | recovery | maskrom | dead
    bank: Optional[str]
    mcu: str                               # running | rolled_back | downloader | dead
    mouse: str                             # assisted | passthrough | direct
    video: str                             # tap | direct
    pwa: str                               # full | recovery | none
    led: str                               # indication state key
    profiles_kept: bool
    notes: list = field(default_factory=list)


def _mcu(u: Unit) -> tuple[str, list]:
    if u.mcu_ok["A"]:
        return "running", []
    if u.mcu_ok["B"]:
        return "rolled_back", ["the MCU bootloader falls back to bank B"]
    return ("downloader", ["no valid MCU image: the ROM serial downloader waits"]) if u.rom_mcu_ok else ("dead", ["the MCU is dead"])


def _kept(u: Unit) -> bool:
    """Do the profiles come back? They need an intact data partition AND the secure element that holds the keys they are sealed under."""
    return bool(u.profiles and u.data_ok and u.se_ok)


def boot(u: Unit) -> Boot:
    """What the unit does when it is powered. The hardware fail-safe is always what the mouse and the monitor fall back on."""
    notes: list = []
    if not u.supply_ok:
        return Boot("dead", None, "dead", "direct", "direct", "none", "OFF", _kept(u), ["no external power: contacts released, everything direct"])
    mcu, n = _mcu(u)
    notes += n
    mcu_up = mcu in ("running", "rolled_back")
    degraded = "passthrough" if mcu_up else "direct"
    # --- the SoM
    if not (u.rom_som_ok and u.emmc_ok):
        return Boot("dead", None, mcu, degraded, "direct", "none", "DEGRADED" if mcu_up else "BYPASS", False,
                    notes + ["the SoM or its eMMC is dead: a return to the manufacturer"])
    if not u.loader_ok:
        return Boot("maskrom", None, mcu, degraded, "direct", "none", "DEGRADED" if mcu_up else "BYPASS", _kept(u),
                    notes + ["no loader: the boot ROM waits for the support tool on the service port"])
    want_recovery = u.chord
    bank, rolled = None, False
    for b in (u.active, "B" if u.active == "A" else "A"):
        if u.bank_ok.get(b):
            bank, rolled = b, b != u.active
            break
    if bank is None:
        want_recovery = True
    if want_recovery:
        if not u.golden_ok:
            return Boot("maskrom", None, mcu, degraded, "direct", "none", "DEGRADED" if mcu_up else "BYPASS", _kept(u),
                        notes + ["the golden image is damaged: only the boot ROM and the support tool can help"])
        return Boot("recovery", None, mcu, degraded, "direct", "recovery", "RECOVERY", _kept(u),
                    notes + ["the signed recovery system runs from the golden image; the profiles are not touched"])
    if rolled:
        notes.append(f"bank {u.active} is bad: bank {bank} boots (the probation of an update ends after {MAX_BOOTS} boots)")
    kept = _kept(u)
    if not kept:
        notes.append("the data partition or the secure element does not give the profiles back: an empty device: the profile never leaves a device, so there is no copy to bring back; the person calibrates again")
    return Boot("normal", bank, mcu, "assisted" if mcu_up else "direct", "tap", "full", "READY" if mcu_up else "BYPASS", kept, notes)


# --------------------------------------------------------------------------------------------------------------------------- the ways back
class RecoveryError(RuntimeError):
    """`key`: stage | presence | port | button | golden | floor | token | hardware"""

    def __init__(self, key: str, msg: str = "") -> None:
        super().__init__(msg or key)
        self.key = key


def restore_banks_from_golden(u: Unit, *, presence: bool) -> list:
    """From the recovery system: both banks are written again from the golden image, on the person's button. Needs no computer."""
    if boot(u).stage != "recovery":
        raise RecoveryError("stage", "not in the recovery system")
    if not presence:
        raise RecoveryError("presence", "press CONFIRM on the device to restore")
    if not u.golden_ok:
        raise RecoveryError("golden")
    if u.golden_version < u.floor:
        raise RecoveryError("floor", "the golden image is older than the anti-rollback floor: a newer one has to be downloaded by the support tool")
    u.bank_ok = {"A": True, "B": True}
    u.active = "A"
    u.chord = False
    return ["both banks rewritten from the golden image", "the data partition was not touched"]


def reflash_mcu_from_som(u: Unit) -> list:
    """The SoM drives BOOT / RESET / UART of the bridge MCU and writes its signed image again (the ROM serial downloader needs nothing else)."""
    if boot(u).stage not in ("normal", "recovery"):
        raise RecoveryError("stage", "the SoM is not running")
    if not u.rom_mcu_ok:
        raise RecoveryError("hardware", "the MCU boot ROM does not answer")
    u.mcu_ok = {"A": True, "B": True}
    return ["both MCU banks rewritten over the ISP line"]


def maskrom_flash(u: Unit, *, service_port: bool, recovery_button: bool, image_floor_ok: bool = True) -> list:
    """The support tool, through the service data lines of the POWER port and the recessed RECOVERY button: loader, golden image, banks."""
    st = boot(u).stage
    if st not in ("maskrom", "dead") or not u.rom_som_ok or not u.emmc_ok:
        raise RecoveryError("hardware" if st == "dead" else "stage", "the boot ROM is not waiting for a tool")
    if not service_port:
        raise RecoveryError("port", "connect the computer to the POWER port")
    if not recovery_button:
        raise RecoveryError("button", "hold RECOVERY with a paper clip while connecting")
    if not image_floor_ok:
        raise RecoveryError("floor", "the image offered is older than the anti-rollback floor")
    u.loader_ok = u.golden_ok = True
    u.bank_ok = {"A": True, "B": True}
    u.active = "A"
    u.golden_version = max(u.golden_version, u.floor)
    return ["loader and golden image written again", "banks A and B written", "the data partition was not touched" if u.data_ok else "the data partition has to be created again"]


def download_golden(u: Unit, version: int) -> None:
    """Support downloads a newer golden image (signed, never older than the floor) so that a restore is allowed again."""
    if version < u.floor:
        raise RecoveryError("floor")
    u.golden_version = version


# ------------------------------------------------------------------------------------------------------------------------------ scenarios
@dataclass
class Scenario:
    key: str
    event: str
    damage: Callable[[Unit], None]
    procedure: tuple                       # (action key, human text)


def _dmg(**kw):
    def f(u: Unit):
        for k, v in kw.items():
            setattr(u, k, v)
    return f


def _bank_b_bad(u: Unit):
    u.active = "B"
    u.bank_ok = {"A": True, "B": False}


SCENARIOS = (
    Scenario("ota_bad_image", "обновление загрузилось, но не работает (банк B с пробой)", _bank_b_bad,
             (("auto", "само: после трёх неподтверждённых загрузок возвращается банк A"),)),
    Scenario("ota_power_cut", "питание пропало посреди записи обновления", _dmg(),
             (("auto", "ничего: работающий банк не трогался, обновление можно загрузить снова"),)),
    Scenario("som_banks_bad", "оба банка SoM повреждены, загрузчик цел", _dmg(bank_ok={"A": False, "B": False}),
             (("restore", "само запускается система восстановления; нажать CONFIRM: банки пишутся из золотого образа"),)),
    Scenario("som_chord", "прошивка SoM ведёт себя странно, человек сам просит восстановление", _dmg(chord=True),
             (("restore", "удержать SLOT+CONFIRM при подаче питания ≥ 5 с, затем CONFIRM: банки пишутся из золотого образа"),)),
    Scenario("som_loader_dead", "загрузчик SoM повреждён", _dmg(loader_ok=False),
             (("maskrom", "компьютер на порт питания + скрепка в RECOVERY: утилита поддержки пишет загрузчик, золотой образ и банки"),)),
    Scenario("som_golden_dead", "повреждены золотой образ и оба банка", _dmg(golden_ok=False, bank_ok={"A": False, "B": False}),
             (("maskrom", "компьютер на порт питания + скрепка в RECOVERY: утилита поддержки"),)),
    Scenario("data_dead", "раздел данных (профили) не читается", _dmg(data_ok=False, profiles=False),
             (("fresh", "устройство загружается пустым; профиль наружу не выгружается, копии нет: калибровка заново"),)),
    Scenario("mcu_bad_image", "образ MCU моста A плохой", _dmg(mcu_ok={"A": False, "B": True}),
             (("auto", "само: загрузчик MCU берёт банк B; SoM позже пишет A заново"),)),
    Scenario("mcu_dead", "оба банка MCU моста повреждены", _dmg(mcu_ok={"A": False, "B": False}),
             (("mcu", "мышь и экран при этом напрямую; SoM сам пишет образ MCU по ISP-линии (из системы восстановления или в обычной работе)"),)),
    Scenario("som_mcu_dead", "повреждены и SoM (загрузчик), и MCU моста", _dmg(loader_ok=False, mcu_ok={"A": False, "B": False}),
             (("maskrom", "сначала SoM через порт питания и RECOVERY"), ("mcu", "затем SoM пишет MCU по ISP"))),
    Scenario("golden_old", "золотой образ старше порога анти-отката, а банки повреждены", _dmg(bank_ok={"A": False, "B": False}, golden_version=1, floor=3),
             (("download", "восстановление отказывает (откат запрещён): поддержка загружает новый золотой образ"), ("restore", "затем CONFIRM"))),
    Scenario("lost_phone", "потерян телефон или сопряжение", _dmg(),
             (("pair", "удержать CONFIRM 3-10 с: окно сопряжения 120 с; при необходимости заводской сброс (L3)"),)),
    Scenario("se_dead", "secure element не отвечает (аппаратный отказ)", _dmg(se_ok=False),
             (("rma", "возврат производителю: ключи и профили недоступны, мышь и экран работают напрямую"),)),
    Scenario("supply_dead", "блок питания не работает или не подключён", _dmg(supply_ok=False),
             (("supply", "заменить блок питания (5 В / 3 А); прошивка ни при чём, мышь и экран работают напрямую"),)),
)


@dataclass
class Outcome:
    scenario: Scenario
    first: Boot
    final: Boot
    computer: bool                         # does any step need a computer on the service port?
    profiles_kept: bool                    # do the profiles survive the repair? (there is no copy to bring them back from: docs/RESIDENCY.md)
    rma: bool                              # is the unit lost to the field (return to the manufacturer)?
    steps: list


def run(s: Scenario) -> Outcome:
    u = Unit()
    s.damage(u)
    first = boot(u)
    computer, steps = False, []
    for key, _text in s.procedure:
        if key == "restore":
            if boot(u).stage != "recovery":
                u.chord = True                              # entering recovery is the first half of 'restore'
            steps += restore_banks_from_golden(u, presence=True)
        elif key == "maskrom":
            steps += maskrom_flash(u, service_port=True, recovery_button=True)
            computer = True
        elif key == "mcu":
            steps += reflash_mcu_from_som(u)
        elif key == "download":
            download_golden(u, 3)
            computer = True
            steps.append("a newer golden image downloaded")
        elif key == "fresh":
            u.data_ok, u.profiles = True, False             # a clean partition: an empty device, the person calibrates again
            steps.append("the device starts empty; the person calibrates again")
        elif key in ("pair", "supply"):
            if key == "supply":
                u.supply_ok = True
        elif key == "rma":
            steps.append("return to the manufacturer")
    final = boot(u)
    rma = any(k == "rma" for k, _ in s.procedure)
    return Outcome(s, first, final, computer, bool(first.profiles_kept), rma, steps)


def always_direct(u: Unit) -> bool:
    """The promise of layer 0: whatever the damage, the mouse and the monitor can reach the PC. A broken relay is the one thing it can not cover."""
    return u.relays_ok


# ------------------------------------------------------------------------------------------------------------------------- full return
def factory_return(u: Unit, *, golden_ok: bool = True) -> list:
    """L4 on the model: the banks of the SoM and of the MCU are written again from the golden image; the floor and the serial are untouched."""
    if boot(u).stage not in ("recovery", "normal"):
        raise RecoveryError("stage")
    if not golden_ok or not u.golden_ok:
        raise RecoveryError("golden")
    if u.golden_version < u.floor:
        raise RecoveryError("floor")
    u.bank_ok, u.active, u.mcu_ok = {"A": True, "B": True}, "A", {"A": True, "B": True}
    u.profiles, u.bonds = False, 0
    return ["SoM banks A and B and MCU banks A and B written from the golden image", "profiles, trust list, bonds erased", "serial, DAK, attestation and floor unchanged"]


def copy_unit(u: Unit) -> Unit:
    return deepcopy(u)
