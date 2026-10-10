"""The control gateway: the first service on the compute module. It is the only thing the phone talks to, and it talks to the bridge.

    phone <-- CtlLink chunks (BLE / WebSocket, not this module's business) --> Gateway --BridgeLink frames (SPI)--> bridge

What it owns: the user's settings (assistance on/off, two knob levels), the hardware slots (each: a profile, its levels, a layout, a name),
calibration, the 'try it, keep it or undo it' timer, the front panel (two buttons, four LEDs), the firmware banks, and the keep-alive duties
toward the bridge (HELLO, CMD, parameter blobs). What it never does: put motion into the pointer path
(that is the bridge's job and the bridge only subtracts), accept a raw parameter blob from the phone, or lift a latch the user's hand set.

Rules (docs/PWA.md section 5):
  * lowering help is free and immediate; raising it is a TRIAL: applied at once, undone by itself after `trial_s` unless confirmed
  * 'turn assistance off' needs no session and no manifest and is persisted: a gateway restart never turns assistance back on
  * the phone disconnecting changes nothing about how the device works
  * everything the phone asks is checked against the same manifest that was sent to it
  * switching context (slot) never turns assistance on; a slot the person has not yet kept in work is switched to as a trial
  * the device's own files are encrypted at rest, each slot under its own key; a settings file leaves the device only sealed to ONE
    device and opens only on that one
    (docs/SECURITY.md); whatever could move the personal profile elsewhere, or throw it away, needs the button on the device itself
"""
from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..bioprofile.profile import ProfileError, ProfileState, ProfileView
from ..bioprofile.progress import Progress, profile_progress
from ..bridge import protocol as BP
from . import bundle as B
from . import panel as PN
from . import protocol as P
from . import residency as RS
from . import seal as SL
from . import tuning as T
from .firmware import FirmwareError, SlotManager, Upload
from .identity import FileKeyStore, KeyStore, load_identity, provision
from .manifest import Manifest, validate_manifest
from .slots import SLOT_COUNT, SettingsStore, Slot, SlotSet, clean_name, migrate_legacy
from .vault import Vault, VaultError
from ..updates.channels import MAGIC_A, UpdateError, channel_of
from ..updates.manager import PackageManager, Refusal

__all__ = ["Gateway", "Settings", "SettingsStore", "Learner", "SLOT_COUNT"]

FW = "ctl-2"
CAP_CALIBRATION, CAP_BUNDLE, CAP_SLOTS, CAP_FIRMWARE, CAP_PACKAGES = 1, 2, 4, 8, 16
FW_IDLE_US = 60_000_000                 # an upload nobody has touched for this long is dropped
MAX_TRUSTED = 16
SEAL_CODES = {"damaged": (P.E.BAD_BUNDLE, "damaged"), "version": (P.E.BAD_BUNDLE, "version"), "unsupported": (P.E.UNSUPPORTED, "section"),
              "wrong_device": (P.E.WRONG_DEVICE, "wrong_device"), "bad_signature": (P.E.BAD_SIGNATURE, "bad_signature"),
              "replay": (P.E.REPLAY, "replay"), "plain_refused": (P.E.PLAIN_REFUSED, "plain")}


class Learner(Protocol):
    """Whatever feeds the BioProfile engine during calibration. On a device this is the bridge telemetry plus the UI-detector scene (not
    wired yet, docs/PWA.md section 0); the simulator replays a simulated person."""

    def start(self, now_us: int) -> None: ...
    def stop(self, now_us: int) -> None: ...
    def poll(self, now_us: int) -> None: ...
    def snapshot(self) -> ProfileState: ...


# ---------------------------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Settings:
    assist_wanted: bool = False          # a restart never turns assistance on by itself
    strength: int = T.LEVEL_DEFAULT
    tremor: int = T.LEVEL_DEFAULT


@dataclass
class _Trial:
    deadline_us: int
    snap: Settings
    profile_prev: Optional[bytes]        # the profile at the start of the trial (None: there was none)
    profile_changed: bool = False
    slot: Optional[int] = None           # set when the trial began with a switch of slot: where 'undo' goes back to
    prof_slot: int = 0                   # the slot `profile_prev` belongs to


def _lvl(v: int) -> int:
    return min(max(int(v), T.LEVEL_MIN), T.LEVEL_MAX)


class Gateway:
    def __init__(self, directory: str | Path, *, spi: Callable[[bytes], bytes], notify: Callable[[bytes], None],
                 notify_status: Callable[[bytes], None] = lambda b: None, clock_us: Optional[Callable[[], int]] = None,
                 manifest: Optional[Manifest] = None, learner: Optional[Learner] = None, keystore: Optional[KeyStore] = None,
                 allow_plain_import: bool = False, trial_s: int = 20, spi_period_us: int = 10_000, params_period_us: int = 1_000_000,
                 cmd_period_us: int = 1_000_000, status_period_us: int = 250_000, on_forget: Callable[[], None] = lambda: None,
                 chunk_cap: int = P.CHUNK_DEFAULT, vendor_pub: Optional[bytes] = None, hw_id: bytes = b"DOHW0001",
                 factory_image: Optional[bytes] = None, fw_confirm_s: int = 10,
                 on_reboot: Callable[[str], None] = lambda why: None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spi, self.notify, self.notify_status = spi, notify, notify_status
        self.clock = clock_us or (lambda: time.monotonic_ns() // 1000)
        self.learner = learner
        self.allow_plain_import = allow_plain_import
        self.keystore = keystore or FileKeyStore(self.dir / "keys" / "keys.json")
        self.identity = load_identity(self.keystore)
        self.vendor_pub, self.hw_id = vendor_pub, hw_id
        self.egress_blocked = 0          # answers the residency check refused to send (docs/RESIDENCY.md); a healthy device keeps it at 0
        # the factory record, when this device was provisioned (docs/PROVISIONING.md)
        self.device = self._open_device()
        self._attach_device()
        self.vault = Vault(self.identity.storage_key)
        self.trial_s = trial_s
        self.spi_period_us, self.params_period_us, self.cmd_period_us, self.status_period_us = (
            spi_period_us, params_period_us, cmd_period_us, status_period_us)
        self.on_forget = on_forget
        self.on_reboot = on_reboot
        self.fw_confirm_us = fw_confirm_s * 1_000_000
        self.chunk_cap = min(max(chunk_cap, P.CHUNK_MIN), P.CHUNK_MAX)       # what the transport can really deliver (BLE: ATT MTU - 3)
        self.started_us = self.clock()
        self.session = False
        # persisted state
        self.base_manifest = manifest or Manifest()
        self._open_stores()
        saved = self.settings_store.load()
        self.active = saved["slot"] if isinstance(saved.get("slot"), int) and 0 <= saved["slot"] < SLOT_COUNT else 0
        self.settings = Settings(bool(saved.get("assist_wanted", False)), self.slot.strength, self.slot.tremor)
        self.epoch = (int(saved.get("epoch", 0)) + 1) & 0xFFFF           # generations never go backwards across restarts
        self.manifest = self._slot_manifest()
        self.view: Optional[ProfileView] = None
        self.profile_error = ""
        self._load_profile()
        self.trial: Optional[_Trial] = None
        self._recover_trial(saved.get("trial"))
        self._persist()
        # front panel and firmware banks
        self.panel = PN.Panel()
        self.error_until = -1
        self.fw: Optional[SlotManager] = None
        self.fw_upload: Optional[Upload] = None
        self.fw_ok_since: Optional[int] = None
        self.fw_dir = self.dir / "fw"
        if vendor_pub is not None:
            self.fw = SlotManager.load(self.fw_dir, vendor_pub, hw_id)
            self.fw.require_approval = True             # a staged update waits for the person's button; a power cut does not apply it
            if factory_image is not None and self.fw.slots[self.fw.active].image is None:
                self.fw.install_factory(factory_image)
            self.fw.reboot()                    # what the bootloader decides at this boot (counts a boot of an unconfirmed update)
            self.fw.save(self.fw_dir)
        self.pkg = PackageManager(self)               # channel B: sealed packages for the slots (docs/UPDATES.md)
        # bridge side
        self.gen_counter = 0
        self.gen = {"asc": 0, "tremor": 0}
        self.blob_bytes: dict[str, bytes] = {}
        self.seq = 0
        self.serial = 0
        self.bridge: Optional[dict] = None
        self.bridge_t = -10**12
        self.link_bad = 0
        self.next_spi = self.next_params = self.next_cmd = 0
        self.cmd_sent: Optional[int] = None
        self.bypass_deadline: Optional[int] = None
        self.next_bypass = 0
        self.derived = T.derive(None)
        self.dirty_params = True
        self._rederive()
        # phone side
        self.reasm = P.Reassembler()
        self.session = False
        self.chunk = P.CHUNK_DEFAULT
        self.out_seq = 0
        self.state_rev = 0
        self.last_state: dict = {}
        self.last_status = b""
        self.next_status = 0
        self.calibrating = False
        self.live_progress: Optional[Progress] = None
        self.next_live = 0
        self.physical_until = -1
        self.pairing_until = -1
        self.bad_messages = 0
        self.last_error = ""
        self.pkg.recover(self.clock())               # an apply that a restart cut in the middle is finished
        self._publish(self.clock(), force_status=True)

    # ------------------------------------------------------------------------------------------------------ factory identity
    def _open_device(self):
        if self.vendor_pub is None or not (self.dir / "otp" / "otp.json").exists():
            return None
        from ..provisioning.device import DeviceAgent
        agent = DeviceAgent(self.dir, hw_id=self.hw_id, vendor_pub=self.vendor_pub)
        return agent if agent.record is not None else None

    def _attach_device(self) -> None:
        """The card this device shows carries the manufacturer's chain for ITS CURRENT owner keys: the secure element
        signs it again whenever they change."""
        if self.device is None:
            self.identity.device = None
            return
        self.identity.device = self.device.cert(self.identity.digest).to_json()
        self.device.first_boot()

    @property
    def device_serial(self) -> str:
        rec = self.device.record if self.device is not None else None
        return rec.serial if rec is not None else ""

    # ------------------------------------------------------------------------------------------------------ persistence
    def _open_stores(self) -> None:
        self.settings_store = SettingsStore(self.dir / "settings", self.vault)
        self.trust_store = SettingsStore(self.dir / "trust", self.vault)
        for base in ("settings", "trust"):                     # a device that predates the vault: seal what is still plain, both files
            for slot in ("a", "b"):
                self.vault.migrate(self.dir / f"{base}.{slot}", base)
        legacy = self.settings_store.load()                    # a device that predates slots: its one profile becomes slot 0
        self.slotset = SlotSet(self.dir, self.identity)
        levels = None
        if "strength" in legacy:
            levels = (_lvl(legacy.get("strength", T.LEVEL_DEFAULT)), _lvl(legacy.get("tremor", T.LEVEL_DEFAULT)))
        migrate_legacy(self.dir, self.vault, self.slotset, levels)
        s0 = self.slotset[0]
        if levels is not None and not s0.meta_store.load():
            s0.strength, s0.tremor = levels
            s0.save_meta()
        t = self.trust_store.load()
        self.trust: dict = {"senders": dict(t["senders"]) if isinstance(t.get("senders"), dict) else {}}

    # -- the active slot, as the rest of the gateway sees it
    @property
    def slot(self) -> Slot:
        return self.slotset[self.active]

    @property
    def profile_store(self):
        return self.slot.profile

    @property
    def prev_path(self) -> Path:
        return self.slot.prev_path

    @property
    def custom_manifest(self) -> bool:
        return self.manifest is not self.base_manifest

    def _slot_manifest(self) -> Manifest:
        """The active slot's own layout if it has a valid one, else the device's built-in (or injected) one."""
        m = self.slot.manifest
        if m is not None and not validate_manifest(m):
            return Manifest(m)
        return self.base_manifest

    def _set_manifest(self, m: Manifest, k: Optional[int] = None) -> None:
        k = self.active if k is None else k
        self.slotset[k].set_manifest(m.m)
        if k == self.active:
            self.manifest = m
            self._manifest_changed()

    def _manifest_changed(self) -> None:
        if self.session:                                     # the phone is told, and fetches the new one (it caches by hash)
            self._send(P.pack_json(P.T_EVENT, 0, {"rev": self.state_rev, "state": {}, "manifest": self.manifest.hash.hex()}))

    def _save_trust(self) -> None:
        self.trust_store.save(self.trust)

    def _persist(self) -> None:
        t = self.trial
        trial = None
        if t is not None:
            trial = {"snap": asdict(t.snap), "changed": t.profile_changed, "prev": t.profile_prev.hex() if t.profile_prev else None,
                     "slot": t.slot, "prof_slot": t.prof_slot}
        sl = self.slot
        if (sl.strength, sl.tremor) != (self.settings.strength, self.settings.tremor):
            sl.strength, sl.tremor = self.settings.strength, self.settings.tremor
            sl.save_meta()
        self.settings_store.save({"assist_wanted": self.settings.assist_wanted, "slot": self.active, "epoch": self.epoch, "trial": trial})

    def _recover_trial(self, saved: Optional[dict]) -> None:
        """A restart in the middle of a trial is a 'no': the kept state comes back (a trial is never made permanent by a crash)."""
        if not saved:
            return
        try:
            s = saved["snap"]
            prev = bytes.fromhex(saved["prev"]) if saved.get("prev") else None
            snap = Settings(bool(s["assist_wanted"]), _lvl(s["strength"]), _lvl(s["tremor"]))
            back = saved.get("slot")
            back = back if isinstance(back, int) and 0 <= back < SLOT_COUNT else None
            ps = saved.get("prof_slot")
            ps = ps if isinstance(ps, int) and 0 <= ps < SLOT_COUNT else self.active
            self.trial = _Trial(0, snap, prev, bool(saved.get("changed")), back, ps)
        except (KeyError, TypeError, ValueError):
            self.settings = replace(self.settings, assist_wanted=False)          # unreadable: the safe state
            return
        self._end_trial(False, 0)

    def _load_profile(self) -> None:
        try:
            st = self.profile_store.load()
            self.view = ProfileView(st.pack()) if st is not None else None
            self.profile_error = ""
        except ProfileError as e:                      # a newer or damaged profile: stay without one, never overwrite
            self.view, self.profile_error = None, str(e)
        self.slot.refresh()

    @staticmethod
    def _pack_of(sl: Slot) -> Optional[bytes]:
        try:
            st = sl.profile.load()
        except ProfileError:
            return None
        return st.pack() if st is not None else None

    def _install_profile(self, state: ProfileState, now: int, k: Optional[int] = None) -> None:
        """Make `state` the profile of slot `k` (the active one by default). The one it replaces is kept for 'restore the previous profile'.
        In the active slot with assistance on it is a trial; in a slot that is not in use nothing running changes, and the slot is marked
        as not yet kept in work, so the first switch to it with assistance on is a trial too."""
        k = self.active if k is None else k
        sl = self.slotset[k]
        prev = self._pack_of(sl)
        if prev is not None:
            sl.prev_path.write_bytes(sl.vault.seal("profile.prev", prev))
        sl.profile.save(state)
        sl.refresh()
        sl.vetted = False
        sl.save_meta()
        if k != self.active:
            return
        if self.settings.assist_wanted:
            if self.trial is None:
                self.trial = _Trial(now + self.trial_s * 1_000_000, self.settings, prev, True, None, k)
            else:
                if not self.trial.profile_changed:
                    self.trial.prof_slot = k
                self.trial.profile_changed = True
                self.trial.deadline_us = now + self.trial_s * 1_000_000
        self._load_profile()
        self._rederive()
        self._persist()

    # ------------------------------------------------------------------------------------------------------ derived blobs
    def _rederive(self) -> None:
        self.derived = T.derive(self.view, self.settings.strength, self.settings.tremor)
        self.dirty_params = True

    def _blob_frames(self, now: int) -> list[BP.Frame]:
        d = self.derived
        frames: list[BP.Frame] = []
        for kind, fixed, make in (("asc", d.asc, lambda g, s: BP.asc_frames(d.asc, g, d.profile_id, d.ppc, serial=s)),
                                  ("tremor", d.tremor, lambda g, s: BP.tremor_frames(d.tremor, g, d.profile_id, serial=s))):
            raw = repr(fixed).encode() + repr((d.ppc, d.profile_id)).encode()
            if self.blob_bytes.get(kind) != raw:                 # new content: next generation (same content is a keepalive)
                self.blob_bytes[kind] = raw
                self.gen_counter = (self.gen_counter + 1) & 0xFFFF
                self.gen[kind] = ((self.epoch << 16) | self.gen_counter) & 0xFFFFFFFF
            self.serial = (self.serial + 1) & 15
            frames += make(self.gen[kind], self.serial)
        return frames

    # ------------------------------------------------------------------------------------------------------ bridge side
    def _xfer(self, f: BP.Frame) -> None:
        self.seq = (self.seq + 1) & 0xFFFF
        f.seq = self.seq
        rx = BP.unpack_frame(self.spi(BP.pack_frame(f)))
        if rx is None:
            self.link_bad += 1
            return
        if rx.kind == BP.LK_STATUS:
            self.bridge, self.bridge_t = BP.parse_status(rx), self.clock()
        elif rx.kind == BP.LK_TELEM and self.calibrating and hasattr(self.learner, "on_telem"):
            self.learner.on_telem(BP.parse_telem(rx)[0])

    def _cmd_now(self) -> int:
        return BP.CMD_ASSIST if (self.settings.assist_wanted and not self.calibrating) else BP.CMD_PASSTHRU

    def _send_cmd(self, cmd: int, now: int) -> None:
        self._xfer(BP.cmd_frame(cmd))
        self.cmd_sent, self.next_cmd = cmd, now + self.cmd_period_us

    def _bridge_tick(self, now: int) -> None:
        if now < self.next_spi:
            return
        self.next_spi = now + self.spi_period_us
        if self.bypass_deadline is not None:
            b = self.bridge or {}
            if b.get("latch_hw") or b.get("hw") == 0 or now >= self.bypass_deadline:
                self.bypass_deadline = None               # latched (or gave up): from here on only PASSTHRU is ever sent
            elif now >= self.next_bypass:
                self._send_cmd(BP.CMD_BYPASS, now)
                self.next_bypass = now + 100_000
        if self.bypass_deadline is None and (self.cmd_sent != self._cmd_now() or now >= self.next_cmd):
            self._send_cmd(self._cmd_now(), now)
        if self.dirty_params or now >= self.next_params:
            for f in self._blob_frames(now):
                self._xfer(f)
            self.dirty_params, self.next_params = False, now + self.params_period_us
        self._xfer(BP.Frame(BP.LK_HELLO))

    # ------------------------------------------------------------------------------------------------------ the knobs
    def _raised(self, snap: Settings, new: Settings, profile_changed: bool) -> bool:
        if not new.assist_wanted:
            return False
        return (not snap.assist_wanted) or new.strength > snap.strength or new.tremor > snap.tremor or profile_changed

    def _apply(self, new: Settings, now: int) -> None:
        old = self.settings
        if new == old:
            return
        if self.trial is None:
            if self._raised(old, new, False):
                self.trial = _Trial(now + self.trial_s * 1_000_000, old, None)
        elif self._raised(self.trial.snap, new, self.trial.profile_changed or self.trial.slot is not None):
            self.trial.deadline_us = now + self.trial_s * 1_000_000      # still above what was kept: the timer starts over
        else:
            self.trial = None                                           # back at or below the kept state: nothing left to confirm
        self.settings = new
        self._persist()
        self._rederive()

    def _end_trial(self, keep: bool, now: int) -> None:
        t, self.trial = self.trial, None
        if t is None:
            return
        if keep:
            self.slot.vetted = True                                    # kept in work: the next switch to it is not a trial
            self.slot.save_meta()
            self._persist()
            return
        snap = t.snap
        if t.profile_changed:
            if t.profile_prev is not None:
                sl = self.slotset[t.prof_slot]
                sl.profile.save(ProfileState.unpack(t.profile_prev))
                sl.refresh()
                if t.prof_slot == self.active:
                    self._load_profile()
            else:                                                      # there was no profile before: the safe way back is 'off'
                snap = replace(snap, assist_wanted=False)
        if t.slot is not None and t.slot != self.active:               # the trial began with a switch of slot: back to where it came from
            self._activate(t.slot)
        self.settings = snap
        self._persist()
        self._rederive()

    # ------------------------------------------------------------------------------------------------------ slots
    def _activate(self, k: int) -> None:
        """Make slot `k` the active one: its profile, its levels, its layout. Does not touch whether assistance is wanted."""
        self.active = k
        self.settings = replace(self.settings, strength=self.slot.strength, tremor=self.slot.tremor)
        self._load_profile()
        old = self.manifest
        self.manifest = self._slot_manifest()
        if self.manifest.hash != old.hash:
            self._manifest_changed()

    def _switch_slot(self, k: int, now: int) -> None:
        """Go to slot `k`. Assistance is neither switched on nor off by this. A slot the person has kept in work before is switched to at
        once; anything newer (a file, a calibration) with assistance on is a trial, and 'undo' comes back to the slot it left."""
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < SLOT_COUNT:
            raise _Refuse(P.E.BAD_VALUE, f"0..{SLOT_COUNT - 1} was expected")
        if k == self.active:
            return
        if self.calibrating:
            raise _Refuse(P.E.BUSY, "calibrating")
        if self.trial is not None:
            raise _Refuse(P.E.BUSY, "trial")
        old_settings, left = self.settings, self.slot
        if old_settings.assist_wanted and not left.vetted:             # it was in work and nobody objected: it counts as kept
            left.vetted = True
            left.save_meta()
        self._activate(k)
        if old_settings.assist_wanted and self.slot.has and not self.slot.vetted:
            self.trial = _Trial(now + self.trial_s * 1_000_000, old_settings, None, False, left.k, k)
        self._persist()
        self._rederive()
        self._send_cmd(self._cmd_now(), now)

    def _next_slot(self) -> int:
        """The slot the SLOT button goes to: the next one in use (so a person with two contexts toggles between exactly those), or the next
        one at all when fewer than two are in use."""
        used = [s.k for s in self.slotset if s.used]
        pool = used if len(used) >= 2 else list(range(SLOT_COUNT))
        after = [k for k in pool if k > self.active]
        return after[0] if after else pool[0]

    def _clear_slot(self, now: int) -> None:
        """Clear the active slot: a new key for it (the old ciphertext is dead), nothing in it, assistance off."""
        self._stop_calibration(now)
        self.trial = None
        self.slotset.clear(self.active)
        self.pkg.touch()
        self.settings = Settings(False, T.LEVEL_DEFAULT, T.LEVEL_DEFAULT)
        self._load_profile()
        old, self.manifest = self.manifest, self._slot_manifest()
        self._persist()
        self._rederive()
        self._send_cmd(BP.CMD_PASSTHRU, now)
        if old.hash != self.manifest.hash:
            self._manifest_changed()

    def _stop_calibration(self, now: int) -> None:
        if self.calibrating:
            self.calibrating = False
            if self.learner is not None:
                self.learner.stop(now)
            self.live_progress = None

    def _stop(self, now: int) -> None:
        self.trial = None
        self.settings = replace(self.settings, assist_wanted=False)
        self._persist()
        self._send_cmd(BP.CMD_PASSTHRU, now)                           # right now, not at the next tick
        self.bypass_deadline = None

    def _hard_bypass(self, now: int) -> None:
        self.trial = None
        self.settings = replace(self.settings, assist_wanted=False)
        self._persist()
        self.bypass_deadline, self.next_bypass = now + 1_000_000, now
        self._send_cmd(BP.CMD_BYPASS, now)
        self.next_bypass = now + 100_000

    # ------------------------------------------------------------------------------------------------------ calibration
    def _calib(self, run: bool, now: int) -> None:
        if self.learner is None:
            raise _Refuse(P.E.UNSUPPORTED, "unsupported")
        if run and not self.calibrating:
            self.learner.start(now)
            self.calibrating = True
            self.live_progress = Progress(0, False, "collecting", {})       # the NEW profile's progress starts at zero
        elif not run and self.calibrating:
            self.calibrating = False
            self.learner.stop(now)
            st = self.learner.snapshot()
            self.live_progress = None
            if any(s.valid for s in st.stats.values()):
                self._install_profile(st, now)

    # ------------------------------------------------------------------------------------------------------ state for the phone
    def progress(self) -> Progress:
        return self.live_progress or profile_progress(self.view)

    def state_tree(self, now: Optional[int] = None) -> dict:
        now = self.clock() if now is None else now
        pr = self.progress()
        layers = "both" if pr.asc_ready and pr.tremor_ready else "asc" if pr.asc_ready else "tremor" if pr.tremor_ready else "none"
        return {"assist.on": self.settings.assist_wanted, "assist.strength": self.settings.strength, "tremor.level": self.settings.tremor,
                "calib.running": self.calibrating, "profile.fill": pr.fill, "profile.layers": layers, "profile.tremor": pr.tremor,
                "device.id": self.identity.id, "device.serial": self.device_serial, "trusted.count": len(self.trust["senders"]),
                "trial.left_s": self._trial_left(now),
                "pairing.open": now < self.pairing_until, "slot.active": self.active, "slot.name": self.slot.name,
                **{f"slot.{s.k}.name": s.name for s in self.slotset},
                "fw.version": self._fw_version(), "fw.state": self._fw_state(), **self.pkg.state()}

    def _fw_version(self) -> int:
        if self.fw is None:
            return 0
        img = self.fw.slots[self.fw.running].image
        return img.version if img else 0

    def _fw_state(self) -> str:
        if self.fw is None:
            return "unsupported"
        if self.fw.trial is None:
            return "current"
        return "trial" if self.fw.running == self.fw.trial else "staged"

    def _trial_left(self, now: int) -> int:
        return 0 if self.trial is None else max(0, -(-(self.trial.deadline_us - now) // 1_000_000))

    def status(self, now: Optional[int] = None) -> P.StatusSnapshot:
        now = self.clock() if now is None else now
        b = self.bridge if (self.bridge is not None and now - self.bridge_t < 1_500_000) else None
        pr = self.progress()
        flags = (P.SF_ASSIST_WANTED if self.settings.assist_wanted else 0) | (P.SF_CALIBRATING if self.calibrating else 0)
        flags |= P.SF_TRIAL if self.trial else 0
        if b is not None:
            flags |= P.SF_BRIDGE_SEEN
            for k, bit in (("latch_soft", P.SF_LATCH_SOFT), ("latch_hw", P.SF_LATCH_HW), ("attach", P.SF_PC_ATTACHED),
                           ("link_ok", P.SF_LINK_OK), ("healthy", P.SF_HEALTHY)):
                flags |= bit if b[k] else 0
            flags |= P.SF_PARAMS_OK if (b["params_asc_ok"] or b["params_trm_ok"]) else 0
        ready = (P.RB_ASC if pr.asc_ready else 0) | (P.RB_TREMOR if pr.tremor_ready else 0)
        ready |= P.RB_TREMOR_NOT_NEEDED if pr.tremor == "not_needed" else 0
        return P.StatusSnapshot(int(b is not None), b["state"] if b else P.MODE_UNKNOWN, b["reason"] if b else 0, flags, pr.fill, ready,
                                self.state_rev, self.manifest.rev, max(0, (now - self.started_us) // 60_000_000), self._trial_left(now),
                                self.settings.strength, self.settings.tremor, self.active, self.slotset.mask())

    def _publish(self, now: int, force_status: bool = False) -> None:
        tree = self.state_tree(now)
        diff = {k: v for k, v in tree.items() if k != "trial.left_s" and self.last_state.get(k) != v}
        if diff:
            self.state_rev = (self.state_rev + 1) & 0xFFFF
            self.last_state = {**self.last_state, **diff}
            if self.session:
                ev = {"rev": self.state_rev, "state": diff}
                try:
                    RS.scrub_event(ev)
                except RS.ResidencyViolation:
                    self.egress_blocked += 1
                else:
                    self._send(P.pack_json(P.T_EVENT, 0, ev))
        st = self.status(now).pack()
        if force_status or (st != self.last_status and (diff or now >= self.next_status)):
            self.last_status, self.next_status = st, now + self.status_period_us
            self.notify_status(st)

    def read_status(self) -> bytes:
        return self.status().pack()

    def read_info(self) -> bytes:
        caps = ((CAP_CALIBRATION if self.learner is not None else 0) | CAP_BUNDLE | CAP_SLOTS | CAP_PACKAGES
                | (CAP_FIRMWARE if self.fw else 0))
        return P.Info(caps, self.identity.digest[:4], self.manifest.hash, self.manifest.rev, self.chunk_cap).pack()

    # ------------------------------------------------------------------------------------------------------ main loop
    def tick(self, now: Optional[int] = None) -> None:
        now = self.clock() if now is None else now
        self._bridge_tick(now)
        if self.trial is not None and now >= self.trial.deadline_us:
            self._end_trial(False, now)
            self._send_cmd(self._cmd_now(), now)
        if self.calibrating and self.learner is not None:
            self.learner.poll(now)
            if now >= self.next_live:
                self.next_live = now + 250_000
                try:
                    self.live_progress = profile_progress(ProfileView(self.learner.snapshot().pack()))
                except ProfileError:
                    self.live_progress = None
        self._fw_tick(now)
        self.pkg.tick(now)
        self._publish(now)

    def _fw_tick(self, now: int) -> None:
        if self.fw_upload is not None and now - self.fw_upload.last_us > FW_IDLE_US:
            self.fw_upload = None                                       # nobody is sending any more: the half image is dropped
        fw = self.fw
        if fw is None or fw.trial is None or fw.running != fw.trial:
            self.fw_ok_since = None
            return
        b = self.bridge if (self.bridge is not None and now - self.bridge_t < 1_500_000) else None
        # talks to the bridge, the PC sees the mouse
        if b is not None and b["healthy"] and b["link_ok"] and b["attach"] and b["state"] >= 2:
            self.fw_ok_since = now if self.fw_ok_since is None else self.fw_ok_since
            if now - self.fw_ok_since >= self.fw_confirm_us:
                fw.confirm()
                fw.save(self.fw_dir)
                self.fw_ok_since = None
        else:
            self.fw_ok_since = None

    def physical_press(self, now: Optional[int] = None) -> None:
        """The button on the device itself (the pairing button): actions that must not be doable from the phone alone listen for it."""
        self.physical_until = (self.clock() if now is None else now) + 30_000_000

    # ------------------------------------------------------------------------------------------------------ the front panel
    def button(self, name: str, down: bool, now: Optional[int] = None) -> None:
        """A hardware button edge ('slot' or 'confirm'). What a press means is in panel.py; this is what the device does about it."""
        now = self.clock() if now is None else now
        if down:
            self.panel.press(name, now)
            return
        ev = self.panel.release(name, now)
        try:
            if ev == PN.SLOT_NEXT:
                self._switch_slot(self._next_slot(), now)
            elif ev == PN.CONFIRM_SHORT:
                if self.trial is not None:
                    self._end_trial(True, now)
                    self._send_cmd(self._cmd_now(), now)
                else:
                    self.physical_press(now)
            elif ev == PN.PAIR:
                self.pairing_until = now + PN.PAIR_WINDOW_US            # the phone may bond now (the BlueZ side is not written: state only)
            elif ev in (PN.ERASE, PN.FACTORY):
                self._erase(now, factory=ev == PN.FACTORY)
        except _Refuse:
            self.error_until = now + 600_000                           # the LEDs say 'no'
        self._publish(now)

    def led_mode(self, now: Optional[int] = None) -> str:
        now = self.clock() if now is None else now
        if now < self.error_until:
            return PN.M_ERROR
        warn = self.panel.hold_warning(now)
        if warn:
            return warn
        if self.trial is not None:
            return PN.M_TRIAL
        if now < self.pairing_until:
            return PN.M_PAIR
        if now <= self.physical_until:
            return PN.M_WINDOW
        return PN.M_CALIB if self.calibrating else PN.M_STEADY

    def leds(self, now: Optional[int] = None) -> tuple[bool, ...]:
        now = self.clock() if now is None else now
        return PN.leds(self.led_mode(now), self.active, now, SLOT_COUNT)

    # ------------------------------------------------------------------------------------------------------ the phone
    def on_connect(self) -> None:
        self.session = False
        self.reasm.reset()
        self.chunk, self.out_seq = P.CHUNK_DEFAULT, 0

    def on_disconnect(self) -> None:
        self.session = False
        self.reasm.reset()

    def _send(self, raw: bytes) -> None:
        if len(raw) < P.HDR or raw[1] not in RS.OUTGOING_TYPES:       # the one door out: only the answers on the list leave the device
            self.egress_blocked += 1
            return
        chunks, self.out_seq = P.chunk_message(raw, self.chunk, self.out_seq)
        for c in chunks:
            self.notify(c)

    def _ack(self, req: int, **extra) -> None:
        self._publish(self.clock())              # the state change goes out BEFORE the answer: when `set()` resolves it is current
        self._send(P.pack_json(P.T_ACK, req, {"ok": True, "rev": self.state_rev, "trial": self._trial_left(self.clock()), **extra}))

    def _err(self, req: int, code: int, detail: str = "") -> None:
        name = next((k.lower() for k, v in P.ERRORS.items() if v == code), "unknown")
        detail = str(detail)[:RS.MAX_ERR_DETAIL]
        self.last_error = f"{name}:{detail}"
        self._send(P.pack_json(P.T_ERR, req, {"code": code, "key": f"err.{name}", "detail": detail}))

    def on_write(self, chunk: bytes) -> None:
        now = self.clock()
        raw = self.reasm.feed(chunk, now)
        if raw is None:
            return
        msg = P.unpack_message(raw)
        if msg is None:
            self.bad_messages += 1
            self._err(0, P.E.BAD_MSG, "crc/size")
            return
        try:
            self._dispatch(msg, now)
        except _Refuse as r:
            self._err(msg.req, r.code, r.detail)
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
            self._err(msg.req, P.E.BAD_MSG, type(e).__name__)
        self._publish(now)

    def _dispatch(self, m: P.Message, now: int) -> None:
        t = m.type
        if t == P.T_PING:
            self._send(P.pack_message(P.T_PONG, m.req, m.body[:64]))
            return
        if t == P.T_STOP:
            self._stop(now)
            self._ack(m.req)
            return
        if t == P.T_HARD_BYPASS:
            self._hard_bypass(now)
            self._ack(m.req)
            return
        if t == P.T_HELLO:
            j = m.json()
            if j.get("v") != P.VER:
                raise _Refuse(P.E.BAD_VERSION, f"v={j.get('v')}")
            self.chunk = min(max(int(j.get("chunk", P.CHUNK_DEFAULT)), P.CHUNK_MIN), self.chunk_cap)
            self.session = True
            self._send(P.pack_json(P.T_HELLO_R, m.req, {"v": P.VER, "chunk": self.chunk, "device": self.identity.id, "fw": FW,
                                                          "manifest_rev": self.manifest.rev, "manifest_hash": self.manifest.hash.hex(),
                                                          "trial_s": self.trial_s}))
            return
        if not self.session:
            raise _Refuse(P.E.NO_SESSION, "say HELLO first")
        if t == P.T_GET:
            self._get(m)
        elif t == P.T_SET:
            j = m.json()
            # like STOP: whatever layout a slot brings, the way to the next slot stays
            if j.get("key") == "slot.active":
                self._switch_slot(j.get("value"), now)
                self._ack(m.req)
                return
            v = self.manifest.check_set(j.get("key"), j.get("value"))
            if not v.ok:
                raise _Refuse(v.code, v.detail)
            self._do_set(j["key"], j["value"], now)
            self._ack(m.req)
        elif t == P.T_ACT:
            j = m.json()
            v = self.manifest.check_act(j.get("key"), bool(j.get("confirmed")))
            if not v.ok:
                raise _Refuse(v.code, v.detail)
            self._do_act(j["key"], now)
            self._ack(m.req)
        elif t == P.T_CONFIRM:
            if self.trial is None:
                raise _Refuse(P.E.NOT_ALLOWED, "no trial")
            keep = bool(m.json().get("keep"))
            self._end_trial(keep, now)
            self._send_cmd(self._cmd_now(), now)
            self._ack(m.req)
        elif t == P.T_BUNDLE_PUT:
            if len(m.body) > SL.MAX_FILE:
                raise _Refuse(P.E.TOO_BIG, "bundle")
            self._import_bundle(m.body, now)
            self._ack(m.req)
        elif t in (P.T_FW_BEGIN, P.T_FW_CHUNK, P.T_FW_END):
            self._firmware_stream(m, now)
        elif t in (P.T_PKG_BEGIN, P.T_PKG_CHUNK, P.T_PKG_END):
            self._package_stream(m, now)
        else:
            raise _Refuse(P.E.UNSUPPORTED, f"type {t:#x}")

    def _get(self, m: P.Message) -> None:
        """What the phone can ask the device to SAY: a closed list (`P.GET_KINDS`). Asking for the profile or the model, under any name, is
        refused with RESIDENT: that is not a missing feature, it is the rule."""
        what = m.json().get("what")
        if what in RS.ASK_FOR_SECRET:
            raise _Refuse(P.E.RESIDENT, "resident")
        if what not in P.GET_KINDS:
            raise _Refuse(P.E.BAD_KEY, "get what?")
        now = self.clock()
        if what == P.GET_MANIFEST:
            body = self.manifest.raw
        else:
            if what == P.GET_STATE:
                obj = {"rev": self.state_rev, "state": self.state_tree(now)}
            elif what == P.GET_IDENTITY:
                obj = self.identity.card().to_json()
            elif what == P.GET_SLOTS:
                obj = {"active": self.active, "count": SLOT_COUNT, "slots": [s.info() for s in self.slotset]}
            elif what == P.GET_FIRMWARE:
                obj = self._fw_info()
            else:
                obj = self.pkg.info()
            try:
                RS.scrub(what, obj)
            except RS.ResidencyViolation:
                self.egress_blocked += 1
                raise _Refuse(P.E.RESIDENT, "egress") from None
            body = P.pack_json(P.T_DATA, m.req, obj)
            if len(body) > RS.MAX_ANSWER + P.HDR + 4:
                self.egress_blocked += 1
                raise _Refuse(P.E.RESIDENT, "egress")
            self._send(body)
            return
        self._send(P.pack_message(P.T_DATA, m.req, body))

    def _do_set(self, key: str, value, now: int) -> None:
        s = self.settings
        if key == "assist.on":
            if value and self.calibrating:
                raise _Refuse(P.E.BUSY, "calibrating")
            self._apply(replace(s, assist_wanted=bool(value)), now)
            self._send_cmd(self._cmd_now(), now)
        elif key == "assist.strength":
            self._apply(replace(s, strength=value), now)
        elif key == "tremor.level":
            self._apply(replace(s, tremor=value), now)
        elif key == "calib.running":
            self._calib(bool(value), now)
            self._send_cmd(self._cmd_now(), now)
        elif key == "slot.name":
            self.slot.name = clean_name(value)
            self.slot.save_meta()
        else:
            raise _Refuse(P.E.BAD_KEY, key)

    def _need_button(self, now: int, detail: str) -> None:
        """Spend the press of the button on the device (one press, one use, 30 s), or refuse and say what is being asked for."""
        if now > self.physical_until:
            raise _Refuse(P.E.PHYSICAL, detail)
        self.physical_until = -1

    def _do_act(self, key: str, now: int) -> None:
        if key == "profile.restore":
            try:
                prev = ProfileState.unpack(self.slot.vault.read(self.prev_path, "profile.prev"))
            except (OSError, ProfileError, VaultError):
                raise _Refuse(P.E.NO_PROFILE, "nothing to restore") from None
            self._install_profile(prev, now)
        elif key == "pairing.forget":
            self._need_button(now, "press the button on the device")
            self.on_forget()
        elif key in ("erase.profile", "factory.reset"):
            self._need_button(now, "erase")
            self._erase(now, factory=key == "factory.reset")
        elif key == "slot.clear":
            self._need_button(now, f"slot:{self.active + 1}")
            self._clear_slot(now)
        elif key == "fw.apply":
            self._fw_apply(now)
        elif key == "fw.rollback":
            self._fw_rollback(now)
        elif key in ("pkg.apply", "pkg.discard", "pkg.revert", "model.clear", "trust.clear"):
            self._pkg_act(key, now)
        else:
            raise _Refuse(P.E.BAD_KEY, key)

    # ------------------------------------------------------------------------------------------------------ the sealed file
    # There is no export here, on purpose (docs/RESIDENCY.md): the profile and the weights of the model never leave the device, so this
    # gateway has no function that seals them for anybody, itself included. Files only come IN.
    def _import_bundle(self, raw: bytes, now: int) -> None:
        if raw[:4] == b"DOBN":
            if not self.allow_plain_import:
                raise _Refuse(P.E.PLAIN_REFUSED, "plain")
            self._import_plain(raw, now)
            return
        if raw[:4] != SL.MAGIC:
            raise _Refuse(P.E.BAD_BUNDLE, "damaged")
        try:
            o = SL.open_sealed(raw, self.identity)
        except SL.SealError as e:
            code, detail = SEAL_CODES[e.key]
            raise _Refuse(code, detail) from None
        if self.calibrating:
            raise _Refuse(P.E.BUSY, "calibrating")
        man = None
        if o.manifest is not None:
            if validate_manifest(o.manifest):
                raise _Refuse(P.E.BAD_BUNDLE, "damaged")
            man = Manifest(o.manifest)
        for sd in o.slots:                                       # every layout is checked before any slot is touched
            if sd.manifest is not None and validate_manifest(sd.manifest):
                raise _Refuse(P.E.BAD_BUNDLE, "damaged")
        if o.is_self:                  # this device makes no files (docs/RESIDENCY.md): one that says it did is a theft of its keys
            raise _Refuse(P.E.BAD_BUNDLE, "own_file")
        fp = o.sender_digest.hex()
        rec = self.trust["senders"].get(fp)                      # everybody else needs to be known, the first time by the button here
        if rec is not None and o.seq <= rec["last_seq"]:
            raise _Refuse(P.E.REPLAY, "replay")
        if rec is None:
            if len(self.trust["senders"]) >= MAX_TRUSTED:
                raise _Refuse(P.E.NOT_ALLOWED, "trust list full")
            self._need_button(now, f"trust:{o.sender_id}")
            rec = {"id": o.sender_id}
            self.trust["senders"][fp] = rec
        rec["last_seq"] = o.seq
        self._save_trust()
        if o.slots:                                              # a file about named slots: each goes to the slot with its number
            for sd in o.slots:
                self._install_slot(sd, now)
            return
        if o.profile is not None:
            self._install_profile(o.profile, now)
        if o.tuning is not None:
            self._apply(replace(self.settings, strength=o.tuning[0], tremor=o.tuning[1]), now)
        if man is not None:
            self._set_manifest(man)

    def _install_slot(self, sd: "SL.SlotData", now: int) -> None:
        k, sl = sd.n, self.slotset[sd.n]
        if sd.profile is not None:
            self._install_profile(sd.profile, now, k)
        if sd.tuning is not None:
            if k == self.active:
                self._apply(replace(self.settings, strength=sd.tuning[0], tremor=sd.tuning[1]), now)
            else:
                sl.strength, sl.tremor = sd.tuning
                sl.save_meta()
        if sd.manifest is not None:
            self._set_manifest(Manifest(sd.manifest), k)
        if sd.name:
            sl.name = sd.name
            sl.save_meta()

    def _import_plain(self, raw: bytes, now: int) -> None:
        """The old open format: only when the device was configured to take it (a migration aid, never the default)."""
        try:
            b = B.unpack(raw)
        except B.BundleError as e:
            raise _Refuse(P.E.BAD_BUNDLE, e.key) from None
        if self.calibrating:
            raise _Refuse(P.E.BUSY, "calibrating")
        if b.profile is not None:
            self._install_profile(b.profile, now)
        self._apply(replace(self.settings, strength=b.strength, tremor=b.tremor), now)

    # ------------------------------------------------------------------------------------------------------ erase
    def _erase(self, now: int, factory: bool) -> None:
        """Throw away everything personal, in EVERY slot. The storage key changes, so older ciphertext can never be read again even where
        the flash keeps old blocks; a factory reset also makes the device someone else (new keys, new ID, every file sealed to the old one
        is dead). The firmware banks are not personal and stay."""
        self._stop_calibration(now)
        self.trial = None
        for pattern in ("settings.*", "trust.*", "profile.*", "manifest.json*"):          # the last two: what a pre-slot device left behind
            for p in self.dir.glob(pattern):
                try:
                    p.unlink()
                except OSError:
                    pass
        shutil.rmtree(self.dir / "slots", ignore_errors=True)
        self.pkg.erase()                                              # a package waiting to be applied is personal too
        if factory:
            self.identity = provision(self.keystore)
            self._attach_device()                                    # same serial, same DAK, a new owner card signed by it
            self.on_forget()                                         # and the paired phones are forgotten
        else:
            self.identity.rotate_storage_key()
        self.vault = Vault(self.identity.storage_key)
        self._open_stores()
        self.active = 0
        self.view, self.profile_error = None, ""
        self.settings = Settings()
        self.manifest = self.base_manifest
        self._persist()
        self._rederive()
        self._send_cmd(BP.CMD_PASSTHRU, now)
        self._manifest_changed()

    # ------------------------------------------------------------------------------------------------------ firmware
    def _fw_info(self) -> dict:
        fw = self.fw
        if fw is None:
            return {"supported": False}
        up = self.fw_upload
        return {"supported": True, "running": fw.running, "active": fw.active, "trial": fw.trial, "boots": fw.boots, "floor": fw.floor,
                "versions": {k: (s.image.version if s.image else None) for k, s in fw.slots.items()},
                "upload": None if up is None else {"next": up.next, "size": up.size}, "max_size": fw.slot_size}

    def _fw_refuse(self, e: FirmwareError):
        raise _Refuse(P.E.FW_REJECTED, e.key)

    def _firmware_stream(self, m: P.Message, now: int) -> None:
        """FW_BEGIN {size, sha256}, FW_CHUNK (u32 offset + data)*, FW_END: the image goes into the bank that is NOT running, after it has
        been verified as a whole (manufacturer signature, hardware, anti-rollback). Nothing here changes what runs."""
        fw = self.fw
        if fw is None:
            raise _Refuse(P.E.FW_REJECTED, "unsupported")
        try:
            if m.type == P.T_FW_BEGIN:
                j = m.json()
                if fw.trial is not None:
                    raise FirmwareError("no_trial", "the previous update is not confirmed yet")
                sha = bytes.fromhex(j["sha256"]) if "sha256" in j else None
                self.fw_upload = Upload(j["size"], sha, fw.slot_size + 256, now, clean_name(j.get("name", "")))
                self._ack(m.req, next=0)
                return
            up = self.fw_upload
            if up is None:
                raise FirmwareError("sequence", "no upload in progress")
            if m.type == P.T_FW_CHUNK:
                if len(m.body) < 5:
                    raise FirmwareError("damaged", "empty chunk")
                if int.from_bytes(m.body[:4], "little") == 0 and len(m.body) >= 8 and m.body[4:8] != MAGIC_A:
                    self.fw_upload = None                     # the first bytes say what a file is: a package is not a system image
                    raise FirmwareError("wrong_channel" if channel_of(m.body[4:8]) == "B" else "damaged", "not a system image")
                nxt = up.chunk(int.from_bytes(m.body[:4], "little"), m.body[4:], now)
                self._ack(m.req, next=nxt)
                return
            try:
                raw = up.finish()
            except FirmwareError as e:
                if e.key == "damaged":
                    self.fw_upload = None
                raise
            img = fw.stage(raw)
            fw.save(self.fw_dir)
            self.fw_upload = None
            self._ack(m.req, staged=img.version)
        except (FirmwareError, KeyError, TypeError, ValueError) as e:
            key = e.key if isinstance(e, FirmwareError) else "damaged"
            raise _Refuse(P.E.FW_REJECTED, key) from None

    # ------------------------------------------------------------------------------------------------------ packages (channel B)
    def _package_stream(self, m: P.Message, now: int) -> None:
        """PKG_BEGIN {size}, PKG_CHUNK (u32 offset + data)*, PKG_END: a sealed package for this device. The header is judged as soon as
        it is here (signature, recipient, replay, schema), the rest is checked as it arrives; nothing is applied by receiving it."""
        try:
            if m.type == P.T_PKG_BEGIN:
                self.pkg.begin(m.json(), now)
                self._ack(m.req, next=0)
            elif m.type == P.T_PKG_CHUNK:
                self._ack(m.req, next=self.pkg.chunk(m.body, now))
            else:
                self._ack(m.req, pending=self.pkg.end(now))
        except UpdateError as e:
            raise _Refuse(P.E.PKG_REJECTED, e.key) from None
        except (KeyError, TypeError, ValueError):
            raise _Refuse(P.E.PKG_REJECTED, "damaged") from None

    def _pkg_act(self, key: str, now: int) -> None:
        try:
            if key == "pkg.apply":
                self.pkg.apply(now)
            elif key == "pkg.discard":
                self.pkg.discard()
            elif key == "model.clear":
                self.pkg.clear_model()
            elif key == "trust.clear":
                self.pkg.forget_senders(now)
            else:
                self.pkg.revert(now)
        except UpdateError as e:
            raise _Refuse(P.E.PKG_REJECTED, e.key) from None
        except Refusal as r:
            raise _Refuse(r.code, r.detail) from None

    def _reboot(self, why: str) -> None:
        """On a board: the bootloader takes over right here. In the simulation the world swaps in a fresh gateway on the same directory."""
        self._persist()
        if self.fw is not None:
            self.fw.save(self.fw_dir)
        self.on_reboot(why)

    def _fw_apply(self, now: int) -> None:
        fw = self.fw
        if fw is None:
            raise _Refuse(P.E.FW_REJECTED, "unsupported")
        if fw.trial is None or fw.running == fw.trial:
            raise _Refuse(P.E.FW_REJECTED, "no_trial")
        self._need_button(now, "fw.apply")
        fw.approve()
        self._reboot("update")

    def _fw_rollback(self, now: int) -> None:
        fw = self.fw
        if fw is None:
            raise _Refuse(P.E.FW_REJECTED, "unsupported")
        try:
            fw.check_switch_back()
        except FirmwareError as e:
            raise _Refuse(P.E.FW_REJECTED, e.key) from None
        self._need_button(now, "fw.rollback")
        fw.switch_back()
        self._reboot("rollback")


class _Refuse(Exception):
    def __init__(self, code: int, detail: str = "") -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail
