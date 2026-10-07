"""The control gateway: the first service on the compute module. It is the only thing the phone talks to, and it talks to the bridge.

    phone <-- CtlLink chunks (BLE / WebSocket, not this module's business) --> Gateway --BridgeLink frames (SPI)--> bridge

What it owns: the user's settings (assistance on/off, two knob levels), the profile (A/B store), calibration, the 'try it, keep it or undo
it' timer, and the keep-alive duties toward the bridge (HELLO, CMD, parameter blobs). What it never does: put motion into the pointer path
(that is the bridge's job and the bridge only subtracts), accept a raw parameter blob from the phone, or lift a latch the user's hand set.

Rules (docs/PWA.md section 5):
  * lowering help is free and immediate; raising it is a TRIAL: applied at once, undone by itself after `trial_s` unless confirmed
  * 'turn assistance off' needs no session and no manifest and is persisted: a gateway restart never turns assistance back on
  * the phone disconnecting changes nothing about how the device works
  * everything the phone asks is checked against the same manifest that was sent to it
  * the device's own files are encrypted at rest; a settings file leaves the device only sealed to ONE device and opens only on that one
    (docs/SECURITY.md); whatever could move the personal profile elsewhere, or throw it away, needs the button on the device itself
"""
from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..bioprofile.profile import ProfileError, ProfileState, ProfileView
from ..bioprofile.progress import Progress, profile_progress
from ..bioprofile.store import ProfileStore
from ..bridge import protocol as BP
from . import bundle as B
from . import protocol as P
from . import seal as SL
from . import tuning as T
from .identity import Card, CardError, FileKeyStore, KeyStore, load_identity, provision
from .manifest import Manifest, validate_manifest
from .vault import Vault, VaultError

FW = "ctl-2"
CAP_CALIBRATION, CAP_BUNDLE = 1, 2
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


class SettingsStore:
    """Two slot files, newest valid wins (same idea as ProfileStore), JSON + CRC32 inside; encrypted at rest when given a vault."""

    def __init__(self, base: Path, vault: Optional[Vault] = None) -> None:
        self.base, self.vault = base, vault
        self.slots = [base.with_name(base.name + ".a"), base.with_name(base.name + ".b")]

    def _read(self, p: Path) -> Optional[dict]:
        try:
            raw = self.vault.read(p, self.base.name) if self.vault else p.read_bytes()
            if len(raw) < 5 or struct.unpack_from("<I", raw, len(raw) - 4)[0] != P.crc32(raw[:-4]):
                return None
            d = json.loads(raw[:-4].decode("utf-8"))
            return d if isinstance(d, dict) and isinstance(d.get("n"), int) else None
        except (OSError, ValueError, VaultError):
            return None

    def load(self) -> dict:
        best: dict = {}
        for p in self.slots:
            d = self._read(p)
            if d is not None and (not best or d["n"] >= best["n"]):
                best = d
        return best

    def save(self, data: dict) -> None:
        cur = self.load()
        data = {**data, "n": cur.get("n", 0) + 1}
        older = min(self.slots, key=lambda p: (self._read(p) or {"n": -1})["n"])
        body = json.dumps(data, separators=(",", ":")).encode()
        tmp = older.with_name(older.name + ".tmp")
        blob = body + struct.pack("<I", P.crc32(body))
        with open(tmp, "wb") as f:
            f.write(self.vault.seal(self.base.name, blob) if self.vault else blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, older)


@dataclass
class _Trial:
    deadline_us: int
    snap: Settings
    profile_prev: Optional[bytes]        # the profile at the start of the trial (None: there was none)
    profile_changed: bool = False


def _lvl(v: int) -> int:
    return min(max(int(v), T.LEVEL_MIN), T.LEVEL_MAX)


class Gateway:
    def __init__(self, directory: str | Path, *, spi: Callable[[bytes], bytes], notify: Callable[[bytes], None],
                 notify_status: Callable[[bytes], None] = lambda b: None, clock_us: Optional[Callable[[], int]] = None,
                 manifest: Optional[Manifest] = None, learner: Optional[Learner] = None, keystore: Optional[KeyStore] = None,
                 allow_plain_import: bool = False, trial_s: int = 20, spi_period_us: int = 10_000, params_period_us: int = 1_000_000,
                 cmd_period_us: int = 1_000_000, status_period_us: int = 250_000, on_forget: Callable[[], None] = lambda: None,
                 chunk_cap: int = P.CHUNK_DEFAULT) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spi, self.notify, self.notify_status = spi, notify, notify_status
        self.clock = clock_us or (lambda: time.monotonic_ns() // 1000)
        self.learner = learner
        self.allow_plain_import = allow_plain_import
        self.keystore = keystore or FileKeyStore(self.dir / "keys" / "keys.json")
        self.identity = load_identity(self.keystore)
        self.vault = Vault(self.identity.storage_key)
        self.trial_s = trial_s
        self.spi_period_us, self.params_period_us, self.cmd_period_us, self.status_period_us = (
            spi_period_us, params_period_us, cmd_period_us, status_period_us)
        self.on_forget = on_forget
        self.chunk_cap = min(max(chunk_cap, P.CHUNK_MIN), P.CHUNK_MAX)       # what the transport can really deliver (BLE: ATT MTU - 3)
        self.started_us = self.clock()
        # persisted state
        self._open_stores()
        saved = self.settings_store.load()
        self.settings = Settings(bool(saved.get("assist_wanted", False)), _lvl(saved.get("strength", T.LEVEL_DEFAULT)),
                                 _lvl(saved.get("tremor", T.LEVEL_DEFAULT)))
        self.epoch = (int(saved.get("epoch", 0)) + 1) & 0xFFFF           # generations never go backwards across restarts
        self.manifest = manifest or self._load_manifest()
        self.custom_manifest = manifest is None and self._manifest_path().exists()
        self.view: Optional[ProfileView] = None
        self.profile_error = ""
        self._load_profile()
        self.trial: Optional[_Trial] = None
        self._recover_trial(saved.get("trial"))
        self._persist()
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
        self.bad_messages = 0
        self.last_error = ""
        self._publish(self.clock(), force_status=True)

    # ------------------------------------------------------------------------------------------------------ persistence
    def _open_stores(self) -> None:
        self.settings_store = SettingsStore(self.dir / "settings", self.vault)
        self.profile_store = ProfileStore(self.dir / "profile", codec=self.vault)
        self.trust_store = SettingsStore(self.dir / "trust", self.vault)
        self.prev_path = self.dir / "profile.prev"
        for base in ("profile", "settings", "trust"):          # a device that predates the vault: seal what is still plain, both slots
            for slot in ("a", "b"):
                self.vault.migrate(self.dir / f"{base}.{slot}", base)
        self.vault.migrate(self.prev_path, "profile.prev")
        t = self.trust_store.load()
        self.trust: dict = {"senders": dict(t["senders"]) if isinstance(t.get("senders"), dict) else {}}

    def _manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def _load_manifest(self) -> Manifest:
        """The manifest the device was given by a trusted sender, else the built-in one."""
        try:
            m = json.loads(self.vault.read(self._manifest_path(), "manifest.json").decode("utf-8"))
            if not validate_manifest(m):
                return Manifest(m)
        except (OSError, ValueError, VaultError):
            pass
        return Manifest()

    def _set_manifest(self, m: Manifest) -> None:
        tmp = self._manifest_path().with_name("manifest.json.tmp")
        tmp.write_bytes(self.vault.seal("manifest.json", m.raw))
        os.replace(tmp, self._manifest_path())
        self.manifest, self.custom_manifest = m, True
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
            trial = {"snap": asdict(t.snap), "changed": t.profile_changed, "prev": t.profile_prev.hex() if t.profile_prev else None}
        self.settings_store.save({**asdict(self.settings), "epoch": self.epoch, "trial": trial})

    def _recover_trial(self, saved: Optional[dict]) -> None:
        """A restart in the middle of a trial is a 'no': the kept state comes back (a trial is never made permanent by a crash)."""
        if not saved:
            return
        try:
            s = saved["snap"]
            prev = bytes.fromhex(saved["prev"]) if saved.get("prev") else None
            snap = Settings(bool(s["assist_wanted"]), _lvl(s["strength"]), _lvl(s["tremor"]))
            self.trial = _Trial(0, snap, prev, bool(saved.get("changed")))
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

    def _profile_pack(self) -> Optional[bytes]:
        try:
            st = self.profile_store.load()
        except ProfileError:
            return None
        return st.pack() if st is not None else None

    def _install_profile(self, state: ProfileState, now: int) -> None:
        """Make `state` the profile. The one it replaces is kept for 'restore the previous profile'; with assistance on it is a trial."""
        prev = self._profile_pack()
        if prev is not None:
            self.prev_path.write_bytes(self.vault.seal("profile.prev", prev))
        if self.settings.assist_wanted:
            if self.trial is None:
                self.trial = _Trial(now + self.trial_s * 1_000_000, self.settings, prev, True)
            else:
                self.trial.profile_changed = True
                self.trial.deadline_us = now + self.trial_s * 1_000_000
        self.profile_store.save(state)
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
        elif self._raised(self.trial.snap, new, self.trial.profile_changed):
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
            self._persist()
            return
        snap = t.snap
        if t.profile_changed:
            if t.profile_prev is not None:
                self.trial = None
                self.profile_store.save(ProfileState.unpack(t.profile_prev))
                self._load_profile()
            else:                                                      # there was no profile before: the safe way back is 'off'
                snap = replace(snap, assist_wanted=False)
        self.settings = snap
        self._persist()
        self._rederive()

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
                "device.id": self.identity.id, "trusted.count": len(self.trust["senders"]), "trial.left_s": self._trial_left(now)}

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
                                self.state_rev, self.manifest.rev, max(0, (now - self.started_us) // 1_000_000), self._trial_left(now),
                                self.settings.strength, self.settings.tremor)

    def _publish(self, now: int, force_status: bool = False) -> None:
        tree = self.state_tree(now)
        diff = {k: v for k, v in tree.items() if k != "trial.left_s" and self.last_state.get(k) != v}
        if diff:
            self.state_rev = (self.state_rev + 1) & 0xFFFF
            self.last_state = {**self.last_state, **diff}
            if self.session:
                self._send(P.pack_json(P.T_EVENT, 0, {"rev": self.state_rev, "state": diff}))
        st = self.status(now).pack()
        if force_status or (st != self.last_status and (diff or now >= self.next_status)):
            self.last_status, self.next_status = st, now + self.status_period_us
            self.notify_status(st)

    def read_status(self) -> bytes:
        return self.status().pack()

    def read_info(self) -> bytes:
        caps = (CAP_CALIBRATION if self.learner is not None else 0) | CAP_BUNDLE
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
        self._publish(now)

    def physical_press(self, now: Optional[int] = None) -> None:
        """The button on the device itself (the pairing button): actions that must not be doable from the phone alone listen for it."""
        self.physical_until = (self.clock() if now is None else now) + 30_000_000

    # ------------------------------------------------------------------------------------------------------ the phone
    def on_connect(self) -> None:
        self.session = False
        self.reasm.reset()
        self.chunk, self.out_seq = P.CHUNK_DEFAULT, 0

    def on_disconnect(self) -> None:
        self.session = False
        self.reasm.reset()

    def _send(self, raw: bytes) -> None:
        chunks, self.out_seq = P.chunk_message(raw, self.chunk, self.out_seq)
        for c in chunks:
            self.notify(c)

    def _ack(self, req: int, **extra) -> None:
        self._publish(self.clock())              # the state change goes out BEFORE the answer: when `set()` resolves it is current
        self._send(P.pack_json(P.T_ACK, req, {"ok": True, "rev": self.state_rev, "trial": self._trial_left(self.clock()), **extra}))

    def _err(self, req: int, code: int, detail: str = "") -> None:
        name = next((k.lower() for k, v in P.ERRORS.items() if v == code), "unknown")
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
            what = m.json().get("what")
            if what == P.GET_MANIFEST:
                self._send(P.pack_message(P.T_DATA, m.req, self.manifest.raw))
            elif what == P.GET_STATE:
                self._send(P.pack_json(P.T_DATA, m.req, {"rev": self.state_rev, "state": self.state_tree(now)}))
            elif what == P.GET_IDENTITY:
                self._send(P.pack_json(P.T_DATA, m.req, self.identity.card().to_json()))
            elif what == P.GET_BUNDLE:
                self._send(P.pack_message(P.T_DATA, m.req, self._export_bundle(m.json().get("for", "self"), now)))
            else:
                raise _Refuse(P.E.BAD_KEY, "get what?")
        elif t == P.T_SET:
            j = m.json()
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
        else:
            raise _Refuse(P.E.UNSUPPORTED, f"type {t:#x}")

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
                prev = ProfileState.unpack(self.vault.read(self.prev_path, "profile.prev"))
            except (OSError, ProfileError, VaultError):
                raise _Refuse(P.E.NO_PROFILE, "nothing to restore") from None
            self._install_profile(prev, now)
        elif key == "pairing.forget":
            self._need_button(now, "press the button on the device")
            self.on_forget()
        elif key in ("erase.profile", "factory.reset"):
            self._need_button(now, "erase")
            self._erase(now, factory=key == "factory.reset")
        else:
            raise _Refuse(P.E.BAD_KEY, key)

    # ------------------------------------------------------------------------------------------------------ the sealed file
    def _export_bundle(self, target, now: int) -> bytes:
        """The profile and the two levels, sealed to `target` ('self', or another device's card). Another device's card needs the button
        here: otherwise a stolen, paired phone could carry the personal profile out under somebody else's key."""
        if target == "self":
            card = self.identity.card()
        else:
            try:
                card = Card.from_json(target)
            except CardError as e:
                raise _Refuse(P.E.BAD_BUNDLE, "card") from e
        if card.digest != self.identity.digest:
            self._need_button(now, f"export:{card.id}")
        try:
            st = self.profile_store.load()
        except ProfileError:
            st = None
        man = self.manifest.m if self.custom_manifest else None
        meta = {"name": "DataOpen", "created": time.strftime("%Y-%m-%d", time.gmtime())}
        return SL.seal(self.identity, card, self.identity.next_seq(), profile=st, tuning=(self.settings.strength, self.settings.tremor),
                       manifest=man, meta=meta)

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
        rec = None
        if not o.is_self:                                        # a copy of one's own is always welcome; anything else needs to be known
            fp = o.sender_digest.hex()
            rec = self.trust["senders"].get(fp)
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
        if o.profile is not None:
            self._install_profile(o.profile, now)
        if o.tuning is not None:
            self._apply(replace(self.settings, strength=o.tuning[0], tremor=o.tuning[1]), now)
        if man is not None:
            self._set_manifest(man)

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
        """Throw away everything personal. The storage key changes, so older ciphertext can never be read again even where the flash keeps
        old blocks; a factory reset also makes the device someone else (new keys, new ID, every file sealed to the old one is dead)."""
        if self.calibrating:
            self.calibrating = False
            if self.learner is not None:
                self.learner.stop(now)
            self.live_progress = None
        self.trial = None
        for pattern in ("profile.*", "settings.*", "trust.*", "manifest.json*"):
            for p in self.dir.glob(pattern):
                try:
                    p.unlink()
                except OSError:
                    pass
        if factory:
            self.identity = provision(self.keystore)
        else:
            self.identity.rotate_storage_key()
        self.vault = Vault(self.identity.storage_key)
        self._open_stores()
        self.view, self.profile_error = None, ""
        self.settings = Settings()
        self.manifest, self.custom_manifest = Manifest(), False
        self._persist()
        self._rederive()
        self._send_cmd(BP.CMD_PASSTHRU, now)
        self._manifest_changed()


class _Refuse(Exception):
    def __init__(self, code: int, detail: str = "") -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail
