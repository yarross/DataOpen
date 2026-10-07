"""Everything needed to run the gateway against the REAL bridge core without hardware, a phone or a person:

  World       the bridge simulator (`bridge.sim.Rig`: real C core, simulated mouse, PC, switch, watchdog) with the gateway as the compute
              module on its SPI link, plus a `SimPhone` on the other side of the gateway
  SimPhone    a client written in Python that speaks CtlLink exactly like the browser client does (the two are compared on golden vectors)
  SimLearner  replays a simulated person into the BioProfile engine while 'calibration' runs

What this is NOT: BLE (no radio, no MTU negotiation, no bonding), a real phone, or a real person.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from ..bioprofile.engine import BioProfileEngine, EngineConfig
from ..bioprofile.profile import ProfileState
from ..bioprofile.sim import SimPlayer, simulate
from ..bridge import protocol as BP
from ..bridge.cbridge import S_PASSTHRU
from ..bridge.sim import Rig
from ..bridge.sim_usb import SimMouse
from . import protocol as P
from .gateway import Gateway, Learner
from .manifest import Manifest


class SimLearner:
    """A simulated person who 'uses the mouse' while calibration runs. `speed` = simulated seconds of use per second of device time."""

    def __init__(self, player: Optional[SimPlayer] = None, minutes: float = 6.0, seed: int = 0, speed: float = 120.0,
                 profile_id: int = 0x51D0) -> None:
        self.player = player or SimPlayer(jitter_amp_deg=0.3, t_motor_ms=260.0)
        self.stream = simulate(self.player, minutes, seed)
        self.speed, self.profile_id = speed, profile_id
        self.engine: Optional[BioProfileEngine] = None
        self.t0 = 0
        self.i = 0
        self.last_us = 0
        self.runs = 0

    def start(self, now_us: int) -> None:
        self.engine = BioProfileEngine(EngineConfig(deg_per_count=self.player.dpc, profile_id=self.profile_id))
        self.t0, self.i, self.last_us = now_us, 0, 0
        self.runs += 1

    def poll(self, now_us: int) -> None:
        if self.engine is None:
            return
        t_sim = int((now_us - self.t0) * self.speed)
        ev = self.stream.events
        while self.i < len(ev) and ev[self.i][1] <= t_sim:
            e = ev[self.i]
            self.i += 1
            if e[0] == "mouse":
                self.engine.on_mouse(e[1], e[2], e[3])
            else:
                self.engine.on_target(e[1], e[2])
        self.last_us = min(t_sim, int(self.stream.duration_s * 1e6))
        self.engine.advance(self.last_us)

    def stop(self, now_us: int) -> None:
        self.poll(now_us)

    def snapshot(self) -> ProfileState:
        assert self.engine is not None
        return self.engine.snapshot(clean=True)


class SimPhone:
    """Talks to a Gateway through chunks only, the way a Web Bluetooth page would."""

    def __init__(self, gw: Gateway, chunk: int = 100, clock=lambda: 0) -> None:
        self.gw, self.chunk, self.clock = gw, chunk, clock
        self.reasm = P.Reassembler()
        self.inbox: list[P.Message] = []
        self.status_notes: list[P.StatusSnapshot] = []
        self.bad = 0
        self.req = 0
        self.out_seq = 0
        self.manifest: Optional[dict] = None
        self.state: dict = {}
        gw.notify = self._on_chunk
        gw.notify_status = lambda b: self.status_notes.append(P.StatusSnapshot.unpack(b))

    def _on_chunk(self, c: bytes) -> None:
        raw = self.reasm.feed(c, self.clock())
        if raw is None:
            return
        m = P.unpack_message(raw)
        if m is None:
            self.bad += 1
            return
        self.inbox.append(m)
        if m.type == P.T_EVENT:
            self.state.update(m.json().get("state", {}))

    def connect(self) -> P.Message:
        self.gw.on_connect()
        return self.call(P.T_HELLO, {"v": P.VER, "chunk": self.chunk, "lang": "ru"})

    def write(self, raw: bytes) -> None:
        chunks, self.out_seq = P.chunk_message(raw, max(P.CHUNK_MIN, min(self.chunk, P.CHUNK_MAX)), self.out_seq)
        for c in chunks:
            self.gw.on_write(c)

    def call(self, type_: int, obj: Optional[dict] = None, body: Optional[bytes] = None) -> P.Message:
        """Send and return the reply that carries the same req id (replies are synchronous in the simulator)."""
        self.req = (self.req % 0xFFFF) + 1
        raw = P.pack_message(type_, self.req, body) if body is not None else P.pack_json(type_, self.req, obj)
        mark = len(self.inbox)
        self.write(raw)
        for m in self.inbox[mark:]:
            if m.req == self.req and m.type != P.T_EVENT:
                return m
        raise AssertionError("no reply")

    # conveniences
    def get_manifest(self) -> dict:
        r = self.call(P.T_GET, {"what": "manifest"})
        assert r.type == P.T_DATA
        self.manifest = json.loads(r.body.decode("utf-8"))
        return self.manifest

    def get_state(self) -> dict:
        r = self.call(P.T_GET, {"what": "state"})
        self.state = r.json()["state"]
        return self.state

    def set(self, key: str, value) -> P.Message:
        return self.call(P.T_SET, {"key": key, "value": value})

    def act(self, key: str, confirmed: bool = False) -> P.Message:
        return self.call(P.T_ACT, {"key": key, "confirmed": confirmed})

    def confirm(self, keep: bool) -> P.Message:
        return self.call(P.T_CONFIRM, {"keep": keep})

    def stop(self) -> P.Message:
        return self.call(P.T_STOP, body=b"")

    def hard_bypass(self) -> P.Message:
        return self.call(P.T_HARD_BYPASS, body=b"")

    def try_bundle(self, target="self") -> P.Message:
        """`target`: 'self' or another device's card (a dict). The reply is DATA (the sealed file) or ERR."""
        return self.call(P.T_GET, {"what": "bundle", "for": target})

    def get_bundle(self, target="self") -> bytes:
        r = self.try_bundle(target)
        assert r.type == P.T_DATA, r.body
        return r.body

    def get_identity(self) -> dict:
        r = self.call(P.T_GET, {"what": "identity"})
        assert r.type == P.T_DATA
        return r.json()

    def put_bundle(self, raw: bytes) -> P.Message:
        return self.call(P.T_BUNDLE_PUT, body=raw)

    def disconnect(self) -> None:
        self.gw.on_disconnect()


class World:
    """The gateway, the real bridge core and a simulated mouse and PC, advancing together in simulated microseconds."""

    def __init__(self, directory: str | Path, *, kind: str = "m16", learner: Optional[Learner] = None, manifest: Optional[Manifest] = None,
                 trial_s: int = 20, cfg: Optional[dict] = None, chunk: int = 100, seed: int = 0, testing: bool = False,
                 start: bool = True, spi_period_us: int = 10_000, **gw_kw) -> None:
        self.kind, self.seed, self.chunk = kind, seed, chunk
        self.rig = Rig(SimMouse(kind), cfg=cfg, module=False, testing=testing)
        self.pre = BP.pack_frame(BP.Frame(BP.LK_NOP))
        self.spi_loss = 0.0
        self.rng = random.Random(seed)
        self.dir = Path(directory)
        self.gw_args = dict(chunk_cap=P.CHUNK_MAX, manifest=manifest, learner=learner, trial_s=trial_s,
                            spi_period_us=spi_period_us, **gw_kw)
        self.gw: Gateway
        self.phone: SimPhone
        self.rig.module = _Ticker(self)  # type: ignore[assignment]
        self.make_gateway()
        if start:
            self.rig.run_until(lambda r: r.b.status(r.t).state >= S_PASSTHRU, 20000)

    def make_gateway(self) -> None:
        """(Re)start the compute-module side: the same directory, so the persisted settings and the profile come back."""
        self.gw = Gateway(self.dir, spi=self._spi, notify=lambda b: None, clock_us=lambda: self.rig.t, **self.gw_args)
        self.phone = SimPhone(self.gw, chunk=self.chunk, clock=lambda: self.rig.t)
        self.pre = BP.pack_frame(BP.Frame(BP.LK_NOP))

    def _spi(self, frame: bytes) -> bytes:
        """One full-duplex SPI transaction: the bridge's preloaded frame comes back, the bridge consumes ours."""
        if self.spi_loss and self.rng.random() < self.spi_loss:
            return BP.pack_frame(BP.Frame(BP.LK_NOP))
        t = self.rig.t
        rx, self.pre = self.pre, b""
        self.rig.b.link_rx(t, frame)
        self.pre = self.rig.b.link_tx(t)
        return rx

    # shortcuts
    @property
    def t(self) -> int:
        return self.rig.t

    def run(self, ms: float) -> None:
        self.rig.run(ms)

    def bridge(self):
        return self.rig.status()

    def reason(self) -> str:
        from ..bridge.cbridge import REASONS
        return REASONS[self.rig.status().reason]


class _Ticker:
    """What `Rig.step` calls on its 'module': here, the gateway's own loop."""

    def __init__(self, w: World) -> None:
        self.w = w

    def tick(self, t: int) -> None:
        if getattr(self.w, "gw", None) is not None and not getattr(self.w, "gw_dead", False):
            self.w.gw.tick(t)

    def param_frames(self):          # used by BridgeAssist only
        return []


def seed_profile(directory: str | Path, persona: str = "tremor", minutes: float = 8.0, seed: int = 1):
    """Put a profile learned from a simulated person (the real BioProfile engine) into a gateway directory before it starts."""
    from ..assist.sim_user import PERSONAS, build_profile
    from ..bioprofile.store import ProfileStore
    view = build_profile(PERSONAS[persona], minutes=minutes, seed=seed)
    Path(directory).mkdir(parents=True, exist_ok=True)
    ProfileStore(Path(directory) / "profile").save(view._state)
    return view
