"""Simulator of the whole rig around the bridge core: the analog switch with its watchdog and panic loop, the real mouse, the PC that
enumerates and reads reports, the compute module on the SPI link, and fault injection. Time is simulated (microseconds).

What it is NOT: a model of electricity, of USB timing below the microframe, or of any vendor's firmware. The re-enumeration gap a PC
needs after the switch moves is a constant (`reenum_ms`, an ASSUMPTION; real PCs take about 0.2 - 2 s)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from ..assist.fixed import FixedParams
from ..assist.tremor_fixed import FixedTremorParams
from . import protocol as P
from .cbridge import HW_PROBE, PX_FORWARD, PX_LOCAL, PX_SERVE, RESET_POWER, S_ASSIST, CBridge
from .sim_usb import OK, SimMouse, SimPC


class HwModel:
    """The analog switch (mouse <-> PC when disengaged), the external watchdog supervisor, the panic loop and the mechanical switch.
    ENGAGE = powered AND mcu_engage AND watchdog ok AND panic loop ok AND mechanical switch on. Everything else is firmware."""

    def __init__(self, wdg_ms: int = 100, rc_ms: int = 3000) -> None:
        self.wdg_ms, self.rc_ms = wdg_ms, rc_ms
        self.powered = True
        self.switch_on = True
        self.mcu_engage = False
        self.last_kick: Optional[int] = None
        self.panic_since: Optional[int] = None
        self.panic_wire_cut = False

    def kick(self, t: int) -> None:
        self.last_kick = t

    def wdg_ok(self, t: int) -> bool:
        return self.last_kick is not None and t - self.last_kick <= self.wdg_ms * 1000

    def panic_loop_ok(self, t: int) -> bool:
        return not self.panic_wire_cut and not (self.panic_since is not None and t - self.panic_since >= self.rc_ms * 1000)

    def engage(self, t: int) -> bool:
        return self.powered and self.switch_on and self.mcu_engage and self.wdg_ok(t) and self.panic_loop_ok(t)


class BridgeFront:
    """The bridge as the PC sees it: answers control requests from the cache or by forwarding to the real mouse."""

    def __init__(self, rig: "Rig") -> None:
        self.rig = rig

    def control(self, bm, req, value, index, length, data=b""):
        rig = self.rig
        kind, served = rig.b.pc_setup(bm, req, value, index, length)
        if kind == PX_SERVE:
            return OK, served
        if kind == PX_LOCAL:
            return OK, b""
        assert kind == PX_FORWARD
        try:
            st, d = rig.mouse.control(bm, req, value, index, length, data)
        except TimeoutError:
            rig.b.usb_error(rig.t)
            raise
        if st == OK:
            rig.b.pc_forwarded_ok(rig.t, bm, req, value, index, length)
        return st, d


class SimModule:
    """The compute module on the SPI link (master). Sends params/scene/keepalive, receives telemetry/status. Faults on request."""

    def __init__(
        self,
        rig: "Rig",
        asc: Optional[FixedParams] = None,
        tremor: Optional[FixedTremorParams] = None,
        ppc: float = 1.0,
        period_ms: int = 5,
        scene_fn: Optional[Callable] = None,
        scene_ms: int = 33,
        seed: int = 0,
    ) -> None:
        self.rig, self.asc, self.tremor, self.ppc = rig, asc, tremor, ppc
        self.period_us, self.scene_fn, self.scene_us = period_ms * 1000, scene_fn, scene_ms * 1000
        self.generation = 0
        self.silent = False
        self.loss = 0.0
        self.corrupt = 0.0
        self.rng = random.Random(seed)
        self.next_t = 0
        self.next_scene = 0
        self.next_params = 0
        self.seq = 0
        self.serial = 0
        self.pre = P.pack_frame(P.Frame(P.LK_NOP))
        self.telemetry: list[P.Sample] = []
        self.statuses: list[dict] = []
        self.bad_rx = 0
        self.rx_seqs: list[int] = []
        self.tsync: list[dict] = []
        self.outbox: list[P.Frame] = []

    def set_params(self, asc: Optional[FixedParams], tremor: Optional[FixedTremorParams], ppc: float = 1.0) -> None:
        self.asc, self.tremor, self.ppc = asc, tremor, ppc
        self.generation += 1
        self.next_params = 0

    def param_frames(self) -> list[P.Frame]:
        self.serial = (self.serial + 1) & 15
        out = []
        if self.asc is not None:
            out += P.asc_frames(self.asc, self.generation, ppc=self.ppc, serial=self.serial)
        if self.tremor is not None:
            out += P.tremor_frames(self.tremor, self.generation, serial=self.serial)
        return out

    def exchange(self, t: int, frame: bytes) -> None:
        """One full-duplex SPI transaction: what the bridge had preloaded comes back, the bridge consumes ours."""
        if self.loss and self.rng.random() < self.loss:
            return
        if self.corrupt and self.rng.random() < self.corrupt:
            b = bytearray(frame)
            b[self.rng.randrange(len(b))] ^= 1 << self.rng.randrange(8)
            frame = bytes(b)
        rx, self.pre = self.pre, None
        self.rig.b.link_rx(t, frame)
        self.pre = self.rig.b.link_tx(t)
        f = P.unpack_frame(rx)
        if f is None:
            self.bad_rx += 1
            return
        self.rx_seqs.append(f.seq)
        if f.kind == P.LK_TELEM:
            self.telemetry += P.parse_telem(f)[0]
        elif f.kind == P.LK_STATUS:
            self.statuses.append(P.parse_status(f))
        elif f.kind == P.LK_TSYNC_REPLY:
            self.tsync.append(P.parse_tsync_reply(f))

    def send(self, t: int, frames: list[P.Frame]) -> None:
        for f in frames:
            self.seq = (self.seq + 1) & 0xFFFF
            f.seq = self.seq
            self.exchange(t, P.pack_frame(f))

    def tick(self, t: int) -> None:
        if self.silent or t < self.next_t:
            return
        self.next_t = t + self.period_us
        frames: list[P.Frame] = []
        if t >= self.next_params and (self.asc is not None or self.tremor is not None):
            frames += self.param_frames()
            self.next_params = t + 1_000_000
        if self.scene_fn is not None and t >= self.next_scene:
            self.next_scene = t + self.scene_us
            sc = self.scene_fn(t)
            tcap, objs = sc if isinstance(sc, tuple) else (t, sc)  # a detector reports WHEN it looked, not when it finished
            frames.append(P.scene_frame(tcap, objs))
        frames.append(P.Frame(P.LK_HELLO))
        self.send(t, frames[:1] if len(frames) == 1 else frames)


class Rig:
    def __init__(
        self,
        mouse: Optional[SimMouse] = None,
        cfg: Optional[dict] = None,
        step_us: int = 1000,
        reenum_ms: int = 300,
        testing: bool = False,
        module: bool = True,
        asc: Optional[FixedParams] = None,
        tremor: Optional[FixedTremorParams] = None,
        ppc: float = 1.0,
        scene_fn: Optional[Callable] = None,
        enum_ms: int = 150,
    ) -> None:
        self.cfg, self.testing = cfg, testing
        self.b = CBridge(cfg, testing=testing)
        self.hw = HwModel()
        self.t = 0
        self.step_us, self.reenum_us, self.enum_us = step_us, reenum_ms * 1000, enum_ms * 1000
        self.probe_t: Optional[int] = None
        self.mouse: Optional[SimMouse] = None
        self.alive = True
        self.route = "bypass"
        self.pc_conn: Optional[str] = None
        self.pc_ready_at: Optional[int] = None
        self.pc_reports: list[tuple] = []
        self.report_delay_us: list[int] = []  # per routed report: time from `move()` to the step that handled it (docs/LATENCY.md)
        self._enq: dict[int, list[int]] = {}
        self.transcripts: dict[str, list] = {}
        self.pc_enumerations: list[tuple[int, str, list]] = []   # every enumeration by the PC: (time, "direct" | "bridge", transcript)
        self.front = BridgeFront(self)
        self.module = SimModule(self, asc, tremor, ppc, scene_fn=scene_fn) if module else None
        self.image_built = False
        self.dropped = 0
        self.pc_never_configures = False
        self.events: list[tuple[int, str]] = []
        self.nv_crashes = 0
        self.b.boot(0, RESET_POWER, 0)
        if mouse is not None:
            self.plug(mouse)

    # -- the world
    def plug(self, mouse: SimMouse) -> None:
        self.mouse = mouse
        self._enq.clear()
        self.b.set_speed(2 if mouse.speed == "HS" else 1)
        self.b.dev_present(self.t, True)
        self.pc_conn = None
        self.pc_ready_at = self.t + self.reenum_us
        self.log("mouse plugged")

    def unplug(self) -> None:
        self.mouse = None
        self._enq.clear()
        self.b.dev_present(self.t, False)
        self.pc_conn = None
        self.pc_ready_at = None
        self.image_built = False
        self.log("mouse unplugged")

    def log(self, s: str) -> None:
        self.events.append((self.t, s))

    def panic(self, pressed: bool) -> None:
        self.hw.panic_since = self.t if pressed else None
        if self.alive and self.hw.powered:
            self.b.panic(self.t, pressed)

    def kill_firmware(self) -> None:
        self.alive = False
        self.log("firmware hung")

    def power_off(self) -> None:
        self.hw.powered = False
        self.alive = False
        self.log("power lost")

    def reboot_firmware(self, cause: int = 1) -> None:
        crashes = self.nv_crashes
        self.b = CBridge(self.cfg, testing=self.testing)
        if self.mouse is not None:
            self.b.set_speed(2 if self.mouse.speed == "HS" else 1)
        self.b.boot(self.t, cause, crashes)
        n = self.b.nv_take()
        if n is not None:
            self.nv_crashes = n
        if self.mouse is not None:
            self.b.dev_present(self.t, True)
        self.alive = True
        self.hw.powered = True
        self.image_built = False
        self.log("firmware rebooted")

    # -- one time step
    def _build_image(self, t: int) -> None:
        m = self.mouse
        try:
            self.b.img_device(m.dev)
            self.b.img_config(m.cfg)
            plain = [f for f in m.ifaces if f.alt == 0]
            for i in range(min(self.b.img_n_if(), len(plain))):
                n = self.b.img_rd_wanted(i)
                if n:
                    st, rd = m.control(0x81, 6, 0x2200, plain[i].num, n)
                    if st != OK:
                        return
                    self.b.img_report_desc(i, rd)
            self.b.img_done(t)
            self.image_built = True
        except TimeoutError:
            return

    def step(self) -> None:
        t = self.t
        hw = self.hw
        if hw.powered and self.alive:
            self.b.poll(t)
            if self.b.wants_kick:
                hw.kick(t)
            hw.mcu_engage = self.b.hw_select != 0
            engage = hw.engage(t)
            if self.b.hw_select == HW_PROBE:
                if self.probe_t is None:
                    self.probe_t = t
                if not self.image_built and self.mouse is not None and engage and t - self.probe_t >= self.enum_us:
                    self._build_image(t)
            else:
                self.image_built = False
                self.probe_t = None
            n = self.b.nv_take()  # the port persists the crash counter when the core says so
            if n is not None:
                self.nv_crashes = n
            if self.module is not None:
                self.module.tick(t)
        else:
            engage = hw.engage(t) if hw.powered else False
        attached = bool(hw.powered and self.alive and self.b.attach) or (
            hw.powered and not self.alive and engage and self.route == "bridge"
        )
        route = "bypass" if not engage else ("bridge" if attached else "none")
        if route != self.route:
            self.log(f"route {self.route} -> {route}")
            self.route = route
            self.pc_conn = None
            self.pc_ready_at = t + self.reenum_us if (route in ("bypass", "bridge") and self.mouse is not None) else None
            if route == "bridge":
                self.b.pc_bus_reset(t) if self.alive else None
        if self.pc_ready_at is not None and t >= self.pc_ready_at and self.mouse is not None:
            self.pc_ready_at = None
            if self.route == "bypass":
                self.transcripts["direct"] = SimPC.enumerate(self.mouse)
                self.pc_enumerations.append((t, "direct", self.transcripts["direct"]))
                self.pc_conn = "direct"
            elif self.route == "bridge" and self.alive and self.pc_never_configures:
                self.pc_conn = None  # the PC never finishes enumerating us (a host that gave up)
            elif self.route == "bridge" and self.alive:
                try:
                    self.transcripts["bridge"] = SimPC.enumerate(self.front)
                    self.pc_enumerations.append((t, "bridge", self.transcripts["bridge"]))
                    self.pc_conn = "bridge"
                except TimeoutError:
                    self.pc_ready_at = t + 100_000
        if self.mouse is not None:
            for ep in list(self.mouse.in_q):
                while True:
                    r = self.mouse.pop_in(ep)
                    if r is None:
                        break
                    q = self._enq.get(ep)
                    t_enq = q.pop(0) if q else t  # a report put in by someone else than move(): no stamp, no delay
                    if self.pc_conn == "bridge" and self.route == "bridge" and self.alive:
                        out, _ = self.b.mouse_in(t, ep, r)
                        self.pc_reports.append((t, "bridge", ep, out, r))
                        self.report_delay_us.append(t - t_enq)
                    elif self.pc_conn == "direct" and self.route == "bypass":
                        self.pc_reports.append((t, "direct", ep, r, r))
                        self.report_delay_us.append(t - t_enq)
                    else:
                        self.dropped += 1
        self.t += self.step_us

    def run(self, ms: float) -> None:
        end = self.t + int(ms * 1000)
        while self.t < end:
            self.step()

    def run_until(self, cond: Callable[["Rig"], bool], max_ms: int = 20000) -> bool:
        end = self.t + max_ms * 1000
        while self.t < end:
            if cond(self):
                return True
            self.step()
        return cond(self)

    def engage_fast(self, max_ms: int = 20000) -> bool:
        return self.run_until(lambda r: r.b.status(r.t).state == S_ASSIST, max_ms)

    def status(self):
        return self.b.status(self.t)

    def pc_out(self, ep: int, data: bytes) -> bool:
        """The PC writes to an interrupt OUT endpoint (HID++ shaped traffic): goes to the mouse if the path is up."""
        if self.pc_conn in ("direct", "bridge") and self.mouse is not None:
            self.mouse.interrupt_out(ep, data)
            return True
        return False

    # -- the person
    def move(self, dx: int, dy: int, buttons: int = 0, wheel: int = 0, pan: int = 0, ep: Optional[int] = None) -> None:
        ep = ep or self.mouse.motion_ep
        self.mouse.push_in(ep, self.mouse.pack(buttons, dx, dy, wheel, pan))
        self._enq.setdefault(ep, []).append(self.t)


@dataclass
class _Out:
    dx: int
    dy: int
    k: float


class BridgeAssist:
    """The bridge core behind the `tick(t, dx, dy, px, py, obj)` interface of the correction chain, so the closed-loop person
    simulator (assist/sim_user.py) can run THROUGH the bridge: report bytes in, report bytes out. The scene (the object relative to the
    cursor) goes over the real link frames. Time is offset so the rig can finish engaging first."""

    def __init__(
        self, asc, tremor, kind: str = "m16", hold_overlap: float = 0.75, ppc: float = 1.0, cfg: Optional[dict] = None, speed: str = "FS"
    ) -> None:
        from ..assist.chain import overlap_hold
        from ..assist.fixed import FixedParams as FP
        from ..assist.tremor_fixed import FixedTremorParams as FT

        asc2 = overlap_hold(asc, tremor, hold_overlap)
        self.mouse = SimMouse(kind, speed=speed)
        self.rig = Rig(self.mouse, cfg=cfg, asc=FP.from_params(asc2), tremor=FT.from_params(tremor), ppc=ppc)
        if not self.rig.engage_fast():
            raise RuntimeError(f"the bridge did not engage: {self.rig.status().reason}")
        self.t0 = ((self.rig.t // 1_000_000) + 2) * 1_000_000
        self.b, self.last_ms = self.rig.b, None
        self.ppc = ppc
        self.next_keepalive = 0
        self.seq = 0

    def _link(self, t: int, frames) -> None:
        for f in frames:
            self.seq = (self.seq + 1) & 0xFFFF
            f.seq = self.seq
            self.b.link_rx(t, P.pack_frame(f))

    def tick(self, t_us: int, dx: int, dy: int, px: float, py: float, obj) -> _Out:
        t = t_us + self.t0
        if self.last_ms is None:
            self.last_ms = t
        while self.last_ms + 1000 < t:  # the timer between samples (1 ms grid)
            self.last_ms += 1000
            self.b.poll(self.last_ms)
        self.last_ms = t
        if t >= self.next_keepalive:
            self.next_keepalive = t + 100_000
            self._link(t, [P.Frame(P.LK_HELLO)])
            if t % 1_000_000 < 100_000 or self.rig.module is None:
                self._link(t, self.rig.module.param_frames())
        self.b.poll(t)
        if obj is not None:
            ta = None if obj.t_appear_us is None else obj.t_appear_us + self.t0
            self._link(t, [P.scene_frame(t, [P.SceneObject(obj.id, obj.x - px, obj.y - py, obj.radius, ta)])])
        ox, oy = 0, 0
        if dx or dy:
            out, _ = self.b.mouse_in(t, 0x81, self.mouse.pack(0, dx, dy))
            ox, oy = self.mouse_xy(out)
        s = self.b.status(t)
        return _Out(ox, oy, s.k_q16 / 65536)

    def mouse_xy(self, report: bytes) -> tuple[int, int]:
        import struct

        if self.mouse.kind == "m16":
            _, x, y, _, _ = struct.unpack("<Bhhbb", report)
            return x, y
        if self.mouse.kind == "boot":
            _, x, y = struct.unpack("<Bbb", report)
            return x, y
        raise ValueError(self.mouse.kind)
