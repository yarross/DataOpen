"""Measurements on the simulation (docs/LATENCY.md). Everything here runs in VIRTUAL time: a number does not depend on the speed of the
machine the test runs on. What it can not know (RK3588, the NPU, the real USB host, the monitor) is not in here; see `budget.py` for the
stages and for the basis of each number.

    pipe_timeline     frames -> a queue -> the detector -> the tracker's confirmation (discrete events; the same queue rules as
                      runtime.frames.QueueSource: capacity, producer-side drop, `drain_latest`)
    core_report_delay the REAL C bridge core behind a simulated mouse: time from the mouse report to the report the PC gets
    poll_resample     the cascade of two USB poll grids (the bridge polls the mouse, the PC polls the bridge)
    chain_delay       the REAL core's ASC + tremor chain: how late the output follows a deliberate reach
    scene_effect      the REAL core with a scene of a given age: how much help is left (compensation and the 100 ms TTL)
"""

from __future__ import annotations

import math
import random
import struct
from collections import deque
from dataclasses import dataclass
from typing import Optional


def _pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(q / 100.0 * len(s)))] if s else 0.0


def stats(v: list[float]) -> dict[str, float]:
    return {"n": len(v), "min": min(v) if v else 0.0, "mean": sum(v) / len(v) if v else 0.0, "p50": _pct(v, 50), "p95": _pct(v, 95),
            "p99": _pct(v, 99), "max": max(v) if v else 0.0}


# ------------------------------------------------------------------------------------------------ frames -> detector -> confirmed
@dataclass(frozen=True)
class Pipe:
    """One detector service. `frame_ms`: the video frame period. `infer_ms`: inference + decode. `update_ms`: the tracker. `policy`: 'fifo'
    (the oldest frame first: what `UiService` does by default) or 'latest' (`latest_only=True`). `capacity`: the frame queue (QueueSource)."""

    frame_ms: float
    infer_ms: float
    update_ms: float = 0.3
    policy: str = "fifo"
    capacity: int = 4
    confirm_hits: int = 2


def pipe_timeline(p: Pipe, n_frames: int = 400, first_k0: int = 40, n_k0: int = 200) -> dict[str, list[float]]:
    """Frames become ready every `frame_ms`; one detector takes them one by one. For an object that is first in frame `k0` (for many `k0`,
    so the detector's phase against the frames varies) report, in milliseconds counted from the moment frame `k0` was ready:
    `wait` (until the detector starts on the first frame that holds the object), `infer`, `update`, `confirm` (the extra detection cycles the
    tracker needs: `confirm_hits - 1`), `total` (the target is confirmed) and `taken_age` (how old the first used frame was when taken)."""
    if p.policy not in ("fifo", "latest"):
        raise ValueError(p.policy)
    ready = [k * p.frame_ms for k in range(n_frames)]
    q: deque[int] = deque()
    done: list[tuple[int, float, float]] = []  # (frame, start, finish)
    t, a, dropped = 0.0, 0, 0
    while True:
        while a < n_frames and ready[a] <= t:
            if len(q) >= p.capacity:
                dropped += 1  # the producer never blocks: the NEW frame is dropped
            else:
                q.append(a)
            a += 1
        if not q:
            if a >= n_frames:
                break
            t = ready[a]
            continue
        if p.policy == "latest":
            k = q.pop()
            q.clear()
        else:
            k = q.popleft()
        done.append((k, t, t + p.infer_ms + p.update_ms))
        t += p.infer_ms + p.update_ms
    out: dict[str, list[float]] = {"wait": [], "infer": [], "update": [], "confirm": [], "total": [], "taken_age": []}
    for k0 in range(first_k0, first_k0 + n_k0):
        seen = [d for d in done if d[0] >= k0]
        if len(seen) < p.confirm_hits:
            continue
        k1, s1, f1 = seen[0]
        fc = seen[p.confirm_hits - 1][2]
        out["wait"].append(s1 - ready[k0])
        out["infer"].append(p.infer_ms)
        out["update"].append(p.update_ms)
        out["confirm"].append(fc - f1)
        out["total"].append(fc - ready[k0])
        out["taken_age"].append(s1 - ready[k1])
    return out


# ------------------------------------------------------------------------------------------------ the bridge core, report by report
def _rig(persona: str, speed: str = "FS", interval: int = 1, kind: str = "m16", seed: int = 1):
    from ..assist.fixed import FixedParams
    from ..assist.params import AscParams
    from ..assist.sim_user import PERSONAS, build_profile
    from ..assist.tremor import TremorParams
    from ..assist.tremor_fixed import FixedTremorParams
    from ..bridge.sim import Rig
    from ..bridge.sim_usb import SimMouse

    v = build_profile(PERSONAS[persona], seed=seed)
    ap, tp = AscParams.from_view(v), TremorParams.from_view(v)
    r = Rig(SimMouse(kind, speed=speed, interval=interval), asc=FixedParams.from_params(ap), tremor=FixedTremorParams.from_params(tp))
    if not r.engage_fast():
        raise RuntimeError(f"the bridge did not engage: {r.status().reason}")
    r.run(300)
    return r, ap, tp


def core_report_delay(speed: str = "FS", interval: int = 1, persona: str = "tremor", n: int = 600) -> dict:
    """What the core itself adds: the time between the moment the mouse report is there and the step that handed the changed report to the
    PC. The core is called per report and answers synchronously, so in the simulation this is 0 (the step grid is the only resolution)."""
    r, _, _ = _rig(persona, speed, interval)
    r.report_delay_us.clear()
    for k in range(n):
        r.move(2 + (k % 3), (k % 2))
        r.step()
    d = [x / 1000.0 for x in r.report_delay_us]
    return {"speed": speed, "interval": interval, "reports": len(d), "delay_ms": stats(d), "step_ms": r.step_us / 1000.0}


# ------------------------------------------------------------------------------------------------ two poll grids in cascade
def poll_resample(mouse_hz: float, trials: int = 4000, seed: int = 1, bridge_poll_hz: Optional[float] = None) -> dict:
    """The bridge polls the mouse on a grid of `1/mouse_hz`; the PC polls the bridge's endpoint on a grid of the same period (the bridge
    presents the mouse's own `bInterval`). A report that is ready at a random time waits for the bridge's poll and then for the PC's poll;
    without the bridge it waits for the PC's poll only. Result: the added delay in ms, `added = with_bridge - direct`. `bridge_poll_hz`:
    the bridge polls the mouse faster than the mouse's own interval (not what the USB spec asks of a host; see docs/LATENCY.md)."""
    rng = random.Random(seed)
    q = 1000.0 / mouse_hz  # the PC's poll period of the endpoint it sees (also the period without the bridge)
    p = 1000.0 / (bridge_poll_hz or mouse_hz)
    added, direct, bridged = [], [], []
    for _ in range(trials):
        t_ready = rng.random() * 100.0
        ph_b, ph_pc = rng.random() * p, rng.random() * q
        t_b = t_ready + ((ph_b - t_ready) % p)  # the bridge's poll that sees the report
        d_direct = (ph_pc - t_ready) % q
        d_bridge = (t_b - t_ready) + ((ph_pc - t_b) % q)
        direct.append(d_direct)
        bridged.append(d_bridge)
        added.append(d_bridge - d_direct)
    return {"mouse_hz": mouse_hz, "bridge_poll_hz": bridge_poll_hz or mouse_hz, "period_ms": q, "added_ms": stats(added),
            "direct_ms": stats(direct), "bridged_ms": stats(bridged)}


# ------------------------------------------------------------------------------------------------ the chain: ASC + tremor
def _reach(dist: float, dur_ms: float, k: int) -> float:
    tau = min(k / dur_ms, 1.0)
    return dist * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)  # minimum-jerk: how a deliberate reach is shaped


def chain_delay(persona: str = "tremor", amp_counts: float = 0.0, tremor_hz: float = 6.0, dist: float = 600.0, dur_ms: float = 600.0) -> dict:
    """A minimum-jerk reach (plus an optional tremor sinusoid) goes through the real core with no scene (K = 1: only the tremor stage acts).
    For the fractions 10/25/50/75/90 % of the whole output: how many ms later the output reaches that fraction than the input does. Also
    `attenuation`: how much of the net motion the chain took away. The tremor filter is one-sided (it can only remove), so this is the price
    of removing involuntary motion, not an error."""
    r, _, tp = _rig(persona)
    n0 = len(r.pc_reports)
    ci, cin = 0, []
    prev = carry = 0.0
    for k in range(int(dur_ms) + 400):
        pos = _reach(dist, dur_ms, k) + amp_counts * math.sin(2 * math.pi * tremor_hz * k / 1000.0)
        carry += pos - prev
        prev = pos
        i = int(round(carry))
        carry -= i
        r.move(i, 0)
        r.step()
        ci += i
        cin.append(ci)
    co, c = [], 0
    for _t, _route, _ep, out, _rr in r.pc_reports[n0:]:
        c += struct.unpack_from("<h", out, 1)[0]
        co.append(c)

    def t_at(cum: list[float], frac: float) -> float:
        goal = frac * cum[-1]
        for i, v in enumerate(cum):
            if v >= goal:
                return (i - 1) + (goal - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9) if i else 0.0
        return float(len(cum))

    fr = (0.1, 0.25, 0.5, 0.75, 0.9)
    delays = {f: t_at(co, f) - t_at(cin, f) for f in fr}
    return {"persona": persona, "tremor_filter": bool(tp.enabled), "in_counts": cin[-1], "out_counts": co[-1],
            "attenuation": 1.0 - co[-1] / cin[-1] if cin[-1] else 0.0, "delay_ms": delays,
            "max_deficit_counts": max(a - b for a, b in zip(cin, co))}


# ------------------------------------------------------------------------------------------------ a scene of a given age
def scene_effect(age_ms: float, refresh_ms: int = 20, persona: str = "overshooter", dist: float = 700.0, target: float = 600.0,
                 radius: float = 24.0, dur_ms: int = 450) -> dict:
    """A reach towards a target at `target` counts. A scene (the target relative to the pointer AT CAPTURE TIME) is sent every `refresh_ms`,
    each one captured `age_ms` before it is sent. Result: the lowest K (1 = no help), the number of ticks with K < 0.9, where the pointer
    ended, when K first dropped below 0.9, and the steepest FALL of K per second (the core slews K). A scene older than the core's TTL is never used."""
    from ..bridge import protocol as P

    r, _, _ = _rig(persona)
    cum0 = r.status().cum_x
    hist: list[tuple[int, int]] = []
    ks: list[float] = []
    prev = carry = 0.0
    for k in range(dur_ms + 200):
        pos = _reach(dist, dur_ms, k)
        carry += pos - prev
        prev = pos
        i = int(round(carry))
        carry -= i
        r.move(i, 0)
        r.step()
        st = r.status()
        hist.append((r.t, st.cum_x - cum0))
        ks.append(st.k_q16 / 65536.0)
        if k % refresh_ms == 0:
            t_cap = r.t - int(age_ms * 1000)
            cum_cap = next((c for (t, c) in reversed(hist) if t <= t_cap), 0)
            r.module.send(r.t, [P.scene_frame(t_cap, [P.SceneObject(1, target - cum_cap, 0.0, radius, None)])])
    per_s = 1000.0 / (r.step_us / 1000.0)
    fall = max((a - b for a, b in zip(ks, ks[1:])), default=0.0) * per_s  # K going DOWN (the core slews it); a closed guard resets it to 1
    first = next((i for i, x in enumerate(ks) if x < 0.9), None)
    return {"age_ms": age_ms, "min_k": min(ks), "ticks_braked": sum(1 for x in ks if x < 0.9), "end_counts": hist[-1][1],
            "ttl_ms": r.b.cfg.scene_ttl_ms, "max_fall_per_s": fall, "first_brake_ms": None if first is None else first * r.step_us / 1000.0}
