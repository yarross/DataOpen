"""Closed-loop test bench: a simulated person reaches for a target with and without ASC in the loop.

The person is a feedback controller built from minimum-jerk submovements: after a reaction time they plan a movement to what they SEE
(the cursor as it was `vis_delay_ms` ago), execute it open loop (counts out of the hand regardless of what the cursor does), pause, and
correct from what they see, until the cursor looks inside the target. Overshoot (gain error of the first movement) and tremor (an
8 Hz oscillation added to the hand's motion) are per-person. The profile ASC reads is produced by the real BioProfile engine from a
simulated session of the same person, so the whole chain (hand -> profile -> params -> correction -> cursor) is exercised.
This is a model of the structure of the task, not recorded human data: it can show that the mechanism does what it should, not how
people will feel about it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..bioprofile.engine import BioProfileEngine, EngineConfig
from ..bioprofile.profile import ProfileView
from ..bioprofile.sim import SimPlayer, simulate
from .model import AdaptiveSensitivity
from .types import ObjectOfInterest


@dataclass
class Persona:
    name: str
    t_motor_ms: float = 230.0
    t_motor_sigma_ms: float = 25.0
    overshoot_mean: float = 0.05
    overshoot_sigma: float = 0.04
    final_err_deg: float = 0.25          # accuracy of the corrective movements (feeds the BioProfile 'miss' statistics)
    tremor_deg: float = 0.10             # amplitude of the oscillation added to the hand's motion
    tremor_hz: float = 8.0
    speed_factor: float = 1.0
    dpc: float = 0.02                    # degrees per count
    fatigue_t_ms_per_h: float = 0.0
    fatigue_err_per_h: float = 0.0

    def sim_player(self) -> SimPlayer:
        return SimPlayer(t_motor_ms=self.t_motor_ms, t_motor_sigma_ms=self.t_motor_sigma_ms, overshoot_mean=self.overshoot_mean,
                         overshoot_sigma=self.overshoot_sigma, final_err_deg=self.final_err_deg, jitter_hz=self.tremor_hz,
                         jitter_amp_deg=self.tremor_deg, speed_factor=self.speed_factor, dpc=self.dpc,
                         t_motor_drift_ms_per_h=self.fatigue_t_ms_per_h, err_drift_deg_per_h=self.fatigue_err_per_h)


PERSONAS = {
    "steady": Persona("steady"),
    "overshooter": Persona("overshooter", overshoot_mean=0.30, overshoot_sigma=0.08, final_err_deg=0.8, tremor_deg=0.12),
    "tremor": Persona("tremor", t_motor_ms=290.0, t_motor_sigma_ms=45.0, overshoot_mean=0.12, overshoot_sigma=0.08, final_err_deg=0.6,
                      tremor_deg=0.45, tremor_hz=7.0, speed_factor=0.7),
}


def build_profile(persona: Persona, minutes: float = 8.0, seed: int = 0) -> ProfileView:
    """Run the BioProfile engine on a simulated session of this person and read the profile back through the public view."""
    p = persona.sim_player()
    s = simulate(p, minutes, seed)
    eng = BioProfileEngine(EngineConfig(deg_per_count=p.dpc))
    for e in s.events:
        if e[0] == "mouse":
            eng.on_mouse(e[1], e[2], e[3])
        else:
            eng.on_target(e[1], e[2])
    eng.advance(int(s.duration_s * 1e6) + 3_000_000)
    return ProfileView(eng.snapshot(clean=True).pack())


@dataclass
class Trial:
    acquired: bool = False
    t_acquire_ms: float = math.nan
    overshoot_px: float = 0.0
    final_err_px: float = math.nan
    hold_rms_px: float = math.nan
    n_sub: int = 0
    k_mean_moving: float = 1.0
    k_min: float = 1.0
    frac_assisted: float = 0.0           # share of moving ticks with K < 0.9
    guard_violations: int = 0            # K != 1 before the person moved
    amp_violations: int = 0              # |out| > |raw| or other sign
    k_trace: list = field(default_factory=list)


def _mj(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _mj_vel(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return 30 * tau**2 - 60 * tau**3 + 30 * tau**4


def run_trial(persona: Persona, asc: Optional[AdaptiveSensitivity], rng: np.random.Generator, dist_px: float = 600.0,
              radius_px: float = 30.0, px_per_count: float = 1.0, vis_delay_ms: int = 90, hold_ms: int = 500, max_ms: int = 3500,
              trace: bool = False, record: Optional[list] = None, adapt: bool = True) -> Trial:
    """One reach along +x from the origin to a target `dist_px` away. With `asc` None the cursor moves 1:1 with the hand."""
    out = Trial()
    pos = np.zeros(2)
    hist = [pos.copy()] * (vis_delay_ms + 1)                          # what the person sees is the cursor `vis_delay_ms` ago
    tgt = np.array([dist_px, 0.0])
    obj = ObjectOfInterest(1, float(dist_px), 0.0, radius_px, t_appear_us=0)
    t_motor = max(110.0, rng.normal(persona.t_motor_ms, persona.t_motor_sigma_ms))
    tremor_counts = persona.tremor_deg / persona.dpc
    tph = rng.uniform(0, 2 * math.pi)
    carry = np.zeros(2)
    sub: Optional[dict] = None
    rho = 1.0                                   # the person's estimate of how much of an intended movement the cursor delivers
    next_plan = t_motor
    in_ms, k_sum, k_n, assisted = 0, 0.0, 0, 0
    acquire_t: Optional[int] = None
    still_run = 0
    first_motion = None
    hold_pos: list = []
    max_proj = 0.0
    n_ticks = max_ms + hold_ms
    for ms in range(1, n_ticks + 1):
        seen = hist[0]
        # --- the hand (counts per ms, open loop within a submovement)
        raw = np.zeros(2)
        if sub is None and ms >= next_plan and out.n_sub < 8:
            err = tgt - seen
            dist = float(np.hypot(*err))
            if dist > 0.6 * radius_px or out.n_sub == 0:
                eps = rng.normal(persona.overshoot_mean if out.n_sub == 0 else 0.0,
                                 persona.overshoot_sigma if out.n_sub == 0 else persona.overshoot_sigma * 0.5)
                eps = float(np.clip(eps, -0.4, 0.8))
                gain = min(max(rho, 0.15), 1.0) if adapt else 1.0
                amp = dist * (1 + eps) / (px_per_count * gain)
                dur = max(60.0, (0.10 + 0.0035 * dist * persona.dpc * px_per_count) / persona.speed_factor * 1000.0)
                if out.n_sub > 0:
                    dur = max(60.0, dur * 0.6)
                sub = {"t0": ms, "dur": dur, "amp": amp, "dir": err / max(dist, 1e-9), "p0": pos.copy()}
                out.n_sub += 1
            else:
                next_plan = ms + 20                                   # looks done: keep watching
        if sub is not None:
            tau = (ms - sub["t0"]) / sub["dur"]
            if tau >= 1.0:
                if adapt and sub["amp"] > 8:                           # people adapt to a changed gain: what did that movement deliver?
                    got = float((pos - sub["p0"]) @ sub["dir"]) / (sub["amp"] * px_per_count)
                    rho = 0.5 * rho + 0.5 * min(max(got, 0.05), 1.3)
                sub = None
                next_plan = ms + rng.uniform(60, 110) + vis_delay_ms
            else:
                rate = sub["amp"] * _mj_vel(tau) / sub["dur"]            # counts per ms along the movement direction
                raw = rate * sub["dir"]
        # tremor rides on the hand's motion at all times once the person is acting (it exists at rest too)
        if ms >= t_motor - 50:
            raw = raw + np.array([math.cos(2 * math.pi * persona.tremor_hz * ms / 1000.0 + tph),
                                  0.4 * math.sin(2 * math.pi * persona.tremor_hz * ms / 1000.0 + tph)]) \
                * tremor_counts * 2 * math.pi * persona.tremor_hz / 1000.0
        total = raw + carry
        ir = np.round(total).astype(int)
        carry = total - ir
        # --- the correction in the loop
        if record is not None:
            record.append((ms * 1000, int(ir[0]), int(ir[1]), float(pos[0]), float(pos[1])))
        if asc is None:
            odx, ody, k = int(ir[0]), int(ir[1]), 1.0
        else:
            tk = asc.tick(ms * 1000, int(ir[0]), int(ir[1]), float(pos[0]), float(pos[1]), obj)
            odx, ody, k = tk.dx, tk.dy, tk.k
            if abs(odx) > abs(ir[0]) or abs(ody) > abs(ir[1]) or (odx != 0 and (odx > 0) != (ir[0] > 0)) \
                    or (ody != 0 and (ody > 0) != (ir[1] > 0)):
                out.amp_violations += 1
        if first_motion is None and (ir != 0).any() and np.hypot(*raw) > 0.5:
            first_motion = ms
        if first_motion is None and k != 1.0:
            out.guard_violations += 1
        pos = pos + np.array([odx, ody]) * px_per_count
        hist.append(pos.copy())
        hist.pop(0)
        max_proj = max(max_proj, pos[0] - dist_px)
        if first_motion is not None and acquire_t is None:
            in_ms += 1
            k_sum += k
            k_n += 1
            assisted += k < 0.9
            if trace:
                out.k_trace.append(k)
            if math.hypot(*(pos - tgt)) <= radius_px:                    # selected by dwelling inside the target
                still_run += 1
                if still_run >= 80:
                    acquire_t = ms - 80
            else:
                still_run = 0
        if acquire_t is not None and ms >= acquire_t + 80 + 100:
            hold_pos.append(pos.copy())
        out.k_min = min(out.k_min, k)
        if acquire_t is not None and ms >= acquire_t + 80 + hold_ms:
            break
    out.acquired = acquire_t is not None
    out.t_acquire_ms = float(acquire_t - (first_motion or 0)) if out.acquired else math.nan
    out.overshoot_px = max(0.0, max_proj)
    out.final_err_px = float(np.hypot(*(pos - tgt)))
    if len(hold_pos) > 20:
        h = np.array(hold_pos)
        out.hold_rms_px = float(np.sqrt(((h - h.mean(axis=0)) ** 2).sum(axis=1).mean()))
    out.k_mean_moving = k_sum / k_n if k_n else 1.0
    out.frac_assisted = assisted / k_n if k_n else 0.0
    return out


def summarize(trials: list[Trial], radius_px: float) -> dict:
    ok = [t for t in trials if t.acquired]
    f = lambda a: float(np.nanmean(a)) if len(a) else math.nan   # noqa: E731
    return {"n": len(trials), "acquired": len(ok) / max(1, len(trials)),
            "t_acquire_ms": f([t.t_acquire_ms for t in ok]),
            "overshoot_rate": float(np.mean([t.overshoot_px > radius_px for t in trials])),
            "overshoot_px": f([t.overshoot_px for t in trials]),
            "final_err_px": f([t.final_err_px for t in trials]),
            "hold_rms_px": f([t.hold_rms_px for t in ok]),
            "k_mean": f([t.k_mean_moving for t in trials]), "assisted": f([t.frac_assisted for t in trials]),
            "n_sub": f([t.n_sub for t in trials]),
            "guard_violations": int(sum(t.guard_violations for t in trials)),
            "amp_violations": int(sum(t.amp_violations for t in trials))}


def compare(persona: Persona, n: int = 60, seed: int = 0, params=None, view: Optional[ProfileView] = None, dist_px: float = 600.0,
            radius_px: float = 30.0) -> tuple[dict, dict, ProfileView]:
    """The same hand (same random draws) with and without assistance."""
    from .params import AscParams
    view = view or build_profile(persona, seed=seed)
    params = params or AscParams.from_view(view)
    base, assisted = [], []
    for i in range(n):
        base.append(run_trial(persona, None, np.random.default_rng(seed * 1000 + i), dist_px, radius_px))
        asc = AdaptiveSensitivity(params)
        assisted.append(run_trial(persona, asc, np.random.default_rng(seed * 1000 + i), dist_px, radius_px))
    return summarize(base, radius_px), summarize(assisted, radius_px), view


def compare_all(persona: Persona, n: int = 40, seed: int = 1, view: Optional[ProfileView] = None, dist_px: float = 600.0,
                radius_px: float = 30.0) -> dict:
    """The same hand under four conditions: nothing, ASC alone, tremor suppression alone, the chain (ASC then tremor suppression)."""
    from .chain import AssistChain
    from .params import AscParams
    from .tremor import TremorParams
    view = view or build_profile(persona, seed=seed)
    ap, tp = AscParams.from_view(view), TremorParams.from_view(view)
    makers = {"none": lambda: None,
              "asc": lambda: AssistChain.build(ap, TremorParams.disabled(), "float"),
              "tremor": lambda: AssistChain.build(AscParams.disabled(), tp, "float"),
              "chain": lambda: AssistChain.build(ap, tp, "float")}
    out = {}
    for name, mk in makers.items():
        out[name] = summarize([run_trial(persona, mk(), np.random.default_rng(seed * 1000 + i), dist_px, radius_px) for i in range(n)],
                              radius_px)
    return out
