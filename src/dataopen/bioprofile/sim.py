"""A synthetic human for testing and demos: generates mouse motion and target observations with KNOWN ground truth so the estimators can
be checked against it. Movements are minimum-jerk submovements (a flick, a pause, a corrective submovement), tracking follows the target
with a delay plus an oscillatory micro-correction; reaction time, overshoot, speed and jitter are drawn from per-player parameters that
can drift over the session (fatigue). This is a model of the *shape* of human motion, not recorded human data."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .types import TargetObs

FRAME_US = 4167                    # 240 Hz detector frames


@dataclass
class SimPlayer:
    t_motor_ms: float = 230.0
    t_motor_sigma_ms: float = 25.0
    lowvis_extra_ms: float = 80.0           # slower reaction when the target is hard to see
    lowvis_detect_delay_ms: float = 40.0    # and the detector sees it later (a bias the engine cannot know)
    speed_factor: float = 1.0
    overshoot_mean: float = 0.06
    overshoot_sigma: float = 0.05
    final_err_deg: float = 0.25
    phase_lag_ms: float = 70.0
    jitter_hz: float = 8.0
    jitter_amp_deg: float = 0.10
    anticipation_rate: float = 0.03
    heading_err_prob: float = 0.0            # chance that the first movement starts in a wrong direction
    heading_err_deg: float = 60.0
    lapse_prob: float = 0.0                  # chance that the player does not react at all (the engine must call it a lapse)
    dpc: float = 0.02                       # degrees per mouse count
    # fatigue: drift per hour of session time
    t_motor_drift_ms_per_h: float = 0.0
    err_drift_deg_per_h: float = 0.0


@dataclass
class Truth:
    kind: str                               # flick | track | censored
    t_appear_us: int
    surprise: bool
    lowvis: bool
    d0: float = 0.0
    t_motor_ms: float = math.nan
    v_max: float = math.nan
    d_brake_frac: float = math.nan
    overshoot: float = math.nan
    heading_dev: float = 0.0
    undershoot: bool = False
    err_final: float = math.nan
    phase_lag_ms: float = math.nan
    jitter_hz: float = math.nan
    jitter_amp: float = math.nan
    anticipation: bool = False


def _mj(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


@dataclass
class SimStream:
    """Time-ordered events: ('mouse', t_us, dx, dy) with integer counts and ('target', t_us, TargetObs | None)."""
    events: list = field(default_factory=list)
    truth: list[Truth] = field(default_factory=list)
    duration_s: float = 0.0


def _smooth_path(rng: np.random.Generator):
    """A smooth 2-D target path p(s) with p(0) = 0 and speeds up to a few tens of deg/s."""
    f1, f2, ph1, ph2 = rng.uniform(0.3, 0.9), rng.uniform(1.0, 1.8), rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    a1, a2 = rng.uniform(4, 14), rng.uniform(1, 5)
    ang = rng.uniform(0, 2 * math.pi)
    u = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-u[1], u[0]])

    def p(s: np.ndarray) -> np.ndarray:
        s = np.clip(s, 0, None)
        w = a1 * (np.sin(2 * math.pi * f1 * s + ph1) - math.sin(ph1)) + a2 * (np.sin(2 * math.pi * f2 * s + ph2) - math.sin(ph2))
        return w[:, None] * u[None] + 0.4 * w[:, None] * n[None]
    return p, u


@dataclass
class _Ep:
    pos: np.ndarray                  # (N, 2) view at ms t0+1 .. t0+N
    tgt: np.ndarray                  # (N, 2) target in world angles
    appear_i: int                    # index of the first sample at which the target is on screen
    gone_i: int
    truth: Truth
    lowvis: bool


def simulate(player: SimPlayer, minutes: float = 10.0, seed: int = 0, mix: Optional[dict] = None) -> SimStream:
    """mix: probabilities of episode kinds: surprise (hand still), lowvis (surprise under bad visibility), track (already following),
    premove (hand moving when a far target appears: must be left out of the reaction-time statistics)."""
    mix = {"surprise": 0.55, "lowvis": 0.15, "track": 0.2, "premove": 0.1, **(mix or {})}
    kinds, probs = list(mix), np.array(list(mix.values()), dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    out = SimStream()
    view, carry, t_ms = np.zeros(2), np.zeros(2), 0.0
    end_ms = minutes * 60_000.0
    dpc = player.dpc

    def emit(seg: np.ndarray, t0: float) -> None:
        nonlocal carry, view
        cum = np.cumsum(np.diff(np.vstack([view[None], seg]), axis=0), axis=0) / dpc + carry   # position in counts + remainder
        q = np.round(cum)
        counts = np.diff(np.vstack([np.zeros((1, 2)), q]), axis=0)
        carry = cum[-1] - q[-1]
        for i in np.nonzero((counts != 0).any(axis=1))[0]:
            out.events.append(("mouse", int((t0 + i + 1) * 1000), int(counts[i, 0]), int(counts[i, 1])))
        view = seg[-1].copy()

    while t_ms < end_ms:
        hours = t_ms / 3.6e6
        idle = int(rng.uniform(900, 2200))
        seg = np.repeat(view[None], idle, axis=0)
        for j in rng.integers(0, idle, size=int(rng.integers(0, 3))):
            seg[j:] += rng.choice([-1, 1], size=2) * dpc * (rng.random(2) < 0.5)
        emit(seg, t_ms)
        t_ms += idle
        kind = str(rng.choice(kinds, p=probs))
        ep = {"track": _track, "premove": _premove}.get(kind, _flick)(player, rng, kind, view.copy(), hours)
        emit(ep.pos, t_ms)
        ep.truth.t_appear_us = int((t_ms + ep.appear_i + 1) * 1000)
        # detector frames on their own clock; a low-visibility target is detected later (a bias the engine cannot know)
        delay = player.lowvis_detect_delay_ms if ep.lowvis else 0.0
        vis = 0.25 if ep.lowvis else 0.9
        first_us = int((t_ms + ep.appear_i + 1 + delay) * 1000)
        t_us = (first_us // FRAME_US + 1) * FRAME_US
        last_us = int((t_ms + ep.gone_i + 1) * 1000)
        while t_us < last_us:
            i = int(t_us / 1000.0 - t_ms) - 1
            e = ep.tgt[i] - ep.pos[i] + rng.normal(0, 0.03, 2)
            out.events.append(("target", t_us, TargetObs(t_us, float(e[0]), float(e[1]), vis, ep.lowvis, 0.5)))
            t_us += FRAME_US
        out.events.append(("target", t_us, None))
        out.truth.append(ep.truth)
        t_ms += len(ep.pos)
        view = ep.pos[-1].copy()
    out.events.sort(key=lambda e: e[1])
    out.duration_s = t_ms / 1000.0
    return out


def _flick(p: SimPlayer, rng: np.random.Generator, kind: str, view0: np.ndarray, hours: float) -> _Ep:
    lowvis = kind == "lowvis"
    anticip = rng.random() < p.anticipation_rate
    t_motor = float(rng.uniform(30, 85)) if anticip else max(
        110.0, rng.normal(p.t_motor_ms + p.t_motor_drift_ms_per_h * hours + (p.lowvis_extra_ms if lowvis else 0.0),
                          p.t_motor_sigma_ms * (1.4 if lowvis else 1.0)))
    sigma_f = max(0.02, p.final_err_deg + p.err_drift_deg_per_h * hours)
    lag, track_ms = max(10.0, rng.normal(p.phase_lag_ms, 6.0)), float(rng.uniform(500, 1400))
    jf, ja, jph = max(2.0, rng.normal(p.jitter_hz, 0.5)), max(0.02, rng.normal(p.jitter_amp_deg, 0.015)), rng.uniform(0, 6.28)
    path, u = _smooth_path(rng)
    ju = np.array([math.cos(math.atan2(u[1], u[0]) + 1.0), math.sin(math.atan2(u[1], u[0]) + 1.0)])
    d0 = float(rng.uniform(14, 55))
    tgt0 = view0 + d0 * u
    eps = float(np.clip(rng.normal(p.overshoot_mean, p.overshoot_sigma), -0.5, 0.5))
    a1, t1 = d0 * (1 + eps), (0.10 + 0.0035 * d0) / p.speed_factor
    dev = p.heading_err_deg * (1 if rng.random() < 0.5 else -1) if rng.random() < p.heading_err_prob else 0.0
    um = np.array([[math.cos(math.radians(dev)), -math.sin(math.radians(dev))],
                   [math.sin(math.radians(dev)), math.cos(math.radians(dev))]]) @ u      # direction the hand actually starts in
    lapse = rng.random() < p.lapse_prob
    pause, t2 = float(rng.uniform(0.05, 0.11)), float(rng.uniform(0.07, 0.11))
    r = rng.normal(0, sigma_f, 2)
    corr = (tgt0 - (view0 + a1 * um)) + r
    move_ms = (t1 + pause + t2) * 1000
    if lapse:                                           # no reaction: the hand stays where it is while the target is visible
        n = 1500
        tr = Truth("lapse", 0, True, lowvis, d0=d0)
        return _Ep(np.repeat(view0[None], n, axis=0), np.repeat(tgt0[None], n, axis=0), 0, n - 50, tr, lowvis)
    n = int(t_motor + move_ms + 40 + track_ms + 250)
    ms = np.arange(1, n + 1, dtype=float)
    s = (ms - 1 - t_motor) / 1000.0                     # sample ms is absolute t0 + ms; the target is on screen from t0 + 1
    pos = view0[None] + (a1 * _mj(s / t1))[:, None] * um[None] + _mj((s - t1 - pause) / t2)[:, None] * corr[None]
    trk = (ms - 1 - (t_motor + move_ms + 40)) / 1000.0
    tpath = path(trk)
    tgt = tgt0[None] + tpath
    follow = path(trk - lag / 1000.0) * (trk > lag / 1000.0)[:, None]
    pos = pos + follow + (ja * np.sin(2 * math.pi * jf * trk + jph))[:, None] * ju[None] * (trk > 0)[:, None]
    d_brake = float(np.hypot(*(tgt0 - (view0 + 0.5 * a1 * um)))) / d0        # distance left at the speed peak
    tr = Truth("flick", 0, True, lowvis, d0=d0, t_motor_ms=t_motor, v_max=1.875 * a1 / t1, d_brake_frac=d_brake,
               overshoot=max(0.0, eps), heading_dev=abs(dev), undershoot=a1 < 0.8 * d0, err_final=float(np.hypot(*r)),
               phase_lag_ms=lag, jitter_hz=jf, jitter_amp=ja, anticipation=anticip)
    return _Ep(pos, tgt, 0, n - 150, tr, lowvis)


def _track(p: SimPlayer, rng: np.random.Generator, kind: str, view0: np.ndarray, hours: float) -> _Ep:
    lag = max(10.0, rng.normal(p.phase_lag_ms, 6.0))
    jf, ja, jph = max(2.0, rng.normal(p.jitter_hz, 0.5)), max(0.02, rng.normal(p.jitter_amp_deg, 0.015)), rng.uniform(0, 6.28)
    path, u = _smooth_path(rng)
    ju = np.array([math.cos(math.atan2(u[1], u[0]) + 1.0), math.sin(math.atan2(u[1], u[0]) + 1.0)])
    pre, dur = 700, int(rng.uniform(900, 1600))
    n = pre + dur
    s = (np.arange(1, n + 1) - pre) / 1000.0 + 0.7                  # the path already runs before the target "re-appears"
    tgt = view0[None] + path(s) - path(np.array([0.0]))
    pos = view0[None] + path(s - lag / 1000.0) + (ja * np.sin(2 * math.pi * jf * s + jph))[:, None] * ju[None] \
        + rng.normal(0, 0.003, (n, 2))
    off = (view0 + 0.02) - pos[0]
    pos += off[None]
    tgt += off[None] + (rng.uniform(0.8, 2.2) * u)[None] * 0
    tr = Truth("track", 0, False, False, d0=float(np.hypot(*(tgt[pre] - pos[pre]))), phase_lag_ms=lag, jitter_hz=jf, jitter_amp=ja)
    return _Ep(pos, tgt, pre, n - 100, tr, False)


def _premove(p: SimPlayer, rng: np.random.Generator, kind: str, view0: np.ndarray, hours: float) -> _Ep:
    """The hand is already sweeping (a speed far above 'still') when a distant target appears."""
    ang = rng.uniform(0, 2 * math.pi)
    u = np.array([math.cos(ang), math.sin(ang)])
    n = 800
    ms = np.arange(1, n + 1, dtype=float)
    pos = view0[None] + (40.0 * _mj(ms / 600.0))[:, None] * u[None]
    tgt = np.repeat((view0 + 60.0 * np.array([-u[1], u[0]]))[None], n, axis=0)
    tr = Truth("censored", 0, False, False, d0=60.0)
    return _Ep(pos, tgt, 260, n - 50, tr, False)
