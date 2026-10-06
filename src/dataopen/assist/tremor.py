"""Tremor suppression (docs/ASSIST.md): a band-limited, ONE-SIDED filter that only takes tremor away from the person's own motion.

Why one-sided. A linear filter would keep ringing after the hand stops and so would emit counts out of nothing; here the output is the
input pulled toward its low-frequency part, then clamped between zero and the input itself. Per axis and per tick: |out| <= |in|, the
same sign or zero, nothing at all when the input is zero, and at most `trim_cap` counts removed from any single report (never more
than the person's own tremor velocity). The module knows nothing about objects or screens: only dx, dy and the profile.

    lp   = two cascaded one-pole low-passes at fc (about 0.4 x the tremor frequency, 1.5 .. 4 Hz): the intended, slow motion
    hp   = x - lp
    band = hp through two one-pole low-passes at 16 Hz: what lies in 3..16 Hz, the physiological tremor band
    r    = E[band] / (E[band] + w * E[lp] + eps): how much of the current motion is tremor (a flick makes r ~ 0, a hold r ~ 1)
    s    = s_max * smoothstep((r - r_lo) / (r_hi - r_lo)) * (1 - smoothstep(|x| / v_t - big_lo) / (big_hi - big_lo)))
    y    = clamp(x - s * hp, between 0 and x), |x - y| <= trim_cap
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..bioprofile.profile import ProfileView
from .model import smoothstep


@dataclass(frozen=True)
class TremorConfig:
    f_band_hi_hz: float = 16.0
    fc_ratio: float = 0.4                # fc = ratio * tremor frequency ...
    fc_min_hz: float = 1.5
    fc_max_hz: float = 4.0               # ... clamped to this range
    default_f_hz: float = 8.0
    energy_tau_ms: float = 100.0
    lp_weight: float = 1.0               # intended motion counts this much against tremor in the tremor-ness ratio r
    eps: float = 0.0025                  # (counts/ms)^2: below this the signal is quantization noise, not tremor
    r_lo: float = 0.5
    r_hi: float = 0.9
    big_lo: float = 1.0                  # an input larger than this many tremor-velocity amplitudes cannot be tremor: ...
    big_hi: float = 3.0                  # ... suppression is fully off from here
    s_cap: float = 0.9
    amp_lo_deg: float = 0.08             # tremor amplitude where suppression starts (below: filter off)
    amp_hi_deg: float = 0.40             # ... and where it reaches s_cap
    cap_gain: float = 1.5                # trim_cap = ceil(cap_gain * tremor velocity amplitude, counts per report)
    reset_ms: int = 300                  # this long without any input: all state is forgotten (zeros between sparse reports are not idle)
    min_confidence_n: int = 12


@dataclass(frozen=True)
class TremorParams:
    enabled: bool = False
    a_lp: float = 0.0                    # one-pole coefficient 1 - exp(-2 pi fc dt) at dt = 1 ms
    a_band: float = 0.0
    a_e: float = 0.0
    s_max: float = 0.0
    trim_cap: int = 1
    v_t: float = 1.0                     # the person's tremor velocity amplitude, counts per report
    cfg: TremorConfig = field(default_factory=TremorConfig)

    @staticmethod
    def disabled(cfg: Optional[TremorConfig] = None) -> "TremorParams":
        return TremorParams(cfg=cfg or TremorConfig())

    @staticmethod
    def from_view(view: ProfileView, cfg: Optional[TremorConfig] = None, deg_per_count: Optional[float] = None) -> "TremorParams":
        """Personalize from the jitter part of a BioProfile; without enough evidence or with a barely visible tremor the filter is OFF."""
        cfg = cfg or TremorConfig()
        dpc = deg_per_count if deg_per_count else view.deg_per_count
        sure = view.confident("jitter_amp", cfg.min_confidence_n) and view.confident("jitter_hz", cfg.min_confidence_n)
        if not dpc or dpc <= 0 or not sure:
            return TremorParams.disabled(cfg)
        amp_deg = view.stat("jitter_amp").median
        f = view.stat("jitter_hz").median
        if not (3.0 <= f <= 16.0):
            f = cfg.default_f_hz
        s_max = cfg.s_cap * smoothstep((amp_deg - cfg.amp_lo_deg) / (cfg.amp_hi_deg - cfg.amp_lo_deg))
        if s_max < 0.02:
            return TremorParams.disabled(cfg)
        fc = min(max(cfg.fc_ratio * f, cfg.fc_min_hz), cfg.fc_max_hz)
        amp_counts = amp_deg / dpc
        cap = max(1, math.ceil(cfg.cap_gain * 2.0 * math.pi * f * amp_counts / 1000.0))
        one_pole = lambda hz: 1.0 - math.exp(-2.0 * math.pi * hz / 1000.0)    # noqa: E731
        v_t = max(2.0 * math.pi * f * amp_counts / 1000.0, 0.25)
        return TremorParams(True, one_pole(fc), one_pole(cfg.f_band_hi_hz), 1.0 - math.exp(-1.0 / cfg.energy_tau_ms), s_max, cap, v_t, cfg)


@dataclass
class _Axis:
    l1: float = 0.0
    l2: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    carry: float = 0.0


class TremorSuppressor:
    def __init__(self, params: Optional[TremorParams] = None) -> None:
        self.p = params or TremorParams.disabled()
        self.reset()

    def set_params(self, params: TremorParams) -> None:
        self.p = params

    def reset(self) -> None:
        self.ax = [_Axis(), _Axis()]
        self.e_band = self.e_lp = 0.0
        self.last_t: Optional[int] = None
        self.zero_ms = 0
        self.s = 0.0
        self.r = 0.0

    # ------------------------------------------------------------------
    def tick(self, t_us: int, dx: int, dy: int) -> tuple[int, int]:
        p = self.p
        gap = 0
        if self.last_t is not None:
            dt = t_us - self.last_t
            if dt > p.cfg.reset_ms * 1000:
                self._forget()
            elif dt > 1500:
                gap = min((dt + 500) // 1000 - 1, p.cfg.reset_ms)           # reports are not guaranteed every ms: silent ms are zeros
        self.last_t = t_us
        if not p.enabled:
            return dx, dy
        for _ in range(gap):
            self._step(0, 0)
        self.zero_ms += gap
        if dx == 0 and dy == 0:
            self.zero_ms += 1
            self._step(0, 0)
            if self.zero_ms >= p.cfg.reset_ms:
                self._forget()
            self.ax[0].carry = self.ax[1].carry = 0.0                       # nothing in, nothing out; no stored motion is released later
            return 0, 0
        self.zero_ms = 0
        hp = self._step(dx, dy)
        big = max(abs(dx), abs(dy)) / p.v_t                               # far above the person's own tremor: not tremor, leave it alone
        s_eff = self.s * (1.0 - smoothstep((big - p.cfg.big_lo) / (p.cfg.big_hi - p.cfg.big_lo)))
        out = []
        for i, x in enumerate((dx, dy)):
            if x == 0:
                out.append(0)
                continue
            y = x - s_eff * hp[i]
            lo, hi = (0.0, float(x)) if x > 0 else (float(x), 0.0)
            y = min(max(y, lo), hi)
            y = min(max(y, x - p.trim_cap), x + p.trim_cap)                 # never remove more than the tremor's own size
            y = min(max(y, lo), hi)
            v = y + self.ax[i].carry
            o = int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)
            if abs(o) > abs(x):
                o = x
            elif o != 0 and (o > 0) != (x > 0):
                o = 0
            self.ax[i].carry = v - o
            out.append(o)
        return out[0], out[1]

    # ------------------------------------------------------------------
    def _forget(self) -> None:
        self.ax = [_Axis(), _Axis()]
        self.e_band = self.e_lp = 0.0
        self.zero_ms = 0
        self.s = self.r = 0.0

    def _step(self, dx: int, dy: int) -> tuple[float, float]:
        """Advance the filters by one millisecond; returns the high-pass part of the input (per axis)."""
        p, c = self.p, self.p.cfg
        hps, bsq, lsq = [], 0.0, 0.0
        for a, x in zip(self.ax, (dx, dy)):
            a.l1 += p.a_lp * (x - a.l1)
            a.l2 += p.a_lp * (a.l1 - a.l2)
            hp = x - a.l2
            a.b1 += p.a_band * (hp - a.b1)
            a.b2 += p.a_band * (a.b1 - a.b2)
            hps.append(hp)
            bsq += a.b2 * a.b2
            lsq += a.l2 * a.l2
        self.e_band += p.a_e * (bsq - self.e_band)
        self.e_lp += p.a_e * (lsq - self.e_lp)
        self.r = self.e_band / (self.e_band + c.lp_weight * self.e_lp + c.eps)
        self.s = p.s_max * smoothstep((self.r - c.r_lo) / (c.r_hi - c.r_lo))
        return hps[0], hps[1]
