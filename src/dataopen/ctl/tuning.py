"""The two user-facing knobs and what they turn (docs/PWA.md section 5).

`strength` (0..10, default 5) scales the movement-assistance layer, `tremor` (0..10, default 5) the tremor filter. Level 5 is exactly what
the profile recommends (factor 1.0), level 0 switches the layer off, level 10 is x1.5 (for the tremor filter its share moves toward the
cap of 0.95 instead). The phone never sends raw parameter blobs: the gateway derives them here, so every blob that reaches the bridge
is built by code that knows the profile and passes the bridge's range tables, and the bridge's output clamp (|out| <= |in|) holds for
whatever the knobs say.

Honest limits: the scale of the factors is a design choice checked for safety and monotonicity on the simulator, not for how it feels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from ..assist.chain import overlap_hold
from ..assist.fixed import FixedParams
from ..assist.params import AscConfig, AscParams
from ..assist.tremor import TremorConfig, TremorParams
from ..assist.tremor_fixed import FixedTremorParams
from ..bioprofile.profile import ProfileView

LEVEL_MIN, LEVEL_MAX, LEVEL_DEFAULT = 0, 10, 5
S_MAX_CAP = 0.95                 # the tremor filter never removes more than this share (bridge range: <= 1.0)
TRIM_CAP_MAX = 64                # the bridge's range table limit


def factor(level: int) -> float:
    """0 -> 0, 5 -> 1.0, 10 -> 1.5, linear in between and beyond."""
    level = min(max(int(level), LEVEL_MIN), LEVEL_MAX)
    return level / 5.0 if level <= 5 else 1.0 + 0.1 * (level - 5)


def scale_asc(base: AscParams, level: int) -> AscParams:
    """Stronger = steeper braking, deeper floor, a hold well that is not weakened. Level 0 or a disabled base: layer OFF."""
    f = factor(level)
    if not base.enabled or f <= 0.0:
        return AscParams.disabled(base.cfg)
    k_floor = min(max(1.0 - f * (1.0 - base.cfg.k_floor), base.cfg.k_floor), 1.0)
    cfg = replace(base.cfg, k_floor=k_floor)
    return replace(base, s_brake=min(base.s_brake * f, cfg.s_cap), hold_scale=min(base.hold_scale * f, 1.0), cfg=cfg)


def scale_tremor(base: TremorParams, level: int) -> TremorParams:
    f = factor(level)
    if not base.enabled or f <= 0.0:
        return TremorParams.disabled(base.cfg)
    # below the recommendation the share scales down; above it, it moves toward the cap, so that the top steps always mean something
    s_max = base.s_max * f if f <= 1.0 else base.s_max + (max(S_MAX_CAP, base.s_max) - base.s_max) * (f - 1.0) / 0.5
    return replace(base, s_max=min(s_max, S_MAX_CAP), trim_cap=min(max(1, math.ceil(base.trim_cap * f)), TRIM_CAP_MAX))


def neutral_asc() -> FixedParams:
    """A switched-off ASC blob that still passes the bridge's range tables (a blob of zeros would be rejected: v_on must be >= 1)."""
    p = AscParams(enabled=False, v_on=0.01, v_still=0.008, t_lo_us=100_000, ramp_us=20_000, f_b=0.5, v_ref=1.0, v_leave=0.6)
    return replace(FixedParams.from_params(p), enabled=0)


def neutral_tremor() -> FixedTremorParams:
    return FixedTremorParams.from_params(TremorParams.disabled())


@dataclass(frozen=True)
class Derived:
    asc: FixedParams
    tremor: FixedTremorParams
    asc_on: bool
    tremor_on: bool
    ppc: float
    profile_id: int
    profile_gen: int

    @property
    def any_on(self) -> bool:
        return self.asc_on or self.tremor_on


def derive(view: Optional[ProfileView], strength: int = LEVEL_DEFAULT, tremor: int = LEVEL_DEFAULT, hold_overlap: float = 0.75,
           asc_cfg: Optional[AscConfig] = None, tremor_cfg: Optional[TremorConfig] = None) -> Derived:
    """profile + two levels -> the exact fixed-point blobs for the bridge (both always valid; a layer that is off is a neutral blob)."""
    if view is None:
        return Derived(neutral_asc(), neutral_tremor(), False, False, (asc_cfg or AscConfig()).px_per_count, 0, 0)
    asc = scale_asc(AscParams.from_view(view, asc_cfg), strength)
    trm = scale_tremor(TremorParams.from_view(view, tremor_cfg), tremor)
    asc = overlap_hold(asc, trm, hold_overlap)               # the two layers act on the same signal: avoid double damping
    fa = FixedParams.from_params(asc) if asc.enabled else neutral_asc()
    ft = FixedTremorParams.from_params(trm) if trm.enabled else neutral_tremor()
    return Derived(fa, ft, bool(asc.enabled), bool(trm.enabled), asc.cfg.px_per_count, view.profile_id, view.generation)
