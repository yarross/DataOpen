"""The assistive chain: ASC first, tremor suppression second.

    raw HID delta --> ASC (K from the object geometry; sees the person's UNTOUCHED motion, so its guard and direction logic can not
                      be shifted by the filter) --> tremor suppression (one-sided, only removes) --> pointer

Every stage can only shrink the delta, so |final| <= |ASC out| <= |raw| per axis, with the sign of the raw delta or zero,
and zero in gives zero out.
The cursor position given to the next tick must be the FINAL (post-chain) position.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..bioprofile.profile import ProfileView
from .fixed import FixedAsc, FixedParams
from .model import AdaptiveSensitivity
from .params import AscConfig, AscParams
from .tremor import TremorConfig, TremorParams, TremorSuppressor
from .tremor_fixed import FixedTremor, FixedTremorParams
from .types import Guard, ObjectOfInterest, Reason


@dataclass(frozen=True)
class ChainOut:
    k: float
    dx: int                  # final delta (after both stages)
    dy: int
    asc_dx: int              # after ASC only
    asc_dy: int
    guard: Guard
    reason: Reason
    tremor_s: float          # tremor suppression strength used (0 = none)


def overlap_hold(asc: AscParams, tremor: TremorParams, overlap: float) -> AscParams:
    if not (tremor.enabled and asc.enabled) or overlap <= 0:
        return asc
    return replace(asc, hold_scale=asc.hold_scale * (1.0 - overlap * tremor.s_max / max(tremor.cfg.s_cap, 1e-9)))


class AssistChain:
    def __init__(self, asc, tremor) -> None:
        self.asc, self.tremor = asc, tremor

    @staticmethod
    def build(asc_params: AscParams, tremor_params: TremorParams, impl: str = "fixed", hold_overlap: float = 0.75) -> "AssistChain":
        """impl: 'float' (reference), 'fixed' (Python golden) or 'c' (compiled core; an error if there is no compiler).
        `hold_overlap`: ASC's hold well and the tremor filter act on the same signal (tremor at the object): with the filter at
        full strength the well is scaled by (1 - overlap), because both together over-damp (measured on the simulator)."""
        asc_params = overlap_hold(asc_params, tremor_params, hold_overlap)
        if impl == "float":
            return AssistChain(AdaptiveSensitivity(asc_params), TremorSuppressor(tremor_params))
        fa, ft = FixedParams.from_params(asc_params), FixedTremorParams.from_params(tremor_params)
        if impl == "fixed":
            return AssistChain(FixedAsc(fa), FixedTremor(ft))
        if impl == "c":
            from .cimpl import CAsc, CTremor
            return AssistChain(CAsc(fa), CTremor(ft))
        raise ValueError("impl must be 'float', 'fixed' or 'c'")

    @staticmethod
    def from_view(view: ProfileView, impl: str = "fixed", asc_cfg: Optional[AscConfig] = None,
                  tremor_cfg: Optional[TremorConfig] = None) -> "AssistChain":
        return AssistChain.build(AscParams.from_view(view, asc_cfg), TremorParams.from_view(view, tremor_cfg), impl)

    def set_params_from(self, asc_params: AscParams, tremor_params: TremorParams, hold_overlap: float = 0.75) -> None:
        self.set_params(overlap_hold(asc_params, tremor_params, hold_overlap), tremor_params)

    def set_params(self, asc_params: AscParams, tremor_params: TremorParams) -> None:
        if isinstance(self.asc, AdaptiveSensitivity):
            self.asc.set_params(asc_params)
            self.tremor.set_params(tremor_params)
        else:
            self.asc.set_params(FixedParams.from_params(asc_params))
            self.tremor.set_params(FixedTremorParams.from_params(tremor_params))

    def reset(self) -> None:
        self.asc.reset()
        self.tremor.reset()

    def tick(self, t_us: int, dx: int, dy: int, px: float, py: float, obj: Optional[ObjectOfInterest]) -> ChainOut:
        a = self.asc.tick(t_us, dx, dy, px, py, obj)
        fx, fy = self.tremor.tick(t_us, a.dx, a.dy)
        s = getattr(self.tremor, "s", 0)
        s = s / 65536 if isinstance(s, int) else float(s)
        return ChainOut(a.k, fx, fy, a.dx, a.dy, a.guard, a.reason, s)
