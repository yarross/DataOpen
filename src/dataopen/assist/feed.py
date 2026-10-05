"""Glue around the core: keep the parameters current from a live BioProfile, and two tiny object providers for tests and demos.

`ProfileFeed` is outside the 1 kHz path: poll it every ~100 ms. A profile change never makes the sensitivity jump: new parameters are
blended in over `blend_s` seconds, and a missing/invalid profile switches assistance off (K = 1.0) only after `ttl_s` of silence.
"""
from __future__ import annotations

import math
import time
from dataclasses import fields, replace
from typing import Callable, Optional, Sequence

from ..bioprofile.profile import ProfileView
from .params import AscConfig, AscParams
from .types import ObjectOfInterest

_BLEND = ("v_on", "v_still", "f_b", "ov_rate", "ov_med", "s_brake", "tremor_counts", "hold_scale", "v_ref", "v_leave", "need", "fatigue")
_INT = ("t_lo_us", "ramp_us")


def blend_params(a: AscParams, b: AscParams, w: float) -> AscParams:
    """Interpolate numeric parameters (w = 0 -> a, 1 -> b). Assistance stays off if either side is off: a profile that is not
    trustworthy must not be half-applied."""
    if not (a.enabled and b.enabled):
        return b if w >= 1.0 else (a if not a.enabled else replace(a, enabled=False))
    out = {}
    for f in fields(AscParams):
        if f.name in _BLEND:
            out[f.name] = (1 - w) * getattr(a, f.name) + w * getattr(b, f.name)
        elif f.name in _INT:
            out[f.name] = int(round((1 - w) * getattr(a, f.name) + w * getattr(b, f.name)))
    return replace(b, **out)


class ProfileFeed:
    def __init__(self, source: Callable[[], Optional[ProfileView]], cfg: Optional[AscConfig] = None, blend_s: float = 1.0,
                 ttl_s: float = 600.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.source, self.cfg, self.blend_s, self.ttl_s, self.clock = source, cfg or AscConfig(), blend_s, ttl_s, clock
        self.current = AscParams.disabled(self.cfg)
        self._target = self.current
        self._from = self.current
        self._t0 = clock()
        self._gen: Optional[int] = None
        self._last_ok = clock()

    def update(self) -> AscParams:
        """Poll the profile source and return the parameters to use now."""
        now = self.clock()
        view = None
        try:
            view = self.source()
        except Exception:                                   # a failing reader must never break pointing
            view = None
        if view is not None:
            self._last_ok = now
            if view.generation != self._gen or view.profile_id != getattr(self, "_pid", None):
                self._gen, self._pid = view.generation, view.profile_id
                self._from, self._target, self._t0 = self.current, AscParams.from_view(view, self.cfg), now
        elif now - self._last_ok > self.ttl_s and self._target.enabled:
            self._from, self._target, self._t0 = self.current, AscParams.disabled(self.cfg), now
        w = 1.0 if self.blend_s <= 0 else min(max((now - self._t0) / self.blend_s, 0.0), 1.0)
        self.current = blend_params(self._from, self._target, w)
        return self.current


class StaticObjects:
    """A fixed list of objects: `nearest()` returns the one closest to the cursor (what a UI adapter would do)."""

    def __init__(self, objects: Sequence[ObjectOfInterest]) -> None:
        self.objects = list(objects)

    def nearest(self, x: float, y: float) -> Optional[ObjectOfInterest]:
        if not self.objects:
            return None
        return min(self.objects, key=lambda o: math.hypot(o.x - x, o.y - y) - o.radius)
