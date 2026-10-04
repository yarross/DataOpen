"""Detector output -> target observations: turns the runtime's `KeypointArray` into `TargetObs` (angles from the crosshair).

The camera model is rectilinear: a pixel offset `dx` from the crosshair is `atan(dx / f)` with `f = (W/2) / tan(fov/2)`. The target the
player is engaging is taken to be the detection whose aim point is closest to the crosshair (a heuristic: gaze is not available).
`KeypointArray.timestamp_us` is 32 bits (wraps every 71.6 minutes): the adapter unwraps it into a monotonic 64-bit clock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..detector import structs
from ..runtime.loop import FLAG_EMPTY_ERROR
from .types import TargetObs


@dataclass
class AdapterConfig:
    aim_index: int = 1                  # keypoint index of the aim point (shooter12: head_center)
    width: int = 640
    height: int = 640
    fov_h_deg: float = 90.0             # horizontal field of view of the captured image
    cx: Optional[float] = None          # crosshair in image pixels (default: the centre)
    cy: Optional[float] = None
    min_kp_conf: float = 0.3            # aim-point confidence below this: not a usable observation
    head_frac: float = 0.07             # aim-region radius as a fraction of the box height
    dark_brightness: int = 40           # scene brightness below this marks the observation degraded


class DetectorAdapter:
    def __init__(self, cfg: Optional[AdapterConfig] = None, n_kpt: int = 12) -> None:
        self.cfg = cfg or AdapterConfig()
        self.n_kpt = n_kpt
        c = self.cfg
        self.f = (c.width / 2.0) / math.tan(math.radians(c.fov_h_deg) / 2.0)
        self._last32: Optional[int] = None
        self._wraps = 0

    def _unwrap(self, ts32: int) -> int:
        if self._last32 is not None and ts32 < self._last32 - (1 << 31):
            self._wraps += 1
        self._last32 = ts32
        return ts32 + (self._wraps << 32)

    def observe(self, arr) -> tuple[int, Optional[TargetObs]]:
        """-> (unwrapped capture time in us, the target observation or None when nothing usable was detected / inference failed)."""
        c = self.cfg
        t_us = self._unwrap(int(arr.timestamp_us))
        if arr.flags & FLAG_EMPTY_ERROR:
            return t_us, None
        cx = c.width / 2.0 if c.cx is None else c.cx
        cy = c.height / 2.0 if c.cy is None else c.cy
        best, best_d = None, math.inf
        for d in structs.unpack(arr, self.n_kpt):
            kc = d["kscore"][c.aim_index]
            if kc < c.min_kp_conf:
                continue
            px, py = d["kxy"][c.aim_index]
            dist = math.hypot(px - cx, py - cy)
            if dist < best_d:
                best, best_d = (d, px, py, kc), dist
        if best is None:
            return t_us, None
        d, px, py, kc = best
        h_px = max(float(d["box"][3] - d["box"][1]), 1.0)
        radius = math.degrees(math.atan2(max(1.0, c.head_frac * h_px), self.f))
        return t_us, TargetObs(t_us, math.degrees(math.atan2(px - cx, self.f)), math.degrees(math.atan2(py - cy, self.f)),
                               vis=float(kc * d["score"]), degraded=int(arr.avg_scene_brightness) < c.dark_brightness,
                               radius_deg=radius)
