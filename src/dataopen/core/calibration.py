"""Calibration cross-checks: compare the engine's own projection (Probe.screen) with ours.

No ground-truth images needed: a mod reports a few world points together with where *the engine*
puts them on screen. If our projection of the same points (from the reported camera pose) disagrees,
the camera basis, FOV, units or axis conventions in the mod are wrong, and every label would be
silently shifted. We detect that per frame, and diagnose the likely cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import FrameSnapshot
from .projection import in_frame_mask, project


@dataclass
class ProbeResult:
    compared: int = 0
    missing: int = 0               # we see it clearly on screen, the engine says off-screen
    max_err: float = 0.0
    median_err: float = 0.0
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)  # (core, engine)

    def ok(self, tol_px: float) -> bool:
        return self.compared >= 2 and self.missing == 0 and self.max_err <= tol_px


def check_probes(snap: FrameSnapshot, margin_px: float = 20.0) -> Optional[ProbeResult]:
    """None if the mod sent no probes."""
    if not snap.probes:
        return None
    cam = snap.camera
    errs: list[float] = []
    res = ProbeResult()
    for p in snap.probes:
        uv, z = project(np.asarray(p.world, dtype=float), cam)
        core_in = bool(in_frame_mask(uv, z, cam))
        if p.screen is None:
            # only a real disagreement if we are comfortably inside the frame
            inside = core_in and margin_px <= uv[0] <= cam.width - margin_px and margin_px <= uv[1] <= cam.height - margin_px
            res.missing += int(inside)
            continue
        if not core_in:
            continue  # engine sees it, we say it is behind/outside: counted via a large error below
        e = float(np.hypot(uv[0] - p.screen[0], uv[1] - p.screen[1]))
        errs.append(e)
        res.pairs.append(((float(uv[0]), float(uv[1])), (float(p.screen[0]), float(p.screen[1]))))
    res.compared = len(errs)
    if errs:
        res.max_err, res.median_err = float(max(errs)), float(np.median(errs))
    return res


def diagnose(res: ProbeResult, cam_width: int, cam_height: int, tol_px: float = 3.0) -> str:
    """Best-guess explanation of a probe mismatch, from simple transform hypotheses."""
    if len(res.pairs) < 2:
        return "too few comparable probes; check that the mod reports probes inside the frame"
    core = np.array([c for c, _ in res.pairs])
    eng = np.array([e for _, e in res.pairs])

    def fits(pred: np.ndarray) -> bool:
        return float(np.abs(pred - eng).max()) <= tol_px

    if fits(core * [1, -1] + [0, cam_height]):
        return "the engine's Y axis is flipped relative to the reported camera (v = H - v): check the camera 'up' vector sign"
    if fits(core * [-1, 1] + [cam_width, 0]):
        return "the engine's X axis is mirrored (u = W - u): check the camera 'right' vector sign / handedness"
    if fits(core[:, ::-1]):
        return "U and V look swapped: the mod reports right/up in the wrong order"
    c = np.array([cam_width / 2.0, cam_height / 2.0])
    d_core, d_eng = (core - c).reshape(-1), (eng - c).reshape(-1)
    denom = float(d_core @ d_core)
    if denom > 1e-9:
        k = float(d_core @ d_eng) / denom
        if abs(k - 1.0) > 0.02 and fits((core - c) * k + c):
            return (f"focal length differs: the engine's projection is {k:.3f}x ours. Usually horizontal vs vertical "
                    f"FOV mixed up (report fov_v_deg or fov_h_deg correctly) or an aspect-ratio/zoom setting")
    return ("no simple transform explains it: check the camera pose basis (forward/right/up must be unit world-space "
            "directions of the SCREEN axes), units (meters) and that the pose is read in the same frame as the render")
