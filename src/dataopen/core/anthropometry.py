"""Typical adult segment lengths (meters): catches wrong bone mapping (doctor) and broken animation (quality)."""
from __future__ import annotations

import numpy as np

from .schema import SkeletonSchema

# (joint a, joint b, min m, max m)
SEGMENTS = (("l_shoulder", "l_elbow", 0.15, 0.45), ("l_elbow", "l_wrist", 0.15, 0.42),
            ("pelvis", "l_knee", 0.28, 0.65), ("l_knee", "l_ankle", 0.28, 0.62), ("head", "neck", 0.06, 0.40))
# left/right counterparts must have similar length
SYMMETRIC = (("l_shoulder", "l_elbow", "r_shoulder", "r_elbow"), ("l_elbow", "l_wrist", "r_elbow", "r_wrist"),
             ("l_knee", "l_ankle", "r_knee", "r_ankle"))


def segment_issues(skeleton_world: np.ndarray, valid: np.ndarray, schema: SkeletonSchema,
                   slack: float = 1.6) -> list[str]:
    """Issues in ONE skeleton. `slack` widens the typical ranges (children, tall characters, stylized rigs)."""
    out: list[str] = []
    kp = schema.keypoints

    def length(a: str, b: str):
        if a not in kp or b not in kp:
            return None
        ia, ib = schema.index(a), schema.index(b)
        if not (valid[ia] and valid[ib]):
            return None
        d = float(np.linalg.norm(skeleton_world[ia] - skeleton_world[ib]))
        return d if np.isfinite(d) else float("nan")

    def side_of(name: str, side: str) -> str:
        return side + name[2:] if name.startswith("l_") else name

    for a, b, lo, hi in SEGMENTS:
        sides = ("l_", "r_") if (a.startswith("l_") or b.startswith("l_")) else ("",)
        for side in sides:
            aa, bb = side_of(a, side), side_of(b, side)
            d = length(aa, bb)
            if d is None:
                continue
            if not np.isfinite(d):
                out.append(f"{aa}-{bb} not finite")
            elif d < lo / slack or d > hi * slack:
                out.append(f"{aa}-{bb} length {d:.2f}m outside {lo / slack:.2f}-{hi * slack:.2f}m")
    for la, lb, ra, rb in SYMMETRIC:
        dl, dr = length(la, lb), length(ra, rb)
        if dl and dr and np.isfinite(dl) and np.isfinite(dr) and not (0.55 <= dl / dr <= 1.8):
            out.append(f"{la}-{lb} vs {ra}-{rb} asymmetric ({dl:.2f}m vs {dr:.2f}m)")
    return out
