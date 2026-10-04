"""Geometric self-occlusion estimate (torso capsule + head sphere), engine-agnostic.

Engine raycasts and occluder-only depth passes cannot see a person occluding THEMSELVES
(a wrist behind the torso). This models the torso as a capsule (neck -> pelvis) and the head
as a sphere and marks a limb joint occluded when the camera ray to it passes through them
before reaching the joint. It is a heuristic with explicit radii, not ground truth: it removes
the most common label error (back-facing arms/legs marked visible) at ~zero cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Visibility
from .schema import SkeletonSchema

# joints that can legitimately be hidden by the torso/head; torso-attached joints are never tested
_TESTED = ("l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_knee", "r_knee", "l_ankle", "r_ankle")


@dataclass(frozen=True)
class SelfOcclusionConfig:
    torso_radius_ratio: float = 0.35  # capsule radius = ratio * shoulder width
    head_radius: float = 0.11         # meters
    margin: float = 0.03              # joint must be at least this far behind the surface


def _closest_segment_params(p1, q1, p2, q2):
    """Closest points between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection).
    Returns (s, t, distance) with s on segment 1 and t on segment 2, both in [0, 1]."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return 0.0, 0.0, float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            s, t = np.clip(-c / a, 0.0, 1.0), 0.0
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(s), float(t), float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def apply_self_occlusion(
    skeleton_world: np.ndarray,
    flags: np.ndarray,
    camera_pos: np.ndarray,
    schema: SkeletonSchema,
    cfg: SelfOcclusionConfig = SelfOcclusionConfig(),
) -> np.ndarray:
    """Return a copy of `flags` with VISIBLE limb joints downgraded to OCCLUDED where the
    torso capsule or head sphere lies on the camera ray in front of them."""
    out = flags.copy()
    roles = {r: schema.role(r) for r in ("neck", "pelvis", "head", "l_shoulder", "r_shoulder")}
    if any(v is None for v in roles.values()):
        return out
    idx = {r: schema.index(n) for r, n in roles.items() if n is not None}
    neck, pelvis = skeleton_world[idx["neck"]], skeleton_world[idx["pelvis"]]
    head = skeleton_world[idx["head"]]
    sw = float(np.linalg.norm(skeleton_world[idx["l_shoulder"]] - skeleton_world[idx["r_shoulder"]]))
    r_torso = cfg.torso_radius_ratio * sw
    if r_torso <= 0 or not np.isfinite(r_torso):
        return out
    cam = np.asarray(camera_pos, dtype=np.float64)

    for name in _TESTED:
        if name not in schema.keypoints:
            continue
        i = schema.index(name)
        if out[i] != Visibility.VISIBLE:
            continue
        j = skeleton_world[i]
        ray_len = float(np.linalg.norm(j - cam))
        if ray_len < 1e-6:
            continue
        # torso capsule
        s, t, dist = _closest_segment_params(neck, pelvis, cam, j)
        if dist < r_torso and t < 1.0 - (cfg.margin + r_torso) / ray_len:
            # the joint itself must lie outside the capsule (otherwise it is attached to it)
            _, _, jd = _closest_segment_params(neck, pelvis, j, j)
            if jd > r_torso * 1.05:
                out[i] = Visibility.OCCLUDED
                continue
        # head sphere
        to_head = head - cam
        proj = float(to_head @ (j - cam)) / ray_len
        if 0 < proj < ray_len - cfg.margin:
            if float(np.linalg.norm(to_head - proj * (j - cam) / ray_len)) < cfg.head_radius:
                out[i] = Visibility.OCCLUDED
    return out
