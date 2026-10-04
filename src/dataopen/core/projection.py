"""3D -> 2D projection and keypoint visibility flags (vectorized, engine-agnostic).

Single source of truth for ground-truth math: it is implemented and unit-tested once
here instead of once per game.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .models import CameraModel, Visibility


def project(points_world: np.ndarray, cam: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """(..., 3) world points -> ((..., 2) pixel coords, (...) camera-space z in meters)."""
    pts = np.asarray(points_world, dtype=np.float64)
    pc = pts @ cam.world_to_camera[:3, :3].T + cam.world_to_camera[:3, 3]
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * pc[..., 0] / z + cam.cx
        v = cam.fy * pc[..., 1] / z + cam.cy
    return np.stack([u, v], axis=-1), z


def in_frame_mask(uv: np.ndarray, z: np.ndarray, cam: CameraModel) -> np.ndarray:
    u, v = uv[..., 0], uv[..., 1]
    with np.errstate(invalid="ignore"):
        return (z > cam.near) & np.isfinite(u) & np.isfinite(v) & \
            (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)


def visibility_flags(
    uv: np.ndarray,
    z: np.ndarray,
    cam: CameraModel,
    joint_valid: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    depth_tol: float = 0.05,
    engine_visibility: Optional[np.ndarray] = None,
) -> np.ndarray:
    """2 = visible, 1 = occluded (in frame, hidden), 0 = out of frame / invalid.

    Occlusion source priority:
      1. `engine_visibility` (engine raycasts against body-part colliders; handles self-occlusion)
      2. `depth`: z-depth of *occluders* (render the depth pass without the target actors, or
         raise `depth_tol` above body radius); joint is occluded if depth < z - tol
      3. neither: in-frame joints are reported visible (an upper bound; the core warns)
    """
    inside = in_frame_mask(uv, z, cam)
    flags = np.where(inside, int(Visibility.VISIBLE), int(Visibility.OUT_OF_FRAME)).astype(np.int8)

    if engine_visibility is not None:
        ev = np.asarray(engine_visibility)
        flags = np.where(inside & (ev < int(Visibility.VISIBLE)), int(Visibility.OCCLUDED), flags)
    elif depth is not None:
        dh, dw = depth.shape
        iu = np.nan_to_num(np.floor(uv[..., 0] * dw / cam.width), nan=0).astype(int).clip(0, dw - 1)
        iv = np.nan_to_num(np.floor(uv[..., 1] * dh / cam.height), nan=0).astype(int).clip(0, dh - 1)
        d = depth[iv, iu]
        occluded = inside & np.isfinite(d) & (d < z - depth_tol)
        flags = np.where(occluded, int(Visibility.OCCLUDED), flags)

    if joint_valid is not None:
        flags = np.where(joint_valid, flags, int(Visibility.OUT_OF_FRAME))
    return flags.astype(np.int8)
