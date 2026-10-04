"""FrameSnapshot -> Annotations: projection, visibility, bbox, per-entity verdicts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .models import Annotation, EntityState, FrameSnapshot, Visibility
from .projection import project, visibility_flags
from .self_occlusion import apply_self_occlusion
from .schema import SkeletonSchema


class Verdict(str, Enum):
    ACCEPT = "accept"
    ABSENT = "absent"              # nothing of this entity is visible in the image: not annotated, fine
    NEGLIGIBLE = "negligible"                        # a few pixels tall: below any detector's resolution, ignored
                                                     # without rejecting the frame
    IGNORE_TOO_SMALL = "ignore_too_small"            # visible, too small to label reliably but too big to ignore
    IGNORE_TOO_OCCLUDED = "ignore_too_occluded"      # a few joints visible only
    IGNORE_BAD_BBOX = "ignore_bad_bbox"              # degenerate / absurd aspect ratio


@dataclass
class AnnotationConfig:
    bbox_padding: float = 0.08           # fraction of bbox size, used when no hull points
    min_bbox_height_px: float = 24.0     # smaller (but not negligible) visible persons reject the frame
    min_bbox_width_px: float = 8.0
    negligible_height_px: float = 12.0   # visible persons shorter than this are simply not annotated
    min_visible_keypoints: int = 4       # joints with v == 2
    max_aspect: float = 8.0
    bbox_include_occluded: bool = True   # amodal-in-frame bbox vs visible-only
    depth_tol: float = 0.05
    self_occlusion: bool = True          # torso/head capsule heuristic (engines can't see self-occlusion)


@dataclass
class BuildResult:
    annotations: list[Annotation] = field(default_factory=list)
    verdicts: dict[int, Verdict] = field(default_factory=dict)  # entity_id -> verdict
    ignored_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # persons present but not labeled
    warnings: list[str] = field(default_factory=list)


class AnnotationBuilder:
    def __init__(self, schema: SkeletonSchema, config: Optional[AnnotationConfig] = None) -> None:
        self.schema = schema
        self.cfg = config or AnnotationConfig()
        self._last_bbox: Optional[tuple[float, float, float, float]] = None

    def build(self, snap: FrameSnapshot) -> BuildResult:
        res = BuildResult()
        if snap.depth is None and any(e.engine_visibility is None for e in snap.entities):
            res.warnings.append("no occlusion source: in-frame joints reported as visible")
        for ent in snap.entities:
            ann, verdict = self._entity(ent, snap)
            res.verdicts[ent.entity_id] = verdict
            if ann is not None:
                res.annotations.append(ann)
            elif verdict is not Verdict.ABSENT and self._last_bbox is not None:
                res.ignored_boxes.append(self._last_bbox)
        return res

    def _entity(self, ent: EntityState, snap: FrameSnapshot) -> tuple[Optional[Annotation], Verdict]:
        self._last_bbox = None
        cam, cfg = snap.camera, self.cfg
        uv, z = project(ent.skeleton_world, cam)
        flags = visibility_flags(uv, z, cam, ent.joint_valid, snap.depth, cfg.depth_tol,
                                 ent.engine_visibility)
        if cfg.self_occlusion:
            flags = apply_self_occlusion(ent.skeleton_world, flags, cam.position, self.schema)
        n_vis = int((flags == Visibility.VISIBLE).sum())
        if n_vis == 0:
            return None, Verdict.ABSENT

        bbox = self._bbox(ent, uv, z, flags, cam)
        if bbox is None:
            return None, Verdict.IGNORE_BAD_BBOX
        x, y, w, h = bbox
        self._last_bbox = bbox
        if h < cfg.negligible_height_px:
            return None, Verdict.NEGLIGIBLE
        if h < cfg.min_bbox_height_px or w < cfg.min_bbox_width_px:
            return None, Verdict.IGNORE_TOO_SMALL
        if max(w / h, h / w) > cfg.max_aspect:
            return None, Verdict.IGNORE_BAD_BBOX
        if n_vis < cfg.min_visible_keypoints:
            return None, Verdict.IGNORE_TOO_OCCLUDED

        kp = np.zeros((self.schema.num_keypoints, 3), dtype=np.float64)
        keep = flags > 0
        kp[keep, :2] = uv[keep]          # v == 0 keeps x = y = 0 (COCO convention)
        kp[:, 2] = flags
        return Annotation(ent.entity_id, kp, bbox, {"rig_id": ent.rig_id, **ent.meta}), Verdict.ACCEPT

    def _bbox(self, ent, uv, z, flags, cam) -> Optional[tuple[float, float, float, float]]:
        cfg = self.cfg
        if ent.hull_points_world is not None and len(ent.hull_points_world):
            huv, hz = project(ent.hull_points_world, cam)
            ok = (hz > cam.near) & np.isfinite(huv).all(axis=-1)
            pts, pad = huv[ok], 0.0                     # hull is already the real extent
        else:
            use = (flags >= (Visibility.OCCLUDED if cfg.bbox_include_occluded else Visibility.VISIBLE))
            pts, pad = uv[use], cfg.bbox_padding
        if len(pts) == 0:
            return None
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py
        x0, x1 = np.clip([x0, x1], 0, cam.width)
        y0, y1 = np.clip([y0, y1], 0, cam.height)
        if x1 <= x0 or y1 <= y0:
            return None
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)
