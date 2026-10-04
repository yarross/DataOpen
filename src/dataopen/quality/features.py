"""Model-independent frame features: can a person in this picture be perceived at all?

This is what separates "the model is blind because the picture is unusable" (poison: drop) from "the model
struggles on a perfectly visible person" (a hard example: keep, and generate more like it). The model's own score
cannot tell these apart, the pixels can: contrast between the person's visible limbs and their surroundings,
brightness, size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.anthropometry import segment_issues
from ..core.models import Annotation, EntityState
from ..core.schema import SkeletonSchema
from .types import FrameFeatures, PersonFeatures


@dataclass
class FeatureConfig:
    contrast_ref: float = 0.10     # colour distance (0..1 of full scale) at which a person counts as fully distinguishable
    dark_floor: float = 0.03       # limb luma below this = invisible
    dark_ref: float = 0.12         # limb luma above this = bright enough
    size_ref: float = 40.0         # bbox height (px) above which size no longer limits perceptibility
    ring_frac: float = 0.15        # surrounding-ring thickness as a fraction of the larger bbox side
    stride: int = 4                # sub-sampling of the global statistics
    focus_radius_rel: float = 0.05  # radius of the aim-point patch as a fraction of the person's bbox height
    focus_size_ref: float = 4.0    # patch radius (px) above which size no longer limits the aim point's perceptibility


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0


def _limb_pixels(img: np.ndarray, ann: Annotation, schema: SkeletonSchema) -> np.ndarray:
    """Colours sampled on the person's VISIBLE joints and the bones between visible joints."""
    h, w, _ = img.shape
    kp = ann.keypoints
    pts = [kp[i, :2] for i in range(len(kp)) if kp[i, 2] == 2]
    for a, b in schema.edges:
        ia, ib = schema.index(a), schema.index(b)
        if kp[ia, 2] == 2 and kp[ib, 2] == 2:
            for t in (0.2, 0.4, 0.6, 0.8):
                pts.append(kp[ia, :2] * (1 - t) + kp[ib, :2] * t)
    if not pts:
        return np.zeros((0, 3))
    p = np.asarray(pts)
    xs = np.clip(p[:, 0].astype(int), 0, w - 1)
    ys = np.clip(p[:, 1].astype(int), 0, h - 1)
    return img[ys, xs].astype(np.float64)


def _ring_mean(img: np.ndarray, bbox, frac: float) -> Optional[np.ndarray]:
    h, w, _ = img.shape
    x, y, bw, bh = bbox
    t = max(3, int(frac * max(bw, bh)))
    x0, y0, x1, y1 = int(max(0, x)), int(max(0, y)), int(min(w, x + bw)), int(min(h, y + bh))
    strips = []
    for (a0, b0, a1, b1) in ((max(0, x0 - t), max(0, y0 - t), min(w, x1 + t), y0),          # above
                             (max(0, x0 - t), y1, min(w, x1 + t), min(h, y1 + t)),            # below
                             (max(0, x0 - t), y0, x0, y1), (x1, y0, min(w, x1 + t), y1)):      # left, right
        if a1 > a0 and b1 > b0:
            strips.append(img[b0:b1:2, a0:a1:2].reshape(-1, 3))
    if not strips:
        return None
    return np.concatenate(strips).astype(np.float64).mean(axis=0)


def _sharpness(img: np.ndarray, bbox) -> float:
    x, y, bw, bh = (int(round(v)) for v in bbox)
    crop = _luma(img[max(0, y):y + bh:2, max(0, x):x + bw:2])
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return 0.0
    lap = crop[1:-1, 1:-1] * 4 - crop[:-2, 1:-1] - crop[2:, 1:-1] - crop[1:-1, :-2] - crop[1:-1, 2:]
    return float(lap.var())


def _person_difficulty(perceptibility: float, occlusion: float, size_px: float) -> float:
    small = 1.0 - float(np.clip(size_px / 120.0, 0.0, 1.0))
    return float(np.clip(0.45 * (1.0 - perceptibility) + 0.35 * occlusion + 0.20 * small, 0.0, 1.0))


def _focus_features(img: np.ndarray, ann: Annotation, schema: SkeletonSchema, cfg: FeatureConfig, pf: PersonFeatures) -> None:
    """Perceptibility of the schema's primary keypoint(s): a patch around the point against the ring around the patch."""
    prim = [i for i in schema.primary_idx() if ann.keypoints[i, 2] > 0]
    if not prim:
        return
    h, w, _ = img.shape
    pf.has_focus = True
    pf.focus_visibility = float((ann.keypoints[prim, 2] == 2).mean())
    r = max(2.0, cfg.focus_radius_rel * ann.bbox[3])
    pf.focus_radius_px = float(r)
    patches, rings = [], []
    for i in prim:
        x, y = ann.keypoints[i, :2]
        x0, x1, y0, y1 = (int(max(0, x - r)), int(min(w, x + r + 1)), int(max(0, y - r)), int(min(h, y + r + 1)))
        X0, X1, Y0, Y1 = (int(max(0, x - 2.2 * r)), int(min(w, x + 2.2 * r + 1)),
                          int(max(0, y - 2.2 * r)), int(min(h, y + 2.2 * r + 1)))
        if x1 <= x0 or y1 <= y0 or X1 <= X0 or Y1 <= Y0:
            continue
        outer = img[Y0:Y1, X0:X1].astype(np.float64)
        mask = np.ones(outer.shape[:2], dtype=bool)
        mask[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        if not mask.any():
            continue
        patches.append(img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64).mean(axis=0))
        rings.append(outer[mask].mean(axis=0))
    if patches:
        patch, ring = np.mean(patches, axis=0), np.mean(rings, axis=0)
        pf.focus_contrast = float(np.linalg.norm(patch - ring) / (255.0 * np.sqrt(3.0)))
        pf.focus_brightness = float(_luma(patch))
        c = np.clip(pf.focus_contrast / cfg.contrast_ref, 0.0, 1.0)
        b = np.clip((pf.focus_brightness - cfg.dark_floor) / (cfg.dark_ref - cfg.dark_floor), 0.0, 1.0)
        pf.focus_perceptibility = float(c * b * (0.5 + 0.5 * np.clip(r / cfg.focus_size_ref, 0.0, 1.0)))
    s = float(np.clip(r / cfg.focus_size_ref, 0.0, 1.0))
    pf.focus_difficulty = float(np.clip(0.45 * (1.0 - pf.focus_perceptibility) + 0.35 * (1.0 - pf.focus_visibility)
                                        + 0.20 * (1.0 - s), 0.0, 1.0))


def person_features(img: np.ndarray, ann: Annotation, schema: SkeletonSchema, cfg: FeatureConfig) -> PersonFeatures:
    kp = ann.keypoints
    pf = PersonFeatures(ann.entity_id, size_px=float(ann.bbox[3]))
    pf.occlusion_index = float(1.0 - (kp[:, 2] == 2).sum() / len(kp))
    _focus_features(img, ann, schema, cfg, pf)
    limbs = _limb_pixels(img, ann, schema)
    ring = _ring_mean(img, ann.bbox, cfg.ring_frac)
    if len(limbs) == 0 or ring is None:
        pf.limiting_factor = "visibility"
        pf.difficulty = _person_difficulty(0.0, pf.occlusion_index, pf.size_px)
        return pf                                     # perceptibility stays 0
    limb_mean = limbs.mean(axis=0)
    pf.contrast_rate = float(np.linalg.norm(limb_mean - ring) / (255.0 * np.sqrt(3.0)))
    pf.brightness = float(_luma(limb_mean))
    pf.sharpness = _sharpness(img, ann.bbox)
    c = np.clip(pf.contrast_rate / cfg.contrast_ref, 0.0, 1.0)
    b = np.clip((pf.brightness - cfg.dark_floor) / (cfg.dark_ref - cfg.dark_floor), 0.0, 1.0)
    s = np.clip(pf.size_px / cfg.size_ref, 0.0, 1.0)
    sz = 0.5 + 0.5 * s
    pf.perceptibility = float(c * b * sz)
    pf.limiting_factor = min((("contrast", c), ("brightness", b), ("size", sz)), key=lambda kv: kv[1])[0]
    pf.difficulty = _person_difficulty(pf.perceptibility, pf.occlusion_index, pf.size_px)
    return pf


def compute_features(img: Optional[np.ndarray], annotations: Sequence[Annotation], entities: Sequence[EntityState],
                     schema: SkeletonSchema, cfg: Optional[FeatureConfig] = None,
                     rig_schema: Optional[SkeletonSchema] = None) -> FrameFeatures:
    """`entities` are as the mod reported them (in `rig_schema`); `annotations` are in the dataset's `schema`."""
    cfg = cfg or FeatureConfig()
    f = FrameFeatures()
    ids = {a.entity_id for a in annotations}
    for e in entities:                                # 3D sanity of the labeled skeletons (needs no pixels)
        if e.entity_id in ids:
            f.gt_issues += [f"entity {e.entity_id}: {m}"
                            for m in segment_issues(e.skeleton_world, e.joint_valid, rig_schema or schema)]
    if img is None:
        return f
    f.has_pixels = True
    sub = img[::cfg.stride, ::cfg.stride]
    luma = _luma(sub)
    f.luma_mean, f.luma_std = float(luma.mean()), float(luma.std())
    f.saturated_frac = float(((luma < 0.02) | (luma > 0.98)).mean())
    f.persons = [person_features(img, a, schema, cfg) for a in annotations]
    return f


def static_difficulty(f: FrameFeatures) -> float:
    """0..1 from pixels and labels only: low perceptibility, heavy occlusion, small people."""
    if not f.persons:
        return 0.0
    deficit = 1.0 - float(np.mean([p.perceptibility for p in f.persons]))
    small = 1.0 - float(np.clip(np.mean([p.size_px for p in f.persons]) / 120.0, 0.0, 1.0))
    return float(np.clip(0.45 * deficit + 0.35 * f.mean_occlusion + 0.20 * small, 0.0, 1.0))
