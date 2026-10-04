"""Host-side post-processing (numpy only): dense per-level outputs -> at most `max_det` detections, no NMS.

The same function is used by the evaluator in the closed loop, by the training validation and (ported 1:1) by the board
runtime (`csrc/apollo_post.c`, checked against this file by a test)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..quality.types import Prediction
from .layout import DecodeConfig, HeadLayout


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def grid_anchors(h: int, w: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    return (gx.reshape(-1) + 0.5) * stride, (gy.reshape(-1) + 0.5) * stride


def decode_level(raw: np.ndarray, stride: int, lay: HeadLayout) -> dict[str, np.ndarray]:
    """raw: (C, H, W) or (1, C, H, W) -> per-anchor arrays (N = H*W)."""
    r = np.asarray(raw, dtype=np.float32)
    r = r[0] if r.ndim == 4 else r
    c, h, w = r.shape
    if c != lay.channels:
        raise ValueError(f"level tensor has {c} channels, the layout needs {lay.channels} "
                         f"(n_cls={lay.n_cls}, n_kpt={lay.n_kpt})")
    flat = r.reshape(c, h * w).T                                          # (N, C)
    ax, ay = grid_anchors(h, w, stride)
    s = lay.slices
    d = np.exp(np.clip(flat[:, s["box"]], -6.0, 6.0)) * stride            # l, t, r, b
    box = np.stack([ax - d[:, 0], ay - d[:, 1], ax + d[:, 2], ay + d[:, 3]], axis=1)
    k = lay.n_kpt
    off = flat[:, s["kp"]].reshape(-1, k, 2) * lay.offset_scale * stride
    kxy = np.stack([ax[:, None] + off[:, :, 0], ay[:, None] + off[:, :, 1]], axis=2)
    return {"cls": _sigmoid(flat[:, s["cls"]]), "box": box, "kxy": kxy, "kscore": _sigmoid(flat[:, s["kp_score"]]),
            "kvis": _sigmoid(flat[:, s["kp_vis"]])}


def _refine_aim(kxy: np.ndarray, hm: np.ndarray, off: np.ndarray, stride: int, radius_px: float, prim: Sequence[int]
                ) -> np.ndarray:
    """Soft-argmax of the stride-4 heatmap around the regressed primary point; offsets give sub-cell precision."""
    out = kxy.copy()
    h, w = hm.shape[-2:]
    for pi, kp in enumerate(prim):
        x, y = kxy[kp]
        cx, cy = int(x / stride), int(y / stride)
        r = max(1, int(np.ceil(radius_px / stride)))
        x0, x1, y0, y1 = max(0, cx - r), min(w, cx + r + 1), max(0, cy - r), min(h, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        win = hm[pi, y0:y1, x0:x1]
        e = np.exp(win - win.max())
        e /= e.sum()
        gy, gx = np.mgrid[y0:y1, x0:x1]
        px = float((e * ((gx + 0.5 + off[pi, y0:y1, x0:x1]) * stride)).sum())
        py = float((e * ((gy + 0.5 + off[len(prim) + pi, y0:y1, x0:x1]) * stride)).sum())
        if np.hypot(px - x, py - y) <= radius_px:
            out[kp] = (px, py)
    return out


def decode_dense(levels: Sequence[np.ndarray], lay: HeadLayout, cfg: Optional[DecodeConfig] = None,
                 refine: Optional[np.ndarray] = None) -> list[dict]:
    """The one-to-one head's dense outputs -> detections in MODEL-INPUT pixels, best first, at most cfg.max_det.
    Each detection: dict(score, cls, box xyxy, kxy (K,2), kscore (K,), kvis (K,))."""
    cfg = cfg or DecodeConfig()
    if len(levels) != len(lay.strides):
        raise ValueError(f"expected {len(lay.strides)} level tensors, got {len(levels)}")
    parts = [decode_level(t, s, lay) for t, s in zip(levels, lay.strides)]
    cls = np.concatenate([p["cls"] for p in parts])
    score = cls.max(axis=1)
    order = np.where(score >= cfg.conf_thr)[0]
    if order.size == 0:
        return []
    order = order[np.argsort(-score[order], kind="stable")]
    box = np.concatenate([p["box"] for p in parts])
    kxy = np.concatenate([p["kxy"] for p in parts])
    ksc = np.concatenate([p["kscore"] for p in parts])
    kvi = np.concatenate([p["kvis"] for p in parts])
    lvl_of = np.concatenate([np.full(len(p["cls"]), i) for i, p in enumerate(parts)])
    if cfg.nms_iou is not None:                                           # off by default: the o2o head does not duplicate
        from ..quality.evaluators.decode import nms
        order = order[nms(box[order], score[order], cfg.nms_iou)]
    hm = off = None
    if cfg.refine and refine is not None and lay.refine_stride and lay.primary:
        r = np.asarray(refine, dtype=np.float32)
        r = r[0] if r.ndim == 4 else r
        p = len(lay.primary)
        hm, off = r[:p], r[p:3 * p]
    dets = []
    for i in order[:cfg.max_det]:
        k = kxy[i]
        if hm is not None:
            k = _refine_aim(k, hm, off, lay.refine_stride, cfg.refine_radius_px, lay.primary)
        dets.append({"score": float(score[i]), "cls": int(cls[i].argmax()), "box": box[i].copy(), "kxy": k,
                     "kscore": ksc[i].copy(), "kvis": kvi[i].copy(), "level": int(lvl_of[i])})
    return dets


def to_predictions(dets: list[dict], lb, lay: HeadLayout) -> list[Prediction]:
    """Detections in input pixels -> Prediction in ORIGINAL image pixels (undo the letterbox). keypoint conf = score of the
    point (calibrated similarity), visibility = the vis head."""
    preds = []
    for d in dets:
        x1, y1, x2, y2 = (d["box"] - [lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y]) / lb.scale
        x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
        x2, y2 = min(float(lb.orig_w), float(x2)), min(float(lb.orig_h), float(y2))
        kp = np.zeros((lay.n_kpt, 3), dtype=np.float64)
        kp[:, 0] = (d["kxy"][:, 0] - lb.pad_x) / lb.scale
        kp[:, 1] = (d["kxy"][:, 1] - lb.pad_y) / lb.scale
        kp[:, 2] = d["kscore"]
        preds.append(Prediction((x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)), d["score"], kp, d["cls"],
                                d["kvis"].astype(np.float64)))
    return preds
