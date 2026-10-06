"""Host-side decoding of UiNet outputs (numpy only: the same code serves evaluation, the ONNX service and tests)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..quality.evaluators.decode import nms
from .model import STRIDES


@dataclass(frozen=True)
class Detection:
    cls: int
    score: float
    box: tuple[float, float, float, float]  # xyxy in input pixels


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def decode(
    levels: list[np.ndarray], n_cls: int, conf: float = 0.3, nms_iou: float = 0.5, max_det: int = 60, strides=STRIDES
) -> list[Detection]:
    """levels: per stride a (n_cls + 4, H, W) array [class logits | raw ltrb] for ONE image."""
    boxes, scores, classes = [], [], []
    for lv, s in zip(levels, strides):
        h, w = lv.shape[1:]
        p = _sigmoid(lv[:n_cls])  # (C, H, W)
        best, cid = p.max(axis=0), p.argmax(axis=0)
        ys, xs = np.nonzero(best >= conf)
        if len(ys) == 0:
            continue
        d = np.exp(np.clip(lv[n_cls:, ys, xs], -6, 6)) * s
        cx, cy = (xs + 0.5) * s, (ys + 0.5) * s
        boxes.append(np.stack([cx - d[0], cy - d[1], cx + d[2], cy + d[3]], axis=1))
        scores.append(best[ys, xs])
        classes.append(cid[ys, xs])
    if not boxes:
        return []
    b, sc, cl = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
    out: list[Detection] = []
    for c in np.unique(cl):  # NMS inside each class
        idx = np.nonzero(cl == c)[0]
        for k in nms(b[idx], sc[idx], nms_iou):
            j = idx[k]
            out.append(Detection(int(c), float(sc[j]), tuple(float(v) for v in b[j])))
    out.sort(key=lambda d: -d.score)
    return out[:max_det]
