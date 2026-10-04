"""OKS / IoU and ground-truth <-> prediction matching (vectorized, COCO conventions)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.models import Annotation
from ..core.schema import SkeletonSchema
from .interfaces import IQualityMetricCalculator
from .types import MatchedPerson, Prediction, QualityMetrics


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def oks(gt_kp: np.ndarray, pred_kp: np.ndarray, area: float, sigmas: Sequence[float]) -> float:
    """Object Keypoint Similarity (COCO). gt_kp, pred_kp: (K, 3); only GT keypoints with v > 0 count."""
    labeled = gt_kp[:, 2] > 0
    if not labeled.any() or area <= 0:
        return 0.0
    k2 = (2.0 * np.asarray(sigmas, dtype=np.float64)) ** 2
    d2 = ((gt_kp[:, 0] - pred_kp[:, 0]) ** 2 + (gt_kp[:, 1] - pred_kp[:, 1]) ** 2)
    e = d2 / (2.0 * area * k2 + np.finfo(float).eps)
    return float(np.exp(-e)[labeled].mean())


class DefaultMetricCalculator(IQualityMetricCalculator):
    """Greedy matching by OKS (IoU for box-only models), highest score first."""

    def __init__(self, schema: SkeletonSchema, match_conf: float = 0.25, oks_match_thr: float = 0.5,
                 phantom_conf: float = 0.6, iou_explain: float = 0.3) -> None:
        self.schema, self.match_conf, self.oks_thr = schema, match_conf, oks_match_thr
        self.phantom_conf, self.iou_explain = phantom_conf, iou_explain
        self.sigmas = schema.oks_sigmas()

    def compute(self, gt: Sequence[Annotation], preds: Optional[Sequence[Prediction]], backend: str = "",
                latency_ms: float = 0.0, explained_boxes: Sequence[Sequence[float]] = ()) -> QualityMetrics:
        m = QualityMetrics(evaluated=preds is not None, backend=backend, latency_ms=latency_ms, n_gt=len(gt))
        if preds is None:
            return m
        preds = sorted(preds, key=lambda p: -p.score)
        confident = [p for p in preds if p.score >= self.match_conf]
        m.n_pred = len(confident)
        used: set[int] = set()
        box_only = bool(confident) and all(p.keypoints is None for p in confident)
        for a in gt:
            best_key, best_i = (-1.0, -1.0), -1
            for i, p in enumerate(confident):
                if i in used:
                    continue
                iou = iou_xywh(a.bbox, p.bbox)
                sim = iou if p.keypoints is None else oks(a.keypoints, p.keypoints, a.area, self.sigmas)
                if (sim, iou) > best_key:
                    best_key, best_i = (sim, iou), i
            person = MatchedPerson(a.entity_id)
            if best_i >= 0 and (best_key[0] > 0 or best_key[1] > 0):
                p = confident[best_i]
                person.oks, person.iou, person.score = best_key[0], best_key[1], p.score
                labeled = a.keypoints[:, 2] > 0
                if p.keypoints is not None and labeled.any():
                    person.keypoint_conf_mean = float(np.mean(p.keypoints[labeled, 2]))
                if best_key[0] >= self.oks_thr:
                    used.add(best_i)
            m.persons.append(person)
        # confident predictions that overlap a GT box but put the keypoints somewhere else
        for p in confident:
            if p.keypoints is None:
                continue
            for a in gt:
                if iou_xywh(a.bbox, p.bbox) >= 0.5 and oks(a.keypoints, p.keypoints, a.area, self.sigmas) < 0.3:
                    m.max_disagreement_score = max(m.max_disagreement_score, p.score)
        explained = [a.bbox for a in gt] + [tuple(b) for b in explained_boxes]
        m.n_phantom = sum(1 for p in preds if p.score >= self.phantom_conf
                          and not any(iou_xywh(e, p.bbox) >= self.iou_explain for e in explained))
        if m.persons:
            vals = np.array([p.oks for p in m.persons])      # for box-only models "oks" falls back to IoU
            m.mean_oks, m.min_oks, m.oks_spread = float(vals.mean()), float(vals.min()), float(vals.std())
            m.recall = float(np.mean([p.iou >= 0.5 for p in m.persons])) if box_only else float((vals >= self.oks_thr).mean())
            m.mean_iou = float(np.mean([p.iou for p in m.persons]))
            m.mean_conf = float(np.mean([p.score for p in m.persons]))
        return m
