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


def keypoint_similarity(gt_kp: np.ndarray, pred_kp: np.ndarray, area: float, sigmas: Sequence[float]) -> np.ndarray:
    """Per-keypoint COCO similarity exp(-d^2 / (2 s^2 k^2)), (K,). Meaningful where the GT keypoint has v > 0."""
    k2 = (2.0 * np.asarray(sigmas, dtype=np.float64)) ** 2
    d2 = ((gt_kp[:, 0] - pred_kp[:, 0]) ** 2 + (gt_kp[:, 1] - pred_kp[:, 1]) ** 2)
    return np.exp(-d2 / (2.0 * max(area, 1e-9) * k2 + np.finfo(float).eps))


def oks(gt_kp: np.ndarray, pred_kp: np.ndarray, area: float, sigmas: Sequence[float],
        weights: Optional[Sequence[float]] = None) -> float:
    """Object Keypoint Similarity (COCO). gt_kp, pred_kp: (K, 3); only GT keypoints with v > 0 count.
    `weights` (per keypoint, e.g. SkeletonSchema.oks_weights()) turn the mean into a weighted mean: the aim point can
    count more than a derived midpoint. None = the plain COCO mean."""
    labeled = gt_kp[:, 2] > 0
    if not labeled.any() or area <= 0:
        return 0.0
    sim = keypoint_similarity(gt_kp, pred_kp, area, sigmas)
    if weights is None:
        return float(sim[labeled].mean())
    w = np.asarray(weights, dtype=np.float64)[labeled]
    return float((sim[labeled] * w).sum() / w.sum()) if w.sum() > 0 else 0.0


class DefaultMetricCalculator(IQualityMetricCalculator):
    """Greedy matching by OKS (IoU for box-only models), highest score first."""

    def __init__(self, schema: SkeletonSchema, match_conf: float = 0.25, oks_match_thr: float = 0.5,
                 phantom_conf: float = 0.6, iou_explain: float = 0.3, focus_hit_rel: float = 0.04) -> None:
        self.schema, self.match_conf, self.oks_thr = schema, match_conf, oks_match_thr
        self.phantom_conf, self.iou_explain, self.focus_hit_rel = phantom_conf, iou_explain, focus_hit_rel
        self.sigmas = schema.oks_sigmas()
        self.weights = schema.oks_weights()            # derived points (neck, pelvis, ...) count less, the aim point more
        self.primary = schema.primary_idx()

    def _oks(self, a: Annotation, p: Prediction) -> float:
        return oks(a.keypoints, p.keypoints, a.area, self.sigmas, self.weights)

    def _focus(self, person: MatchedPerson, a: Annotation, p: Optional[Prediction]) -> None:
        lab = [i for i in self.primary if a.keypoints[i, 2] > 0]
        if not lab:
            return
        if p is None:                                  # nothing found at all: the aim point is lost
            person.focus_oks, person.focus_conf, person.focus_hit = 0.0, 0.0, False
            return
        if p.keypoints is None:                        # a box-only model has no aim point to judge
            return
        sim = keypoint_similarity(a.keypoints, p.keypoints, a.area, self.sigmas)
        err = np.hypot(*(a.keypoints[lab, :2] - p.keypoints[lab, :2]).T).mean()
        person.focus_oks = float(sim[lab].mean())
        person.focus_conf = float(p.keypoints[lab, 2].mean())
        person.focus_err_rel = float(err / max(a.bbox[3], 1e-9))
        person.focus_hit = bool(person.focus_err_rel <= self.focus_hit_rel)

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
                sim = iou if p.keypoints is None else self._oks(a, p)
                if (sim, iou) > best_key:
                    best_key, best_i = (sim, iou), i
            person = MatchedPerson(a.entity_id)
            matched: Optional[Prediction] = None
            if best_i >= 0 and (best_key[0] > 0 or best_key[1] > 0):
                p = matched = confident[best_i]
                person.oks, person.iou, person.score = best_key[0], best_key[1], p.score
                labeled = a.keypoints[:, 2] > 0
                if p.keypoints is not None and labeled.any():
                    person.keypoint_conf_mean = float(np.mean(p.keypoints[labeled, 2]))
                    sim = keypoint_similarity(a.keypoints, p.keypoints, a.area, self.sigmas)
                    person.kp_sim = [round(float(v), 3) if labeled[j] else None for j, v in enumerate(sim)]
                if p.class_id is not None and len(self.schema.classes) > 1:
                    person.class_ok = bool(p.class_id == a.class_id)
                if p.visibility is not None and labeled.any():
                    person.vis_acc = float(np.mean((np.asarray(p.visibility)[labeled] > 0.5) == (a.keypoints[labeled, 2] == 2)))
                if best_key[0] >= self.oks_thr:
                    used.add(best_i)
            self._focus(person, a, matched)
            m.persons.append(person)
        # confident predictions that overlap a GT box but put the keypoints somewhere else
        for p in confident:
            if p.keypoints is None:
                continue
            for a in gt:
                if iou_xywh(a.bbox, p.bbox) >= 0.5 and self._oks(a, p) < 0.3:
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
            foc = [p for p in m.persons if p.focus_oks is not None]
            if foc:
                m.focus_evaluated = True
                f = np.array([p.focus_oks for p in foc])
                m.mean_focus_oks, m.min_focus_oks = float(f.mean()), float(f.min())
                m.focus_hit_rate = float(np.mean([bool(p.focus_hit) for p in foc]))
            cls = [p.class_ok for p in m.persons if p.class_ok is not None and p.oks >= self.oks_thr]
            if cls:
                m.class_accuracy = float(np.mean(cls))
            va = [p.vis_acc for p in m.persons if p.vis_acc is not None]
            if va:
                m.mean_vis_acc = float(np.mean(va))
        return m
