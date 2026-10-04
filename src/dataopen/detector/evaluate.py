"""Detector metrics (numpy only): the targets from the brief are `map50` > 0.90 (bbox) and `kp_ap_oks50` > 0.85 (keypoints).

  map50          class-aware COCO AP at IoU 0.5 (101-point interpolation), mean over classes that occur
  ap50_person    the same, ignoring the class (a wrong team does not hurt)
  kp_ap_oks50    COCO keypoint AP at OKS 0.5, OKS weighted by the SCHEMA (aim point 3x), class-agnostic
  kp_ap_oks50_u  the same with the plain COCO (unweighted) OKS
  recall_oks50   share of ground-truth persons found with weighted OKS >= 0.5 at the operating point (score >= conf)
  aim_hit_rate   share of persons whose primary keypoint is within `hit_rel` of the person's height
  class_acc      share of found persons with the right class
Every number is also reported for the HARD subset (images whose closed-loop weight >= hard_weight) so progress on the frames
that matter is visible; hard frames are where a model that is merely good on average fails.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.models import Annotation
from ..core.schema import SkeletonSchema
from ..quality.interfaces import IModelEvaluator
from ..quality.metrics import iou_xywh, oks
from ..quality.types import Prediction


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO AP: 101-point interpolated precision over recall."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(np.float64)
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(rec, r, side="left")
        ap += prec[idx] if idx < len(prec) else 0.0
    return float(ap / 101.0)


def _match_image(gts: Sequence[Annotation], preds: Sequence[Prediction], sim, thr: float, same_class: bool):
    """Greedy by score: each prediction takes the best still-free ground truth with sim >= thr. -> tp flags aligned with preds."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    used: set[int] = set()
    tp = np.zeros(len(preds), dtype=bool)
    for i in order:
        best, bj = thr, -1
        for j, g in enumerate(gts):
            if j in used or (same_class and preds[i].class_id is not None and preds[i].class_id != g.class_id):
                continue
            s = sim(g, preds[i])
            if s >= best:
                best, bj = s, j
        if bj >= 0:
            used.add(bj)
            tp[i] = True
    return tp


def evaluate_predictions(gts: Sequence[Sequence[Annotation]], preds: Sequence[Sequence[Prediction]], schema: SkeletonSchema,
                         conf: float = 0.25, hit_rel: float = 0.04, weights: Optional[Sequence[float]] = None,
                         hard_weight: float = 1.5) -> dict:
    """gts/preds: per image. `weights`: per-image closed-loop weight (None = no hard subset)."""
    out = _evaluate(gts, preds, schema, conf, hit_rel)
    if weights is not None:
        hard = [i for i, w in enumerate(weights) if w >= hard_weight]
        out["n_hard_images"] = len(hard)
        if hard:
            sub = _evaluate([gts[i] for i in hard], [preds[i] for i in hard], schema, conf, hit_rel)
            out["hard"] = {k: v for k, v in sub.items() if k not in ("n_images", "n_gt")}
    return out


def _evaluate(gts, preds, schema: SkeletonSchema, conf: float, hit_rel: float) -> dict:
    sig, w = schema.oks_sigmas(), schema.oks_weights()
    prim = schema.primary_idx()
    n_cls = len(schema.classes)

    def iou_sim(g, p):
        return iou_xywh(g.bbox, p.bbox)

    def oks_w(g, p):
        return oks(g.keypoints, p.keypoints, g.area, sig, w) if p.keypoints is not None else 0.0

    def oks_u(g, p):
        return oks(g.keypoints, p.keypoints, g.area, sig, None) if p.keypoints is not None else 0.0

    def ap_for(sim, thr, per_class: Optional[int]):
        scores, tps, n_gt = [], [], 0
        for g_img, p_img in zip(gts, preds):
            gi = [g for g in g_img if per_class is None or g.class_id == per_class]
            pi = [p for p in p_img if per_class is None or p.class_id == per_class or p.class_id is None]
            n_gt += len(gi)
            if pi:
                tps.append(_match_image(gi, pi, sim, thr, same_class=False))
                scores.append(np.array([p.score for p in pi]))
        if n_gt == 0:
            return float("nan")
        return average_precision(np.concatenate(scores) if scores else np.zeros(0), np.concatenate(tps) if tps else np.zeros(0, bool), n_gt)

    per_cls = [ap_for(iou_sim, 0.5, c) for c in range(n_cls)]
    per_cls = [a for a in per_cls if not np.isnan(a)]
    res = {"n_images": len(gts), "n_gt": int(sum(len(g) for g in gts)),
           "map50": float(np.mean(per_cls)) if per_cls else float("nan"),
           "ap50_person": ap_for(iou_sim, 0.5, None),
           "kp_ap_oks50": ap_for(oks_w, 0.5, None), "kp_ap_oks50_u": ap_for(oks_u, 0.5, None)}
    # operating point: detections with score >= conf
    found = hits = cls_ok = cls_n = total = 0
    for g_img, p_img in zip(gts, preds):
        conf_p = sorted([p for p in p_img if p.score >= conf], key=lambda p: -p.score)
        used: set[int] = set()
        for g in g_img:
            total += 1
            best, bj = 0.5, -1
            for j, p in enumerate(conf_p):
                if j in used or p.keypoints is None:
                    continue
                s = oks_w(g, p)
                if s >= best:
                    best, bj = s, j
            if bj < 0:
                continue
            used.add(bj)
            found += 1
            p = conf_p[bj]
            if p.class_id is not None and n_cls > 1:
                cls_n += 1
                cls_ok += int(p.class_id == g.class_id)
            lab = [i for i in prim if g.keypoints[i, 2] > 0]
            if lab:
                err = np.hypot(*(g.keypoints[lab, :2] - p.keypoints[lab, :2]).T).mean() / max(g.bbox[3], 1e-9)
                hits += int(err <= hit_rel)
    res["recall_oks50"] = found / total if total else float("nan")
    res["aim_hit_rate"] = hits / total if (total and prim) else float("nan")
    res["class_acc"] = cls_ok / cls_n if cls_n else float("nan")
    return {k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else v) for k, v in res.items()}


def evaluate_evaluator(ev: IModelEvaluator, items, schema: SkeletonSchema, conf: float = 0.25, batch: int = 1,
                       load=None) -> dict:
    """Run any `IModelEvaluator` (torch, ONNX, INT8 QDQ, RKNN-simulated...) over dataset items and score it."""
    from .data import load_image
    load = load or load_image
    gts, preds, weights = [], [], []
    for it in items:
        img = load(it.path)
        preds.append(ev.predict([img])[0])
        gts.append([Annotation(i, it.kpts[i], (float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])), {}, int(it.cls[i]))
                    for i, b in enumerate(it.boxes)])
        weights.append(it.weight)
    return evaluate_predictions(gts, preds, schema, conf, weights=weights)


def check_targets(metrics: dict, map50: float = 0.90, kp_oks50: float = 0.85) -> list[str]:
    """Failures against the brief's targets (empty = all met)."""
    out = []
    if not metrics.get("map50", 0.0) > map50:
        out.append(f"map50 {metrics.get('map50')} <= {map50}")
    if not metrics.get("kp_ap_oks50", 0.0) > kp_oks50:
        out.append(f"kp_ap_oks50 {metrics.get('kp_ap_oks50')} <= {kp_oks50}")
    return out
