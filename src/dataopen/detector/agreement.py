"""Annotation double-check: IoU agreement between two independent label sources (target: > 0.85 for matched persons).

The engine's labels come from the skeleton and mesh hull. A second, independent source is any of: instance-mask boxes rendered by
the game, a second annotation pass, or a trained detector's detections. `compare_coco` takes two COCO Keypoints files;
`agreement_with_evaluator` uses a model as the second source. Disagreements are listed per image so they can go to review.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..quality.metrics import iou_xywh


def load_coco_boxes(path: str | Path) -> dict[str, list[dict]]:
    d = json.loads(Path(path).read_text())
    imgs = {im["id"]: im["file_name"] for im in d["images"]}
    out: dict[str, list[dict]] = {name: [] for name in imgs.values()}
    for a in d["annotations"]:
        out[imgs[a["image_id"]]].append({"bbox": a["bbox"], "cls": a.get("category_id", 1) - 1,
                                         "kpts": np.asarray(a.get("keypoints", []), dtype=float).reshape(-1, 3)})
    return out


def _greedy_match(a: Sequence[dict], b: Sequence[dict]):
    pairs, used = [], set()
    cand = sorted(((iou_xywh(x["bbox"], y["bbox"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    ua = set()
    for iou, i, j in cand:
        if iou <= 0 or i in ua or j in used:
            continue
        ua.add(i)
        used.add(j)
        pairs.append((i, j, iou))
    return pairs


def compare(a: dict[str, list[dict]], b: dict[str, list[dict]], thr: float = 0.85) -> dict:
    ious, per_image, only_a, only_b, cls_mismatch = [], [], 0, 0, 0
    names = sorted(set(a) & set(b))
    for name in names:
        pairs = _greedy_match(a[name], b[name])
        only_a += len(a[name]) - len(pairs)
        only_b += len(b[name]) - len(pairs)
        for i, j, iou in pairs:
            ious.append(iou)
            cls_mismatch += int(a[name][i]["cls"] != b[name][j]["cls"])
        low = [round(iou, 3) for _, _, iou in pairs if iou < thr]
        if low or len(pairs) < max(len(a[name]), len(b[name])):     # a low IoU, or someone is left unmatched
            per_image.append({"image": name, "n_a": len(a[name]), "n_b": len(b[name]), "low_iou": low})
    arr = np.array(ious) if ious else np.zeros(0)
    rep = {"images_compared": len(names), "images_only_in_one": len(set(a) ^ set(b)), "matched_persons": len(ious),
           "mean_iou": round(float(arr.mean()), 4) if len(arr) else None,
           "share_iou_ge_thr": round(float((arr >= thr).mean()), 4) if len(arr) else None, "thr": thr,
           "unmatched_in_a": only_a, "unmatched_in_b": only_b, "class_mismatch": cls_mismatch,
           "images_to_review": per_image[:200]}
    total = len(ious) + only_a + only_b
    rep["agreement_rate"] = round(float((arr >= thr).sum() / max(1, len(ious) + only_a)), 4) if total else None
    rep["passes"] = bool(rep["share_iou_ge_thr"] is not None and rep["share_iou_ge_thr"] > thr and only_a == 0 and only_b == 0)
    return rep


def compare_coco(path_a: str | Path, path_b: str | Path, thr: float = 0.85) -> dict:
    return compare(load_coco_boxes(path_a), load_coco_boxes(path_b), thr)


def agreement_with_evaluator(items, evaluator, thr: float = 0.85, conf: float = 0.25, limit: Optional[int] = None) -> dict:
    """A detector as the second annotator: IoU of each labeled person with the model's best box (class-agnostic)."""
    from .data import load_image
    a, b = {}, {}
    for it in (items[:limit] if limit else items):
        name = str(it.path)
        a[name] = [{"bbox": [float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])], "cls": int(c)}
                   for x, c in zip(it.boxes, it.cls)]
        preds = [p for p in evaluator.predict([load_image(it.path)])[0] if p.score >= conf]
        b[name] = [{"bbox": list(p.bbox), "cls": int(p.class_id or 0)} for p in preds]
    return compare(a, b, thr)
