"""Evaluation of the UI detector: AP50 per class and overall, and the numbers the assistive use cares about (localisation of the pointer, how many
false elements per screen, how small an element is still found)."""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .infer import Detection
from .taxonomy import NAMES, TARGET_NAMES


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores)
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]  # monotone envelope
    pts = np.linspace(0, 1, 101)
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in pts]))


@dataclass
class Report:
    map50: float
    ap50: dict[str, float]
    recall_targets: float  # fraction of ground-truth interface elements found (any class confusion counts as a miss)
    fp_per_screen: float  # confident (>= 0.5) detections of target classes that match nothing, per screen
    cursor_err_px: float  # median distance between detected and true pointer top-left, in INPUT pixels
    cursor_found: float
    by_size: dict[str, float] = field(default_factory=dict)  # recall by the longer side of the box (input px)
    n_images: int = 0


def evaluate(preds: list[list[Detection]], gts: list[tuple[np.ndarray, np.ndarray]], iou_thr: float = 0.5) -> Report:
    """preds[i]: detections of image i (input pixels); gts[i] = (boxes (n,4), classes (n,))."""
    n_cls = len(NAMES)
    sc: list[list[float]] = [[] for _ in range(n_cls)]
    tpl: list[list[int]] = [[] for _ in range(n_cls)]
    n_gt = np.zeros(n_cls, int)
    found = total = fp_hi = 0
    cur_err, cur_hit, cur_n = [], 0, 0
    size_hit = {"<16": [0, 0], "16-32": [0, 0], "32-64": [0, 0], "64-128": [0, 0], ">=128": [0, 0]}
    cid = NAMES.index("cursor")
    for p, (gb, gc) in zip(preds, gts):
        for c in range(n_cls):
            n_gt[c] += int((gc == c).sum())
        pb = np.array([d.box for d in p], float).reshape(-1, 4)
        pc = np.array([d.cls for d in p], int)
        ps = np.array([d.score for d in p], float)
        used = np.zeros(len(gb), bool)
        order = np.argsort(-ps)
        io = iou_matrix(pb, gb) if len(pb) and len(gb) else np.zeros((len(pb), len(gb)))
        for i in order:
            c = pc[i]
            cand = [j for j in range(len(gb)) if gc[j] == c and not used[j] and io[i, j] >= iou_thr]
            sc[c].append(ps[i])
            if cand:
                j = max(cand, key=lambda j: io[i, j])
                used[j] = True
                tpl[c].append(1)
            else:
                tpl[c].append(0)
                if NAMES[c] in TARGET_NAMES and ps[i] >= 0.5 and not (io[i] >= iou_thr).any():
                    fp_hi += 1
        for j in range(len(gb)):
            if NAMES[gc[j]] in TARGET_NAMES:
                total += 1
                side = max(gb[j, 2] - gb[j, 0], gb[j, 3] - gb[j, 1])
                k = "<16" if side < 16 else "16-32" if side < 32 else "32-64" if side < 64 else "64-128" if side < 128 else ">=128"
                size_hit[k][1] += 1
                if used[j]:
                    found += 1
                    size_hit[k][0] += 1
            elif gc[j] == cid:
                cur_n += 1
                cands = [i for i in range(len(pb)) if pc[i] == cid and ps[i] >= 0.4]
                if cands:
                    i = max(cands, key=lambda i: ps[i])
                    cur_hit += 1
                    cur_err.append(float(np.hypot(pb[i, 0] - gb[j, 0], pb[i, 1] - gb[j, 1])))
    ap = {NAMES[c]: average_precision(np.array(sc[c]), np.array(tpl[c]), int(n_gt[c])) for c in range(n_cls)}
    vals = [v for v in ap.values() if not np.isnan(v)]
    return Report(
        float(np.mean(vals)) if vals else float("nan"),
        ap,
        found / max(total, 1),
        fp_hi / max(len(preds), 1),
        float(np.median(cur_err)) if cur_err else float("nan"),
        cur_hit / max(cur_n, 1),
        {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in size_hit.items()},
        len(preds),
    )
