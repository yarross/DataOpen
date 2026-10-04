"""Pre/post-processing shared by every inference backend (ONNX Runtime, TensorRT, RKNN): letterbox, YOLOv8-pose and
D-FINE decoding, NMS, and mapping a model's keypoints onto the dataset schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..types import Prediction


@dataclass
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float
    in_w: int
    in_h: int
    orig_w: int
    orig_h: int


def resize_bilinear(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Pillow when present (fast, exact), otherwise a vectorized numpy bilinear resize."""
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    except ImportError:
        h, w, c = img.shape
        ys = (np.arange(new_h) + 0.5) * h / new_h - 0.5
        xs = (np.arange(new_w) + 0.5) * w / new_w - 0.5
        y0, x0 = np.clip(np.floor(ys).astype(int), 0, h - 1), np.clip(np.floor(xs).astype(int), 0, w - 1)
        y1, x1 = np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)
        wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
        f = img.astype(np.float32)
        top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
        bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
        return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


def letterbox(img: np.ndarray, in_w: int, in_h: int, pad_value: int = 114) -> tuple[np.ndarray, Letterbox]:
    h, w, _ = img.shape
    r = min(in_w / w, in_h / h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    resized = img if (nw, nh) == (w, h) else resize_bilinear(img, nw, nh)
    out = np.full((in_h, in_w, 3), pad_value, dtype=np.uint8)
    px, py = (in_w - nw) // 2, (in_h - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, Letterbox(r, px, py, in_w, in_h, w, h)


def to_chw_float(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_r = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


class KeypointMap:
    """Our keypoint j = weighted mean of model keypoints (index, weight); confidence = min over the parts."""

    def __init__(self, parts: Sequence[Sequence[tuple[int, float]]]) -> None:
        self.parts = [list(p) for p in parts]

    @staticmethod
    def identity(n: int) -> "KeypointMap":
        return KeypointMap([[(i, 1.0)] for i in range(n)])

    @staticmethod
    def coco17_to_human13() -> "KeypointMap":
        """head=ears, neck=mid-shoulders, pelvis=mid-hips (COCO has neither a neck nor a pelvis point)."""
        return KeypointMap([
            [(3, .5), (4, .5)], [(5, .5), (6, .5)], [(5, 1)], [(6, 1)], [(7, 1)], [(8, 1)], [(9, 1)], [(10, 1)],
            [(11, .5), (12, .5)], [(13, 1)], [(14, 1)], [(15, 1)], [(16, 1)]])

    def apply(self, kp: np.ndarray) -> np.ndarray:
        """kp: (M, 3) model keypoints -> (K, 3) schema keypoints."""
        out = np.zeros((len(self.parts), 3), dtype=np.float64)
        for j, parts in enumerate(self.parts):
            w = np.array([p[1] for p in parts])
            idx = [p[0] for p in parts]
            out[j, :2] = (kp[idx, :2] * w[:, None]).sum(axis=0) / w.sum()
            out[j, 2] = kp[idx, 2].min()
        return out


def decode_yolov8_pose(raw: np.ndarray, lb: Letterbox, conf_thr: float, iou_thr: float,
                       kmap: Optional[KeypointMap] = None, max_det: int = 100) -> list[Prediction]:
    """raw: (1, 4+1+3*M, N) or (1, N, 4+1+3*M): cx, cy, w, h, score, then (x, y, conf) per keypoint, input-pixel units."""
    a = np.asarray(raw)[0]
    # The channel axis is the one with 5 + 3*M entries (M keypoints). Do not guess from "which axis is longer": a model
    # with built-in NMS can have fewer detections than channels.
    axes = [ax for ax in (0, 1) if a.shape[ax] >= 8 and (a.shape[ax] - 5) % 3 == 0]
    if not axes:
        raise ValueError(f"unexpected YOLOv8-pose output shape {a.shape} (need 4 box + 1 score + 3 per keypoint channels)")
    ch_axis = min(axes, key=lambda ax: a.shape[ax])
    if ch_axis == 1:                   # (N, C) -> (C, N)
        a = a.T
    c = a.shape[0]
    nk = (c - 5) // 3
    score = a[4]
    sel = np.where(score >= conf_thr)[0]
    if sel.size == 0:
        return []
    cx, cy, w, h = a[0, sel], a[1, sel], a[2, sel], a[3, sel]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms(xyxy, score[sel], iou_thr)[:max_det]
    preds = []
    for k in keep:
        i = sel[k]
        x1, y1, x2, y2 = (xyxy[k] - [lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y]) / lb.scale
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(lb.orig_w), x2), min(float(lb.orig_h), y2)
        kp = a[5:, i].reshape(nk, 3).astype(np.float64).copy()
        kp[:, 0] = (kp[:, 0] - lb.pad_x) / lb.scale
        kp[:, 1] = (kp[:, 1] - lb.pad_y) / lb.scale
        if kmap is not None:
            kp = kmap.apply(kp)
        preds.append(Prediction((float(x1), float(y1), float(x2 - x1), float(y2 - y1)), float(score[i]), kp))
    return preds


def decode_dfine(labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, conf_thr: float,
                 person_label: int = 0) -> list[Prediction]:
    """D-FINE ONNX export: boxes are xyxy in ORIGINAL pixels (the export scales by orig_target_sizes). Box-only."""
    out = []
    lab, box, sc = np.asarray(labels)[0], np.asarray(boxes)[0], np.asarray(scores)[0]
    for l, b, s in zip(lab, box, sc):
        if int(l) == person_label and float(s) >= conf_thr:
            out.append(Prediction((float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])), float(s), None))
    return out
