"""Skeleton overlays (numpy only): used by `doctor` and `preview` so a human can eyeball labels."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .models import Annotation
from .schema import SkeletonSchema

LEFT, RIGHT, CENTER = (60, 120, 255), (255, 70, 70), (60, 220, 90)   # blue / red / green
BOX = (255, 220, 0)


def _color(name: str) -> tuple[int, int, int]:
    return LEFT if name.startswith("l_") else RIGHT if name.startswith("r_") else CENTER


def _line(img: np.ndarray, p0, p1, color, thickness: int = 1) -> None:
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    xs = np.linspace(p0[0], p1[0], n).round().astype(int)
    ys = np.linspace(p0[1], p1[1], n).round().astype(int)
    h, w, _ = img.shape
    r = thickness // 2
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            x, y = xs + dx, ys + dy
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            img[y[ok], x[ok]] = color


def _dot(img: np.ndarray, x: float, y: float, color, r: int, hollow: bool) -> None:
    h, w, _ = img.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dx * dx + dy * dy
            if d2 > r * r or (hollow and d2 < (r - 1) ** 2):
                continue
            xx, yy = int(round(x)) + dx, int(round(y)) + dy
            if 0 <= xx < w and 0 <= yy < h:
                img[yy, xx] = color


def draw_annotations(img: np.ndarray, annotations: Sequence[Annotation], schema: SkeletonSchema,
                     draw_boxes: bool = True) -> np.ndarray:
    """Returns a copy. Visible joints are filled dots, occluded ones hollow; left blue, right red."""
    out = np.ascontiguousarray(img, dtype=np.uint8).copy()
    scale = max(1, round(min(out.shape[:2]) / 360))
    for a in annotations:
        if draw_boxes:
            x, y, w, h = (int(round(v)) for v in a.bbox)
            for p0, p1 in (((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)),
                           ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))):
                _line(out, p0, p1, BOX, 1)
        kp = a.keypoints
        for ka, kb in schema.edges:
            ia, ib = schema.index(ka), schema.index(kb)
            if kp[ia, 2] > 0 and kp[ib, 2] > 0:
                _line(out, kp[ia, :2], kp[ib, :2], _color(kb if kb[:2] in ("l_", "r_") else ka), scale)
        for i, name in enumerate(schema.keypoints):
            if kp[i, 2] > 0:
                _dot(out, kp[i, 0], kp[i, 1], _color(name), 2 * scale, hollow=kp[i, 2] == 1)
    return out


def contact_sheet(tiles: Sequence[np.ndarray], cols: int = 4, tile_width: int = 480) -> np.ndarray:
    """Nearest-neighbour resize to a common width and tile into a grid."""
    resized = []
    for t in tiles:
        h, w, _ = t.shape
        th = max(1, round(h * tile_width / w))
        ys = (np.arange(th) * h / th).astype(int)
        xs = (np.arange(tile_width) * w / tile_width).astype(int)
        resized.append(t[ys][:, xs])
    th = max(r.shape[0] for r in resized)
    rows = []
    for i in range(0, len(resized), cols):
        row = resized[i:i + cols]
        row = [np.pad(r, ((0, th - r.shape[0]), (0, 0), (0, 0))) for r in row]
        row += [np.zeros((th, tile_width, 3), dtype=np.uint8)] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)
