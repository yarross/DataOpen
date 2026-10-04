"""COCO Keypoints dataset with the augmentations the detector is trained with (numpy + Pillow; tensors at the very end).

Reads what `dataopen collect` writes (`annotations/coco_<split>.json`, `images/`, `DATASET_CARD.json`): the 12-point schema,
classes, per-point OKS sigmas/weights and the closed loop's per-image `weight` (hard examples are sampled more often).
Augmentations: mosaic, mixup, random affine (scale/translate/rotate), HSV, horizontal flip (keypoints swapped through the
schema's flip_idx, so left/right stay correct), photometric smoke and flash overlays (labels untouched).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..core.card import schema_from_card_dict
from ..core.schema import SkeletonSchema
from .config import TrainConfig

PAD = 114


class DataError(ValueError):
    pass


@dataclass
class Item:
    path: Path
    w: int
    h: int
    boxes: np.ndarray        # (N, 4) xyxy, original pixels
    cls: np.ndarray          # (N,) int
    kpts: np.ndarray         # (N, K, 3) x, y, v
    weight: float = 1.0
    meta: Optional[dict] = None


def load_schema_from_dataset(root: Path) -> SkeletonSchema:
    card = Path(root) / "DATASET_CARD.json"
    if not card.exists():
        raise DataError(f"{root} has no DATASET_CARD.json (not a DataOpen dataset)")
    return schema_from_card_dict(json.loads(card.read_text())["skeleton"])


def read_coco(root: Path, split: str, n_kpt: int) -> list[Item]:
    p = Path(root) / "annotations" / f"coco_{split}.json"
    if not p.exists():
        raise DataError(f"{p} not found (did the dataset get exported with the coco format?)")
    d = json.loads(p.read_text())
    by_img: dict[int, list[dict]] = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    items = []
    for im in d["images"]:
        anns = by_img.get(im["id"], [])
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns],
                         dtype=np.float32).reshape(-1, 4)
        cls = np.array([a["category_id"] - 1 for a in anns], dtype=np.int64)
        kp = np.array([a["keypoints"] for a in anns], dtype=np.float32).reshape(-1, n_kpt, 3)
        items.append(Item(Path(root) / im["file_name"], im["width"], im["height"], boxes, cls, kp,
                          float(im.get("weight", 1.0)) or 1.0, {k: im.get(k) for k in ("verdict", "difficulty_score")}))
    return items


def load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except ImportError:
        from ..core.imageio import read_image
        return read_image(path)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from ..quality.evaluators.decode import resize_bilinear
    return img if img.shape[1] == w and img.shape[0] == h else resize_bilinear(img, w, h)


def letterbox(img: np.ndarray, size: int):
    from ..quality.evaluators.decode import letterbox as lb
    return lb(img, size, size, PAD)


# ---- geometry on (boxes, kpts) ---------------------------------------------------------------------------------------

def affine_matrix(src_w: int, src_h: int, out: int, scale: float, deg: float, tx: float, ty: float,
                  pre_shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """3x3 matrix mapping source pixels to output pixels: centre the source, rotate/scale, place at the output centre + shift."""
    c = np.array([[1, 0, -src_w / 2 - pre_shift[0]], [0, 1, -src_h / 2 - pre_shift[1]], [0, 0, 1]], dtype=np.float64)
    a = math.radians(deg)
    r = np.array([[scale * math.cos(a), -scale * math.sin(a), 0], [scale * math.sin(a), scale * math.cos(a), 0], [0, 0, 1]])
    t = np.array([[1, 0, out / 2 + tx], [0, 1, out / 2 + ty], [0, 0, 1]], dtype=np.float64)
    return t @ r @ c


def warp(img: np.ndarray, m: np.ndarray, out: int) -> np.ndarray:
    from PIL import Image
    inv = np.linalg.inv(m)[:2].reshape(-1)
    im = Image.fromarray(img).transform((out, out), Image.AFFINE, tuple(inv), resample=Image.BILINEAR, fillcolor=(PAD,) * 3)
    return np.asarray(im)


def transform_labels(boxes: np.ndarray, kpts: np.ndarray, m: np.ndarray, out: int, min_side: float = 4.0,
                     min_area_ratio: float = 0.2):
    """Apply m to boxes (corner-wise) and keypoints; drop vanished boxes, zero keypoints that left the image (v = 0)."""
    n = len(boxes)
    if n == 0:
        return boxes.reshape(0, 4), kpts.reshape(0, kpts.shape[1], 3), np.zeros(0, bool)
    corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1).reshape(-1, 2)
    pc = (np.c_[corners, np.ones(len(corners))] @ m.T)[:, :2].reshape(n, 4, 2)
    nb = np.concatenate([pc.min(1), pc.max(1)], axis=1)
    area0 = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
    nb_c = nb.copy()
    nb_c[:, [0, 2]] = nb_c[:, [0, 2]].clip(0, out)
    nb_c[:, [1, 3]] = nb_c[:, [1, 3]].clip(0, out)
    area1 = (nb_c[:, 2] - nb_c[:, 0]).clip(0) * (nb_c[:, 3] - nb_c[:, 1]).clip(0)
    keep = ((nb_c[:, 2] - nb_c[:, 0]) > min_side) & ((nb_c[:, 3] - nb_c[:, 1]) > min_side) & (area1 > min_area_ratio * area0)
    k = kpts.copy()
    xy = (np.concatenate([k[..., :2], np.ones(k.shape[:2] + (1,))], axis=-1) @ m.T)[..., :2]
    inside = (xy[..., 0] >= 0) & (xy[..., 0] < out) & (xy[..., 1] >= 0) & (xy[..., 1] < out) & (k[..., 2] > 0)
    k[..., :2] = np.where(inside[..., None], xy, 0.0)
    k[..., 2] = np.where(inside, k[..., 2], 0.0)
    return nb_c[keep], k[keep], keep


def hflip(img: np.ndarray, boxes: np.ndarray, kpts: np.ndarray, flip_idx: Sequence[int]):
    w = img.shape[1]
    out = img[:, ::-1].copy()
    b = boxes.copy()
    b[:, [0, 2]] = w - boxes[:, [2, 0]]
    k = kpts.copy()
    lab = k[..., 2] > 0
    k[..., 0] = np.where(lab, w - k[..., 0], 0.0)
    k = k[:, list(flip_idx)]
    return out, b, k


def hsv_jitter(img: np.ndarray, rng: np.random.Generator, h: float, s: float, v: float) -> np.ndarray:
    from PIL import Image
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.float32)
    g = rng.uniform(-1, 1, 3) * np.array([h, s, v]) + 1.0
    hsv[..., 0] = (hsv[..., 0] * g[0]) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * g[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * g[2], 0, 255)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))


def smoke_flash(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap stand-ins for what the simulator renders: a haze blob or an overexposed flash. Labels are NOT changed."""
    h, w, _ = img.shape
    yy, xx = np.ogrid[:h, :w]
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    r = rng.uniform(0.08, 0.35) * max(h, w)
    a = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))[..., None]
    if rng.random() < 0.5:
        out = img * (1 - 0.85 * a * rng.uniform(0.3, 1.0)) + 172 * 0.85 * a * rng.uniform(0.3, 1.0)
    else:
        out = img * (1 - a * rng.uniform(0.4, 1.0)) + 255 * a * rng.uniform(0.4, 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


class PoseDataset:
    """Index -> (uint8 CHW image tensor-ready array, labels). Torch is only needed by `collate` and the DataLoader."""

    def __init__(self, items: list[Item], cfg: TrainConfig, flip_idx: Sequence[int], n_kpt: int, train: bool = True,
                 imgsz: Optional[int] = None) -> None:
        self.items, self.cfg, self.flip_idx, self.n_kpt, self.train = items, cfg, list(flip_idx), n_kpt, train
        self.imgsz = imgsz or cfg.imgsz
        self.epoch = 0
        self.mosaic_on = True

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int, close_mosaic: bool = False) -> None:
        self.epoch, self.mosaic_on = epoch, not close_mosaic

    def sample_weights(self) -> np.ndarray:
        w = np.array([max(it.weight, 1e-3) for it in self.items], dtype=np.float64) ** self.cfg.hard_sampling
        return w / w.sum()

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng([self.cfg.seed, self.epoch, idx, 7])

    # ---- single image, scaled so its long side is imgsz ----
    def _load(self, idx: int):
        it = self.items[idx]
        img = load_image(it.path)
        r = self.imgsz / max(it.h, it.w)
        nw, nh = max(1, round(it.w * r)), max(1, round(it.h * r))
        img = resize(img, nw, nh)
        b, k = it.boxes * r, it.kpts.copy()
        k[..., :2] *= r
        return img, b, it.cls.copy(), k

    def _mosaic(self, idx: int, rng: np.random.Generator):
        s = self.imgsz
        ids = [idx] + [int(rng.integers(0, len(self.items))) for _ in range(3)]
        xc, yc = int(rng.uniform(0.5 * s, 1.5 * s)), int(rng.uniform(0.5 * s, 1.5 * s))
        canvas = np.full((2 * s, 2 * s, 3), PAD, np.uint8)
        B, C, K = [], [], []
        for i, j in enumerate(ids):
            img, b, c, k = self._load(j)
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a, x1b, y1b = max(xc - w, 0), max(yc - h, 0), xc, yc, w - (xc - max(xc - w, 0)), h - (yc - max(yc - h, 0))
            elif i == 1:
                x1a, y1a, x2a, y2a, x1b, y1b = xc, max(yc - h, 0), min(xc + w, 2 * s), yc, 0, h - (yc - max(yc - h, 0))
            elif i == 2:
                x1a, y1a, x2a, y2a, x1b, y1b = max(xc - w, 0), yc, xc, min(2 * s, yc + h), w - (xc - max(xc - w, 0)), 0
            else:
                x1a, y1a, x2a, y2a, x1b, y1b = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h), 0, 0
            cw, ch = x2a - x1a, y2a - y1a
            if cw <= 0 or ch <= 0:
                continue
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y1b + ch, x1b:x1b + cw]
            dx, dy = x1a - x1b, y1a - y1b
            b = b + [dx, dy, dx, dy]
            k = k.copy()
            lab = k[..., 2] > 0
            k[..., 0] = np.where(lab, k[..., 0] + dx, 0.0)
            k[..., 1] = np.where(lab, k[..., 1] + dy, 0.0)
            B.append(b)
            C.append(c)
            K.append(k)
        boxes = np.concatenate(B) if B else np.zeros((0, 4), np.float32)
        return canvas, boxes, (np.concatenate(C) if C else np.zeros(0, np.int64)), \
            (np.concatenate(K) if K else np.zeros((0, self.n_kpt, 3), np.float32)), True

    def _plain(self, idx: int):
        img, b, c, k = self._load(idx)
        s = self.imgsz
        h, w = img.shape[:2]
        canvas = np.full((s, s, 3), PAD, np.uint8)
        dx, dy = (s - w) // 2, (s - h) // 2
        canvas[dy:dy + h, dx:dx + w] = img
        b = b + [dx, dy, dx, dy]
        k = k.copy()
        lab = k[..., 2] > 0
        k[..., 0] = np.where(lab, k[..., 0] + dx, 0.0)
        k[..., 1] = np.where(lab, k[..., 1] + dy, 0.0)
        return canvas, b, c, k, False

    def _sample(self, idx: int, rng: np.random.Generator):
        cfg, s = self.cfg, self.imgsz
        use_mosaic = self.train and self.mosaic_on and rng.random() < cfg.mosaic
        img, b, c, k, mosaic = self._mosaic(idx, rng) if use_mosaic else self._plain(idx)
        src = img.shape[0]                                   # imgsz (plain) or 2*imgsz (mosaic): the centre is cropped
        sc = float(rng.uniform(1 - cfg.scale, 1 + cfg.scale))
        deg = float(rng.uniform(-cfg.degrees, cfg.degrees))
        tx, ty = (float(rng.uniform(-cfg.translate, cfg.translate)) * s for _ in range(2))
        m = affine_matrix(src, src, s, sc, deg, tx, ty)
        img = warp(img, m, s)
        b, k, keep = transform_labels(b, k, m, s)
        c = c[keep]
        return img, b, c, k

    def get(self, idx: int):
        rng = self._rng(idx)
        cfg = self.cfg
        if not self.train:
            it = self.items[idx]
            img = load_image(it.path)
            lb_img, lb = letterbox(img, self.imgsz)
            return lb_img, lb, it
        img, b, c, k = self._sample(idx, rng)
        if rng.random() < cfg.mixup:
            j = int(rng.integers(0, len(self.items)))
            img2, b2, c2, k2 = self._sample(j, rng)
            a = float(rng.beta(32.0, 32.0))
            img = (img * a + img2 * (1 - a)).astype(np.uint8)
            b, c, k = np.concatenate([b, b2]), np.concatenate([c, c2]), np.concatenate([k, k2])
        img = hsv_jitter(img, rng, cfg.hsv_h, cfg.hsv_s, cfg.hsv_v)
        if rng.random() < cfg.flip:
            img, b, k = hflip(img, b, k, self.flip_idx)
        if rng.random() < cfg.smoke_flash:
            img = smoke_flash(img, rng)
        return img, b.astype(np.float32), c.astype(np.int64), k.astype(np.float32)

    def __getitem__(self, idx: int):
        return self.get(idx)


def collate(batch):
    """Training batch -> dict of tensors (images stay uint8 until the device)."""
    import torch
    imgs = np.stack([b[0] for b in batch]).transpose(0, 3, 1, 2)
    m = max(1, max(len(b[1]) for b in batch))
    K = batch[0][3].shape[1]
    boxes = np.zeros((len(batch), m, 4), np.float32)
    cls = np.full((len(batch), m), -1, np.int64)
    kp = np.zeros((len(batch), m, K, 3), np.float32)
    valid = np.zeros((len(batch), m), bool)
    for i, (_, b, c, k) in enumerate(batch):
        n = len(b)
        boxes[i, :n], cls[i, :n], kp[i, :n], valid[i, :n] = b, c, k, True
    return {"images": torch.from_numpy(np.ascontiguousarray(imgs)), "boxes": torch.from_numpy(boxes),
            "cls": torch.from_numpy(cls), "kpts": torch.from_numpy(kp), "valid": torch.from_numpy(valid)}


def collate_val(batch):
    import torch
    imgs = np.stack([b[0] for b in batch]).transpose(0, 3, 1, 2)
    return {"images": torch.from_numpy(np.ascontiguousarray(imgs)), "lb": [b[1] for b in batch], "items": [b[2] for b in batch]}
