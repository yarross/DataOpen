"""Training / evaluation data: synthetic screens rendered at screen resolution and prepared through the SAME path as deployment (the C
frame-prep core: area downscale + letterbox, optionally a ROI crop), so there is no train/serve gap in how a screen becomes a 640x640 input.
Real screenshots (labelled in YOLO txt format, screen coordinates) go through the same path: see `RealScreens`."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..video.prep import FramePrep, Geometry
from . import synth
from .taxonomy import NAMES


def prepare(img: np.ndarray, boxes: np.ndarray, roi: Optional[tuple[int, int, int, int]] = None, out: int = 640):
    """Screen image + boxes (n,4) xyxy in screen pixels -> (out, out, 3) input and boxes in input pixels. Boxes cut by the ROI are clipped;
    ones that lose most of their area are dropped."""
    h, w = img.shape[:2]
    fp = FramePrep(w, h, "rgb24", roi, out)
    g: Geometry = fp.geometry
    x, _, _ = fp.frame(img)
    if len(boxes) == 0:
        return x, np.zeros((0, 4), np.float32), g, np.zeros(0, bool)
    b = boxes.astype(np.float64).copy()
    area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    cx0, cy0, cw, ch = g.crop_x, g.crop_y, g.crop_w, g.crop_h
    b[:, [0, 2]] = b[:, [0, 2]].clip(cx0, cx0 + cw)
    b[:, [1, 3]] = b[:, [1, 3]].clip(cy0, cy0 + ch)
    keep = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) >= 0.5 * area0
    xs0, ys0 = (b[:, 0] - cx0) * g.content_w / cw + g.pad_x, (b[:, 1] - cy0) * g.content_h / ch + g.pad_y
    xs1, ys1 = (b[:, 2] - cx0) * g.content_w / cw + g.pad_x, (b[:, 3] - cy0) * g.content_h / ch + g.pad_y
    out_b = np.stack([xs0, ys0, xs1, ys1], axis=1).astype(np.float32)
    keep &= ((out_b[:, 2] - out_b[:, 0]) >= 2.5) & ((out_b[:, 3] - out_b[:, 1]) >= 2.5)
    return x, out_b, g, keep


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric only (geometry would break the pixel-exact link between screen and input): brightness / contrast / saturation, noise,
    a little blur, JPEG."""
    f = x.astype(np.float32)
    f = (f - 128) * rng.uniform(0.8, 1.2) + 128 + rng.uniform(-20, 20)
    gray = f.mean(axis=2, keepdims=True)
    f = gray + (f - gray) * rng.uniform(0.7, 1.3)
    if rng.random() < 0.3:
        f += rng.normal(0, rng.uniform(1, 6), f.shape)
    im = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
    if rng.random() < 0.15:
        from PIL import ImageFilter

        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    if rng.random() < 0.25:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(rng.integers(55, 95)))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.asarray(im)


def random_roi(rng: np.random.Generator, w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    """A third of the samples look at a region of the screen at larger scale, as the video path does with a ROI."""
    if rng.random() > 0.3:
        return None
    cw = int(rng.integers(640, max(641, min(w, 1400))))
    ch = int(rng.integers(480, max(481, min(h, int(cw * 0.8) + 1))))
    return int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1)), cw, ch


class SynthUi:
    """Map-style dataset of `n` deterministic samples (sample i is always the same picture for a given seed)."""

    def __init__(self, n: int, seed: int = 0, families=synth.TRAIN_FAMILIES, train: bool = True) -> None:
        self.n, self.seed, self.families, self.train = n, seed, families, train

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, i])
        scr = synth.render_screen(rng, None, self.families)
        boxes = np.array([e.box for e in scr.elements], np.float32).reshape(-1, 4)
        cls = np.array([e.cls for e in scr.elements], np.int64)
        h, w = scr.image.shape[:2]
        x, b, g, keep = prepare(scr.image, boxes, random_roi(rng, w, h) if self.train else None)
        if self.train:
            x = augment(x, rng)
        return np.ascontiguousarray(x), b[keep], cls[keep], g, scr.meta


class RealScreens:
    """Labelled real screenshots: a folder with images (png/jpg) and, next to each, a YOLO-format .txt (class cx cy w h, normalised to the
    screenshot) plus a `classes.txt` naming the classes (must be the UI taxonomy). The images go through the same preparation as everything else."""  # noqa: E501

    def __init__(self, folder: str | Path, train: bool = True, seed: int = 0) -> None:
        self.dir = Path(folder)
        self.files = sorted(
            p for p in self.dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.with_suffix(".txt").exists()
        )
        names = [ln.strip() for ln in (self.dir / "classes.txt").read_text().splitlines() if ln.strip()]
        from .taxonomy import require_ui_layout

        require_ui_layout(names)
        self.names, self.train, self.seed = names, train, seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, i])
        p = self.files[i % len(self.files)]
        img = np.asarray(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        rows = [ln.split() for ln in p.with_suffix(".txt").read_text().splitlines() if ln.strip()]
        cls = np.array([NAMES.index(self.names[int(r[0])]) for r in rows], np.int64)
        v = np.array([[float(t) for t in r[1:5]] for r in rows], np.float32).reshape(-1, 4)
        boxes = np.stack(
            [(v[:, 0] - v[:, 2] / 2) * w, (v[:, 1] - v[:, 3] / 2) * h, (v[:, 0] + v[:, 2] / 2) * w, (v[:, 1] + v[:, 3] / 2) * h], axis=1
        )
        x, b, g, keep = prepare(img, boxes, random_roi(rng, w, h) if self.train else None)
        if self.train:
            x = augment(x, rng)
        return np.ascontiguousarray(x), b[keep], cls[keep], g, {"file": str(p)}


def collate(batch):
    import torch

    x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).contiguous()
    gts = [(torch.from_numpy(b[1]), torch.from_numpy(b[2])) for b in batch]
    return x, gts, [b[3] for b in batch], [b[4] for b in batch]
