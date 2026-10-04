"""A tiny deterministic COCO-keypoints dataset in the 12-point shooter schema: blocky 'players' (team colour body, skin-coloured
head) on a noisy background. Easy enough that a few hundred CPU steps learn it, structured like what `dataopen collect` writes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataopen.core.imageio import write_png
from dataopen.core.schema_io import load_schema

REL = {"head_top": (0.5, 0.0), "head_center": (0.5, 0.09), "l_shoulder": (0.2, 0.22), "neck": (0.5, 0.2),
       "r_shoulder": (0.8, 0.22), "l_elbow": (0.08, 0.42), "spine": (0.5, 0.45), "r_elbow": (0.92, 0.42),
       "l_wrist": (0.06, 0.62), "hip": (0.5, 0.62), "r_wrist": (0.94, 0.62), "center_of_mass": (0.5, 0.5)}
TEAM = {0: (40, 70, 200), 1: (200, 150, 60)}


def make_person(img: np.ndarray, rng: np.random.Generator, cls: int, occupied: list):
    h_img, w_img, _ = img.shape
    for _ in range(30):
        h = int(rng.integers(h_img // 3, h_img // 2 + 10))
        w = int(h * rng.uniform(0.42, 0.55))
        x, y = int(rng.integers(2, w_img - w - 2)), int(rng.integers(2, h_img - h - 2))
        if all(x + w < ox or ox + ow < x or y + h < oy or oy + oh < y for ox, oy, ow, oh in occupied):
            break
    else:
        return None
    occupied.append((x, y, w, h))
    img[y + int(0.15 * h):y + int(0.65 * h), x + int(0.1 * w):x + int(0.9 * w)] = TEAM[cls]
    cy, cx, r = y + int(0.09 * h), x + w // 2, max(2, int(0.075 * h))
    yy, xx = np.ogrid[:h_img, :w_img]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = (225, 175, 140)
    return x, y, w, h


def make_toy_dataset(root: Path, n_train: int = 32, n_val: int = 8, size: int = 128, seed: int = 0) -> Path:
    bundle = load_schema("shooter12")
    schema = bundle.schema
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        images, anns = [], []
        for i in range(n):
            base = np.linspace(70, 110, size)[:, None, None].repeat(size, axis=1).repeat(3, axis=2).astype(np.float32)
            img = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
            occupied: list = []
            people = []
            for _ in range(int(rng.integers(1, 3))):
                cls = int(rng.integers(0, 2))
                p = make_person(img, rng, cls, occupied)
                if p is not None:
                    people.append((cls, p))
            fn = f"images/{split}/{split}_{i:04d}.png"
            (root / fn).parent.mkdir(parents=True, exist_ok=True)
            write_png(root / fn, img)
            images.append({"id": i + 1, "file_name": fn, "width": size, "height": size, "weight": 1.0 + (i % 3 == 0)})
            for cls, (x, y, w, h) in people:
                kp = []
                for name in schema.keypoints:
                    rx, ry = REL[name]
                    kp += [round(x + rx * w, 2), round(y + ry * h, 2), 2]
                anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": cls + 1, "iscrowd": 0, "keypoints": kp,
                             "num_keypoints": 12, "bbox": [x, y, w, h], "area": w * h})
        coco = {"images": images, "annotations": anns,
                "categories": [{"id": k + 1, "name": c, "keypoints": list(schema.keypoints)} for k, c in enumerate(schema.classes)]}
        (root / "annotations" / f"coco_{split}.json").write_text(json.dumps(coco))
    card = {"skeleton": {"name": schema.name, "keypoints": list(schema.keypoints), "edges": [list(e) for e in schema.edges],
                         "flip_pairs": [list(p) for p in schema.flip_pairs], "sigmas": schema.oks_sigmas(),
                         "weights": schema.oks_weights(), "derived": list(schema.derived), "primary": list(schema.primary),
                         "groups": {g: list(m) for g, m in schema.groups}, "roles": dict(schema.roles),
                         "classes": list(schema.classes), "class_key": schema.class_key}}
    (root / "DATASET_CARD.json").write_text(json.dumps(card))
    return root
