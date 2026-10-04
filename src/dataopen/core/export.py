"""Canonical store (JSONL) + COCO Keypoints JSON + YOLO-Pose TXT exporters.

FrameRecord (JSONL) is the source of truth; COCO/YOLO are derived views, so adding a
third format never touches the capture path, and a crash never corrupts a giant JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .models import FrameRecord
from .schema import SkeletonSchema


class CanonicalStore:
    def __init__(self, root: Path) -> None:
        self.dir = root / "annotations"

    def path(self, split: str) -> Path:
        return self.dir / f"{split}.jsonl"

    def append(self, split: str, records: Iterable[FrameRecord]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path(split).open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), separators=(",", ":")) + "\n")

    def load(self, split: str) -> Iterator[FrameRecord]:
        p = self.path(split)
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    yield FrameRecord.from_dict(json.loads(line))

    def splits(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.jsonl"))


def coco_dict(records: Iterable[FrameRecord], schema: SkeletonSchema) -> dict:
    images, anns = [], []
    ann_id = 1
    for img_id, r in enumerate(records, start=1):
        image = {"id": img_id, "file_name": r.file_name, "width": r.width, "height": r.height,
                 "is_negative": r.kind.value == "negative"}
        q = r.meta.get("quality")
        if q:  # frame difficulty metadata (COCO allows extra fields; ignored by standard loaders)
            if q.get("focus_oks") is not None:
                image["focus_oks"] = q["focus_oks"]
            if q.get("focus_difficulty"):
                image["focus_difficulty"] = q["focus_difficulty"]
            image.update({"verdict": q.get("verdict"), "tier": q["tier"], "difficulty_score": q["difficulty"],
                          "difficulty": q["difficulty"], "weight": q["weight"],
                          "occlusion_index": q["occlusion_index"], "contrast_rate": q["contrast_rate"]})
            if q["metrics"]["evaluated"]:
                image["oks_score"] = image["mean_oks"] = q["metrics"]["mean_oks"]
        images.append(image)
        for a in r.annotations:
            kp = a.keypoints
            extra = {k: a.meta[k] for k in ("oks_score", "oks", "difficulty_score", "occlusion_index", "contrast_rate",
                                                  "perceptibility", "focus_oks", "focus_err_rel", "focus_difficulty",
                                                  "focus_visibility", "class_ok") if k in a.meta}
            anns.append({
                **extra,
                "id": ann_id, "image_id": img_id, "category_id": a.class_id + 1, "iscrowd": 0,
                "keypoints": [round(float(x), 2) if i % 3 != 2 else int(x) for i, x in enumerate(kp.reshape(-1))],
                "num_keypoints": a.num_keypoints,
                "bbox": [round(float(x), 2) for x in a.bbox],
                "area": round(a.area, 2),
            })
            ann_id += 1
    return {
        "info": {"description": "DataOpen synthetic human pose", "schema": schema.name},
        "images": images,
        "annotations": anns,
        "categories": [{"id": i + 1, "name": c, "supercategory": "person",
                        "keypoints": list(schema.keypoints), "skeleton": schema.coco_skeleton()}
                       for i, c in enumerate(schema.classes)],
    }


def write_coco(records: Iterable[FrameRecord], schema: SkeletonSchema, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco_dict(records, schema)), encoding="utf-8")


def yolo_pose_lines(record: FrameRecord) -> list[str]:
    """`class xc yc w h  x1 y1 v1 ...`, all coordinates normalized to [0, 1]."""
    W, H = record.width, record.height
    lines = []
    for a in record.annotations:
        x, y, w, h = a.bbox
        parts = [a.class_id, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H]
        for kx, ky, kv in a.keypoints:
            parts += [kx / W, ky / H, int(kv)]
        lines.append(" ".join(str(p) if isinstance(p, int) else f"{p:.6f}" for p in parts))
    return lines


def write_yolo_label(record: FrameRecord, labels_root: Path) -> None:
    p = labels_root / record.split / f"{record.frame_id}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(yolo_pose_lines(record)) + ("\n" if record.annotations else ""), encoding="utf-8")


def write_yolo_yaml(root: Path, schema: SkeletonSchema, splits: Iterable[str]) -> None:
    splits = set(splits)
    lines = [f"path: {root.resolve()}", "train: images/train"]
    if "val" in splits:
        lines.append("val: images/val")
    lines += [f"kpt_shape: [{schema.num_keypoints}, 3]", f"flip_idx: {schema.flip_idx()}",
              "names:", *[f"  {i}: {c}" for i, c in enumerate(schema.classes)], ""]
    (root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")
