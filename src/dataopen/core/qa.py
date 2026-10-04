"""Dataset verification and statistics (`dataopen verify`).

Checks that everything that can silently rot actually agrees: canonical JSONL, image files (existence
and real pixel size), YOLO label files and COCO json; then reports what the data actually covers
(visibility per joint, box sizes, people per frame, time-of-day / weather coverage).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .export import CanonicalStore
from .imageio import read_image_size
from .models import FrameKind, FrameRecord


@dataclass
class Issue:
    severity: str          # "error" | "warning"
    code: str
    where: str
    message: str


@dataclass
class VerifyReport:
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, where: str, message: str) -> None:
        self.issues.append(Issue(severity, code, where, message))


def _keypoint_names(root: Path, records: list[FrameRecord]) -> list[str]:
    card = root / "DATASET_CARD.json"
    if card.exists():
        try:
            return list(json.loads(card.read_text())["skeleton"]["keypoints"])
        except (KeyError, ValueError):
            pass
    for r in records:
        if r.annotations:
            return [f"kp{i}" for i in range(len(r.annotations[0].keypoints))]
    return []


def verify_dataset(root: Path, check_images: bool = True, max_issues_per_code: int = 20) -> VerifyReport:
    root = Path(root)
    rep = VerifyReport()
    store = CanonicalStore(root)
    splits = store.splits()
    if not splits:
        rep.add("error", "no_annotations", str(root / "annotations"), "no annotations/*.jsonl found")
        return rep
    records: list[FrameRecord] = []
    for s in splits:
        try:
            records += list(store.load(s))
        except (ValueError, KeyError) as e:
            rep.add("error", "bad_jsonl", s, f"cannot parse annotations/{s}.jsonl: {e}")
            return rep
    names = _keypoint_names(root, records)
    k = len(names)
    counts: Counter[str] = Counter()

    def issue(sev: str, code: str, where: str, msg: str) -> None:
        counts[code] += 1
        if counts[code] <= max_issues_per_code:
            rep.add(sev, code, where, msg)

    seen: set[str] = set()
    on_disk_images = {p.relative_to(root).as_posix() for p in (root / "images").glob("*/*") if p.is_file()}
    referenced: set[str] = set()
    for r in records:
        if r.frame_id in seen:
            issue("error", "duplicate_frame_id", r.frame_id, "frame_id appears more than once")
        seen.add(r.frame_id)
        referenced.add(r.file_name)
        img = root / r.file_name
        if check_images:
            if not img.exists():
                issue("error", "missing_image", r.file_name, "image file does not exist")
            else:
                try:
                    size = read_image_size(img)
                    if size != (r.width, r.height):
                        issue("error", "image_size_mismatch", r.file_name,
                              f"file is {size[0]}x{size[1]}, record says {r.width}x{r.height}")
                except ValueError as e:
                    issue("error", "unreadable_image", r.file_name, str(e))
        if r.kind is FrameKind.NEGATIVE and r.annotations:
            issue("error", "negative_with_annotations", r.frame_id, "negative frame has annotations")
        if r.kind is FrameKind.POSITIVE and not r.annotations:
            issue("error", "positive_without_annotations", r.frame_id, "positive frame has no annotations")
        for a in r.annotations:
            kp = np.asarray(a.keypoints, dtype=float)
            if k and kp.shape[0] != k:
                issue("error", "wrong_keypoint_count", r.frame_id, f"{kp.shape[0]} keypoints, expected {k}")
                continue
            v = kp[:, 2]
            if not np.isin(v, (0, 1, 2)).all():
                issue("error", "bad_visibility_flag", r.frame_id, f"flags {sorted(set(v.tolist()))}")
            if (kp[v == 0, :2] != 0).any():
                issue("error", "nonzero_coords_for_v0", r.frame_id, "v=0 keypoints must have x=y=0")
            lab = kp[v > 0]
            if len(lab) and ((lab[:, 0] < 0).any() or (lab[:, 0] >= r.width).any()
                             or (lab[:, 1] < 0).any() or (lab[:, 1] >= r.height).any()):
                issue("error", "keypoint_outside_frame", r.frame_id, "labeled keypoint outside the image")
            x, y, w, h = a.bbox
            if w <= 0 or h <= 0 or x < -1e-6 or y < -1e-6 or x + w > r.width + 1e-6 or y + h > r.height + 1e-6:
                issue("error", "bad_bbox", r.frame_id, f"bbox {a.bbox} invalid for {r.width}x{r.height}")
            elif len(lab) and ((lab[:, 0] < x - 2).any() or (lab[:, 0] > x + w + 2).any()
                               or (lab[:, 1] < y - 2).any() or (lab[:, 1] > y + h + 2).any()):
                issue("warning", "keypoint_outside_bbox", r.frame_id, "labeled keypoint lies outside its bbox")
            if (v == 2).sum() == 0:
                issue("warning", "no_visible_keypoint", r.frame_id, "annotation without any visible keypoint")

    for orphan in sorted(on_disk_images - referenced)[:max_issues_per_code]:
        rep.add("warning", "orphan_image", orphan, "image on disk without an annotation record")
    _check_yolo(root, records, rep, issue)
    _check_coco(root, records, splits, rep, issue)
    rep.stats = compute_stats(records, names)
    rep.stats["issue_counts"] = dict(counts)
    return rep


def _check_yolo(root: Path, records: list[FrameRecord], rep: VerifyReport, issue) -> None:
    labels = root / "labels"
    if not labels.exists():
        return
    for r in records:
        p = labels / r.split / f"{r.frame_id}.txt"
        if not p.exists():
            issue("error", "missing_yolo_label", r.frame_id, f"{p.relative_to(root)} does not exist")
            continue
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        if len(lines) != len(r.annotations):
            issue("error", "yolo_count_mismatch", r.frame_id, f"{len(lines)} label lines vs {len(r.annotations)} annotations")
            continue
        for ln, a in zip(lines, r.annotations):
            v = [float(t) for t in ln.split()]
            if len(v) != 5 + 3 * len(a.keypoints):
                issue("error", "yolo_bad_line", r.frame_id, f"{len(v)} fields")
                continue
            x, y, w, h = a.bbox
            exp = [(x + w / 2) / r.width, (y + h / 2) / r.height, w / r.width, h / r.height]
            if max(abs(e - g) for e, g in zip(exp, v[1:5])) > 1e-4:
                issue("error", "yolo_bbox_mismatch", r.frame_id, "YOLO box differs from the canonical record")


def _check_coco(root: Path, records: list[FrameRecord], splits: list[str], rep: VerifyReport, issue) -> None:
    for s in splits:
        p = root / "annotations" / f"coco_{s}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except ValueError as e:
            issue("error", "coco_unreadable", p.name, str(e))
            continue
        recs = [r for r in records if r.split == s]
        if len(d["images"]) != len(recs):
            issue("error", "coco_image_count", p.name, f"{len(d['images'])} images vs {len(recs)} records")
        n_ann = sum(len(r.annotations) for r in recs)
        if len(d["annotations"]) != n_ann:
            issue("error", "coco_annotation_count", p.name, f"{len(d['annotations'])} annotations vs {n_ann}")
        ids = {im["id"] for im in d["images"]}
        if any(a["image_id"] not in ids for a in d["annotations"]):
            issue("error", "coco_dangling_image_id", p.name, "annotation references a missing image")


def compute_stats(records: list[FrameRecord], names: list[str]) -> dict[str, Any]:
    st: dict[str, Any] = {"frames": len(records)}
    st["by_split"] = dict(Counter(r.split for r in records))
    st["by_kind"] = dict(Counter(r.kind.value for r in records))
    st["persons"] = sum(len(r.annotations) for r in records)
    st["persons_per_frame"] = dict(sorted(Counter(len(r.annotations) for r in records).items()))
    anns = [a for r in records for a in r.annotations]
    if anns:
        kp = np.stack([a.keypoints for a in anns])                       # (N, K, 3)
        v = kp[:, :, 2]
        n = len(anns)
        st["keypoint_visibility"] = {
            (names[j] if j < len(names) else f"kp{j}"): {
                "visible": round(float((v[:, j] == 2).sum() / n), 3),
                "occluded": round(float((v[:, j] == 1).sum() / n), 3),
                "out_of_frame": round(float((v[:, j] == 0).sum() / n), 3)}
            for j in range(v.shape[1])}
        hs = np.array([a.bbox[3] for a in anns])
        st["bbox_height_px"] = {f"p{q}": round(float(np.percentile(hs, q)), 1) for q in (5, 25, 50, 75, 95)}
    envs = [r.meta.get("environment", {}) for r in records]
    tod = [e["time_of_day"] for e in envs if "time_of_day" in e]
    if tod:
        hist, _ = np.histogram(np.asarray(tod) % 24, bins=8, range=(0, 24))
        st["time_of_day_hist_3h_bins"] = hist.tolist()
    if any("weather" in e for e in envs):
        st["weather"] = dict(Counter(e.get("weather") for e in envs if "weather" in e))
    errs = [r.meta["probe_max_err_px"] for r in records if "probe_max_err_px" in r.meta]
    if errs:
        st["probe_error_px"] = {"p50": round(float(np.percentile(errs, 50)), 3), "max": round(float(max(errs)), 3)}
    qs = [r.meta["quality"] for r in records if r.meta.get("quality")]
    if qs:
        st["quality"] = {"tiers": dict(Counter(q["tier"] for q in qs)),
                         "difficulty": {f"p{p}": round(float(np.percentile([q["difficulty"] for q in qs], p)), 3)
                                        for p in (10, 50, 90)},
                         "mean_oks": round(float(np.mean([q["metrics"]["mean_oks"] for q in qs if q["metrics"]["evaluated"]])), 3)
                         if any(q["metrics"]["evaluated"] for q in qs) else None,
                         "evaluated_share": round(sum(q["metrics"]["evaluated"] for q in qs) / len(qs), 3)}
    scenes = {r.scene_index for r in records}
    st["scenes"] = len(scenes)
    st["frames_per_scene_mean"] = round(len(records) / max(1, len(scenes)), 2)
    return st


def write_reports(rep: VerifyReport, root: Path) -> None:
    (root / "qa_report.json").write_text(json.dumps(
        {"ok": rep.ok, "issues": [i.__dict__ for i in rep.issues], "stats": rep.stats}, indent=2))
    s = rep.stats
    lines = [f"# QA report: {'OK' if rep.ok else 'FAILED'}", "",
             f"- frames: {s.get('frames')}  persons: {s.get('persons')}  scenes: {s.get('scenes')}",
             f"- split: {s.get('by_split')}  kind: {s.get('by_kind')}",
             f"- persons per frame: {s.get('persons_per_frame')}",
             f"- bbox height px: {s.get('bbox_height_px')}",
             f"- time of day (3h bins from 00:00): {s.get('time_of_day_hist_3h_bins')}",
             f"- quality: {s.get('quality')}", f"- weather: {s.get('weather')}", f"- probe error px: {s.get('probe_error_px')}", "",
             f"## Issues ({len(rep.issues)})"]
    lines += [f"- **{i.severity}** `{i.code}` {i.where}: {i.message}" for i in rep.issues] or ["none"]
    lines += ["", "## Keypoint visibility", "", "| keypoint | visible | occluded | out |", "|---|---|---|---|"]
    for name, d in (s.get("keypoint_visibility") or {}).items():
        lines.append(f"| {name} | {d['visible']} | {d['occluded']} | {d['out_of_frame']} |")
    (root / "qa_report.md").write_text("\n".join(lines) + "\n")
