"""Build the INT8 calibration set from closed-loop datasets: >= 500 frames chosen so the activation ranges the quantizer sees
cover what the model meets in the field, not what is most common.

Frames are stratified on what drives activation statistics: closed-loop verdict (clean / hard), time of day, weather, camera
distance, smoke / flash level, whether the aim point was missed by the baseline, and the team mix. Quota goes round-robin over the
strata that exist, so rare-but-extreme conditions (night, heavy smoke, flash, far) are over-represented relative to their
frequency (that is the point: percentile calibration must not clip them). Output: letterboxed PNGs + `dataset.txt` (RKNN format)
+ `calibration_report.json` (strata counts and coverage).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.export import CanonicalStore
from ..core.imageio import write_png
from ..core.models import FrameRecord


def _bin(v, edges) -> int:
    return int(np.searchsorted(edges, v, side="right"))


def stratum(r: FrameRecord) -> tuple:
    env = r.meta.get("environment", {}) or {}
    cam = r.meta.get("camera", {}) or {}
    q = r.meta.get("quality", {}) or {}
    tod = env.get("time_of_day", 12.0)
    return (q.get("verdict", "keep_clean"),
            "night" if (tod < 5 or tod > 20) else "dusk" if (tod < 8 or tod > 17) else "day",
            _bin(cam.get("distance", 8.0), (6.0, 14.0)),
            _bin(env.get("smoke_density", 0.0), (0.2, 0.6)), _bin(env.get("flash_intensity", 0.0), (0.2, 0.6)),
            bool((q.get("metrics", {}) or {}).get("focus_evaluated") and (q["metrics"].get("min_focus_oks", 1.0) < 0.15)))


def select(records: Sequence[tuple[Path, FrameRecord]], n: int, seed: int = 0, hard_share: float = 0.4):
    """Round-robin over strata; hard (keep_hard) frames get `hard_share` of the quota when available."""
    rng = np.random.default_rng(seed)
    by: dict[tuple, list] = defaultdict(list)
    for item in records:
        by[stratum(item[1])].append(item)
    for v in by.values():
        rng.shuffle(v)
    hard_keys = [k for k in by if k[0] == "keep_hard"]
    other_keys = [k for k in by if k[0] != "keep_hard"]
    chosen: list = []

    def draw(keys, count):
        keys = sorted(keys, key=str)
        i = 0
        got = []
        while len(got) < count and any(by[k] for k in keys):
            k = keys[i % len(keys)]
            if by[k]:
                got.append(by[k].pop())
            i += 1
        return got

    n_hard = min(int(round(n * hard_share)), sum(len(by[k]) for k in hard_keys))
    chosen += draw(hard_keys, n_hard) if hard_keys else []
    chosen += draw(other_keys, n - len(chosen)) if other_keys else []
    if len(chosen) < n and hard_keys:                                   # not enough clean frames: top up with hard ones
        chosen += draw(hard_keys, n - len(chosen))
    return chosen, {k: len(v) for k, v in by.items()}


def build_calibration_set(datasets: Sequence[Path], out: Path, n: int = 500, size: int = 640, seed: int = 0,
                          hard_share: float = 0.4, split: str = "train") -> dict:
    from .data import letterbox, load_image
    records = []
    for d in map(Path, datasets):
        for r in CanonicalStore(d).load(split):
            records.append((d, r))
    if not records:
        raise ValueError(f"no records in split {split!r} of {[str(d) for d in datasets]}")
    chosen, available = select(records, n, seed, hard_share)
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    lines, strata = [], defaultdict(int)
    for i, (d, r) in enumerate(chosen):
        img, _ = letterbox(load_image(d / r.file_name), size)
        p = out / "images" / f"calib_{i:04d}.png"
        write_png(p, img)
        lines.append(f"images/{p.name}")
        strata[stratum(r)] += 1
    (out / "dataset.txt").write_text("\n".join(lines) + "\n")
    rep = {"images": len(chosen), "requested": n, "available_frames": len(records), "size": size,
           "strata_available": len(available), "strata_covered": len(strata),
           "strata_coverage": round(len(strata) / max(1, len(available)), 3),
           "by_verdict": {v: sum(c for k, c in strata.items() if k[0] == v) for v in sorted({k[0] for k in strata})},
           "by_light": {v: sum(c for k, c in strata.items() if k[1] == v) for v in sorted({k[1] for k in strata})},
           "aim_missed_frames": sum(c for k, c in strata.items() if k[5])}
    if len(chosen) < n:
        rep["warning"] = f"only {len(chosen)} frames available (< {n} requested): collect more before quantizing"
    (out / "calibration_report.json").write_text(json.dumps(rep, indent=2))
    return rep
