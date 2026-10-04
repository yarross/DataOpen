"""Re-run the quality policy on a finished dataset, without pixels.

Every kept frame stores the model-independent features and the model's metrics it was judged on (`meta.quality`), and
`IQualityPolicy.decide` is a pure function of those, so new thresholds can be applied to an existing dataset: tighten a
filter, see how many frames would go, produce the exclusion list. The reverse is not possible: a frame that was
rejected at collection time left no pixels behind (only the quarantine keeps them in full).
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from ..core.export import CanonicalStore
from ..core.models import FrameKind
from .interfaces import IQualityPolicy
from .policy import DefaultQualityPolicy
from .types import FrameFeatures, QualityMetrics, Verdict

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    import tomli as tomllib          # type: ignore[no-redef]


def load_policy_spec(path: Path) -> dict[str, Any]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return dict(data.get("policy", data))


def requalify(root: Path, policy_spec: dict[str, Any], policy: Optional[IQualityPolicy] = None) -> dict[str, Any]:
    """Writes quality_requalified.csv and excluded_frames.txt under `root`; returns the summary."""
    root = Path(root)
    pol = policy or DefaultQualityPolicy()
    pol.reconfigure(policy_spec)
    store = CanonicalStore(root)
    transitions: Counter[str] = Counter()
    now: Counter[str] = Counter()
    excluded: list[str] = []
    rows = []
    n = 0
    for split in store.splits():
        for r in store.load(split):
            q = r.meta.get("quality")
            if not q:
                continue
            n += 1
            v = pol.decide(QualityMetrics.from_dict(q["metrics"]), FrameFeatures.from_dict(q["features"]),
                           r.kind is FrameKind.NEGATIVE)
            old = q.get("verdict", "")
            transitions[f"{old}->{v.verdict.value}"] += 1
            now[v.verdict.value] += 1
            if v.verdict in (Verdict.REJECT, Verdict.QUARANTINE):
                excluded.append(r.frame_id)
            rows.append([r.frame_id, r.split, old, q["tier"], v.verdict.value, v.tier.value, round(v.difficulty, 4),
                         v.weight, pol.version, ";".join(v.reasons)])
    with (root / "quality_requalified.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "split", "old_verdict", "old_tier", "new_verdict", "new_tier", "difficulty_score",
                    "weight", "policy_version", "reasons"])
        w.writerows(rows)
    (root / "excluded_frames.txt").write_text("".join(f"{x}\n" for x in excluded), encoding="utf-8")
    return {"frames": n, "verdicts": dict(now), "transitions": dict(transitions), "excluded": len(excluded)}
