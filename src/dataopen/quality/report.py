"""closed_loop_report.{json,md} and quality_index.json: what the validator saw and what the loop learned."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.models import FrameRecord


def quality_index(records: Iterable[FrameRecord]) -> dict[str, Any]:
    """frame_id -> {verdict, tier, difficulty_score, weight, oks_score, ...} for sampling, curricula, weighted training."""
    out = {}
    for r in records:
        q = r.meta.get("quality")
        if q:
            oks = q["metrics"]["mean_oks"] if q["metrics"]["evaluated"] else None
            out[r.frame_id] = {"verdict": q.get("verdict"), "tier": q["tier"], "difficulty_score": q["difficulty"],
                               "difficulty": q["difficulty"], "weight": q["weight"],
                               "occlusion_index": q["occlusion_index"], "contrast_rate": q["contrast_rate"],
                               "oks_score": oks, "mean_oks": oks, "reasons": q["reasons"]}
    return out


FRAME_COLUMNS = ["frame_id", "set", "split", "verdict", "tier", "difficulty_score", "contrast_rate", "occlusion_index",
                 "oks_score", "weight", "policy_version", "n_persons", "reasons", "focus_oks", "focus_difficulty",
                 "class_accuracy"]
PERSON_COLUMNS = ["frame_id", "set", "entity_id", "class_id", "difficulty_score", "contrast_rate", "occlusion_index",
                  "oks_score", "perceptibility", "focus_oks", "focus_err_rel", "focus_visibility", "focus_perceptibility"]


def write_quality_csv(root: Path, kept: Iterable[FrameRecord],
                      quarantined: Iterable[tuple[str, FrameRecord]] = ()) -> None:
    """quality_index.csv / quality_persons.csv: the per-frame and per-person quality metadata, joinable by frame_id.
    YOLO-Pose .txt files must stay strictly standard (extra columns break loaders), so the metadata lives here."""
    import csv
    root = Path(root)
    rows = [("dataset", r) for r in kept] + [("quarantine", r) for _, r in quarantined]
    with (root / "quality_index.csv").open("w", newline="", encoding="utf-8") as f, \
            (root / "quality_persons.csv").open("w", newline="", encoding="utf-8") as g:
        fw, gw = csv.writer(f), csv.writer(g)
        fw.writerow(FRAME_COLUMNS)
        gw.writerow(PERSON_COLUMNS)
        for which, r in rows:
            q = r.meta.get("quality")
            if not q:
                continue
            evaluated = q["metrics"]["evaluated"]
            fw.writerow([r.frame_id, which, r.split, q.get("verdict", ""), q["tier"], q["difficulty"], q["contrast_rate"],
                         q["occlusion_index"], q["metrics"]["mean_oks"] if evaluated else "", q["weight"],
                         q.get("policy_version", 0), len(r.annotations), ";".join(q["reasons"]),
                         "" if q.get("focus_oks") is None else q["focus_oks"], q.get("focus_difficulty", ""),
                         "" if q["metrics"].get("class_accuracy") is None else q["metrics"]["class_accuracy"]])
            for a in r.annotations:
                m = a.meta
                gw.writerow([r.frame_id, which, a.entity_id, a.class_id, m.get("difficulty_score", ""),
                             m.get("contrast_rate", ""), m.get("occlusion_index", ""), m.get("oks_score", ""),
                             m.get("perceptibility", ""), m.get("focus_oks", ""), m.get("focus_err_rel", ""),
                             m.get("focus_visibility", ""), m.get("focus_perceptibility", "")])


def write_closed_loop_report(root: Path, pipeline_stats: Optional[dict[str, Any]], reject_totals: dict[str, int],
                             feedback_report: Optional[dict[str, Any]], records: Iterable[FrameRecord],
                             quarantined: Iterable[tuple[str, FrameRecord]] = ()) -> None:
    root = Path(root)
    records = list(records)
    quarantined = list(quarantined)
    idx = quality_index(records)
    if idx:
        (root / "quality_index.json").write_text(json.dumps(idx))
        write_quality_csv(root, records, quarantined)
    kept = Counter(v["tier"] for v in idx.values())
    verdicts = Counter(v["verdict"] for v in idx.values())
    verdicts["quarantine"] += len(quarantined)
    data = {"pipeline": pipeline_stats, "kept_by_tier": dict(kept), "kept_by_verdict": dict(verdicts),
            "quarantined": len(quarantined), "dropped_by_tier": dict(reject_totals), "feedback": feedback_report}
    (root / "closed_loop_report.json").write_text(json.dumps(data, indent=2))

    lines = ["# Closed-loop validation report", ""]
    if pipeline_stats:
        lines += [f"- evaluator: `{pipeline_stats.get('backend')}`; frames judged: {pipeline_stats['frames']}; "
                  f"run through the model: {pipeline_stats['evaluated_by_model']}; "
                  f"dropped by cheap gates (no inference needed): {pipeline_stats['dropped_by_cheap_gates']}; "
                  f"mean inference: {pipeline_stats['mean_inference_ms']} ms; evaluator errors: {pipeline_stats['evaluator_errors']}",
                  f"- kept: {dict(kept)}", f"- dropped: {dict(reject_totals)}", ""]
    lines += ["How to read it: `keep_hard` frames are visible people the baseline model struggles on (the most valuable "
              "ones, weight > 1 in `quality_index.csv`). `quarantine/` holds frames whose labels and the model contradict "
              "each other, in full, for human review. Rejected frames never enter the dataset; `quality/rejects/` and "
              "`quality/audit/` hold examples. Audit a few: if many dropped frames look usable, loosen the thresholds. "
              "Dropping only on evidence independent of the model (broken skeleton, broken picture, imperceptible person, "
              "an unlabeled person) is what keeps the dataset from being biased toward what the model already knows.", ""]
    if feedback_report:
        lines += [f"## Adaptive randomization ({'active' if feedback_report['adapting'] else 'still warming up'})", "",
                  f"observed frames: {feedback_report['observed_frames']}; remembered edge cases: "
                  f"{feedback_report['remembered_hard_examples']}", ""]
        pc = feedback_report.get("pairwise_appearance_coverage")
        if pc:
            lines += [f"appearance pair coverage: {pc['pairs_seen']}/{pc['pairs_possible']} ({pc['coverage']}) "
                      f"over {', '.join(pc['keys'])}", ""]
        lines += ["### Where the model struggles (bins with >= 10 frames, worst first)", "",
                  "| parameter | bin | frames | mean OKS | hard | dropped | tilt |", "|---|---|---|---|---|---|---|"]
        rows = []
        for key, d in feedback_report["parameters"].items():
            for r in d["rows"]:
                if r["n"] >= 10:
                    rows.append((r["mean_oks"] if r["mean_oks"] is not None else 1.0 - (r["hard_rate"] or 0), key, r))
        for _, key, r in sorted(rows, key=lambda x: x[0])[:25]:
            lines.append(f"| {key} | {r['bin']} | {r['n']} | {r['mean_oks']} | {r['hard_rate']} | {r['drop_rate']} | {r['tilt']} |")
        focus_rows = [(r["focus_blind_rate"], key, r) for key, d in feedback_report["parameters"].items()
                      for r in d["rows"] if r["n"] >= 10 and r.get("focus_blind_rate") is not None]
        if focus_rows:
            lines += ["", "### Where the aim point (primary keypoint) is missed (bins with >= 10 judged frames, worst first)", "",
                      "| parameter | bin | frames | aim OKS | aim missed | tilt |", "|---|---|---|---|---|---|"]
            for _, key, r in sorted(focus_rows, key=lambda x: -x[0])[:25]:
                lines.append(f"| {key} | {r['bin']} | {r['n']} | {r['mean_focus_oks']} | {r['focus_blind_rate']} | {r['tilt']} |")
    (root / "closed_loop_report.md").write_text("\n".join(lines) + "\n")
