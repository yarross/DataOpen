"""Four-class verdicts, quarantine store, explicit size gate, metadata columns, hot-reloadable rules, requalify."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import pytest

from dataopen.adapters.mock import MockGameAdapter
from dataopen.cli import EXIT_OK, main
from dataopen.core.annotation import AnnotationBuilder, Verdict as AnnVerdict
from dataopen.core.orchestrator import DatasetOrchestrator, SessionConfig
from dataopen.core.qa import verify_dataset
from dataopen.core.schema import HUMAN_13
from dataopen.quality.evaluators.simulated import CallableEvaluator, SimulatedEvaluator
from dataopen.quality.pipeline import QualityConfig, QualityPipeline
from dataopen.quality.policy import DefaultQualityPolicy, PolicyConfig
from dataopen.quality.requalify import requalify
from dataopen.quality.types import Prediction
from tests.test_pipeline import _person_snapshot


def run(out, frames=40, evaluator=None, quality_cfg=None, **kw):
    q = QualityPipeline(HUMAN_13, evaluator if evaluator is not None else SimulatedEvaluator(HUMAN_13),
                        quality_cfg or QualityConfig())
    cfg = SessionConfig(out, seed=kw.pop("seed", 3), target_frames=frames, frames_per_scene=6, negative_ratio=0.15, **kw)
    return DatasetOrchestrator(MockGameAdapter(), cfg, quality=q).run()


def jsonl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_person_below_20px_is_rejected_before_anything_is_written():
    b = AnnotationBuilder(HUMAN_13)
    v = lambda h: next(iter(b.build(_person_snapshot(h)).verdicts.values()))     # noqa: E731
    assert v(15) is AnnVerdict.IGNORE_TOO_SMALL and v(19) is AnnVerdict.IGNORE_TOO_SMALL
    assert v(22) is AnnVerdict.ACCEPT and v(8) is AnnVerdict.NEGLIGIBLE


def ghost(images, _n=[0]):    # noqa: B006
    """A model that, on every third frame, is sure it sees a person in the top-left corner where nobody was labeled."""
    out = []
    for _ in images:
        _n[0] += 1
        out.append([Prediction((2.0, 2.0, 18.0, 60.0), 0.95, None)] if _n[0] % 3 == 0 else [])
    return out


def test_quarantine_is_a_full_dataset_of_its_own_and_does_not_count_as_accepted(tmp_path):
    out = tmp_path / "ds"
    rep = run(out, 20, CallableEvaluator(ghost, has_keypoints=False), max_attempt_factor=12)
    q = out / "quarantine"
    assert rep.quarantined > 0 and rep.accepted == 20            # target counts the clean dataset only
    qrecs = [r for p in sorted((q / "annotations").glob("*.jsonl")) for r in jsonl(p)]
    assert len(qrecs) == rep.quarantined
    assert all(r["meta"]["quality"]["verdict"] == "quarantine" for r in qrecs)
    imgs = list(q.glob("images/*/*.png"))
    assert len(imgs) == rep.quarantined                          # pixels kept: a reviewer can look at them
    assert (q / "labels").exists() and list(q.glob("annotations/coco_*.json"))
    main_recs = [r for p in sorted((out / "annotations").glob("*.jsonl")) for r in jsonl(p)]
    assert len(main_recs) == 20 and all(r["meta"]["quality"]["verdict"] != "quarantine" for r in main_recs)
    assert len(list(out.glob("images/*/*.png"))) == 20           # not mixed into the training images
    assert verify_dataset(out).ok                                # the main dataset ignores the quarantine directory
    rows = list(csv.DictReader((out / "quality_index.csv").open()))
    assert {"dataset", "quarantine"} <= {r["set"] for r in rows}
    assert json.loads((out / "closed_loop_report.json").read_text())["quarantined"] == rep.quarantined


def test_metadata_columns_in_coco_csv_and_index(tmp_path):
    out = tmp_path / "ds"
    run(out, 30)
    coco = json.loads(next(out.glob("annotations/coco_train.json")).read_text())
    im = coco["images"][0]
    assert {"verdict", "difficulty_score", "occlusion_index", "contrast_rate", "oks_score", "weight"} <= set(im)
    assert im["verdict"] in ("keep_clean", "keep_hard")
    assert all({"oks_score", "difficulty_score", "occlusion_index", "contrast_rate"} <= set(a) for a in coco["annotations"])
    rows = list(csv.DictReader((out / "quality_index.csv").open()))
    n_main = sum(len(jsonl(p)) for p in (out / "annotations").glob("*.jsonl"))
    assert len([r for r in rows if r["set"] == "dataset"]) == n_main
    assert {"frame_id", "verdict", "tier", "difficulty_score", "contrast_rate", "occlusion_index", "oks_score",
            "weight"} <= set(rows[0])
    assert all(0.0 <= float(r["difficulty_score"]) <= 1.0 for r in rows)
    persons = list(csv.DictReader((out / "quality_persons.csv").open()))
    assert persons and {"entity_id", "oks_score", "difficulty_score"} <= set(persons[0])
    idx = json.loads((out / "quality_index.json").read_text())
    assert all(v["verdict"] in ("keep_clean", "keep_hard") and "oks_score" in v for v in idx.values())
    # YOLO labels stay strictly standard: 5 + 3*K columns, nothing appended
    lines = [ln for f in sorted(out.glob("labels/*/*.txt")) for ln in f.read_text().splitlines()]   # negatives are empty files
    assert lines and all(len(ln.split()) == 5 + 3 * 13 for ln in lines)


def test_policy_hot_reload_changes_the_rules_mid_run_and_a_broken_file_is_ignored(tmp_path):
    pol = tmp_path / "policy.toml"
    pol.write_text("min_contrast_rate = 0.0\n")
    q = QualityPipeline(HUMAN_13, None, QualityConfig(policy_file=str(pol), policy_check_every=1))
    assert q.policy.version == 0
    assert q.reload_policy() and q.policy.version == 1
    assert not q.reload_policy()                                  # unchanged file: nothing happens
    time.sleep(0.01)
    pol.write_text("[policy]\nmin_contrast_rate = 0.9\n")         # `[policy]` table form is accepted too
    assert q.reload_policy() and q.policy.version == 2 and q.policy.cfg.min_contrast_rate == 0.9
    time.sleep(0.01)
    pol.write_text("min_contrast_rat = 0.5\n")                    # typo: must NOT silently keep the old value as "applied"
    assert not q.reload_policy() and q.policy.version == 2 and q.policy.cfg.min_contrast_rate == 0.9
    assert q.policy_changes[-1]["error"] and "unknown policy keys" in q.policy_changes[-1]["error"]
    time.sleep(0.01)
    pol.write_text("this is not toml [")
    assert not q.reload_policy() and q.policy.version == 2
    assert q.stats()["policy_version"] == 2 and len(q.stats()["policy_changes"]) == 4
    q.close()


def test_stricter_rules_loaded_during_a_run_are_stamped_on_later_verdicts(tmp_path):
    out = tmp_path / "ds"
    out.mkdir()
    pol = out / "quality_policy.toml"                             # the file may exist before the run starts
    pol.write_text("min_contrast_rate = 0.0\n")
    run(out, 25, quality_cfg=QualityConfig(policy_file=str(pol), policy_check_every=5))
    versions = {r["meta"]["quality"]["policy_version"] for p in (out / "annotations").glob("*.jsonl") for r in jsonl(p)}
    assert versions == {1}                                        # reloaded before the first frame was judged


def test_requalify_tightens_rules_on_stored_features_without_pixels(tmp_path):
    out = tmp_path / "ds"
    run(out, 40)
    same = requalify(out, {})
    assert same["excluded"] == 0 and sum(same["verdicts"].values()) == same["frames"] == 40
    strict = requalify(out, {"min_contrast_rate": 0.5})            # nobody has this much contrast
    assert strict["excluded"] > 0 and strict["verdicts"].get("reject", 0) == strict["excluded"]
    ex = (out / "excluded_frames.txt").read_text().split()
    assert len(ex) == strict["excluded"]
    rows = list(csv.DictReader((out / "quality_requalified.csv").open()))
    assert {r["new_verdict"] for r in rows} >= {"reject"} and all(r["policy_version"] == "1" for r in rows)


def test_requalify_cli(tmp_path, capsys):
    out = tmp_path / "ds"
    run(out, 12)
    pol = tmp_path / "strict.toml"
    pol.write_text("[policy]\nmin_brightness = 0.99\n")
    with pytest.raises(SystemExit) as e:
        main(["requalify", str(out), "--policy", str(pol)])
    assert e.value.code == EXIT_OK
    assert "excluded=" in capsys.readouterr().out and (out / "excluded_frames.txt").exists()
    bad = tmp_path / "bad.toml"
    bad.write_text("nonsense = 1\n")
    with pytest.raises(SystemExit) as e:
        main(["requalify", str(out), "--policy", str(bad)])
    assert e.value.code != EXIT_OK


def test_policy_reconfigure_rejects_unknown_keys_and_is_pure_on_failure():
    p = DefaultQualityPolicy(PolicyConfig())
    try:
        p.reconfigure({"blind_okss": 0.1})
        raise AssertionError("must raise")
    except ValueError:
        pass
    assert p.version == 0 and p.cfg.blind_oks == PolicyConfig().blind_oks
    p.reconfigure({"blind_oks": 0.3})
    assert p.version == 1 and p.cfg.blind_oks == 0.3 and np.isclose(p.cfg.hard_oks, 0.60)
