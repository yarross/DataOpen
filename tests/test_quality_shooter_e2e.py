"""The whole loop on the 12-point shooter schema (and on a schema defined only as data), through the real CLI."""
import csv
import json
import time
from pathlib import Path

import pytest

from dataopen.cli import EXIT_OK, main


def run_cli(*argv) -> int:
    with pytest.raises(SystemExit) as e:
        main(list(argv))
    return e.value.code


def recs(root: Path):
    return [json.loads(x) for p in sorted((root / "annotations").glob("*.jsonl")) for x in p.read_text().splitlines()]


@pytest.fixture(scope="module")
def shooter_ds(tmp_path_factory):
    out = tmp_path_factory.mktemp("shooter") / "ds"
    code = run_cli("collect", "--game", "mock_shooter", "--frames", "60", "--out", str(out), "--quality-sim", "--adaptive",
                   "--no-doctor", "--seed", "5")
    assert code == EXIT_OK
    return out


def test_dataset_is_labeled_with_12_points_two_classes_and_the_aim_point_metadata(shooter_ds):
    out = shooter_ds
    rs = recs(out)
    assert len(rs) == 60 and all(len(a["keypoints"]) == 36 for r in rs for a in r["annotations"])
    classes = {a["class_id"] for r in rs for a in r["annotations"]}
    assert classes == {0, 1}                                                    # both teams occur
    coco = json.loads(next(out.glob("annotations/coco_train.json")).read_text())
    assert [c["name"] for c in coco["categories"]] == ["player_ct", "player_t"]
    assert len(coco["categories"][0]["keypoints"]) == 12 and coco["categories"][0]["keypoints"][1] == "head_center"
    assert {a["category_id"] for a in coco["annotations"]} == {1, 2}
    assert any("focus_oks" in a and "focus_difficulty" in a for a in coco["annotations"])
    yaml = (out / "data.yaml").read_text()
    assert "kpt_shape: [12, 3]" in yaml and "0: player_ct" in yaml and "1: player_t" in yaml
    assert "flip_idx: [0, 1, 4, 3, 2, 7, 6, 5, 10, 9, 8, 11]" in yaml
    lines = [ln for f in out.glob("labels/*/*.txt") for ln in f.read_text().splitlines()]
    assert lines and all(len(ln.split()) == 5 + 3 * 12 and ln.split()[0] in ("0", "1") for ln in lines)
    rows = list(csv.DictReader((out / "quality_index.csv").open()))
    assert {"focus_oks", "focus_difficulty", "class_accuracy"} <= set(rows[0]) and any(r["focus_oks"] for r in rows)
    persons = list(csv.DictReader((out / "quality_persons.csv").open()))
    assert {"class_id", "focus_oks", "focus_err_rel", "focus_visibility"} <= set(persons[0])
    card = json.loads((out / "DATASET_CARD.json").read_text())
    assert card["skeleton"]["primary"] == ["head_center"] and card["skeleton"]["classes"] == ["player_ct", "player_t"]
    assert card["game"]["capture_schema"] == "human13" and len(card["skeleton"]["weights"]) == 12


def test_verdicts_carry_the_aim_point_and_the_adaptive_report_shows_where_it_fails(shooter_ds):
    rs = recs(shooter_ds)
    q = [r["meta"]["quality"] for r in rs]
    assert any(x["metrics"]["focus_evaluated"] for x in q)
    hard = [x for x in q if x["reasons"] and "focus" in x["reasons"][0]]
    assert hard and all(x["verdict"] == "keep_hard" and x["weight"] > 1.5 for x in hard if "blind" in x["reasons"][0])
    rep = json.loads((shooter_ds / "closed_loop_report.json").read_text())["feedback"]
    row = next(r for d in rep["parameters"].values() for r in d["rows"] if r["n"] >= 5)
    assert {"mean_focus_oks", "focus_blind_rate", "mean_focus_difficulty"} <= set(row)
    assert {"actor.camo", "actor.headgear", "env.smoke_density", "env.flash_intensity", "frame.head_cover"} <= set(rep["parameters"])


def test_verify_preview_requalify_and_merge_work_on_a_12_point_dataset(shooter_ds, tmp_path):
    assert run_cli("verify", str(shooter_ds)) == EXIT_OK
    assert run_cli("preview", str(shooter_ds), "--n", "4", "--out", str(tmp_path / "p.png")) == EXIT_OK
    assert (tmp_path / "p.png").stat().st_size > 1000
    pol = tmp_path / "p.toml"
    pol.write_text("focus_priority = 0.0\n")
    assert run_cli("requalify", str(shooter_ds), "--policy", str(pol)) == EXIT_OK
    shards = [tmp_path / "s0", tmp_path / "s1"]
    for i, d in enumerate(shards):
        assert run_cli("collect", "--game", "mock_shooter", "--frames", "12", "--out", str(d), "--no-doctor", "--seed", "9",
                       "--shard", f"{i}/2") == EXIT_OK
    assert run_cli("merge", str(tmp_path / "m"), *map(str, shards)) == EXIT_OK
    merged = json.loads((tmp_path / "m" / "DATASET_CARD.json").read_text())
    assert merged["skeleton"]["classes"] == ["player_ct", "player_t"]
    assert "player_t" in (tmp_path / "m" / "data.yaml").read_text()


def test_a_head_hard_world_gets_more_aim_point_misses_than_an_easy_one(tmp_path):
    """The mock's head-specific parameters (smoke, flash, camouflage, cover) really do make the aim point harder."""
    from dataopen.adapters.mock import MockGameAdapter
    from dataopen.core.models import CaptureRequest, FrameKind
    from dataopen.core.randomization import DomainRandomizationController
    from dataopen.core.schema_io import resolve_target
    from dataopen.quality.features import compute_features
    from dataopen.core.annotation import AnnotationBuilder
    a = MockGameAdapter(320, 320, variant="shooter")
    _, mapping = resolve_target("shooter12", a.info.schema)
    rz = DomainRandomizationController(1, a.parameter_space())
    builder = AnnotationBuilder(mapping.target)

    def mean_focus_difficulty(**env_override):
        vals = []
        for i in range(6):
            scene = rz.sample_scene(i)
            scene.environment.update(time_of_day=12.0, weather="clear", fog_density=0.0004, cloud_cover=0.1, exposure_ev=0.0,
                                     **env_override)
            for act in scene.actors:
                act.update(camo=env_override.get("camo", 0.0))
            a.environment.apply(scene)
            hs = a.spawner.spawn(scene)
            spec = rz.sample_frame(scene, 0, FrameKind.POSITIVE)
            spec.camera.distance, spec.camera.target_index = 6.0, 0
            for fp in spec.actor_frame:
                fp["head_cover"] = 0.0
            a.spawner.update_actors(hs, spec)
            snap = a.capture.capture(CaptureRequest("f", spec, 320, 320))
            built = builder.build(mapping.convert_snapshot(snap))
            px = a.capture.peek_pixels(snap)
            if built.annotations:
                vals.append(compute_features(px.array, built.annotations, [], mapping.target).focus_difficulty)
            a.capture.discard(snap)
            a.spawner.despawn_all()
        return sum(vals) / len(vals)

    easy = mean_focus_difficulty(smoke_density=0.0, flash_intensity=0.0)
    hard = mean_focus_difficulty(smoke_density=0.0, flash_intensity=0.9)                 # a flash washes out the head
    assert hard > easy + 0.05, (easy, hard)


TINY = """
[schema]
name = "tiny5"
source = "human13"
keypoints = ["crown", "chest", "l_hand", "r_hand", "belly"]
edges = [["crown", "chest"], ["chest", "belly"]]
flip_pairs = [["l_hand", "r_hand"]]
primary = ["crown"]
classes = ["red", "blue"]
class_key = "side"
[schema.roles]
head = "crown"
neck = "chest"
pelvis = "belly"
l_shoulder = "l_hand"
r_shoulder = "r_hand"
[schema.points.crown]
sigma = 0.03
weight = 2.0
rule = "extend_m(neck, head, 0.1)"
[schema.points.chest]
derived = true
rule = "lerp(neck, pelvis, 0.2)"
[schema.points.l_hand]
rule = "copy(l_wrist)"
[schema.points.r_hand]
rule = "copy(r_wrist)"
[schema.points.belly]
rule = "copy(pelvis)"
"""


def test_a_schema_that_exists_only_as_a_toml_file_runs_the_same_loop(tmp_path):
    f = tmp_path / "tiny5.toml"
    f.write_text(TINY)
    out = tmp_path / "ds"
    t0 = time.time()
    assert run_cli("collect", "--game", "mock", "--frames", "25", "--out", str(out), "--quality-sim", "--no-doctor",
                   "--schema", str(f)) == EXIT_OK
    rs = recs(out)
    assert len(rs) == 25 and all(len(a["keypoints"]) == 15 for r in rs for a in r["annotations"])
    assert "kpt_shape: [5, 3]" in (out / "data.yaml").read_text() and "1: blue" in (out / "data.yaml").read_text()
    assert any(r["meta"]["quality"]["metrics"]["focus_evaluated"] for r in rs)             # `crown` is the primary point
    assert run_cli("verify", str(out)) == EXIT_OK
    assert time.time() - t0 < 120
    # a bad schema file is a usage error with a message, not a traceback
    f.write_text(TINY.replace('"copy(l_wrist)"', '"copy(l_wristt)"'))
    assert run_cli("collect", "--game", "mock", "--frames", "5", "--out", str(tmp_path / "x"), "--no-doctor",
                   "--schema", str(f)) == 4
