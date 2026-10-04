import json
import shutil
from pathlib import Path

import pytest

from dataopen.adapters.mock import MockGameAdapter
from dataopen.core.orchestrator import DatasetOrchestrator, SessionConfig
from dataopen.core.qa import verify_dataset, write_reports


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("qa") / "ds"
    DatasetOrchestrator(MockGameAdapter(), SessionConfig(root, seed=11, target_frames=60, frames_per_scene=10,
                                                         negative_ratio=0.15)).run()
    return root


@pytest.fixture
def copy(dataset, tmp_path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(dataset, dst)
    return dst


def codes(rep):
    return {i.code for i in rep.errors}


def test_clean_dataset_verifies_and_has_stats(dataset):
    rep = verify_dataset(dataset)
    assert rep.ok, [i.__dict__ for i in rep.errors]
    st = rep.stats
    assert st["frames"] == 60 and st["persons"] > 0 and st["by_kind"]["negative"] > 0
    assert set(st["keypoint_visibility"]) >= {"head", "l_wrist"} and st["bbox_height_px"]["p50"] > 0
    assert len(st["time_of_day_hist_3h_bins"]) == 8 and "weather" in st


def test_reports_are_written(dataset, tmp_path):
    out = tmp_path / "r"
    shutil.copytree(dataset, out)
    write_reports(verify_dataset(out), out)
    assert json.loads((out / "qa_report.json").read_text())["ok"] is True
    assert "# QA report: OK" in (out / "qa_report.md").read_text()


def test_missing_and_wrong_size_image_are_errors(copy):
    imgs = sorted(copy.glob("images/*/*.png"))
    imgs[0].unlink()
    from dataopen.core.imageio import write_png
    import numpy as np
    write_png(imgs[1], np.zeros((10, 10, 3), dtype=np.uint8))
    assert {"missing_image", "image_size_mismatch"} <= codes(verify_dataset(copy))


def test_yolo_label_tampering_is_detected(copy):
    labels = [p for p in sorted(copy.glob("labels/*/*.txt")) if p.read_text().strip()]
    labels[0].write_text(labels[0].read_text().replace("0.", "0.9", 1))
    labels[1].unlink()
    assert {"yolo_bbox_mismatch", "missing_yolo_label"} & codes(verify_dataset(copy))
    assert "missing_yolo_label" in codes(verify_dataset(copy))


def test_coco_count_mismatch_and_orphan_image(copy):
    p = next(copy.glob("annotations/coco_*.json"))
    d = json.loads(p.read_text())
    d["annotations"] = d["annotations"][:-1]
    p.write_text(json.dumps(d))
    (copy / "images" / "train" / "stray.png").write_bytes(b"x")
    rep = verify_dataset(copy)
    assert "coco_annotation_count" in codes(rep)
    assert any(i.code == "orphan_image" for i in rep.issues)


def test_record_level_corruption_is_detected(copy):
    p = next(copy.glob("annotations/train.jsonl"))
    lines = p.read_text().splitlines()
    rec = next(json.loads(ln) for ln in lines if json.loads(ln)["annotations"])
    bad = json.loads(json.dumps(rec))
    bad["frame_id"] = rec["frame_id"]                     # duplicate id
    bad["annotations"][0]["keypoints"][2] = 7             # invalid flag
    bad["annotations"][0]["bbox"] = [0, 0, -5, 10]        # invalid bbox
    p.write_text("\n".join(lines + [json.dumps(bad)]) + "\n")
    assert {"duplicate_frame_id", "bad_visibility_flag", "bad_bbox"} <= codes(verify_dataset(copy, check_images=False))


def test_empty_directory_is_an_error(tmp_path):
    rep = verify_dataset(tmp_path)
    assert not rep.ok and rep.errors[0].code == "no_annotations"
