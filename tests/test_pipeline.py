import json
from pathlib import Path

import numpy as np
import pytest

from dataopen.adapters.mock import MockGameAdapter
from dataopen.core.annotation import AnnotationBuilder
from dataopen.core.models import CaptureRequest, FrameKind
from dataopen.core.orchestrator import DatasetOrchestrator, SessionConfig
from dataopen.core.projection import project
from dataopen.core.randomization import DomainRandomizationController
from dataopen.core.validation import DuplicateFilter


def load_records(root: Path) -> list[dict]:
    out = []
    for split in ("train", "val"):
        p = root / "annotations" / f"{split}.jsonl"
        if p.exists():
            out += [json.loads(line) for line in p.read_text().splitlines()]
    return out


def run(out: Path, frames=60, **kw):
    cfg = SessionConfig(out, seed=3, target_frames=frames, frames_per_scene=10, negative_ratio=0.2, **kw)
    return DatasetOrchestrator(MockGameAdapter(), cfg).run()


def test_end_to_end_outputs_are_consistent(tmp_path):
    rep = run(tmp_path / "ds")
    assert rep.accepted == 60 and rep.stop_reason == "target_reached"
    root = tmp_path / "ds"
    records = load_records(root)
    assert len(records) == 60
    n_images = len(list(root.glob("images/*/*.png")))
    n_labels = len(list(root.glob("labels/*/*.txt")))
    assert n_images == n_labels == 60                     # rejected frames leave no pixels behind
    negatives = [r for r in records if r["kind"] == "negative"]
    assert negatives and all(r["annotations"] == [] for r in negatives)
    for r in negatives:
        assert (root / "labels" / r["split"] / f"{r['frame_id']}.txt").read_text() == ""
    # no scene is split across train/val (leakage guard)
    split_of = {}
    for r in records:
        assert split_of.setdefault(r["scene_index"], r["split"]) == r["split"]
    # all keypoints / bboxes are inside the frame
    for r in records:
        for a in r["annotations"]:
            x, y, w, h = a["bbox"]
            assert x >= 0 and y >= 0 and x + w <= r["width"] + 1e-6 and y + h <= r["height"] + 1e-6
            kp = np.array(a["keypoints"]).reshape(-1, 3)
            vis = kp[:, 2] > 0
            assert (kp[vis, 0] >= 0).all() and (kp[vis, 0] < r["width"]).all()
            assert (kp[~vis, :2] == 0).all()
    assert (root / "data.yaml").exists() and (root / "annotations" / "coco_train.json").exists()


def test_same_seed_reproduces_identical_annotations(tmp_path):
    run(tmp_path / "a", frames=30)
    run(tmp_path / "b", frames=30)
    for s in ("train", "val"):
        pa, pb = tmp_path / "a" / "annotations" / f"{s}.jsonl", tmp_path / "b" / "annotations" / f"{s}.jsonl"
        assert pa.exists() == pb.exists()
        if pa.exists():
            assert pa.read_text() == pb.read_text()


def test_resume_continues_without_duplicating_scenes(tmp_path):
    out = tmp_path / "ds"
    run(out, frames=20)
    first = json.loads((out / "manifest.json").read_text())
    rep = run(out, frames=40, resume=True)
    assert rep.accepted >= 40
    scenes = [r["scene_index"] for r in load_records(out)]
    assert min(scenes) == 0 and max(scenes) >= first["scenes_done"]
    frame_ids = [r["frame_id"] for r in load_records(out)]
    assert len(frame_ids) == len(set(frame_ids))


def test_refuses_to_overwrite_non_empty_dir(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f").write_text("1")
    with pytest.raises(FileExistsError):
        run(tmp_path / "x")


def test_projected_ground_truth_matches_rendered_joint_pixels():
    """Plumbing check (joint order, flags, pixel coords end to end). The mock renders its dots with
    the same `project()`, so this is NOT an independent check of the math: that is covered by the
    hand-computed cases in test_projection.py and the look-at centring test below."""
    ad = MockGameAdapter()
    ad.connect()
    rz = DomainRandomizationController(1, ad.parameter_space())
    scene = rz.sample_scene(0)
    ad.environment.apply(scene)
    hs = ad.spawner.spawn(scene)
    checked = 0
    for fi in range(30):
        spec = rz.sample_frame(scene, fi, FrameKind.POSITIVE)
        ad.spawner.update_actors(hs, spec)
        snap = ad.capture.capture(CaptureRequest(spec.frame_id, spec, 320, 240))
        img = ad.capture._pending[spec.frame_id]
        built = AnnotationBuilder(ad.info.schema).build(snap)
        for a in built.annotations:
            for x, y, v in a.keypoints:
                if v == 2 and 3 < x < 317 and 3 < y < 237:
                    assert img[int(y), int(x)].tolist() == list(snap.meta["colors"][a.entity_id])
                    checked += 1
        ad.capture.discard(snap)
    assert checked > 50


def test_camera_target_is_centred_when_unrolled():
    ad = MockGameAdapter()
    rz = DomainRandomizationController(1, ad.parameter_space())
    scene = rz.sample_scene(0)
    ad.environment.apply(scene)
    hs = ad.spawner.spawn(scene)
    spec = rz.sample_frame(scene, 0, FrameKind.POSITIVE)
    spec.camera.roll_deg, spec.camera.height_offset = 0.0, 0.0
    ad.spawner.update_actors(hs, spec)
    snap = ad.capture.capture(CaptureRequest("t", spec, 320, 240))
    tgt = next(a for a in ad.spawner.w.actors.items() if a[0] == spec.camera.target_index)[1]
    uv, z = project(np.array([tgt["x"], tgt["y"], 1.0]), snap.camera)
    assert np.allclose(uv, [160, 120], atol=1e-6)


def test_negative_frames_report_no_persons():
    ad = MockGameAdapter()
    rz = DomainRandomizationController(1, ad.parameter_space())
    scene = rz.sample_scene(0)
    ad.environment.apply(scene)
    hs = ad.spawner.spawn(scene)
    spec = rz.sample_frame(scene, 0, FrameKind.NEGATIVE)
    ad.spawner.set_active(hs, False)
    snap = ad.capture.capture(CaptureRequest("n", spec, 320, 240))
    assert snap.entities == []


def test_duplicate_filter_catches_repeated_state_and_stale_pixels():
    ad = MockGameAdapter()
    rz = DomainRandomizationController(1, ad.parameter_space())
    scene = rz.sample_scene(0)
    ad.environment.apply(scene)
    hs = ad.spawner.spawn(scene)
    spec = rz.sample_frame(scene, 0, FrameKind.POSITIVE)
    ad.spawner.update_actors(hs, spec)
    s1 = ad.capture.capture(CaptureRequest("a", spec, 320, 240))
    s2 = ad.capture.capture(CaptureRequest("b", spec, 320, 240))          # same state again
    f = DuplicateFilter()
    assert f.check(spec, s1, None) is None and f.check(spec, s2, None) == "duplicate_state"

    spec2 = rz.sample_frame(scene, 1, FrameKind.POSITIVE)
    ad.spawner.update_actors(hs, spec2)
    s3 = ad.capture.capture(CaptureRequest("c", spec2, 320, 240))
    s3.thumbnail = np.random.default_rng(0).random((8, 9))
    s4 = ad.capture.capture(CaptureRequest("d", rz.sample_frame(scene, 2, FrameKind.POSITIVE), 320, 240))
    s4.thumbnail = s3.thumbnail.copy()                                    # stale readback: pixels unchanged
    g = DuplicateFilter(image_hash=True)
    assert g.check(spec2, s3, None) is None and g.check(spec2, s4, None) == "duplicate_image"


def _person_snapshot(height_px: float, width_px: float = None):
    """A standing person whose projected height is ~height_px, centred in a 640x480 frame."""
    import numpy as np
    from dataopen.core.models import CameraModel, EntityState, FrameSnapshot
    from dataopen.core.schema import HUMAN_13
    # camera at origin looking down +x (Z up); f for vfov 60 at 480px: 415.7 px
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    M = np.eye(4)
    M[:3, :3] = R
    cam = CameraModel.from_vertical_fov(640, 480, 60.0, M)
    dist = 415.7 * 1.8 / height_px                      # 1.8 m tall person
    pos = {"head": (0, 0, 1.7), "neck": (0, 0, 1.5), "l_shoulder": (0, 0.2, 1.45), "r_shoulder": (0, -0.2, 1.45),
           "pelvis": (0, 0, 0.95), "l_elbow": (0, 0.3, 1.2), "r_elbow": (0, -0.3, 1.2), "l_wrist": (0, 0.3, 0.9),
           "r_wrist": (0, -0.3, 0.9), "l_knee": (0, 0.1, 0.5), "r_knee": (0, -0.1, 0.5), "l_ankle": (0, 0.1, 0.05),
           "r_ankle": (0, -0.1, 0.05)}
    sk = np.array([pos[k] for k in HUMAN_13.keypoints], dtype=float) + [dist, 0, -0.9]
    return FrameSnapshot("t", 0, cam, [EntityState(0, "r", sk, np.ones(13, bool))])


def test_person_size_policy_negligible_small_and_labeled():
    from dataopen.core.annotation import AnnotationBuilder, Verdict
    from dataopen.core.schema import HUMAN_13
    b = AnnotationBuilder(HUMAN_13)
    verdict = lambda h: next(iter(b.build(_person_snapshot(h)).verdicts.values()))     # noqa: E731
    assert verdict(8) is Verdict.NEGLIGIBLE                    # tiny: ignored, does not reject the frame
    assert verdict(18) is Verdict.IGNORE_TOO_SMALL             # visible but unreliable: rejects the frame
    assert verdict(60) is Verdict.ACCEPT


def test_negligible_person_does_not_reject_positive_frames_but_does_reject_negatives():
    from dataopen.core.annotation import AnnotationBuilder
    from dataopen.core.models import CameraSpec, FrameKind, FrameSpec
    from dataopen.core.schema import HUMAN_13
    from dataopen.core.validation import NegativeFrameValidator, PositiveFrameValidator
    b = AnnotationBuilder(HUMAN_13)
    snap_far = _person_snapshot(8)
    built = b.build(snap_far)
    pos_spec = FrameSpec(0, 0, 0, FrameKind.POSITIVE, CameraSpec(), [])
    neg_spec = FrameSpec(0, 0, 0, FrameKind.NEGATIVE, CameraSpec(), [])
    assert PositiveFrameValidator().check(pos_spec, snap_far, built) == "positive_without_annotations"
    assert NegativeFrameValidator().check(neg_spec, snap_far, built) == "negative_contains_person"   # still a person
    # a good person plus a negligible one: the frame is kept
    near = _person_snapshot(80)
    far = _person_snapshot(8)
    far.entities[0].entity_id = 1
    near.entities.append(far.entities[0])
    built2 = b.build(near)
    assert len(built2.annotations) == 1 and PositiveFrameValidator().check(pos_spec, near, built2) is None
