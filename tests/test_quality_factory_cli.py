import numpy as np
import pytest

from dataopen.adapters.mock import MockGameAdapter
from dataopen.cli import EXIT_OK, main
from dataopen.core.imageio import write_png
from dataopen.core.schema import HUMAN_13
from dataopen.quality.evaluators.simulated import SimulatedEvaluator
from dataopen.quality.factory import QualityConfigError, build_evaluator, build_quality, build_randomizer
from dataopen.quality.feedback import AdaptiveRandomizer

onnx = pytest.importorskip("onnx")
from onnx import TensorProto  # noqa: E402

from test_quality_decode_onnx import W, H, IN, coco_person, const_model, yolo_tensor  # noqa: E402


def test_build_quality_applies_nested_policy_and_feature_settings():
    q = build_quality({"enabled": True, "evaluator": "simulated", "max_inflight": 4, "sample_every": 3,
                       "policy": {"phantom_action": "hard", "bogus": 1}, "features": {"contrast_ref": 0.2}}, HUMAN_13)
    assert q.cfg.max_inflight == 4 and q.cfg.sample_every == 3 and q.cfg.policy.phantom_action == "hard"
    assert q.cfg.features.contrast_ref == 0.2 and isinstance(q.evaluator, SimulatedEvaluator)
    q.close()
    assert build_quality({}, HUMAN_13) is None and build_quality({"enabled": False, "evaluator": "onnx"}, HUMAN_13) is None
    static = build_quality({"enabled": True, "evaluator": "none"}, HUMAN_13)
    assert static.evaluator is None
    static.close()


def test_evaluator_configuration_errors_are_readable(tmp_path):
    with pytest.raises(QualityConfigError, match="needs `model"):
        build_evaluator({"evaluator": "onnx"}, HUMAN_13)
    with pytest.raises(QualityConfigError, match="not found"):
        build_evaluator({"evaluator": "onnx", "model": str(tmp_path / "x.onnx")}, HUMAN_13)
    with pytest.raises(QualityConfigError, match="unknown evaluator"):
        build_evaluator({"evaluator": "magic"}, HUMAN_13)
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(QualityConfigError, match="cannot load"):
        build_evaluator({"evaluator": "onnx", "model": str(bad)}, HUMAN_13)
    good = tmp_path / "ok.onnx"
    const_model(good, [("output0", np.zeros((1, 56, 3), np.float32))], [("images", TensorProto.FLOAT, [1, 3, IN, IN])])
    with pytest.raises(QualityConfigError, match="keypoint_map"):
        build_evaluator({"evaluator": "onnx", "model": str(good), "keypoint_map": "weird"}, HUMAN_13)
    from dataopen.core.schema import SkeletonSchema
    small = SkeletonSchema("s", ("a", "b"), (), ())
    with pytest.raises(QualityConfigError, match="13-point"):
        build_evaluator({"evaluator": "onnx", "model": str(good), "keypoint_map": "coco17"}, small)


def test_build_randomizer_only_when_adaptive():
    ad = MockGameAdapter()
    assert build_randomizer(ad, 1, False) is None
    r = build_randomizer(ad, 1, True, {"warmup_frames": 7, "unknown": 1})
    assert isinstance(r, AdaptiveRandomizer) and r.fb.warmup_frames == 7


def test_cli_eval_image_draws_detections_from_an_onnx_model(tmp_path, capsys):
    box = (400.0, 200.0, 100.0, 300.0)
    model = tmp_path / "m.onnx"
    const_model(model, [("output0", yolo_tensor([(box, 0.9, coco_person(*box))]))], [("images", TensorProto.FLOAT, [1, 3, IN, IN])])
    img = tmp_path / "i.png"
    write_png(img, np.full((H, W, 3), 90, dtype=np.uint8))
    with pytest.raises(SystemExit) as e:
        main(["eval-image", "--model", str(model), "--image", str(img), "--out", str(tmp_path / "o.png")])
    out = capsys.readouterr().out
    assert e.value.code == EXIT_OK and "1 detections" in out and "keypoints=yes" in out and (tmp_path / "o.png").exists()
