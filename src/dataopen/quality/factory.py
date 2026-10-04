"""Build the quality pipeline / adaptive randomizer from a plain dict (profile `[quality]` table + CLI overrides)."""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

from ..core.interfaces import IGameAdapter
from ..core.schema import SkeletonSchema
from .evaluators.decode import KeypointMap
from .feedback import AdaptiveRandomizer, FeedbackConfig
from .features import FeatureConfig
from .interfaces import IModelEvaluator
from .pipeline import QualityConfig, QualityPipeline
from .policy import PolicyConfig


class QualityConfigError(ValueError):
    pass


def _fields(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def build_evaluator(spec: dict[str, Any], schema: SkeletonSchema) -> Optional[IModelEvaluator]:
    kind = spec.get("evaluator", "none")
    if kind in ("none", "", None):
        return None
    if kind == "simulated":
        from .evaluators.simulated import SimulatedEvaluator
        return SimulatedEvaluator(schema, seed=int(spec.get("seed", 0)))
    if kind == "onnx":
        model = spec.get("model")
        if not model:
            raise QualityConfigError("evaluator = 'onnx' needs `model = \"path/to/model.onnx\"`")
        from pathlib import Path

        from .evaluators.onnx import OnnxEvaluator
        if not Path(str(model)).is_file():
            raise QualityConfigError(f"model file not found: {model}")
        size = spec.get("input_size", [640, 640])
        km = spec.get("keypoint_map", "coco17" if spec.get("format", "yolov8_pose") == "yolov8_pose" else None)
        kmap = {"coco17": KeypointMap.coco17_to_human13, "identity": lambda: KeypointMap.identity(schema.num_keypoints),
                None: lambda: None}.get(km)
        if kmap is None:
            raise QualityConfigError("keypoint_map must be 'coco17' (17-point COCO model -> 13 points) or 'identity'")
        if km == "coco17" and schema.num_keypoints != 13:
            raise QualityConfigError("keypoint_map 'coco17' maps onto the 13-point schema; use 'identity' for a model "
                                     "trained on your own keypoints")
        try:
            return OnnxEvaluator(str(model), spec.get("format", "yolov8_pose"), (int(size[0]), int(size[1])),
                                 spec.get("device", "cpu"), float(spec.get("conf_thr", 0.05)),
                                 float(spec.get("iou_thr", 0.7)), kmap())
        except ImportError as e:
            raise QualityConfigError(f"onnxruntime is not installed: pip install 'dataopen[quality]' ({e})") from e
        except ValueError:
            raise
        except Exception as e:                       # onnxruntime raises its own exception types for bad models
            raise QualityConfigError(f"cannot load {model}: {e}") from e
    raise QualityConfigError(f"unknown evaluator {kind!r}; use 'onnx', 'simulated' or 'none'")


def build_quality(spec: dict[str, Any], schema: SkeletonSchema) -> Optional[QualityPipeline]:
    """None when the section is absent or `enabled = false`. `evaluator = "none"` still gives the cheap gates."""
    if not spec or not spec.get("enabled", False):
        return None
    cfg_kw = {k: v for k, v in spec.items() if k in _fields(QualityConfig) and k not in ("policy", "features", "enabled")}
    policy = PolicyConfig(**{k: v for k, v in (spec.get("policy") or {}).items() if k in _fields(PolicyConfig)})
    features = FeatureConfig(**{k: v for k, v in (spec.get("features") or {}).items() if k in _fields(FeatureConfig)})
    cfg = QualityConfig(policy=policy, features=features, **cfg_kw)
    return QualityPipeline(schema, build_evaluator(spec, schema), cfg)


def build_randomizer(adapter: IGameAdapter, seed: int, adaptive: bool, feedback: Optional[dict[str, Any]] = None):
    """The adaptive randomizer (feedback-driven) or None (the orchestrator then builds the plain controller)."""
    if not adaptive:
        return None
    fb = FeedbackConfig(**{k: v for k, v in (feedback or {}).items() if k in _fields(FeedbackConfig)})
    return AdaptiveRandomizer(seed, adapter.parameter_space(), feedback=fb)
