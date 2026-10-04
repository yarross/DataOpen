"""Build the quality pipeline / adaptive randomizer from a plain dict (profile `[quality]` table + CLI overrides)."""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

from ..core.interfaces import IGameAdapter
from ..core.schema import SkeletonSchema
from .balance import BalanceConfig
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


def _keypoint_map(km: Optional[str], schema: SkeletonSchema, bundle) -> Optional[KeypointMap]:
    """How the model's keypoints become the dataset's: "identity" (the model emits the target schema) or the name of
    a model keypoint set ("coco17") that the schema file knows how to map (`[schema.models.coco17]`)."""
    if km in (None, ""):
        return None
    if km == "identity":
        return KeypointMap.identity(schema.num_keypoints)
    if km == "coco17" and schema.name == "human13":
        return KeypointMap.coco17_to_human13()
    if bundle is not None and km in bundle.model_names():
        return KeypointMap.from_mapping(bundle.model_mapping(km))
    avail = ["identity", *(bundle.model_names() if bundle is not None else [])]
    if km == "coco17" and bundle is None:
        raise QualityConfigError(f"keypoint_map 'coco17' maps onto the 13-point schema only; for schema {schema.name!r} "
                                 f"pass a schema file that has [schema.models.coco17] (--schema), or use 'identity' for a "
                                 f"model trained on this schema")
    raise QualityConfigError(f"keypoint_map {km!r} is unknown for schema {schema.name!r}; use one of {avail}")


def build_evaluator(spec: dict[str, Any], schema: SkeletonSchema, bundle=None) -> Optional[IModelEvaluator]:
    kind = spec.get("evaluator", "none")
    if kind in ("none", "", None):
        return None
    if spec.get("runtime") and kind != "runtime":                  # judge the model through the production runtime path
        return build_evaluator({**spec, "evaluator": "runtime", "inner": kind}, schema, bundle)
    if kind == "runtime":
        from ..runtime.backends import EvaluatorBackend, OrtBackend
        from ..runtime.evaluator import RuntimeEvaluator
        inner = spec.get("inner", "onnx")
        if inner == "onnx" and spec.get("format") == "apollo":
            if not spec.get("model"):
                raise QualityConfigError("runtime evaluator needs `model = \"path/to/model.onnx\"`")
            try:
                backend = OrtBackend(spec["model"], spec.get("device", "cpu"), float(spec.get("conf_thr", 0.25)),
                                     int(spec.get("max_det", 20)), spec.get("post", "auto"))
            except (ValueError, RuntimeError, ImportError) as e:
                raise QualityConfigError(f"cannot load {spec['model']} in the runtime: {e}") from e
            lay = backend.layout
            if lay.keypoints and tuple(lay.keypoints) != tuple(schema.keypoints):
                raise QualityConfigError(f"{spec['model']} was trained for the keypoints {list(lay.keypoints)}, but this run is "
                                         f"labeled with {list(schema.keypoints)}: pass the matching --schema")
        else:
            ev = build_evaluator({**spec, "evaluator": inner, "runtime": False}, schema, bundle)
            if ev is None:
                raise QualityConfigError("runtime evaluator: the inner evaluator is 'none'")
            backend = EvaluatorBackend(ev, n_kpt=schema.num_keypoints)
        return RuntimeEvaluator(backend, window=int(spec.get("runtime_window", 8)), n_kpt=schema.num_keypoints)
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
        fmt = spec.get("format", "yolov8_pose")
        km = spec.get("keypoint_map", {"yolov8_pose": "coco17", "table": "identity", "apollo": "identity"}.get(fmt))
        kmap = _keypoint_map(km, schema, bundle) if fmt != "apollo" else None
        try:
            ev = OnnxEvaluator(str(model), fmt, (int(size[0]), int(size[1])),
                                 spec.get("device", "cpu"), float(spec.get("conf_thr", 0.05)),
                                 float(spec.get("iou_thr", 0.7)), kmap, layout=spec.get("layout"),
                                 coords=spec.get("coords", "pixels"), class_map=spec.get("class_map"),
                                 input_dtype=spec.get("input_dtype", "auto"), input_layout=spec.get("input_layout", "auto"),
                                 max_det=int(spec.get("max_det", 20)), nms=bool(spec.get("nms", False)))
            lay = getattr(ev, "apollo_layout", None)
            if lay is not None and lay.keypoints and tuple(lay.keypoints) != tuple(schema.keypoints):
                raise QualityConfigError(f"{model} was trained for the keypoints {list(lay.keypoints)}, but this run is labeled "
                                         f"with {list(schema.keypoints)}: pass the matching --schema")
            return ev
        except ImportError as e:
            raise QualityConfigError(f"onnxruntime is not installed: pip install 'dataopen[quality]' ({e})") from e
        except (ValueError, QualityConfigError):
            raise
        except Exception as e:                       # onnxruntime raises its own exception types for bad models
            raise QualityConfigError(f"cannot load {model}: {e}") from e
    raise QualityConfigError(f"unknown evaluator {kind!r}; use 'onnx', 'simulated' or 'none'")


def build_quality(spec: dict[str, Any], schema: SkeletonSchema, rig_schema: Optional[SkeletonSchema] = None,
                  bundle=None) -> Optional[QualityPipeline]:
    """None when the section is absent or `enabled = false`. `evaluator = "none"` still gives the cheap gates.
    `schema` = the dataset's (target) keypoints; `rig_schema` = what the mod reports (3D sanity runs on it)."""
    if not spec or not spec.get("enabled", False):
        return None
    cfg_kw = {k: v for k, v in spec.items() if k in _fields(QualityConfig) and k not in ("policy", "features", "enabled")}
    policy = PolicyConfig(**{k: v for k, v in (spec.get("policy") or {}).items() if k in _fields(PolicyConfig)})
    features = FeatureConfig(**{k: v for k, v in (spec.get("features") or {}).items() if k in _fields(FeatureConfig)})
    cfg = QualityConfig(policy=policy, features=features, **cfg_kw)
    return QualityPipeline(schema, build_evaluator(spec, schema, bundle), cfg, rig_schema=rig_schema)


def build_randomizer(adapter: IGameAdapter, seed: int, adaptive: bool, feedback: Optional[dict[str, Any]] = None,
                     balance: Optional[dict[str, Any]] = None):
    """The adaptive randomizer (feedback-driven) or None (the orchestrator then builds the plain controller).
    `balance` is the `[quality.balance]` table (also accepted as `[quality.feedback.balance]`)."""
    if not adaptive:
        return None
    fb_spec = {k: v for k, v in (feedback or {}).items() if k in _fields(FeedbackConfig) and k != "balance"}
    bspec = balance if balance is not None else (feedback or {}).get("balance") or {}
    unknown = sorted(set(bspec) - _fields(BalanceConfig))
    if unknown:
        raise QualityConfigError(f"unknown [quality.balance] keys {unknown}; known: {sorted(_fields(BalanceConfig))}")
    fb = FeedbackConfig(balance=BalanceConfig(**bspec), **fb_spec)
    return AdaptiveRandomizer(seed, adapter.parameter_space(), feedback=fb)
