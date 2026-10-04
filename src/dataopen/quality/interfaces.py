"""Abstractions of the closed-loop quality subsystem.

    IModelEvaluator         runs a baseline model (ONNX Runtime / TensorRT / RKNN / any callable) on frames
    IQualityMetricCalculator GT vs predictions -> OKS / IoU / confidence metrics
    IQualityPolicy          metrics + model-independent features -> verdict (keep / hard / drop / suspect)
    IFeedbackController     verdicts -> adaptation of the Domain Randomizer
    IPixelSource            frames in core memory (host grab, shared memory ring, staged file)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np

from ..core.models import Annotation, FrameSpec, SceneSpec
from .types import Prediction, QualityMetrics, QualityVerdict


class IModelEvaluator(ABC):
    """Baseline model wrapper. Must be thread-safe or serialize internally: the pipeline calls it from a worker pool."""

    name: str = "evaluator"
    has_keypoints: bool = True
    wants_hints: bool = False          # True only for test doubles that peek at the ground truth

    @abstractmethod
    def predict(self, images: Sequence[np.ndarray], hints: Optional[Sequence[Sequence[Annotation]]] = None
                ) -> list[list[Prediction]]:
        """RGB uint8 (H, W, 3) images -> detections per image, in ORIGINAL pixel coordinates."""

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        pass


class IQualityMetricCalculator(ABC):
    @abstractmethod
    def compute(self, gt: Sequence[Annotation], preds: Optional[Sequence[Prediction]], backend: str = "",
                latency_ms: float = 0.0, explained_boxes: Sequence[Sequence[float]] = ()) -> QualityMetrics: ...


class IQualityPolicy(ABC):
    @abstractmethod
    def decide(self, metrics: QualityMetrics, features: Any, is_negative: bool) -> QualityVerdict: ...


class IFeedbackController(ABC):
    """Receives every verdict; adapts the sampling distribution of the Domain Randomizer."""

    @abstractmethod
    def observe(self, scene: SceneSpec, frame: FrameSpec, verdict: QualityVerdict) -> None: ...

    @abstractmethod
    def state(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def report(self) -> dict[str, Any]: ...


class IPixelSource(ABC):
    """Gives the core a frame's pixels without a round trip through disk, ideally without a copy."""

    @abstractmethod
    def peek(self, token: str, max_side: Optional[int] = None) -> Optional["PixelHandle"]: ...


class PixelHandle:
    """Pixels plus an explicit lifetime: `release()` frees the shared-memory slot / buffer."""

    def __init__(self, array: np.ndarray, release=None) -> None:
        self.array = array
        self._release = release

    def release(self) -> None:
        if self._release is not None:
            self._release()
            self._release = None
