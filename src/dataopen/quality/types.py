"""Data structures of the closed-loop quality subsystem (Inference-in-the-Loop)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


@dataclass(eq=False)
class Prediction:
    """One detection from a model, in ORIGINAL image pixel coordinates."""

    bbox: tuple[float, float, float, float]       # x, y, w, h
    score: float
    keypoints: Optional[np.ndarray] = None        # (K, 3): x, y, confidence; None for box-only models (D-FINE)


class Tier(str, Enum):
    KEEP = "keep"                          # normal, usable frame
    KEEP_HARD = "keep_hard"                # perceptible but the baseline model struggles: the most valuable frames
    HARD_NEGATIVE = "hard_negative"        # negative frame the model fires on (kept, tagged)
    DROP_INVISIBLE = "drop_invisible"      # labels say a person is there, pixels say it cannot be seen: poison
    DROP_GT_SANITY = "drop_gt_sanity"      # the 3D skeleton itself is broken (animation glitch)
    DROP_RENDER = "drop_render"            # blank / saturated / otherwise broken picture
    DROP_PHANTOM = "drop_phantom"          # the model confidently sees a person the labels do not contain
    SUSPECT = "suspect"                    # model confidently disagrees with the labels: quarantined for review
    REJECTED = "rejected"                  # stopped by the cheap validators before any quality check (wasted attempt)

    @property
    def is_drop(self) -> bool:
        return self.value.startswith("drop_") or self in (Tier.SUSPECT, Tier.REJECTED)


@dataclass
class MatchedPerson:
    entity_id: int
    oks: float = 0.0               # 0 when the model found nothing
    iou: float = 0.0
    score: float = 0.0
    keypoint_conf_mean: float = 0.0


@dataclass
class QualityMetrics:
    """Ground truth vs. the evaluator's predictions (None fields = no evaluator ran on this frame)."""

    evaluated: bool = False
    backend: str = ""
    latency_ms: float = 0.0
    n_gt: int = 0
    n_pred: int = 0                    # predictions above the match confidence
    n_phantom: int = 0                 # confident predictions not explained by any labeled person
    mean_oks: float = 0.0
    min_oks: float = 0.0
    oks_spread: float = 0.0
    recall: float = 0.0                # fraction of labeled persons found with OKS >= oks_match_thr
    mean_iou: float = 0.0
    mean_conf: float = 0.0
    max_disagreement_score: float = 0.0  # highest-confidence prediction that overlaps a GT box but disagrees on keypoints
    persons: list[MatchedPerson] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "persons"}
        d["persons"] = [{k: round(v, 4) if isinstance(v, float) else v for k, v in p.__dict__.items()} for p in self.persons]
        return d


@dataclass
class PersonFeatures:
    entity_id: int
    contrast_rate: float = 0.0        # colour distance between the person's visible limbs and their surroundings (0..1)
    brightness: float = 0.0           # mean luma of the visible limbs (0..1)
    occlusion_index: float = 0.0      # share of keypoints that are not visible (0..1)
    size_px: float = 0.0              # bbox height
    sharpness: float = 0.0            # local Laplacian variance (blur / fog lowers it)
    perceptibility: float = 0.0       # 0..1: can a model reasonably see this person? (contrast x brightness x size)


@dataclass
class FrameFeatures:
    """Model-independent properties of the picture + labels."""

    has_pixels: bool = False
    luma_mean: float = 0.0
    luma_std: float = 0.0
    saturated_frac: float = 0.0       # share of pure black/white pixels
    persons: list[PersonFeatures] = field(default_factory=list)
    gt_issues: list[str] = field(default_factory=list)

    @property
    def min_perceptibility(self) -> float:
        return min((p.perceptibility for p in self.persons), default=1.0)

    @property
    def mean_occlusion(self) -> float:
        return float(np.mean([p.occlusion_index for p in self.persons])) if self.persons else 0.0

    @property
    def mean_contrast(self) -> float:
        return float(np.mean([p.contrast_rate for p in self.persons])) if self.persons else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"has_pixels": self.has_pixels, "luma_mean": round(self.luma_mean, 4), "luma_std": round(self.luma_std, 4),
                "saturated_frac": round(self.saturated_frac, 4), "gt_issues": self.gt_issues,
                "persons": [{k: round(v, 4) if isinstance(v, float) else v for k, v in p.__dict__.items()}
                            for p in self.persons]}


@dataclass
class QualityVerdict:
    tier: Tier
    reasons: list[str]
    difficulty: float                  # 0 (trivial) .. 1 (very hard), combines static features and the model's trouble
    occlusion_index: float
    contrast_rate: float
    weight: float                      # suggested training sample weight (hard examples > 1)
    utility: float                     # reward for the adaptive randomizer (0 = wasted frame, 1 = ideal edge case)
    metrics: QualityMetrics
    features: FrameFeatures

    def to_dict(self) -> dict[str, Any]:
        return {"tier": self.tier.value, "reasons": self.reasons, "difficulty": round(self.difficulty, 4),
                "occlusion_index": round(self.occlusion_index, 4), "contrast_rate": round(self.contrast_rate, 4),
                "weight": round(self.weight, 3), "utility": round(self.utility, 3),
                "metrics": self.metrics.to_dict(), "features": self.features.to_dict()}


def rejected_verdict(reason: str) -> QualityVerdict:
    """Feedback-only verdict for an attempt the validators rejected: the sampled parameters produced nothing usable."""
    return QualityVerdict(Tier.REJECTED, [reason], 0.0, 0.0, 0.0, 0.0, 0.0, QualityMetrics(), FrameFeatures())
