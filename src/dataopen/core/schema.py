"""Unified skeleton schema and per-rig bone mapping.

The schema is data, not code: everything downstream (projection, COCO/YOLO export,
flip indices) is driven by it, so changing the keypoint set is a one-line change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SkeletonSchema:
    name: str
    keypoints: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    flip_pairs: tuple[tuple[str, str], ...]
    # Per-keypoint OKS falloff constants (COCO convention: how precisely a human annotator can place the point).
    # Empty tuple = a uniform default.
    sigmas: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.sigmas and len(self.sigmas) != len(self.keypoints):
            raise ValueError("sigmas must have one value per keypoint")
        names = set(self.keypoints)
        if len(names) != len(self.keypoints):
            raise ValueError("duplicate keypoint names")
        for a, b in (*self.edges, *self.flip_pairs):
            if a not in names or b not in names:
                raise ValueError(f"unknown keypoint in edge/flip pair: {a}, {b}")

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    def index(self, name: str) -> int:
        return self.keypoints.index(name)

    def flip_idx(self) -> list[int]:
        """Index permutation for horizontal-flip augmentation (YOLO `flip_idx`)."""
        idx = list(range(self.num_keypoints))
        for a, b in self.flip_pairs:
            ia, ib = self.index(a), self.index(b)
            idx[ia], idx[ib] = ib, ia
        return idx

    def oks_sigmas(self) -> list[float]:
        return list(self.sigmas) if self.sigmas else [0.07] * self.num_keypoints

    def coco_skeleton(self) -> list[list[int]]:
        """COCO `skeleton` is 1-based."""
        return [[self.index(a) + 1, self.index(b) + 1] for a, b in self.edges]


# NOTE: the brief says "12 bones", but head, neck, 2x shoulders, 2x elbows, 2x wrists,
# pelvis, 2x knees, 2x ankles is 13 points. The schema is pluggable; 13 is the default.
HUMAN_13 = SkeletonSchema(
    name="human13",
    keypoints=(
        "head", "neck",
        "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
        "pelvis",
        "l_knee", "r_knee", "l_ankle", "r_ankle",
    ),
    edges=(
        ("head", "neck"),
        ("neck", "l_shoulder"), ("neck", "r_shoulder"),
        ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
        ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
        ("neck", "pelvis"),
        ("pelvis", "l_knee"), ("l_knee", "l_ankle"),
        ("pelvis", "r_knee"), ("r_knee", "r_ankle"),
    ),
    flip_pairs=(
        ("l_shoulder", "r_shoulder"), ("l_elbow", "r_elbow"), ("l_wrist", "r_wrist"),
        ("l_knee", "r_knee"), ("l_ankle", "r_ankle"),
    ),
    # COCO sigmas where the point exists (shoulders .079, elbows .072, wrists .062, hips .107, knees .087, ankles .089);
    # head ~ between ears/eyes, neck not in COCO.
    sigmas=(0.04, 0.06, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.087, 0.087, 0.089, 0.089),
)


@dataclass(frozen=True)
class BoneMapping:
    """Maps one rig's internal bone names onto the unified schema.

    Each unified keypoint is a weighted combination of internal bones, so
    "pelvis = mean(left_thigh, right_thigh)" or "neck = 0.5*spine3 + 0.5*head_base"
    need no special-casing in engine code. A single bone is just weight 1.0.
    """

    rig_id: str
    weights: Mapping[str, Sequence[tuple[str, float]]]

    def resolve(
        self, schema: SkeletonSchema, bones: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ((K,3) world positions, (K,) valid mask). Missing bones => invalid joint."""
        k = schema.num_keypoints
        out = np.zeros((k, 3), dtype=np.float64)
        valid = np.zeros(k, dtype=bool)
        for i, kp in enumerate(schema.keypoints):
            parts = self.weights.get(kp)
            if not parts or any(name not in bones for name, _ in parts):
                continue
            total = sum(w for _, w in parts)
            out[i] = sum(np.asarray(bones[name], dtype=np.float64) * w for name, w in parts) / total
            valid[i] = True
        return out, valid
