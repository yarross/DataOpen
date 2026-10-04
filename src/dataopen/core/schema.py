"""Unified skeleton schema and per-rig bone mapping.

The schema is data, not code: everything downstream (projection, COCO/YOLO export,
flip indices) is driven by it, so changing the keypoint set is a one-line change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

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
    # ---- task-specific metadata (all optional; the closed loop reads ONLY these, never keypoint names) ----
    weights: tuple[float, ...] = ()                  # OKS importance per keypoint (empty = all 1.0)
    derived: tuple[str, ...] = ()                    # keypoints that are computed approximations, not anatomical landmarks
    primary: tuple[str, ...] = ()                    # keypoints the loop optimizes for (e.g. the aim point)
    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()   # named keypoint groups ("head", "torso", ...)
    roles: tuple[tuple[str, str], ...] = ()          # geometry roles -> keypoint ("head", "neck", "pelvis", "l_shoulder", ...)
    classes: tuple[str, ...] = ("person",)           # object classes, e.g. ("player_ct", "player_t")
    class_key: str = ""                              # entity meta key that decides the class (e.g. "team"); "" = class 0

    def __post_init__(self) -> None:
        if self.sigmas and len(self.sigmas) != len(self.keypoints):
            raise ValueError("sigmas must have one value per keypoint")
        if self.weights and len(self.weights) != len(self.keypoints):
            raise ValueError("weights must have one value per keypoint")
        names = set(self.keypoints)
        if len(names) != len(self.keypoints):
            raise ValueError("duplicate keypoint names")
        for a, b in (*self.edges, *self.flip_pairs):
            if a not in names or b not in names:
                raise ValueError(f"unknown keypoint in edge/flip pair: {a}, {b}")
        for n in (*self.derived, *self.primary, *(m for _, g in self.groups for m in g), *(n for _, n in self.roles)):
            if n not in names:
                raise ValueError(f"unknown keypoint {n!r} in derived/primary/groups/roles")
        if not self.classes:
            raise ValueError("a schema needs at least one class")

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

    def oks_weights(self) -> list[float]:
        return list(self.weights) if self.weights else [1.0] * self.num_keypoints

    def primary_idx(self) -> list[int]:
        return [self.index(n) for n in self.primary]

    def derived_idx(self) -> list[int]:
        return [self.index(n) for n in self.derived]

    def group_idx(self, group: str) -> list[int]:
        return next(([self.index(n) for n in g] for name, g in self.groups if name == group), [])

    def role(self, role: str) -> Optional[str]:
        """Keypoint playing a geometric role. Falls back to a keypoint literally named like the role."""
        found = next((n for r, n in self.roles if r == role), None)
        return found if found is not None else (role if role in self.keypoints else None)

    def class_of(self, meta: Mapping[str, object]) -> tuple[int, Optional[str]]:
        """(class index, warning). The meta value may be the class name or its suffix ("ct" for "player_ct")."""
        if len(self.classes) == 1 or not self.class_key:
            return 0, None
        raw = meta.get(self.class_key)
        if raw is None:
            return 0, f"entity has no '{self.class_key}': labeled as class {self.classes[0]!r}"
        v = str(raw).lower()
        for i, c in enumerate(self.classes):
            if v == c.lower() or c.lower().endswith("_" + v):
                return i, None
        return 0, f"unknown {self.class_key}={raw!r}: labeled as class {self.classes[0]!r}"

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


COCO_17 = SkeletonSchema(
    name="coco17",
    keypoints=("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
               "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
               "left_ankle", "right_ankle"),
    edges=(("left_ankle", "left_knee"), ("left_knee", "left_hip"), ("right_ankle", "right_knee"),
           ("right_knee", "right_hip"), ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
           ("right_shoulder", "right_hip"), ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
           ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
           ("left_eye", "right_eye"), ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"),
           ("right_eye", "right_ear")),
    flip_pairs=(("left_eye", "right_eye"), ("left_ear", "right_ear"), ("left_shoulder", "right_shoulder"),
                ("left_elbow", "right_elbow"), ("left_wrist", "right_wrist"), ("left_hip", "right_hip"),
                ("left_knee", "right_knee"), ("left_ankle", "right_ankle")),
    sigmas=(0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087,
            0.089, 0.089),
)
"""The COCO person keypoints: what most off-the-shelf baseline models predict. Used as the *model* side of a mapping."""


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
