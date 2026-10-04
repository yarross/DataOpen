"""Wire/data model shared by Universal Core and every Game Adapter.

Conventions (the adapter is responsible for converting engine-native values):
  * units: meters; world frame is arbitrary but must be consistent with `world_to_camera`
  * camera: OpenCV convention (x right, y down, z forward), extrinsics world -> camera
  * pixels: continuous coords, origin at the top-left corner of the top-left pixel
    (a pixel centre is at +0.5), so a point is inside the frame iff 0 <= u < W, 0 <= v < H
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional

import numpy as np


class Visibility(IntEnum):
    """Same numeric values as COCO / YOLO-Pose keypoint flags."""

    OUT_OF_FRAME = 0  # outside the frame, behind the camera, or bone missing
    OCCLUDED = 1      # inside the frame but hidden
    VISIBLE = 2


class FrameKind(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"  # no persons; background / hard-negative training sample


@dataclass(eq=False)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray  # (4, 4)
    near: float = 0.1

    @staticmethod
    def from_vertical_fov(width: int, height: int, fov_deg: float, world_to_camera: np.ndarray,
                          near: float = 0.1) -> "CameraModel":
        f = (height / 2.0) / np.tan(np.radians(fov_deg) / 2.0)  # square pixels
        return CameraModel(width, height, f, f, width / 2.0, height / 2.0,
                           np.asarray(world_to_camera, dtype=np.float64), near)

    @staticmethod
    def from_pose(width: int, height: int, pos, forward, right, up,
                  fov_v_deg: Optional[float] = None, fov_h_deg: Optional[float] = None,
                  near: float = 0.1) -> "CameraModel":
        """Build a camera from an engine pose: position + the world-space directions of screen
        forward / right / up. Handedness-proof: x = (p-pos).right, y = -(p-pos).up, z = (p-pos).forward,
        so every engine (Unity LH Y-up, UE LH Z-up, Source, RED4 RH Z-up) only reports its own basis."""
        pos = np.asarray(pos, dtype=np.float64)
        rows = []
        for v in (right, [-c for c in up], forward):
            v = np.asarray(v, dtype=np.float64)
            n = np.linalg.norm(v)
            if not np.isfinite(n) or n < 1e-9:
                raise ValueError("camera basis vector is zero or not finite")
            rows.append(v / n)
        R = np.stack(rows)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = -R @ pos
        if fov_v_deg is not None:
            f = (height / 2.0) / np.tan(np.radians(fov_v_deg) / 2.0)
        elif fov_h_deg is not None:
            f = (width / 2.0) / np.tan(np.radians(fov_h_deg) / 2.0)
        else:
            raise ValueError("camera needs fov_v_deg or fov_h_deg")
        return CameraModel(width, height, f, f, width / 2.0, height / 2.0, M, near)

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.world_to_camera[:3, :3].T @ self.world_to_camera[:3, 3]

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


# ---- what the core asks the adapter to produce -------------------------------------------

@dataclass
class SpawnArea:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 8.0
    min_separation: float = 1.0


@dataclass
class CameraSpec:
    """Camera relative to the target actor; the adapter resolves it to a world pose
    (it knows the terrain/colliders) and reports the final `CameraModel` back."""

    distance: float = 8.0
    yaw_deg: float = 0.0
    pitch_deg: float = 10.0
    roll_deg: float = 0.0
    fov_deg: float = 60.0
    height_offset: float = 0.0
    target_index: int = 0


@dataclass
class SceneSpec:
    """Expensive-to-change state: environment, level, actor population."""

    scene_index: int
    seed: int
    split: str
    environment: dict[str, Any]
    actors: list[dict[str, Any]]
    area: SpawnArea = field(default_factory=SpawnArea)


@dataclass
class FrameSpec:
    """Cheap-to-change state: camera, per-actor animation/pose, positive vs negative."""

    scene_index: int
    frame_index: int
    seed: int
    kind: FrameKind
    camera: CameraSpec
    actor_frame: list[dict[str, Any]]

    @property
    def frame_id(self) -> str:
        return f"s{self.scene_index:06d}_f{self.frame_index:04d}"


@dataclass
class CaptureRequest:
    frame_id: str
    frame_spec: FrameSpec
    width: int
    height: int
    want_depth: bool = True


# ---- what the adapter returns ------------------------------------------------------------

@dataclass(frozen=True)
class EntityHandle:
    entity_id: int
    rig_id: str
    meta: dict[str, Any] = field(default_factory=dict)  # appearance: skin, outfit, weapon...


@dataclass(eq=False)
class EntityState:
    entity_id: int
    rig_id: str
    skeleton_world: np.ndarray                         # (K, 3), schema order
    joint_valid: np.ndarray                            # (K,) bool
    hull_points_world: Optional[np.ndarray] = None     # (M, 3) mesh bounds -> tight bbox
    engine_visibility: Optional[np.ndarray] = None     # (K,) int, 1/2 from engine raycasts
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Probe:
    """A world point plus where the *engine itself* says it lands on screen (None = off-screen).
    The core re-projects `world` with its own math and compares: this catches wrong FOV, y-flip,
    handedness and unit-scale bugs in an adapter without any ground-truth images."""

    world: tuple[float, float, float]
    screen: Optional[tuple[float, float]]


@dataclass(eq=False)
class FrameSnapshot:
    """Everything captured at ONE engine instant. Pixels stay engine-side behind
    `frame_token` until the core calls `commit` (accepted) or `discard` (rejected)."""

    frame_token: str
    tick: int
    camera: CameraModel
    entities: list[EntityState]
    depth: Optional[np.ndarray] = None       # (h, w) z-depth in meters of occluders, inf = none
    thumbnail: Optional[np.ndarray] = None   # tiny grayscale for near-duplicate hashing
    probes: Optional[list[Probe]] = None
    meta: dict[str, Any] = field(default_factory=dict)


# ---- annotations -------------------------------------------------------------------------

@dataclass(eq=False)
class Annotation:
    entity_id: int
    keypoints: np.ndarray  # (K, 3) float: x, y, v ; x=y=0 when v == 0
    bbox: tuple[float, float, float, float]  # x, y, w, h in pixels (clipped to frame)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_keypoints(self) -> int:
        return int((self.keypoints[:, 2] > 0).sum())

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "keypoints": [round(float(x), 2) if i % 3 != 2 else int(x)
                          for i, x in enumerate(self.keypoints.reshape(-1))],
            "bbox": [round(float(x), 2) for x in self.bbox],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Annotation":
        return Annotation(d["entity_id"], np.asarray(d["keypoints"], dtype=np.float64).reshape(-1, 3),
                          tuple(d["bbox"]), d.get("meta", {}))


@dataclass
class FrameRecord:
    """Canonical, format-agnostic annotation of one image. COCO/YOLO are derived from it."""

    frame_id: str
    scene_index: int
    frame_index: int
    split: str
    file_name: str  # relative to dataset root
    width: int
    height: int
    kind: FrameKind
    annotations: list[Annotation]
    meta: dict[str, Any] = field(default_factory=dict)  # seed, environment, camera, tick

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id, "scene_index": self.scene_index,
            "frame_index": self.frame_index, "split": self.split, "file_name": self.file_name,
            "width": self.width, "height": self.height, "kind": self.kind.value,
            "annotations": [a.to_dict() for a in self.annotations], "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FrameRecord":
        return FrameRecord(
            d["frame_id"], d["scene_index"], d["frame_index"], d["split"], d["file_name"],
            d["width"], d["height"], FrameKind(d["kind"]),
            [Annotation.from_dict(a) for a in d["annotations"]], d.get("meta", {}),
        )
