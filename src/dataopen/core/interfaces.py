"""Contract between the Universal Core and a Game Adapter.

Core never imports anything engine-specific; an adapter never contains sampling,
projection, visibility or serialization logic. Adding a game = implementing this file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from .models import (CaptureRequest, EntityHandle, EntityState, FrameSnapshot, SceneSpec, FrameSpec)
from .randomization import AdapterParameterSpace
from .schema import BoneMapping, SkeletonSchema


class AdapterError(RuntimeError):
    """Recoverable adapter failure (spawn failed, engine hiccup). Core retries / restarts."""


class Capability(str, Enum):
    DEPTH = "depth"                        # returns occluder depth map
    ENGINE_VISIBILITY = "engine_visibility"  # returns per-joint raycast visibility
    HULL_POINTS = "hull_points"            # returns mesh bounds for tight bboxes
    DETERMINISTIC_STEP = "deterministic_step"  # fixed-timestep, stepped simulation


@dataclass
class AdapterInfo:
    name: str
    engine: str                     # "unity" | "unreal" | "mock" | ...
    schema: SkeletonSchema
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    image_size: tuple[int, int] = (1280, 720)


class IEntitySpawner(ABC):
    """Entity Spawner & Randomizer: applies core-sampled values to concrete engine objects."""

    @abstractmethod
    def spawn(self, scene: SceneSpec) -> list[EntityHandle]:
        """Spawn `scene.actors` inside `scene.area`; apply each actor's sampled appearance."""

    @abstractmethod
    def update_actors(self, handles: Sequence[EntityHandle], frame: FrameSpec) -> None:
        """Apply per-frame values (animation clip, phase, stance, FOV-relevant pose)."""

    @abstractmethod
    def set_active(self, handles: Sequence[EntityHandle], active: bool) -> None:
        """Hide/show all actors (negative frames)."""

    @abstractmethod
    def despawn_all(self) -> None: ...


class IEnvironmentController(ABC):
    """Applies environment values: time of day, weather, fog, exposure, level."""

    @abstractmethod
    def apply(self, scene: SceneSpec) -> None: ...


class ISkeletonExtractor(ABC):
    """Skeleton Mapping Engine: engine bones -> unified schema, world space, meters."""

    @property
    @abstractmethod
    def schema(self) -> SkeletonSchema: ...

    @abstractmethod
    def mapping_for(self, rig_id: str) -> BoneMapping: ...

    @abstractmethod
    def extract(self, handles: Sequence[EntityHandle]) -> list[EntityState]:
        """MUST be called from the same engine instant as the pixel capture."""


class ICaptureBridge(ABC):
    """Synchronization Bridge. Guarantees pixels, depth, camera and bone transforms
    in a `FrameSnapshot` all belong to the same engine tick."""

    @abstractmethod
    def capture(self, request: CaptureRequest) -> FrameSnapshot:
        """Place camera from `request.frame_spec.camera`, settle/step the simulation,
        render, read bones. Pixels stay engine-side, keyed by `snapshot.frame_token`."""

    @abstractmethod
    def commit(self, snapshot: FrameSnapshot, dest: Path) -> None:
        """Persist the pixels of an ACCEPTED frame (engine-side encode, no core round-trip)."""

    @abstractmethod
    def discard(self, snapshot: FrameSnapshot) -> None:
        """Release the pixels of a REJECTED frame (no encode, no disk I/O)."""

    def peek_pixels(self, snapshot: FrameSnapshot, max_side: Optional[int] = None):
        """Optional: the frame's pixels in core memory WITHOUT committing it (for in-the-loop validation).
        Returns a `quality.interfaces.PixelHandle` (call .release() when done) or None if unsupported."""
        return None


class IGameAdapter(ABC):
    @property
    @abstractmethod
    def info(self) -> AdapterInfo: ...

    @abstractmethod
    def parameter_space(self) -> AdapterParameterSpace:
        """Declare what this game can randomize; the core samples it."""

    @property
    @abstractmethod
    def spawner(self) -> IEntitySpawner: ...

    @property
    @abstractmethod
    def environment(self) -> IEnvironmentController: ...

    @property
    @abstractmethod
    def extractor(self) -> ISkeletonExtractor: ...

    @property
    @abstractmethod
    def capture(self) -> ICaptureBridge: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def restart(self) -> None:
        """Hard-recover from a crash / leak. Default: close + connect."""
        self.close()
        self.connect()
