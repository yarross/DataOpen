"""Domain Randomization Controller (engine-agnostic).

Design: the *adapter declares* what it can randomize (a `ParameterSpace`: skins, outfits,
animation names...), the *core decides* the values. This keeps sampling strategy,
reproducibility and coverage tracking in one place for every game.

Every parameter is defined through an inverse-CDF `from_unit(u)`, so one Latin-Hypercube
design stratifies continuous and categorical parameters alike. All draws are pure
functions of (session seed, indices) => any frame is reproducible and resumable.
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .models import CameraSpec, FrameKind, FrameSpec, SceneSpec, SpawnArea


def derive_seed(*parts: Any) -> int:
    """Stable (cross-process, cross-platform) seed derivation; Python's hash() is salted."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"|")
    return int.from_bytes(h.digest(), "little")


# ---- parameter types ---------------------------------------------------------------------

class Param(ABC):
    @abstractmethod
    def from_unit(self, u: float) -> Any: ...

    def sample(self, rng: np.random.Generator) -> Any:
        return self.from_unit(float(rng.random()))


@dataclass(frozen=True)
class Uniform(Param):
    lo: float
    hi: float

    def from_unit(self, u: float) -> float:
        return self.lo + (self.hi - self.lo) * u


@dataclass(frozen=True)
class LogUniform(Param):
    lo: float
    hi: float

    def from_unit(self, u: float) -> float:
        return math.exp(math.log(self.lo) + (math.log(self.hi) - math.log(self.lo)) * u)


@dataclass(frozen=True)
class Categorical(Param):
    choices: tuple[Any, ...]
    weights: Optional[tuple[float, ...]] = None

    def from_unit(self, u: float) -> Any:
        w = np.asarray(self.weights if self.weights else [1.0] * len(self.choices), dtype=float)
        cdf = np.cumsum(w / w.sum())
        return self.choices[min(int(np.searchsorted(cdf, u, side="right")), len(self.choices) - 1)]


@dataclass(frozen=True)
class Constant(Param):
    value: Any

    def from_unit(self, u: float) -> Any:
        return self.value


@dataclass
class ParameterSpace:
    params: dict[str, Param] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.params)

    def sample(self, rng: np.random.Generator, units: Optional[Sequence[float]] = None) -> dict[str, Any]:
        names = self.names()
        if units is None:
            units = rng.random(len(names))
        return {n: self.params[n].from_unit(float(u)) for n, u in zip(names, units)}

    def merged(self, other: "ParameterSpace") -> "ParameterSpace":
        """`other` overrides / extends `self` (adapter refines the core defaults)."""
        return ParameterSpace({**self.params, **other.params})


@dataclass
class AdapterParameterSpace:
    """What a game adapter can randomize, by lifecycle stage."""

    environment: ParameterSpace = field(default_factory=ParameterSpace)  # extends core defaults
    actor: ParameterSpace = field(default_factory=ParameterSpace)        # per actor, per scene
    actor_frame: ParameterSpace = field(default_factory=ParameterSpace)  # per actor, per frame


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """(n, d) stratified unit samples: each dimension hits every 1/n stratum exactly once."""
    out = np.empty((n, d))
    for j in range(d):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


# ---- defaults ----------------------------------------------------------------------------

def default_environment_space() -> ParameterSpace:
    return ParameterSpace({
        "time_of_day": Uniform(0.0, 24.0),
        "weather": Categorical(("clear", "overcast", "rain", "fog", "snow"), (0.4, 0.25, 0.15, 0.1, 0.1)),
        "cloud_cover": Uniform(0.0, 1.0),
        "fog_density": LogUniform(1e-4, 0.05),
        "sun_azimuth_deg": Uniform(0.0, 360.0),
        "exposure_ev": Uniform(-1.0, 1.0),
    })


def default_camera_space() -> ParameterSpace:
    return ParameterSpace({
        "distance": LogUniform(2.0, 40.0),
        "yaw_deg": Uniform(-180.0, 180.0),
        "pitch_deg": Uniform(-10.0, 40.0),
        "roll_deg": Uniform(-5.0, 5.0),
        "fov_deg": Uniform(45.0, 100.0),
        "height_offset": Uniform(-0.5, 1.5),
    })


Rule = Callable[[dict[str, Any], np.random.Generator], dict[str, Any]]


def weather_consistency_rule(env: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Keep correlated variables physically plausible (no 'clear sky + heavy rain')."""
    w = env.get("weather")
    if w == "fog":
        env["fog_density"] = max(env["fog_density"], 0.01)
    elif w in ("rain", "snow", "overcast"):
        env["cloud_cover"] = max(env["cloud_cover"], 0.7)
    elif w == "clear":
        env["cloud_cover"] = min(env["cloud_cover"], 0.4)
        env["fog_density"] = min(env["fog_density"], 0.002)
    return env


# ---- controller --------------------------------------------------------------------------

class IDomainRandomizer(ABC):
    @abstractmethod
    def sample_scene(self, scene_index: int) -> SceneSpec: ...

    @abstractmethod
    def sample_frame(self, scene: SceneSpec, frame_index: int, kind: FrameKind) -> FrameSpec: ...


class DomainRandomizationController(IDomainRandomizer):
    def __init__(
        self,
        seed: int,
        adapter_space: Optional[AdapterParameterSpace] = None,
        camera_space: Optional[ParameterSpace] = None,
        rules: Sequence[Rule] = (weather_consistency_rule,),
        actors_per_scene: tuple[int, int] = (1, 4),
        area: Optional[SpawnArea] = None,
        val_fraction: float = 0.1,
        lhs_block: int = 256,
    ) -> None:
        a = adapter_space or AdapterParameterSpace()
        self.seed = seed
        self.env_space = default_environment_space().merged(a.environment)
        self.actor_space = a.actor
        self.actor_frame_space = a.actor_frame
        self.camera_space = camera_space or default_camera_space()
        self.rules = list(rules)
        self.actors_per_scene = actors_per_scene
        self.area = area or SpawnArea()
        self.val_fraction = val_fraction
        self.lhs_block = lhs_block
        self._block_cache: tuple[int, np.ndarray] | None = None

    def _env_units(self, scene_index: int) -> np.ndarray:
        block, pos = divmod(scene_index, self.lhs_block)
        if self._block_cache is None or self._block_cache[0] != block:
            rng = np.random.default_rng(derive_seed(self.seed, "env-lhs", block))
            self._block_cache = (block, latin_hypercube(self.lhs_block, len(self.env_space.params), rng))
        return self._block_cache[1][pos]

    def split_for_scene(self, scene_index: int) -> str:
        # Split by SCENE, never by frame: frames of one scene are near-duplicates.
        u = (derive_seed(self.seed, "split", scene_index) % 10_000) / 10_000
        return "val" if u < self.val_fraction else "train"

    def sample_scene(self, scene_index: int) -> SceneSpec:
        rng = np.random.default_rng(derive_seed(self.seed, "scene", scene_index))
        env = self.env_space.sample(rng, self._env_units(scene_index))
        for rule in self.rules:
            env = rule(env, rng)
        lo, hi = self.actors_per_scene
        n = int(rng.integers(lo, hi + 1))
        actors = [self.actor_space.sample(rng) for _ in range(n)]
        return SceneSpec(scene_index, derive_seed(self.seed, "scene", scene_index),
                         self.split_for_scene(scene_index), env, actors, self.area)

    def sample_frame(self, scene: SceneSpec, frame_index: int, kind: FrameKind) -> FrameSpec:
        seed = derive_seed(self.seed, "frame", scene.scene_index, frame_index)
        rng = np.random.default_rng(seed)
        cam = CameraSpec(**self.camera_space.sample(rng))
        cam.target_index = int(rng.integers(0, max(1, len(scene.actors))))
        actor_frame = [self.actor_frame_space.sample(rng) for _ in scene.actors]
        return FrameSpec(scene.scene_index, frame_index, seed, kind, cam, actor_frame)
