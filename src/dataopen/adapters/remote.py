"""RemoteGameAdapter: the core-side half of every real game integration.

It implements the same IGameAdapter contract as the mock, but each call is a request to a mod running
inside the game (C#, Lua, ...) over the file-mailbox protocol (docs/PROTOCOL.md). Pixels come from one
of two places:

  * ``engine`` mode: the mod renders/encodes the frame itself on ``commit`` (best: no round trip).
  * ``host`` mode:   the mod freezes the game and this process grabs the screen (works with any game
                     whose mod can only read state, e.g. UE4SS/CET Lua mods). Needs ``mss``.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..core.imageio import write_image
from ..core.interfaces import (AdapterError, AdapterInfo, Capability, ICaptureBridge, IEntitySpawner,
                               IEnvironmentController, IGameAdapter, ISkeletonExtractor)
from ..core.models import CaptureRequest, EntityHandle, EntityState, FrameSnapshot, FrameSpec, SceneSpec
from ..core.protocol import (PROTOCOL_VERSION, adapter_space_from_dict, frame_to_dict, scene_to_dict,
                             snapshot_from_wire)
from ..core.randomization import AdapterParameterSpace
from ..core.schema import HUMAN_13, BoneMapping, SkeletonSchema
from ..core.transport import FileMailboxTransport

_CAPS = {"engine_visibility": Capability.ENGINE_VISIBILITY, "hull_points": Capability.HULL_POINTS,
         "deterministic_step": Capability.DETERMINISTIC_STEP, "depth": Capability.DEPTH}


class ScreenGrabber(ABC):
    @abstractmethod
    def grab(self) -> np.ndarray:
        """Return the current game frame as RGB uint8 (H, W, 3)."""


class StaticGrabber(ScreenGrabber):
    """Test double: returns a fixed array, or whatever the callable returns."""

    def __init__(self, source: np.ndarray | Callable[[], np.ndarray]) -> None:
        self._src = source

    def grab(self) -> np.ndarray:
        return self._src() if callable(self._src) else self._src


class MssGrabber(ScreenGrabber):
    """Screen-region grab with `mss` (pip install 'dataopen[capture]'). Run the game borderless
    windowed at exactly the render size with the HUD off; `region` = {left, top, width, height}."""

    def __init__(self, region: Optional[dict[str, int]] = None, monitor: int = 1) -> None:
        self.region, self.monitor = region, monitor

    def grab(self) -> np.ndarray:
        try:
            import mss
        except ImportError as e:
            raise AdapterError("host capture needs mss: pip install 'dataopen[capture]'") from e
        with mss.mss() as sct:
            shot = sct.grab(self.region or sct.monitors[self.monitor])
            return np.asarray(shot)[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB


@dataclass
class RemoteOptions:
    capture_mode: str = "auto"                 # auto | engine | host
    image_size: Optional[tuple[int, int]] = None
    bone_map: dict[str, list[list[Any]]] = field(default_factory=dict)  # keypoint -> [[bone, weight], ...]
    mod_options: dict[str, Any] = field(default_factory=dict)  # population_mode, camera_mode, settle_ticks...
    call_timeout_s: float = 60.0
    capture_timeout_s: float = 120.0
    grab_delay_s: float = 0.05                 # let a frozen frame reach the screen before grabbing
    commit_wait_s: float = 10.0


class _Spawner(IEntitySpawner):
    def __init__(self, a: "RemoteGameAdapter") -> None:
        self.a = a

    def spawn(self, scene: SceneSpec) -> list[EntityHandle]:
        res = self.a._call("begin_scene", {"scene": scene_to_dict(scene)})
        self.a._active = True
        return [EntityHandle(int(h["entity_id"]), str(h.get("rig_id", "")), dict(h.get("meta", {})))
                for h in res.get("handles", [])]

    def update_actors(self, handles: Sequence[EntityHandle], frame: FrameSpec) -> None:
        pass  # per-frame values travel inside capture_frame (one round trip per frame)

    def set_active(self, handles: Sequence[EntityHandle], active: bool) -> None:
        self.a._active = active

    def despawn_all(self) -> None:
        self.a._call("end_scene")


class _Environment(IEnvironmentController):
    def apply(self, scene: SceneSpec) -> None:
        pass  # the environment is part of the scene sent by begin_scene


class _Extractor(ISkeletonExtractor):
    def __init__(self, a: "RemoteGameAdapter") -> None:
        self.a = a

    @property
    def schema(self) -> SkeletonSchema:
        return self.a.schema

    def mapping_for(self, rig_id: str) -> BoneMapping:
        if not self.a.options.bone_map:
            raise AdapterError("bone mapping lives in the game mod; override it via the profile's [bones]")
        return BoneMapping(rig_id, {k: [(b, float(w)) for b, w in v] for k, v in self.a.options.bone_map.items()})

    def extract(self, handles: Sequence[EntityHandle]) -> list[EntityState]:
        raise AdapterError("entities arrive inside capture_frame; there is no separate extract call")


class _Bridge(ICaptureBridge):
    def __init__(self, a: "RemoteGameAdapter") -> None:
        self.a = a
        self._pending: dict[str, Optional[np.ndarray]] = {}

    def capture(self, request: CaptureRequest) -> FrameSnapshot:
        a = self.a
        mode = a.capture_mode
        res = a._call("capture_frame", {
            "frame_id": request.frame_id, "frame": frame_to_dict(request.frame_spec),
            "width": request.width, "height": request.height, "active": a._active, "image_mode": mode,
        }, timeout=a.options.capture_timeout_s)
        try:
            snap = snapshot_from_wire(res, a.schema)
        except (KeyError, ValueError, TypeError) as e:
            a._call("release")
            raise AdapterError(f"malformed capture_frame response: {e}") from e
        snap.meta["image_mode"] = mode
        if mode == "host":
            try:
                if a.options.grab_delay_s:
                    time.sleep(a.options.grab_delay_s)
                img = a.grabber.grab()
            finally:
                a._call("release")  # unfreeze the game even if the grab failed
            cam = snap.camera
            if img.shape != (cam.height, cam.width, 3):
                raise AdapterError(
                    f"screen grab is {img.shape[1]}x{img.shape[0]} but the game camera is "
                    f"{cam.width}x{cam.height}: set the game to borderless windowed at that size "
                    f"and fix the grabber region")
            self._pending[snap.frame_token] = img
        else:
            self._pending[snap.frame_token] = None
        return snap

    def commit(self, snapshot: FrameSnapshot, dest: Path) -> None:
        a = self.a
        if snapshot.meta.get("image_mode") == "host":
            img = self._pending.pop(snapshot.frame_token)
            write_image(dest, img)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        a._call("commit", {"frame_token": snapshot.frame_token, "dest": str(dest)})
        self._pending.pop(snapshot.frame_token, None)
        deadline = time.monotonic() + a.options.commit_wait_s
        while not (dest.exists() and dest.stat().st_size > 0):  # the mod may write asynchronously
            if time.monotonic() > deadline:
                raise AdapterError(f"mod acknowledged commit but {dest} was not written")
            time.sleep(0.01)

    def discard(self, snapshot: FrameSnapshot) -> None:
        self._pending.pop(snapshot.frame_token, None)
        if snapshot.meta.get("image_mode") != "host":
            self.a._call("discard", {"frame_token": snapshot.frame_token})


class RemoteGameAdapter(IGameAdapter):
    def __init__(self, transport: FileMailboxTransport, options: Optional[RemoteOptions] = None,
                 schema: SkeletonSchema = HUMAN_13, grabber: Optional[ScreenGrabber] = None) -> None:
        self.transport, self.options, self.schema, self.grabber = transport, options or RemoteOptions(), schema, grabber
        self.hello: dict[str, Any] = {}
        self.capture_mode = "engine"
        self._info: Optional[AdapterInfo] = None
        self._space: Optional[AdapterParameterSpace] = None
        self._active = True
        self._spawner, self._env = _Spawner(self), _Environment()
        self._extractor, self._bridge = _Extractor(self), _Bridge(self)

    # ---- plumbing ----
    def _call(self, method: str, params: Optional[dict[str, Any]] = None, timeout: Optional[float] = None):
        return self.transport.call(method, params, timeout if timeout is not None else self.options.call_timeout_s)

    @property
    def info(self) -> AdapterInfo:
        if self._info is None:
            raise AdapterError("RemoteGameAdapter is not connected: call connect() first")
        return self._info

    spawner = property(lambda self: self._spawner)
    environment = property(lambda self: self._env)
    extractor = property(lambda self: self._extractor)
    capture = property(lambda self: self._bridge)

    def parameter_space(self) -> AdapterParameterSpace:
        if self._space is None:
            raise AdapterError("RemoteGameAdapter is not connected: call connect() first")
        return self._space

    # ---- lifecycle ----
    def connect(self) -> None:
        o = self.options
        res = self._call("hello", {
            "protocol": PROTOCOL_VERSION,
            "schema": {"name": self.schema.name, "keypoints": list(self.schema.keypoints)},
            "bone_map": o.bone_map, "options": o.mod_options,
            "image": {"width": (o.image_size or (0, 0))[0], "height": (o.image_size or (0, 0))[1]},
        })
        if int(res.get("protocol", -1)) != PROTOCOL_VERSION:
            raise AdapterError(f"mod speaks protocol {res.get('protocol')}, core speaks {PROTOCOL_VERSION}")
        if res.get("schema_errors"):
            raise AdapterError("mod cannot provide the requested skeleton schema: " + "; ".join(res["schema_errors"]))
        caps = set(res.get("capabilities", []))
        self.capture_mode = self._resolve_mode(caps)
        img = res.get("image") or {}
        size = o.image_size or (int(img.get("width", 1280)), int(img.get("height", 720)))
        self.hello = res
        self._info = AdapterInfo(str(res.get("game", "unknown")), str(res.get("engine", "unknown")), self.schema,
                                 frozenset(c for n, c in _CAPS.items() if n in caps), size)
        self._space = adapter_space_from_dict(res.get("parameter_space") or {})

    def _resolve_mode(self, caps: set[str]) -> str:
        want = self.options.capture_mode
        if want == "auto":
            want = "engine" if "image_engine" in caps else "host"
        if want == "engine" and "image_engine" not in caps:
            raise AdapterError("capture_mode=engine but the mod does not offer 'image_engine'")
        if want == "host" and self.grabber is None:
            raise AdapterError("capture_mode=host needs a ScreenGrabber (pip install 'dataopen[capture]')")
        return want

    def close(self) -> None:
        pass  # never stop the user's game; use shutdown() explicitly

    def restart(self) -> None:
        self.connect()  # the mod re-initialises its state on hello

    def health(self) -> dict[str, Any]:
        return self._call("health", timeout=10.0)

    def selftest(self) -> list[dict[str, Any]]:
        return list(self._call("selftest", timeout=self.options.capture_timeout_s).get("checks", []))

    def shutdown(self) -> None:
        self._call("shutdown", timeout=10.0)
