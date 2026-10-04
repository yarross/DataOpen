"""The mock game exposed through the real wire protocol.

This is the "game side" in Python: it answers the same JSON requests a Lua/C# mod would. It lets the
whole stack (RemoteGameAdapter -> mailbox -> orchestrator -> doctor -> verify) run end to end with no
game installed, and it can *simulate adapter bugs* (flipped Y, centimetre units, wrong FOV) so the
calibration checks can be tested against known-bad input.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...core.interfaces import AdapterError
from ...core.models import CaptureRequest, FrameSnapshot
from ...core.projection import project, visibility_flags
from ...core.imageio import write_image
from ...quality.shm import attach
from ...core.protocol import (PROTOCOL_VERSION, adapter_space_to_dict, frame_from_dict, scene_from_dict,
                              snapshot_to_wire, space_to_dict)
from ...core.transport import MailboxServer
from .adapter import MockGameAdapter


@dataclass
class MockServerOptions:
    engine_images: bool = True      # offer 'image_engine'; False forces the host-grab path
    flip_probe_y: bool = False      # engine-native projection disagrees (y flipped)
    wrong_probe_fov: float = 1.0    # multiply the focal length used for probes (FOV bug)
    unit_scale: float = 1.0         # 100 simulates a mod that forgot cm -> m
    swap_lr: bool = False           # left/right bones mapped to each other
    image_peek: bool = True         # offer `peek` (pixels to the core before commit)
    image_shm: bool = True          # ... through shared memory (False = staged file)
    mod_version: str = "mock-1"
    variant: str = ""               # "shooter": the mock world with teams, headgear, camouflage, smoke, flashes, head cover


def native_project(world, cam: dict[str, Any], fov_scale: float = 1.0):
    """Engine-style projection from the pose basis: deliberately a different code path than core."""
    pos, fwd, right, up = (np.asarray(cam[k], dtype=float) for k in ("pos", "forward", "right", "up"))
    d = np.asarray(world, dtype=float) - pos
    z = d @ fwd
    if z <= 0.05:
        return None
    f = (cam["height"] / 2.0) / np.tan(np.radians(cam["fov_v_deg"]) / 2.0) * fov_scale
    u = cam["width"] / 2.0 + f * (d @ right) / z
    v = cam["height"] / 2.0 - f * (d @ up) / z
    return (u, v) if 0 <= u < cam["width"] and 0 <= v < cam["height"] else None


class MockGameServer:
    def __init__(self, options: Optional[MockServerOptions] = None, width: int = 320, height: int = 240,
                 directory: Path | str = ".") -> None:
        self.dir = Path(directory)
        self.opt = options or MockServerOptions()
        self.adapter = MockGameAdapter(width, height, variant=self.opt.variant)
        self.handles: list = []
        self.snaps: dict[str, FrameSnapshot] = {}
        self.frames = 0
        self.stopped = threading.Event()

    def handle(self, method: str, p: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, "m_" + method, None)
        if fn is None:
            raise AdapterError(f"unknown method {method!r}")
        return fn(p)

    # ---- methods ----
    def m_hello(self, p):
        want = p.get("schema", {}).get("keypoints", [])
        errs = [] if list(want) == list(self.adapter.info.schema.keypoints) else \
            [f"schema mismatch: mod has {self.adapter.info.schema.keypoints}, core asked for {want}"]
        caps = ["engine_visibility", "hull_points"] + (["image_engine"] if self.opt.engine_images else [])
        if self.opt.engine_images and self.opt.image_peek:
            caps += ["image_peek"] + (["image_shm"] if self.opt.image_shm else [])
        w, h = self.adapter.info.image_size
        self.handles, self.snaps = [], {}
        return {"protocol": PROTOCOL_VERSION, "game": "mock", "engine": "mock", "game_version": "0",
                "mod_version": self.opt.mod_version, "capabilities": caps, "image": {"width": w, "height": h},
                "schema_errors": errs, "parameter_space": adapter_space_to_dict(self.adapter.parameter_space())}

    def m_begin_scene(self, p):
        scene = scene_from_dict(p["scene"])
        self.adapter.environment.apply(scene)
        self.handles = self.adapter.spawner.spawn(scene)
        return {"handles": [{"entity_id": h.entity_id, "rig_id": h.rig_id, "meta": h.meta} for h in self.handles]}

    def m_capture_frame(self, p):
        spec = frame_from_dict(p["frame"])
        sp, br = self.adapter.spawner, self.adapter.capture
        sp.set_active(self.handles, bool(p.get("active", True)))
        sp.update_actors(self.handles, spec)
        snap = br.capture(CaptureRequest(p["frame_id"], spec, int(p["width"]), int(p["height"])))
        self.frames += 1
        actors = self.adapter.spawner.w.actors
        for e in snap.entities:  # engine-style raycast visibility from the occluder depth map
            uv, z = project(e.skeleton_world, snap.camera)
            f = visibility_flags(uv, z, snap.camera, e.joint_valid, depth=snap.depth)
            e.engine_visibility = np.where(f == 2, 2, 1).astype(np.int8)
            e.meta["forward"] = [float(np.cos(actors[e.entity_id]["yaw"])), float(np.sin(actors[e.entity_id]["yaw"])), 0.0]
        if self.opt.swap_lr:
            flip = self.adapter.info.schema.flip_idx()
            for e in snap.entities:
                e.skeleton_world = e.skeleton_world[flip]
                e.joint_valid = e.joint_valid[flip]
                e.engine_visibility = e.engine_visibility[flip]
        self.snaps[snap.frame_token] = snap
        wire = snapshot_to_wire(snap)
        wire["probes"] = self._probes(wire)
        return self._scaled(wire)

    def _probes(self, wire) -> list[dict[str, Any]]:
        cam = wire["camera"]
        pos, fwd, right, up = (np.asarray(cam[k]) for k in ("pos", "forward", "right", "up"))
        pts = [pos + fwd * 6 + right * 1.5, pos + fwd * 12 - right * 3 + up * 2, pos + fwd * 4 + up * 1.2]
        for e in wire["entities"][:3]:
            pts.append(np.asarray(e["skeleton_world"][:3]))   # head joint
        out = []
        for pt in pts:
            s = native_project(pt, cam, self.opt.wrong_probe_fov)
            if s is not None and self.opt.flip_probe_y:
                s = (s[0], cam["height"] - s[1])
            out.append({"world": [float(c) for c in pt], "screen": list(s) if s else None})
        return out

    def _scaled(self, wire):
        k = self.opt.unit_scale
        if k == 1.0:
            return wire
        wire["camera"]["pos"] = [c * k for c in wire["camera"]["pos"]]
        for e in wire["entities"]:
            e["skeleton_world"] = [c * k for c in e["skeleton_world"]]
            if "hull_points" in e:
                e["hull_points"] = [c * k for c in e["hull_points"]]
        for pr in wire["probes"]:
            pr["world"] = [c * k for c in pr["world"]]
        return wire

    def m_peek(self, p):
        img = self.adapter.capture._pending.get(p["frame_token"])
        if img is None:
            raise AdapterError(f"no pending image for {p['frame_token']}")
        h, w, _ = img.shape
        shm = p.get("shm")
        if shm and self.opt.image_shm:
            if h * w * 3 > int(shm["capacity"]):
                raise AdapterError("frame does not fit the shared-memory slot")
            seg = attach(shm["name"])
            try:
                np.ndarray((h, w, 3), dtype=np.uint8, buffer=seg.buf)[:] = img      # the "game" writes the pixels
            finally:
                seg.close()
            return {"transport": "shm", "width": w, "height": h, "format": "rgb24"}
        rel = f"staging/peek_{p['frame_token']}.png"
        write_image(Path(self.dir) / rel, img)
        return {"transport": "file", "width": w, "height": h, "staged": rel}

    def m_release(self, p):
        return {}

    def m_commit(self, p):
        snap = self.snaps.pop(p["frame_token"])
        self.adapter.capture.commit(snap, Path(p["dest"]))
        return {}

    def m_discard(self, p):
        snap = self.snaps.pop(p["frame_token"], None)
        if snap is not None:
            self.adapter.capture.discard(snap)
        return {}

    def m_end_scene(self, p):
        self.adapter.spawner.despawn_all()
        self.handles = []
        return {}

    def m_selftest(self, p):
        return {"checks": [
            {"name": "bone_mapping", "ok": True, "detail": "13/13 keypoints resolved", "data": {"unmapped": []}},
            {"name": "spawn", "ok": True, "detail": "mock spawner"},
            {"name": "freeze", "ok": True, "detail": "mock engine is stepped, always frozen"},
        ]}

    def m_health(self, p):
        return {"ok": True, "frames": self.frames}

    def m_shutdown(self, p):
        self.stopped.set()
        return {}


def serve_mock(directory: Path | str, options: Optional[MockServerOptions] = None,
               stop: Optional[threading.Event] = None, width: int = 320, height: int = 240) -> MockGameServer:
    """Blocking server loop; returns after `stop` is set or the core sends `shutdown`."""
    game = MockGameServer(options, width, height, directory)
    stop = stop or threading.Event()
    srv = MailboxServer(directory, game.handle)
    while not stop.is_set() and not game.stopped.is_set():
        if not srv.poll_once():
            stop.wait(srv.poll_s)
    return game
