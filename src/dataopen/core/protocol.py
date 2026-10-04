"""Wire protocol v1 between the core and an in-game mod (any language).

Everything is plain JSON so a mod written in C#, Lua or C++ can speak it. Units: meters.
Camera is sent as a *pose* (position + world-space screen forward/right/up + FOV) because that is
the one thing every engine can report correctly regardless of handedness or axis conventions.
See docs/PROTOCOL.md for the human-readable spec.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .models import (CameraModel, CameraSpec, EntityState, FrameKind, FrameSnapshot, FrameSpec, Probe,
                     SceneSpec, SpawnArea)
from .randomization import (AdapterParameterSpace, Categorical, Constant, LogUniform, Param, ParameterSpace,
                            Uniform)
from .schema import SkeletonSchema

PROTOCOL_VERSION = 1

METHODS = ("hello", "begin_scene", "capture_frame", "release", "commit", "discard", "end_scene",
           "selftest", "health", "shutdown", "peek")


# ---- parameter spaces --------------------------------------------------------------------

def param_to_dict(p: Param) -> dict[str, Any]:
    if isinstance(p, Uniform):
        return {"type": "uniform", "lo": p.lo, "hi": p.hi}
    if isinstance(p, LogUniform):
        return {"type": "loguniform", "lo": p.lo, "hi": p.hi}
    if isinstance(p, Categorical):
        return {"type": "categorical", "choices": list(p.choices),
                **({"weights": list(p.weights)} if p.weights else {})}
    if isinstance(p, Constant):
        return {"type": "constant", "value": p.value}
    raise TypeError(f"cannot serialize {type(p).__name__}")


def param_from_dict(d: dict[str, Any]) -> Param:
    t = d["type"]
    if t == "uniform":
        return Uniform(float(d["lo"]), float(d["hi"]))
    if t == "loguniform":
        return LogUniform(float(d["lo"]), float(d["hi"]))
    if t == "categorical":
        w = d.get("weights")
        return Categorical(tuple(d["choices"]), tuple(float(x) for x in w) if w else None)
    if t == "constant":
        return Constant(d["value"])
    raise ValueError(f"unknown parameter type {t!r}")


def space_to_dict(s: ParameterSpace) -> dict[str, Any]:
    return {k: param_to_dict(v) for k, v in s.params.items()}


def space_from_dict(d: Optional[dict[str, Any]]) -> ParameterSpace:
    return ParameterSpace({k: param_from_dict(v) for k, v in (d or {}).items()})


def adapter_space_to_dict(a: AdapterParameterSpace) -> dict[str, Any]:
    return {"environment": space_to_dict(a.environment), "actor": space_to_dict(a.actor),
            "actor_frame": space_to_dict(a.actor_frame)}


def adapter_space_from_dict(d: dict[str, Any]) -> AdapterParameterSpace:
    return AdapterParameterSpace(space_from_dict(d.get("environment")), space_from_dict(d.get("actor")),
                                 space_from_dict(d.get("actor_frame")))


# ---- specs (core -> game) ----------------------------------------------------------------

def scene_to_dict(s: SceneSpec) -> dict[str, Any]:
    return {"scene_index": s.scene_index, "seed": s.seed, "split": s.split, "environment": s.environment,
            "actors": s.actors, "area": {"center": list(s.area.center), "radius": s.area.radius,
                                         "min_separation": s.area.min_separation}}


def scene_from_dict(d: dict[str, Any]) -> SceneSpec:
    a = d.get("area", {})
    return SceneSpec(d["scene_index"], d["seed"], d["split"], d["environment"], d["actors"],
                     SpawnArea(tuple(a.get("center", (0, 0, 0))), a.get("radius", 8.0),
                               a.get("min_separation", 1.0)))


def frame_to_dict(f: FrameSpec) -> dict[str, Any]:
    c = f.camera
    return {"scene_index": f.scene_index, "frame_index": f.frame_index, "seed": f.seed, "kind": f.kind.value,
            "camera": {"distance": c.distance, "yaw_deg": c.yaw_deg, "pitch_deg": c.pitch_deg,
                       "roll_deg": c.roll_deg, "fov_deg": c.fov_deg, "height_offset": c.height_offset,
                       "target_index": c.target_index},
            "actor_frame": f.actor_frame}


def frame_from_dict(d: dict[str, Any]) -> FrameSpec:
    return FrameSpec(d["scene_index"], d["frame_index"], d["seed"], FrameKind(d["kind"]),
                     CameraSpec(**d["camera"]), d["actor_frame"])


# ---- snapshots (game -> core) ------------------------------------------------------------

def camera_to_wire(cam: CameraModel) -> dict[str, Any]:
    R, t = cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3]
    pos = -R.T @ t
    fov_v = float(np.degrees(2 * np.arctan((cam.height / 2.0) / cam.fy)))
    return {"width": cam.width, "height": cam.height, "pos": pos.tolist(), "right": R[0].tolist(),
            "up": (-R[1]).tolist(), "forward": R[2].tolist(), "fov_v_deg": fov_v, "near": cam.near}


def camera_from_wire(d: dict[str, Any]) -> CameraModel:
    return CameraModel.from_pose(int(d["width"]), int(d["height"]), d["pos"], d["forward"], d["right"], d["up"],
                                 d.get("fov_v_deg"), d.get("fov_h_deg"), float(d.get("near", 0.1)))


def entity_from_wire(d: dict[str, Any], schema: SkeletonSchema) -> EntityState:
    k = schema.num_keypoints
    skel = np.asarray(d["skeleton_world"], dtype=np.float64).reshape(-1, 3)
    if skel.shape[0] != k:
        raise ValueError(f"entity {d.get('entity_id')}: got {skel.shape[0]} keypoints, schema needs {k}")
    valid = np.asarray(d.get("joint_valid", [True] * k), dtype=bool)
    if valid.shape != (k,):
        raise ValueError(f"entity {d.get('entity_id')}: joint_valid has {valid.size} items, schema needs {k}")
    ev = d.get("engine_visibility")
    hull = d.get("hull_points")
    return EntityState(
        int(d["entity_id"]), str(d.get("rig_id", "")), skel, valid,
        np.asarray(hull, dtype=np.float64).reshape(-1, 3) if hull else None,
        np.asarray(ev, dtype=np.int8) if ev is not None else None,
        dict(d.get("meta", {})))


def entity_to_wire(e: EntityState) -> dict[str, Any]:
    out: dict[str, Any] = {"entity_id": e.entity_id, "rig_id": e.rig_id,
                           "skeleton_world": e.skeleton_world.reshape(-1).tolist(),
                           "joint_valid": [bool(x) for x in e.joint_valid], "meta": e.meta}
    if e.engine_visibility is not None:
        out["engine_visibility"] = [int(x) for x in e.engine_visibility]
    if e.hull_points_world is not None:
        out["hull_points"] = e.hull_points_world.reshape(-1).tolist()
    return out


def snapshot_from_wire(d: dict[str, Any], schema: SkeletonSchema) -> FrameSnapshot:
    probes: Optional[list[Probe]] = None
    if d.get("probes") is not None:
        probes = [Probe(tuple(p["world"]), tuple(p["screen"]) if p.get("screen") else None)
                  for p in d["probes"]]
    return FrameSnapshot(
        frame_token=str(d["frame_token"]), tick=int(d.get("tick", 0)), camera=camera_from_wire(d["camera"]),
        entities=[entity_from_wire(e, schema) for e in d.get("entities", [])],
        probes=probes, meta={"warnings": list(d.get("warnings", [])), **d.get("meta", {})})


def snapshot_to_wire(s: FrameSnapshot) -> dict[str, Any]:
    out: dict[str, Any] = {"frame_token": s.frame_token, "tick": s.tick, "camera": camera_to_wire(s.camera),
                           "entities": [entity_to_wire(e) for e in s.entities], "meta": {}}
    if s.probes is not None:
        out["probes"] = [{"world": list(p.world), "screen": list(p.screen) if p.screen else None}
                         for p in s.probes]
    return out
