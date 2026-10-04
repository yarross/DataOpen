"""Mock "game": a numpy-simulated world implementing the full adapter contract.

It exists to (1) exercise and test the Universal Core end to end without a game, and
(2) serve as the reference for what a real Unity/Unreal adapter must provide. It uses
non-trivial rig bone names + weighted mapping, a camera-facing wall as occluder, and
Z-up world coordinates so that coordinate conversion bugs would show up in tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...core.interfaces import (AdapterError, AdapterInfo, Capability, ICaptureBridge, IEnvironmentController,
                                IEntitySpawner, IGameAdapter, ISkeletonExtractor)
from ...core.models import (CameraModel, CameraSpec, CaptureRequest, EntityHandle, EntityState, FrameSnapshot,
                            FrameSpec, SceneSpec)
from ...core.randomization import (AdapterParameterSpace, Categorical, ParameterSpace, Uniform, derive_seed)
from ...core.schema import HUMAN_13, BoneMapping, SkeletonSchema
from .png import write_png

# Rig A: engine-style names. Pelvis is not a bone: it is the mean of the two hips.
RIG_A = BoneMapping("mock_a", {
    "head": [("Head", 1.0)], "neck": [("Neck", 1.0)],
    "l_shoulder": [("UpperArm_L", 1.0)], "r_shoulder": [("UpperArm_R", 1.0)],
    "l_elbow": [("Forearm_L", 1.0)], "r_elbow": [("Forearm_R", 1.0)],
    "l_wrist": [("Hand_L", 1.0)], "r_wrist": [("Hand_R", 1.0)],
    "pelvis": [("Thigh_L", 0.5), ("Thigh_R", 0.5)],
    "l_knee": [("Calf_L", 1.0)], "r_knee": [("Calf_R", 1.0)],
    "l_ankle": [("Foot_L", 1.0)], "r_ankle": [("Foot_R", 1.0)],
})
# Rig B: a different naming convention + a real pelvis bone, same unified output.
RIG_B = BoneMapping("mock_b", {
    "head": [("bip_head", 1.0)], "neck": [("bip_neck", 1.0)],
    "l_shoulder": [("bip_l_clavicle", 1.0)], "r_shoulder": [("bip_r_clavicle", 1.0)],
    "l_elbow": [("bip_l_forearm", 1.0)], "r_elbow": [("bip_r_forearm", 1.0)],
    "l_wrist": [("bip_l_hand", 1.0)], "r_wrist": [("bip_r_hand", 1.0)],
    "pelvis": [("bip_pelvis", 1.0)],
    "l_knee": [("bip_l_calf", 1.0)], "r_knee": [("bip_r_calf", 1.0)],
    "l_ankle": [("bip_l_foot", 1.0)], "r_ankle": [("bip_r_foot", 1.0)],
})
_NAMES = {
    "mock_a": dict(head="Head", neck="Neck", ls="UpperArm_L", rs="UpperArm_R", le="Forearm_L", re="Forearm_R",
                   lw="Hand_L", rw="Hand_R", lh="Thigh_L", rh="Thigh_R", lk="Calf_L", rk="Calf_R",
                   la="Foot_L", ra="Foot_R"),
    "mock_b": dict(head="bip_head", neck="bip_neck", ls="bip_l_clavicle", rs="bip_r_clavicle",
                   le="bip_l_forearm", re="bip_r_forearm", lw="bip_l_hand", rw="bip_r_hand",
                   lh="bip_l_thigh", rh="bip_r_thigh", lk="bip_l_calf", rk="bip_r_calf",
                   la="bip_l_foot", ra="bip_r_foot", pelvis="bip_pelvis"),
}


class _World:
    def __init__(self) -> None:
        self.env: dict[str, Any] = {}
        self.actors: dict[int, dict[str, Any]] = {}
        self.active = True
        self.tick = 0
        self.rng = np.random.default_rng(0)


def _pose_bones(rig: str, a: dict[str, Any], fparams: dict[str, Any]) -> dict[str, np.ndarray]:
    """Procedural humanoid in a local frame (+x forward, z up), then rotated/translated."""
    s = a["height_scale"]
    anim, phase = fparams.get("animation", "stand"), fparams.get("phase", 0.0)
    swing = 0.25 * np.sin(2 * np.pi * phase) if anim == "walk" else 0.0
    crouch = 0.35 if anim == "crouch" else 0.0
    zk, za, zp = 0.5 - crouch * 0.4, 0.08, 0.95 - crouch
    p = {  # local joint positions (x fwd, y left, z up)
        "head": (0, 0, 1.70 - crouch), "neck": (0, 0, 1.50 - crouch),
        "ls": (0, 0.20, 1.45 - crouch), "rs": (0, -0.20, 1.45 - crouch),
        "le": (swing, 0.26, 1.15 - crouch), "re": (-swing, -0.26, 1.15 - crouch),
        "lw": (2 * swing, 0.28, 0.90 - crouch), "rw": (-2 * swing, -0.28, 0.90 - crouch),
        "lh": (0, 0.09, zp), "rh": (0, -0.09, zp),
        "lk": (-swing + crouch * 0.4, 0.10, zk), "rk": (swing + crouch * 0.4, -0.10, zk),
        "la": (-2 * swing, 0.10, za), "ra": (2 * swing, -0.10, za),
        "pelvis": (0, 0, zp),
    }
    c, sn = np.cos(a["yaw"]), np.sin(a["yaw"])
    rot = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    base = np.array([a["x"], a["y"], 0.0])
    names = _NAMES[rig]
    return {names[k]: rot @ (np.array(v) * [1, 1, s]) + base for k, v in p.items() if k in names}


class _Spawner(IEntitySpawner):
    def __init__(self, w: _World) -> None:
        self.w = w

    def spawn(self, scene: SceneSpec) -> list[EntityHandle]:
        rng = np.random.default_rng(derive_seed(scene.seed, "mock-spawn"))
        placed: list[tuple[float, float]] = []
        handles = []
        cx, cy, _ = scene.area.center
        for i, params in enumerate(scene.actors):
            for _ in range(50):  # rejection sampling for min separation
                r, th = scene.area.radius * np.sqrt(rng.random()), rng.random() * 2 * np.pi
                x, y = cx + r * np.cos(th), cy + r * np.sin(th)
                if all(np.hypot(x - px, y - py) >= scene.area.min_separation for px, py in placed):
                    break
            else:
                raise AdapterError("could not place actor without overlap")
            placed.append((x, y))
            self.w.actors[i] = {"x": x, "y": y, "yaw": rng.random() * 2 * np.pi, **params}
            handles.append(EntityHandle(i, params["rig"], {k: v for k, v in params.items() if k != "rig"}))
        self.w.active = True
        return handles

    def update_actors(self, handles: Sequence[EntityHandle], frame: FrameSpec) -> None:
        for h, fp in zip(handles, frame.actor_frame):
            self.w.actors[h.entity_id]["frame"] = fp

    def set_active(self, handles, active: bool) -> None:
        self.w.active = active

    def despawn_all(self) -> None:
        self.w.actors.clear()


class _Environment(IEnvironmentController):
    def __init__(self, w: _World) -> None:
        self.w = w

    def apply(self, scene: SceneSpec) -> None:
        self.w.env = dict(scene.environment)


class _Extractor(ISkeletonExtractor):
    def __init__(self, w: _World, schema: SkeletonSchema) -> None:
        self.w, self._schema = w, schema
        self._maps = {"mock_a": RIG_A, "mock_b": RIG_B}

    @property
    def schema(self) -> SkeletonSchema:
        return self._schema

    def mapping_for(self, rig_id: str) -> BoneMapping:
        return self._maps[rig_id]

    def extract(self, handles: Sequence[EntityHandle]) -> list[EntityState]:
        if not self.w.active:
            return []  # hidden actors are not reported (the engine would not render them)
        out = []
        for h in handles:
            a = self.w.actors[h.entity_id]
            bones = _pose_bones(h.rig_id, a, a.get("frame", {}))
            pos, valid = self.mapping_for(h.rig_id).resolve(self._schema, bones)
            hull = np.array(list(bones.values()))
            hull = np.vstack([hull + [dx, dy, dz] for dx in (-.12, .12) for dy in (-.12, .12) for dz in (-.1, .1)])
            out.append(EntityState(h.entity_id, h.rig_id, pos, valid, hull_points_world=hull, meta=dict(h.meta)))
        return out


def _look_at(pos: np.ndarray, target: np.ndarray, roll_deg: float) -> np.ndarray:
    """World (Z-up) -> camera (OpenCV: x right, y down, z forward) 4x4."""
    f = target - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    r = np.radians(roll_deg)
    right, down = np.cos(r) * right + np.sin(r) * down, -np.sin(r) * right + np.cos(r) * down
    R = np.stack([right, down, f])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ pos
    return M


class _Bridge(ICaptureBridge):
    def __init__(self, w: _World, ext: _Extractor) -> None:
        self.w, self.ext = w, ext
        self._pending: dict[str, np.ndarray] = {}

    def _camera(self, spec: CameraSpec, req: CaptureRequest) -> CameraModel:
        w = self.w
        tgt = w.actors[min(spec.target_index, len(w.actors) - 1)]
        target = np.array([tgt["x"], tgt["y"], 1.0])
        y, p = np.radians(spec.yaw_deg), np.radians(spec.pitch_deg)
        pos = target + spec.distance * np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
        pos[2] = max(pos[2] + spec.height_offset, 0.3)  # never below ground
        return CameraModel.from_vertical_fov(req.width, req.height, spec.fov_deg, _look_at(pos, target, spec.roll_deg))

    def capture(self, req: CaptureRequest) -> FrameSnapshot:
        w, spec = self.w, req.frame_spec
        if not w.actors:
            raise AdapterError("no actors spawned")
        w.tick += 1  # real adapters: step N fixed ticks, wait for end-of-frame
        cam = self._camera(spec.camera, req)
        entities = self.ext.extract([EntityHandle(i, a["rig"]) for i, a in w.actors.items()]) if w.active else []
        # NOTE: handles above lose appearance meta; fine for the mock.

        rng = np.random.default_rng(derive_seed(spec.seed, "mock-wall"))
        wall = None
        if entities and rng.random() < 0.4:  # camera-aligned wall between camera and target
            dist = spec.camera.distance * rng.uniform(0.4, 0.8)
            wall = (dist, rng.uniform(-0.6, 0.6), rng.uniform(0.0, 0.8), rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9))
        depth = self._depth(cam, wall)
        img = self._render(cam, entities, wall, depth)
        self._pending[req.frame_id] = img
        gray = img.mean(axis=2)
        thumb = gray[: gray.shape[0] // 8 * 8, : gray.shape[1] // 9 * 9]
        thumb = thumb.reshape(8, thumb.shape[0] // 8, 9, thumb.shape[1] // 9).mean(axis=(1, 3))
        return FrameSnapshot(req.frame_id, w.tick, cam, entities, depth, thumb, {"wall": wall})

    def _depth(self, cam: CameraModel, wall) -> np.ndarray:
        h, wd = cam.height // 2, cam.width // 2  # depth rendered at half resolution
        d = np.full((h, wd), np.inf, dtype=np.float32)
        if wall is not None:
            dist, ox, oy, hw, hh = wall
            u = (np.arange(wd) + 0.5) * 2
            v = (np.arange(h) + 0.5) * 2
            x = (u - cam.cx) / cam.fx * dist
            y = (v - cam.cy) / cam.fy * dist
            inside = (np.abs(x[None, :] - ox) <= hw) & (np.abs(y[:, None] - oy) <= hh)
            d[inside] = dist
        return d

    def _render(self, cam: CameraModel, ents, wall, depth) -> np.ndarray:
        env = self.w.env
        tod = env.get("time_of_day", 12.0)
        light = float(np.clip(np.sin(np.pi * (tod - 6) / 12), 0.12, 1.0)) * 2 ** env.get("exposure_ev", 0)
        fog = 1 - np.exp(-40 * env.get("fog_density", 0.0))
        H, W = cam.height, cam.width
        sky = np.linspace([120, 160, 220], [180, 200, 230], H // 2)
        ground = np.linspace([70, 90, 60], [40, 60, 40], H - H // 2)
        img = np.concatenate([sky, ground])[:, None, :].repeat(W, axis=1)
        img = img * min(light, 1.2) * (1 - fog) + 190 * fog * min(light, 1.0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        from ...core.projection import project
        for e in ents:
            uv, z = project(e.skeleton_world, cam)
            col = np.array([200, 60, 60], dtype=np.uint8)
            for (u, v), zz in zip(uv, z):
                if zz > cam.near and 2 <= u < W - 2 and 2 <= v < H - 2:
                    img[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3] = col
        if wall is not None:
            big = np.repeat(np.repeat(np.isfinite(depth), 2, axis=0), 2, axis=1)[:H, :W]
            img[big] = (60, 60, 70)
        return img

    def commit(self, snapshot: FrameSnapshot, dest: Path) -> None:
        write_png(dest, self._pending.pop(snapshot.frame_token))

    def discard(self, snapshot: FrameSnapshot) -> None:
        self._pending.pop(snapshot.frame_token, None)


class MockGameAdapter(IGameAdapter):
    def __init__(self, width: int = 320, height: int = 240, schema: SkeletonSchema = HUMAN_13) -> None:
        self._world = _World()
        self._info = AdapterInfo("mock", "mock", schema, frozenset({Capability.DEPTH, Capability.HULL_POINTS}),
                                 (width, height))
        self._spawner = _Spawner(self._world)
        self._env = _Environment(self._world)
        self._extractor = _Extractor(self._world, schema)
        self._capture = _Bridge(self._world, self._extractor)

    info = property(lambda self: self._info)
    spawner = property(lambda self: self._spawner)
    environment = property(lambda self: self._env)
    extractor = property(lambda self: self._extractor)
    capture = property(lambda self: self._capture)

    def parameter_space(self) -> AdapterParameterSpace:
        return AdapterParameterSpace(
            actor=ParameterSpace({
                "rig": Categorical(("mock_a", "mock_b")),
                "height_scale": Uniform(0.9, 1.1),
                "outfit": Categorical(("casual", "armor", "uniform")),
                "weapon": Categorical(("none", "rifle", "pistol", "bat")),
            }),
            actor_frame=ParameterSpace({
                "animation": Categorical(("stand", "walk", "crouch"), (0.4, 0.4, 0.2)),
                "phase": Uniform(0.0, 1.0),
            }),
        )

    def connect(self) -> None: ...

    def close(self) -> None: ...
