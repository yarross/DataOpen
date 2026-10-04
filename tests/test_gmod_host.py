"""Runs the real Garry's Mod loader + host + shared runtime on LuaJIT (what GMod embeds) against an API stub
that enforces GMod's hook-only rendering rules, driven by the real Python core over the real mailbox."""
import importlib
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lupa")

from dataopen.adapters.remote import RemoteGameAdapter, RemoteOptions
from dataopen.core.doctor import run_doctor
from dataopen.core.imageio import read_image_size, write_png
from dataopen.core.orchestrator import DatasetOrchestrator, SessionConfig
from dataopen.core.qa import verify_dataset
from dataopen.core.transport import FileMailboxTransport, RemoteError

ROOT = Path(__file__).resolve().parents[1] / "adapters"
ADDON = ROOT / "gmod" / "addon" / "dataopen" / "lua"
W, H = 160, 90


def luajit():
    try:
        return importlib.import_module("lupa.luajit21").LuaRuntime(unpack_returned_tuples=True)
    except Exception:
        pytest.skip("LuaJIT (lupa.luajit21) not available")


class GmodGame:
    def __init__(self, data_dir: Path, png: bytes, opts=None):
        self.data_dir, self.png, self.opts = data_dir, png, opts or {}
        self.stop, self.ready, self.error = threading.Event(), threading.Event(), None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        assert self.ready.wait(10) and self.error is None, self.error

    def _run(self):
        try:
            lua = luajit()
            stub = lua.globals().dofile(str(ROOT / "gmod" / "tests" / "gmod_stub.lua"))
            self.stub, self.lua = stub, lua
            stub.runtime_path = str(ROOT / "lua" / "runtime" / "dataopen_rpc.lua")
            stub.clock = time.monotonic
            stub.setup(str(self.data_dir), self.png, lua.table_from(self.opts))
            lua.globals().include = lambda p: stub.include(p, str(ADDON))
            lua.globals().dofile(str(ADDON / "autorun" / "dataopen_init.lua"))
            self.ready.set()
            while not self.stop.is_set():
                stub.tick()
                time.sleep(0.001)
        except Exception as e:
            self.error = e
            self.ready.set()

    def close(self):
        self.stop.set()
        self.t.join(3)
        assert self.error is None, f"Lua error in the game thread: {self.error}"


@pytest.fixture
def png(tmp_path) -> bytes:
    p = tmp_path / "t.png"
    y, x = np.mgrid[0:H, 0:W]
    write_png(p, np.stack([(x * 3) % 256, (y * 5) % 256, (x + y) % 256], axis=-1).astype(np.uint8))
    return p.read_bytes()


def start(tmp_path, png, game_opts=None, **remote):
    data = tmp_path / "garrysmod" / "data"
    data.mkdir(parents=True)
    g = GmodGame(data, png, game_opts)
    mailbox = data / "dataopen"
    deadline = time.monotonic() + 5
    while not mailbox.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    mailbox.mkdir(exist_ok=True)
    opts = RemoteOptions(image_size=(W, H), call_timeout_s=15, capture_timeout_s=15, **remote)
    return g, RemoteGameAdapter(FileMailboxTransport(mailbox, default_timeout_s=15), opts), mailbox


def test_loader_hello_and_vocabulary(tmp_path, png):
    g, a, _ = start(tmp_path, png)
    try:
        a.connect()
        assert a.info.name == "garrysmod" and a.info.engine == "source" and a.capture_mode == "engine"
        assert a.hello["game_version"] == "240222" and a.info.image_size == (W, H)
        ps = a.parameter_space()
        assert set(ps.actor.params["model"].choices) == {"kleiner", "alyx", "police"}
        assert "none" in ps.actor.params["weapon"].choices and "stance" in ps.actor_frame.params
        assert set(ps.environment.params["weather"].choices) == {"clear", "overcast", "fog"}
        assert "dataopen_status" in [k[4:] for k in g.stub["calls"].keys() if str(k).startswith("cmd_")]
    finally:
        g.close()


def test_doctor_passes_on_gmod_host(tmp_path, png):
    g, a, _ = start(tmp_path, png)
    try:
        rep = run_doctor(a, tmp_path / "out", frames=5)
        assert rep.ok, rep.render()
        status = {c.name: c.status for c in rep.checks}
        assert status["projection"] == "PASS" and status["units"] == "PASS"
        assert status["left_right"] == "PASS" and status["mod:bone_mapping"] == "PASS"
        assert status["mod:render_hook"] == "PASS" and status["mod:data_folder_write"] == "PASS"
        assert (tmp_path / "out" / "doctor" / "overlay_0.png").exists()
    finally:
        g.close()


def test_full_collection_in_gmod_and_cleanup(tmp_path, png):
    g, a, mailbox = start(tmp_path, png)
    try:
        a.connect()
        out = tmp_path / "ds"
        rep = DatasetOrchestrator(a, SessionConfig(out, seed=2, target_frames=20, frames_per_scene=5,
                                                   negative_ratio=0.2)).run()
        assert rep.accepted == 20 and rep.probe_failed == 0
        v = verify_dataset(out)
        assert v.ok, [i.__dict__ for i in v.errors]
        assert len(list(out.glob("images/*/*.png"))) == 20
        assert read_image_size(next(out.glob("images/*/*.png"))) == (W, H)
        assert not [p for p in (mailbox / "staging").iterdir() if p.name != "selftest.dat"]   # staged files moved
        time.sleep(0.1)
        stub = g.stub
        assert all(not e["_valid"] for e in stub["entities"].values())                  # every actor model removed
        rv = stub["calls"]["render_view"]
        assert rv["drawhud"] is False and rv["drawviewmodel"] is False and rv["dopostprocess"] is True
        assert rv["w"] == W and rv["h"] == H
    finally:
        g.close()


def test_wrong_fov_convention_is_caught_with_a_diagnosis(tmp_path, png):
    g, a, _ = start(tmp_path, png, mod_options={"fov_mode": "horizontal"})
    try:
        rep = run_doctor(a, tmp_path / "out", frames=4)
        assert not rep.ok and "focal length differs" in rep.render()
    finally:
        g.close()


def test_no_ground_gives_an_actionable_error(tmp_path, png):
    g, a, _ = start(tmp_path, png, game_opts={"no_ground": True})
    try:
        a.connect()
        from dataopen.core.randomization import DomainRandomizationController
        scene = DomainRandomizationController(0, a.parameter_space()).sample_scene(0)
        with pytest.raises(RemoteError, match="flat, unobstructed ground"):
            a.spawner.spawn(scene)
    finally:
        g.close()


def test_marked_locations_are_used_and_weapons_are_attached(tmp_path, png):
    g, a, mailbox = start(tmp_path, png)
    try:
        (mailbox / "locations.json").write_text("[[5000, 0, 0]]")
        a.connect()
        from dataopen.core.randomization import DomainRandomizationController
        rz = DomainRandomizationController(0, a.parameter_space())
        total_weps = 0
        for i in range(6):
            scene = rz.sample_scene(i)
            hs = a.spawner.spawn(scene)
            time.sleep(0.05)
            actors = [e for e in g.stub["entities"].values() if e["_valid"] and e["path"].startswith("models/player")]
            assert actors and all(abs(e["pos"]["x"] - 5000) < 3 / 0.01905 + 12 / 0.01905 + 5 for e in actors)
            total_weps += sum(1 for e in g.stub["entities"].values() if e["_valid"] and e["path"].startswith("models/weapons"))
            assert {h.rig_id for h in hs} == {"valvebiped"}
            a.spawner.despawn_all()
        assert total_weps > 0
    finally:
        g.close()


def test_visibility_binding_uses_the_world_trace(tmp_path, png):
    g, a, _ = start(tmp_path, png, game_opts={"wall_x": 100})
    try:
        a.connect()
        host = g.lua.globals()  # noqa: F841  (host object lives inside the Lua closure; check via a scene capture)
        from dataopen.core.models import CaptureRequest, FrameKind
        from dataopen.core.randomization import DomainRandomizationController
        rz = DomainRandomizationController(0, a.parameter_space())
        scene = rz.sample_scene(0)
        a.spawner.spawn(scene)
        spec = rz.sample_frame(scene, 0, FrameKind.POSITIVE)
        snap = a.capture.capture(CaptureRequest("f0", spec, W, H))
        assert snap.entities and snap.entities[0].engine_visibility is not None
        assert set(np.unique(snap.entities[0].engine_visibility)) <= {1, 2}
        a.capture.discard(snap)
    finally:
        g.close()


def test_closed_loop_in_gmod_peek_via_staged_dat_file(tmp_path, png):
    from dataopen.core.schema import HUMAN_13
    from dataopen.quality.evaluators.simulated import SimulatedEvaluator
    from dataopen.quality.pipeline import QualityConfig, QualityPipeline
    g, a, mailbox = start(tmp_path, png)
    try:
        a.connect()
        assert "image_peek" in a.caps
        q = QualityPipeline(HUMAN_13, SimulatedEvaluator(HUMAN_13), QualityConfig())
        rep = DatasetOrchestrator(a, SessionConfig(tmp_path / "ds", seed=2, target_frames=12, frames_per_scene=4,
                                                   negative_ratio=0.2), quality=q).run()
        assert rep.accepted == 12 and rep.quality["pipeline"]["frames_without_pixels"] == 0
        assert not [p for p in (mailbox / "staging").iterdir() if p.name != "selftest.dat"]
    finally:
        g.close()


def test_teams_become_model_pools_and_reach_the_dataset_as_classes(tmp_path, png):
    """shooter12 needs a team per player. In Garry's Mod a team is a model pool (police/Combine = ct, citizens = t); the mod reports
    it as entity meta over the wire and the core turns it into the class."""
    import json

    from dataopen.core.schema_io import resolve_target
    g, a, _ = start(tmp_path, png)
    try:
        a.connect()
        ps = a.parameter_space()
        assert set(ps.actor.params["team"].choices) == {"ct", "t"} and "model_u" in ps.actor.params
        _, mapping = resolve_target("shooter12", a.schema)
        out = tmp_path / "ds"
        rep = DatasetOrchestrator(a, SessionConfig(out, seed=5, target_frames=16, frames_per_scene=4, negative_ratio=0.0,
                                                max_attempt_factor=30), target=mapping).run()
        assert rep.accepted == 16
        recs = [json.loads(x) for p in (out / "annotations").glob("*.jsonl") for x in p.read_text().splitlines()]
        pairs = {(an["class_id"], an["meta"]["model"]) for r in recs for an in r["annotations"]}
        assert {c for c, _ in pairs} == {0, 1}                                       # both teams occur
        assert all((m == "police") == (c == 0) for c, m in pairs), pairs               # ct wears police, t wears citizens
        assert all(len(an["keypoints"]) == 36 for r in recs for an in r["annotations"])
    finally:
        g.close()
