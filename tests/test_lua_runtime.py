"""Runs the real Lua runtime (adapters/lua/runtime/dataopen_rpc.lua) against a Lua mock game, driven by
the real Python core over the real file mailbox, on Lua 5.1, 5.4 and LuaJIT (what Garry's Mod, UE4SS and
Cyber Engine Tweaks embed)."""
import importlib
import json
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

ROOT = Path(__file__).resolve().parents[1] / "adapters" / "lua"
FLAVORS = ["lupa.lua51", "lupa.lua54", "lupa.luajit21"]


def lua_runtime(flavor):
    try:
        return importlib.import_module(flavor).LuaRuntime(unpack_returned_tuples=True)
    except Exception:
        pytest.skip(f"{flavor} not available")


class LuaGame:
    """Owns one Lua state in one thread (Lua states are not thread-safe) and polls the runtime like a game tick."""

    def __init__(self, flavor, mailbox: Path, host_opts=None, runtime_opts=None):
        self.flavor, self.mailbox = flavor, mailbox
        self.host_opts, self.runtime_opts = host_opts or {}, runtime_opts or {}
        self.stop, self.ready, self.error = threading.Event(), threading.Event(), None
        self.frozen = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        assert self.ready.wait(10), self.error

    def _run(self):
        try:
            lua = lua_runtime(self.flavor)
            R = lua.globals().dofile(str(ROOT / "runtime" / "dataopen_rpc.lua"))
            build = lua.globals().dofile(str(ROOT / "tests" / "mock_host.lua"))
            opts = lua.table_from({**self.host_opts, "now": time.monotonic})
            host = build(R, str(self.mailbox), opts)
            rt = R.new(host, lua.table_from(self.runtime_opts))
            self.host, self.rt, self.R, self.lua = host, rt, R, lua
            self.ready.set()
            while not self.stop.is_set():
                rt.poll(rt)
                time.sleep(0.001)
        except Exception as e:  # surface Lua errors in the test
            self.error = e
            self.ready.set()

    def close(self):
        self.stop.set()
        self.t.join(3)


@pytest.fixture(params=FLAVORS)
def flavor(request):
    lua_runtime(request.param)  # skip if missing
    return request.param


def png_bytes(tmp_path, w=160, h=90) -> bytes:
    p = tmp_path / "t.png"
    y, x = np.mgrid[0:h, 0:w]
    write_png(p, np.stack([(x * 3) % 256, (y * 5) % 256, (x + y) % 256], axis=-1).astype(np.uint8))
    return p.read_bytes()


def make(flavor, tmp_path, host_opts=None, runtime_opts=None, **remote):
    mb = tmp_path / "mb"
    (mb / "staging").mkdir(parents=True)
    g = LuaGame(flavor, mb, {"png": png_bytes(tmp_path), **(host_opts or {})}, runtime_opts)
    opts = RemoteOptions(call_timeout_s=10, capture_timeout_s=10, **remote)
    a = RemoteGameAdapter(FileMailboxTransport(mb, default_timeout_s=10), opts)
    return g, a


def test_json_roundtrip_and_edge_cases(flavor):
    lua = lua_runtime(flavor)
    R = lua.globals().dofile(str(ROOT / "runtime" / "dataopen_rpc.lua"))
    J = R.json
    text = J.encode(lua.eval('{a = {1, 2.5, -3e-3}, b = "he said \\"hi\\"\\n\\t\\\\", c = true, d = {}}'))
    back = json.loads(text)
    assert back["a"][:2] == [1, 2.5] and abs(back["a"][2] + 0.003) < 1e-12 and back["b"] == 'he said "hi"\n\t\\'
    assert back["c"] is True and back["d"] == {}
    dec = J.decode(json.dumps({"s": "é€😀\u0001", "n": [1, 2, {"x": None, "y": [True, False]}], "f": -1.5e2}))
    assert dec["f"] == -150 and dec["n"][3]["y"][1] is True and dec["n"][3]["y"][2] is False
    assert dec["s"] == "é€😀\u0001"                                       # \u escapes + surrogate pair -> UTF-8
    assert dec["n"][3]["x"] is None                                    # null -> nil
    for bad in ('{"a":', '[1,2', '{"a" 1}', 'nope', '[1,]x'):
        with pytest.raises(Exception, match="unterminated|expected|unexpected|trailing|bad|number"):
            J.decode(bad)
    with pytest.raises(Exception, match="non-finite"):
        J.encode(lua.eval("{x = 0/0}"))                                # NaN is not JSON
    assert J.encode(J.array(lua.table())) == "[]" and J.encode(lua.table()) == "{}"


def test_hello_over_the_wire(flavor, tmp_path):
    g, a = make(flavor, tmp_path)
    try:
        a.connect()
        assert a.info.name == "luamock" and a.hello["game_version"] == "sim-1.0" and a.capture_mode == "engine"
        assert a.info.image_size == (160, 90)
        assert {"rig", "outfit"} <= set(a.parameter_space().actor.params)
        assert "animation" in a.parameter_space().actor_frame.params
        assert a.health()["ok"]
    finally:
        g.close()


def test_doctor_passes_against_the_lua_runtime(flavor, tmp_path):
    g, a = make(flavor, tmp_path)
    try:
        rep = run_doctor(a, tmp_path / "out", frames=5)
        assert rep.ok, rep.render()
        names = {c.name: c.status for c in rep.checks}
        assert names["projection"] == "PASS" and names["units"] == "PASS"          # inches -> meters worked
        assert names["left_right"] == "PASS" and names["mod:bone_mapping"] == "PASS"
        assert (tmp_path / "out" / "doctor" / "overlay_0.png").exists()
    finally:
        g.close()


def test_full_collection_through_the_lua_runtime_with_staged_images(flavor, tmp_path):
    g, a = make(flavor, tmp_path, host_opts={"occlude_below_z": 0.25})
    try:
        a.connect()
        out = tmp_path / "ds"
        rep = DatasetOrchestrator(a, SessionConfig(out, seed=3, target_frames=25, frames_per_scene=6,
                                                   negative_ratio=0.2)).run()
        assert rep.accepted == 25 and rep.probe_failed == 0 and rep.probe_checked >= 25, rep
        v = verify_dataset(out)
        assert v.ok, [i.__dict__ for i in v.errors]
        assert not list((tmp_path / "mb" / "staging").iterdir())                    # every staged file was moved
        imgs = sorted(out.glob("images/*/*.png"))
        assert len(imgs) == 25 and read_image_size(imgs[0]) == (160, 90)
        ankles = v.stats["keypoint_visibility"]["l_ankle"]
        assert ankles["occluded"] > 0                                               # engine visibility reached the labels
        assert g.host["world"]["unfreezes"] > 0 and not g.host["world"]["frozen"]
    finally:
        g.close()


def test_handler_error_is_reported_and_runtime_keeps_working(flavor, tmp_path):
    g, a = make(flavor, tmp_path)
    try:
        a.connect()
        g.host["fail_next"] = "begin_scene"
        from dataopen.core.randomization import DomainRandomizationController
        scene = DomainRandomizationController(0, a.parameter_space()).sample_scene(0)
        with pytest.raises(RemoteError, match="simulated spawn failure"):
            a.spawner.spawn(scene)
        assert len(a.spawner.spawn(scene)) == len(scene.actors)                    # recovered
    finally:
        g.close()


def test_runtime_timeout_when_the_engine_never_finishes(flavor, tmp_path):
    g, a = make(flavor, tmp_path, host_opts={"hang_capture": True}, runtime_opts={"timeout_s": 0.4})
    try:
        a.connect()
        from dataopen.core.models import CaptureRequest
        from dataopen.core.randomization import DomainRandomizationController
        rz = DomainRandomizationController(0, a.parameter_space())
        scene = rz.sample_scene(0)
        a.spawner.spawn(scene)
        from dataopen.core.models import FrameKind
        spec = rz.sample_frame(scene, 0, FrameKind.POSITIVE)
        with pytest.raises(RemoteError, match="timed out"):
            a.capture.capture(CaptureRequest("f0", spec, 160, 90))
        assert a.health()["ok"]                                                    # not wedged
    finally:
        g.close()


def test_unmapped_bones_are_diagnosed_and_fixed_with_a_profile_override(flavor, tmp_path):
    g, a = make(flavor, tmp_path, host_opts={"alt_names": True})
    try:
        a.connect()
        checks = {c["name"]: c for c in a.selftest()}
        bm = checks["bone_mapping"]
        assert not bm["ok"] and "head" in bm["data"]["unmapped"]
        assert "Alt_Head1" in bm["data"]["bones_found"]
        rep = run_doctor(a, tmp_path / "out", frames=2)
        assert not rep.ok and "bone_mapping" in rep.render() and "[bones]" in rep.render()
    finally:
        g.close()
    # the override from the profile fixes it without touching the mod
    from dataopen.core.schema import HUMAN_13
    override = {"head": [["Alt_Head1", 1.0]], "neck": [["Alt_Neck1", 1.0]], "pelvis": [["Alt_Pelvis", 1.0]],
                "l_shoulder": [["Alt_L_UpperArm", 1.0]], "r_shoulder": [["Alt_R_UpperArm", 1.0]],
                "l_elbow": [["Alt_L_Forearm", 1.0]], "r_elbow": [["Alt_R_Forearm", 1.0]],
                "l_wrist": [["Alt_L_Hand", 1.0]], "r_wrist": [["Alt_R_Hand", 1.0]],
                "l_knee": [["Alt_L_Calf", 1.0]], "r_knee": [["Alt_R_Calf", 1.0]],
                "l_ankle": [["Alt_L_Foot", 1.0]], "r_ankle": [["Alt_R_Foot", 1.0]]}
    assert set(override) == set(HUMAN_13.keypoints)
    g2, a2 = make(flavor, tmp_path / "second", host_opts={"alt_names": True}, bone_map=override)
    try:
        a2.connect()
        assert all(c["ok"] for c in a2.selftest())
    finally:
        g2.close()


def test_core_vanishing_mid_frame_unfreezes_the_game(flavor, tmp_path):
    g, a = make(flavor, tmp_path, runtime_opts={"freeze_timeout_s": 0.3})
    try:
        a.connect()
        from dataopen.core.models import CaptureRequest, FrameKind
        from dataopen.core.randomization import DomainRandomizationController
        rz = DomainRandomizationController(0, a.parameter_space())
        scene = rz.sample_scene(0)
        a.spawner.spawn(scene)
        a.capture.capture(CaptureRequest("f0", rz.sample_frame(scene, 0, FrameKind.POSITIVE), 160, 90))
        assert g.host["world"]["frozen"]
        time.sleep(0.8)                                                            # the core "crashed": no release/commit
        assert not g.host["world"]["frozen"]
    finally:
        g.close()


def test_garbage_stale_and_wrong_version_requests(flavor, tmp_path):
    g, a = make(flavor, tmp_path)
    try:
        a.connect()
        mb = tmp_path / "mb"
        (mb / "req.json").write_text('{"v":1,"id":5,"meth')                        # half-written request: ignored
        time.sleep(0.05)
        assert json.loads((mb / "res.json").read_text()).get("id") != 5
        (mb / "req.json").write_text(json.dumps({"v": 99, "id": 6, "method": "hello", "params": {}}))
        time.sleep(0.1)
        assert "protocol version" in json.loads((mb / "res.json").read_text())["error"]["message"]
        (mb / "req.json").write_text(json.dumps({"v": 1, "id": 7, "method": "bogus", "params": {}}))
        time.sleep(0.1)
        assert "unknown method" in json.loads((mb / "res.json").read_text())["error"]["message"]
    finally:
        g.close()
