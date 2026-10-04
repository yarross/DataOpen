import json
import threading
import time

import numpy as np
import pytest

from dataopen.core.interfaces import AdapterError
from dataopen.core.models import (CameraModel, CameraSpec, EntityState, FrameKind, FrameSnapshot, FrameSpec, Probe,
                                  SceneSpec)
from dataopen.core.projection import project
from dataopen.core.protocol import (adapter_space_from_dict, adapter_space_to_dict, camera_from_wire,
                                    camera_to_wire, frame_from_dict, frame_to_dict, scene_from_dict,
                                    scene_to_dict, snapshot_from_wire, snapshot_to_wire)
from dataopen.core.randomization import (AdapterParameterSpace, Categorical, Constant, LogUniform, ParameterSpace,
                                         Uniform)
from dataopen.core.schema import HUMAN_13
from dataopen.core.transport import FileMailboxTransport, MailboxServer, RemoteError, TransportTimeout


def test_camera_from_pose_is_handedness_proof():
    # Same physical camera described in a right-handed Z-up world and in a left-handed Y-up world:
    # the pose basis differs but the pixel for the same physical point must be identical.
    rh = CameraModel.from_pose(640, 480, [0, 0, 0], forward=[1, 0, 0], right=[0, -1, 0], up=[0, 0, 1],
                               fov_v_deg=60)
    lh = CameraModel.from_pose(640, 480, [0, 0, 0], forward=[0, 0, 1], right=[1, 0, 0], up=[0, 1, 0],
                               fov_v_deg=60)
    p_rh = np.array([[10.0, -2.0, 1.0]])   # 10 ahead, 2 right, 1 up
    p_lh = np.array([[2.0, 1.0, 10.0]])    # same point in the LH world
    assert np.allclose(project(p_rh, rh)[0], project(p_lh, lh)[0])
    uv, _ = project(p_rh, rh)
    assert uv[0, 0] > 320 and uv[0, 1] < 240          # right of centre, above centre


def test_camera_from_pose_fov_modes_and_bad_basis():
    v = CameraModel.from_pose(640, 480, [0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1], fov_v_deg=90)
    assert np.isclose(v.fy, 240)
    h = CameraModel.from_pose(640, 480, [0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1], fov_h_deg=90)
    assert np.isclose(h.fx, 320)
    with pytest.raises(ValueError):
        CameraModel.from_pose(640, 480, [0, 0, 0], [0, 0, 0], [0, -1, 0], [0, 0, 1], fov_v_deg=90)
    with pytest.raises(ValueError):
        CameraModel.from_pose(640, 480, [0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1])


def test_camera_wire_roundtrip():
    cam = CameraModel.from_pose(320, 240, [1, 2, 3], [1, 0, 0.2], [0, -1, 0], [-0.2, 0, 1], fov_v_deg=70)
    cam2 = camera_from_wire(json.loads(json.dumps(camera_to_wire(cam))))
    pts = np.array([[8, 1, 3], [5, -3, 4.0]])
    assert np.allclose(project(pts, cam)[0], project(pts, cam2)[0], atol=1e-6)


def test_parameter_space_roundtrip():
    a = AdapterParameterSpace(
        environment=ParameterSpace({"x": Uniform(0, 1)}),
        actor=ParameterSpace({"skin": Categorical(("a", "b"), (1, 3)), "k": Constant(5), "d": LogUniform(1, 10)}),
    )
    b = adapter_space_from_dict(json.loads(json.dumps(adapter_space_to_dict(a))))
    assert b.environment.params == a.environment.params and b.actor.params == a.actor.params


def test_scene_and_frame_roundtrip():
    s = SceneSpec(3, 99, "val", {"weather": "fog"}, [{"rig": "a"}])
    assert scene_from_dict(json.loads(json.dumps(scene_to_dict(s)))) == s
    f = FrameSpec(3, 4, 5, FrameKind.NEGATIVE, CameraSpec(distance=5.0), [{"animation": "walk"}])
    assert frame_from_dict(json.loads(json.dumps(frame_to_dict(f)))) == f


def test_snapshot_roundtrip_and_schema_mismatch():
    cam = CameraModel.from_pose(320, 240, [0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1], fov_v_deg=60)
    ent = EntityState(1, "r", np.arange(39, dtype=float).reshape(13, 3), np.ones(13, bool),
                      engine_visibility=np.full(13, 2, dtype=np.int8))
    snap = FrameSnapshot("tok", 7, cam, [ent], probes=[Probe((5, 0, 0), (160.0, 120.0)), Probe((5, 90, 0), None)])
    back = snapshot_from_wire(json.loads(json.dumps(snapshot_to_wire(snap))), HUMAN_13)
    assert back.frame_token == "tok" and back.tick == 7 and np.allclose(back.entities[0].skeleton_world, ent.skeleton_world)
    assert back.probes[0].screen == (160.0, 120.0) and back.probes[1].screen is None
    wire = snapshot_to_wire(snap)
    wire["entities"][0]["skeleton_world"] = wire["entities"][0]["skeleton_world"][:-3]
    with pytest.raises(ValueError):
        snapshot_from_wire(wire, HUMAN_13)


# ---- transport ----

def serve(tmp_path, handler):
    stop = threading.Event()
    server = MailboxServer(tmp_path, handler)
    t = threading.Thread(target=server.serve_forever, args=(stop,), daemon=True)
    t.start()
    return stop, t


def test_call_roundtrip_error_and_many_sequential_calls(tmp_path):
    def handler(method, params):
        if method == "boom":
            raise RuntimeError("kaput")
        return {"echo": params, "method": method}

    stop, t = serve(tmp_path, handler)
    try:
        tr = FileMailboxTransport(tmp_path)
        for i in range(100):
            assert tr.call("ping", {"i": i}) == {"echo": {"i": i}, "method": "ping"}
        with pytest.raises(RemoteError) as e:
            tr.call("boom")
        assert "kaput" in str(e.value) and isinstance(e.value, AdapterError)
    finally:
        stop.set()
        t.join(2)


def test_timeout_when_nobody_answers(tmp_path):
    tr = FileMailboxTransport(tmp_path, default_timeout_s=0.2)
    t0 = time.monotonic()
    with pytest.raises(TransportTimeout) as e:
        tr.call("hello")
    assert 0.15 < time.monotonic() - t0 < 2 and str(tmp_path) in str(e.value)


def test_stale_and_half_written_responses_are_ignored(tmp_path):
    tr = FileMailboxTransport(tmp_path, default_timeout_s=3)
    stale = {"id": 1, "result": {"stale": True}}
    (tmp_path / "res.json").write_text(json.dumps(stale))        # response to some old request

    def late_writer():
        time.sleep(0.15)
        rid = json.loads((tmp_path / "req.json").read_text())["id"]
        (tmp_path / "res.json").write_text('{"id": %d, "res' % rid)   # half-written
        time.sleep(0.15)
        (tmp_path / "res.json").write_text(json.dumps({"id": rid, "result": {"ok": 1}}))

    th = threading.Thread(target=late_writer)
    th.start()
    assert tr.call("x") == {"ok": 1}
    th.join()


def test_server_rejects_wrong_protocol_version(tmp_path):
    srv = MailboxServer(tmp_path, lambda m, p: {})
    (tmp_path / "req.json").write_text(json.dumps({"v": 99, "id": 5, "method": "hello", "params": {}}))
    assert srv.poll_once()
    reply = json.loads((tmp_path / "res.json").read_text())
    assert reply["id"] == 5 and "protocol version" in reply["error"]["message"]
    assert not srv.poll_once()           # same id is not processed twice
