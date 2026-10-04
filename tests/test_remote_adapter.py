import json
import threading
from pathlib import Path

import numpy as np
import pytest

from dataopen.adapters.mock.server import MockGameServer, MockServerOptions, serve_mock
from dataopen.adapters.remote import RemoteGameAdapter, RemoteOptions, StaticGrabber
from dataopen.core.imageio import read_image_size, write_png
from dataopen.core.interfaces import AdapterError, Capability
from dataopen.core.orchestrator import DatasetOrchestrator, SessionConfig
from dataopen.core.transport import FileMailboxTransport


class Game:
    """Runs the mock game server in a thread for the duration of a test."""

    def __init__(self, directory: Path, **opts):
        self.stop = threading.Event()
        self.t = threading.Thread(target=serve_mock, args=(directory, MockServerOptions(**opts), self.stop), daemon=True)
        self.t.start()

    def close(self):
        self.stop.set()
        self.t.join(3)


@pytest.fixture
def mailbox(tmp_path):
    return tmp_path / "mbox"


def connect(mailbox, grabber=None, **opts) -> RemoteGameAdapter:
    a = RemoteGameAdapter(FileMailboxTransport(mailbox, default_timeout_s=10), RemoteOptions(**opts), grabber=grabber)
    a.connect()
    return a


def records(root: Path):
    out = []
    for split in ("train", "val"):
        p = root / "annotations" / f"{split}.jsonl"
        if p.exists():
            out += [json.loads(line) for line in p.read_text().splitlines()]
    return out


def test_hello_exposes_info_and_parameter_space(mailbox):
    g = Game(mailbox)
    try:
        a = connect(mailbox)
        assert a.info.name == "mock" and Capability.ENGINE_VISIBILITY in a.info.capabilities
        assert a.capture_mode == "engine" and a.info.image_size == (320, 240)
        assert "rig" in a.parameter_space().actor.params and "time_of_day" not in a.parameter_space().environment.params
        assert a.health()["ok"] and all(c["ok"] for c in a.selftest())
    finally:
        g.close()


def test_full_collection_through_the_wire_engine_images(mailbox, tmp_path):
    g = Game(mailbox)
    try:
        a = connect(mailbox)
        out = tmp_path / "ds"
        rep = DatasetOrchestrator(a, SessionConfig(out, seed=5, target_frames=40, frames_per_scene=8,
                                                   negative_ratio=0.2)).run()
        assert rep.accepted == 40 and rep.stop_reason == "target_reached"
        recs = records(out)
        assert len(recs) == 40
        pngs = sorted(out.glob("images/*/*.png"))
        assert len(pngs) == 40 and all(read_image_size(p) == (320, 240) for p in pngs)
        assert any(r["annotations"] for r in recs) and any(r["kind"] == "negative" for r in recs)
        for r in recs:
            if r["kind"] == "negative":
                assert r["annotations"] == []
    finally:
        g.close()


def test_host_capture_mode_uses_the_grabber_and_releases_the_game(mailbox, tmp_path):
    g = Game(mailbox, engine_images=False)
    try:
        frame = np.full((240, 320, 3), 77, dtype=np.uint8)
        a = connect(mailbox, grabber=StaticGrabber(frame), grab_delay_s=0)
        assert a.capture_mode == "host"
        out = tmp_path / "ds"
        rep = DatasetOrchestrator(a, SessionConfig(out, seed=2, target_frames=10, frames_per_scene=5)).run()
        assert rep.accepted == 10
        png = next(out.glob("images/*/*.png"))
        assert read_image_size(png) == (320, 240)
    finally:
        g.close()


def test_host_mode_without_a_grabber_and_wrong_grab_size_fail_loudly(mailbox, tmp_path):
    g = Game(mailbox, engine_images=False)
    try:
        with pytest.raises(AdapterError, match="ScreenGrabber"):
            connect(mailbox)
        a = connect(mailbox, grabber=StaticGrabber(np.zeros((100, 100, 3), np.uint8)), grab_delay_s=0)
        rep_cfg = SessionConfig(tmp_path / "ds", seed=2, target_frames=3, frames_per_scene=3,
                                max_attempt_factor=1.0)
        with pytest.raises(Exception, match="borderless windowed"):
            DatasetOrchestrator(a, rep_cfg).run()
    finally:
        g.close()


def test_engine_mode_rejected_if_mod_lacks_image_engine(mailbox):
    g = Game(mailbox, engine_images=False)
    try:
        with pytest.raises(AdapterError, match="image_engine"):
            connect(mailbox, capture_mode="engine")
    finally:
        g.close()


def test_schema_mismatch_is_reported_by_the_mod(tmp_path):
    g = MockGameServer()
    res = g.handle("hello", {"schema": {"keypoints": ["a", "b"]}})
    assert res["schema_errors"] and "mismatch" in res["schema_errors"][0]


def test_not_connected_adapter_gives_a_clear_error(tmp_path):
    a = RemoteGameAdapter(FileMailboxTransport(tmp_path))
    with pytest.raises(AdapterError, match="connect"):
        _ = a.info
    with pytest.raises(AdapterError, match="connect"):
        a.parameter_space()


def test_imageio_png_roundtrip_header(tmp_path):
    p = tmp_path / "x.png"
    write_png(p, np.random.default_rng(0).integers(0, 255, (17, 29, 3), dtype=np.uint8))
    assert read_image_size(p) == (29, 17)
