import json
import threading
from pathlib import Path

import pytest

from dataopen.adapters.mock import MockGameAdapter
from dataopen.adapters.mock.server import MockServerOptions, serve_mock
from dataopen.adapters.remote import RemoteGameAdapter, RemoteOptions
from dataopen.core.calibration import ProbeResult, diagnose
from dataopen.core.interfaces import AdapterError
from dataopen.core.orchestrator import CalibrationError, DatasetOrchestrator, SessionAborted, SessionConfig
from dataopen.core.transport import FileMailboxTransport


class Game:
    def __init__(self, directory: Path, **opts):
        self.stop = threading.Event()
        self.t = threading.Thread(target=serve_mock, args=(directory, MockServerOptions(**opts), self.stop), daemon=True)
        self.t.start()

    def close(self):
        self.stop.set()
        self.t.join(3)


def remote(mailbox) -> RemoteGameAdapter:
    a = RemoteGameAdapter(FileMailboxTransport(mailbox, default_timeout_s=10), RemoteOptions())
    a.connect()
    return a


def all_records(root: Path):
    out = []
    for p in sorted((root / "annotations").glob("*.jsonl")):
        out += [json.loads(line) for line in p.read_text().splitlines()]
    return out


# ---- calibration: known adapter bugs must be caught and explained ----

@pytest.mark.parametrize("opts,needle", [
    ({"flip_probe_y": True}, "Y axis is flipped"),
    ({"wrong_probe_fov": 1.25}, "focal length differs"),
])
def test_calibration_error_names_the_likely_cause(tmp_path, opts, needle):
    mb = tmp_path / "mb"
    g = Game(mb, **opts)
    try:
        orch = DatasetOrchestrator(remote(mb), SessionConfig(tmp_path / "ds", seed=1, target_frames=30, frames_per_scene=10))
        with pytest.raises(CalibrationError, match=needle):
            orch.run()
        rep = json.loads((tmp_path / "ds" / "report.json").read_text())
        assert rep["stop_reason"] == "aborted" and rep["accepted"] == 0 and needle in rep["error"]
    finally:
        g.close()


def test_correct_probes_pass_and_are_recorded(tmp_path):
    mb = tmp_path / "mb"
    g = Game(mb)
    try:
        rep = DatasetOrchestrator(remote(mb), SessionConfig(tmp_path / "ds", seed=1, target_frames=20,
                                                            frames_per_scene=10)).run()
        assert rep.probe_checked >= 20 and rep.probe_failed == 0
        recs = all_records(tmp_path / "ds")
        assert all(r["meta"].get("probe_max_err_px", 0) < 1e-3 for r in recs)
    finally:
        g.close()


def test_diagnose_unknown_and_too_few_points():
    assert "too few" in diagnose(ProbeResult(), 320, 240)
    pairs = [((10, 10), (200, 5)), ((100, 80), (3, 220)), ((200, 200), (150, 90))]
    assert "no simple transform" in diagnose(ProbeResult(compared=3, pairs=pairs), 320, 240)


# ---- control flow ----

def test_request_stop_ends_gracefully_and_flushes(tmp_path):
    orch = DatasetOrchestrator(MockGameAdapter(), SessionConfig(tmp_path / "ds", seed=1, target_frames=10_000,
                                                                frames_per_scene=10))
    threading.Timer(0.3, orch.request_stop).start()
    rep = orch.run()
    assert rep.stop_reason == "interrupted" and 0 < rep.accepted < 10_000
    assert len(all_records(tmp_path / "ds")) == rep.accepted
    assert len(list((tmp_path / "ds").glob("images/*/*.png"))) == rep.accepted
    assert (tmp_path / "ds" / "DATASET_CARD.json").exists()


def test_shards_take_disjoint_scenes_and_are_reproducible(tmp_path):
    scenes = []
    for i in range(3):
        cfg = SessionConfig(tmp_path / f"s{i}", seed=9, target_frames=25, frames_per_scene=5, shard_index=i, shard_count=3)
        DatasetOrchestrator(MockGameAdapter(), cfg).run()
        s = {r["scene_index"] for r in all_records(tmp_path / f"s{i}")}
        assert s and all(x % 3 == i for x in s)
        scenes.append(s)
    assert not (scenes[0] & scenes[1]) and not (scenes[1] & scenes[2]) and not (scenes[0] & scenes[2])
    names = [p.name for i in range(3) for p in (tmp_path / f"s{i}").glob("images/*/*.png")]
    assert len(names) == len(set(names))                      # frame ids never collide across shards


def test_bad_shard_config_rejected(tmp_path):
    with pytest.raises(ValueError):
        SessionConfig(tmp_path, shard_index=3, shard_count=3)


class FlakyAdapter(MockGameAdapter):
    """Fails on the Nth capture of the session, once."""

    def __init__(self, fail_on: int):
        super().__init__()
        self.n, self.fail_on = 0, fail_on
        real = self.capture.capture

        def capture(req):
            self.n += 1
            if self.n == self.fail_on:
                raise AdapterError("engine hiccup")
            return real(req)
        self.capture.capture = capture


def test_frames_committed_before_a_mid_scene_failure_keep_their_labels(tmp_path):
    cfg = SessionConfig(tmp_path / "ds", seed=4, target_frames=15, frames_per_scene=10)
    rep = DatasetOrchestrator(FlakyAdapter(fail_on=6), cfg).run()
    recs = all_records(tmp_path / "ds")
    assert rep.adapter_errors == 1 and rep.accepted == 15
    assert len(recs) == rep.accepted == len(list((tmp_path / "ds").glob("images/*/*.png")))   # no orphan images


class DeadAdapter(MockGameAdapter):
    def __init__(self):
        super().__init__()
        self.capture.capture = lambda req: (_ for _ in ()).throw(AdapterError("game crashed"))


def test_persistent_adapter_failure_aborts_with_the_reason(tmp_path):
    cfg = SessionConfig(tmp_path / "ds", seed=4, target_frames=15, max_adapter_error_streak=4, adapter_restart_after=2,
                        max_attempt_factor=100)
    with pytest.raises(SessionAborted, match="game crashed"):
        DatasetOrchestrator(DeadAdapter(), cfg).run()
    assert json.loads((tmp_path / "ds" / "report.json").read_text())["stop_reason"] == "aborted"


def test_dataset_card_contents(tmp_path):
    cfg = SessionConfig(tmp_path / "ds", seed=4, target_frames=12, frames_per_scene=6,
                        provenance={"assets": "own CC0 models"})
    DatasetOrchestrator(MockGameAdapter(), cfg).run()
    card = json.loads((tmp_path / "ds" / "DATASET_CARD.json").read_text())
    assert card["game"]["name"] == "mock" and card["counts"]["frames"] == 12
    assert card["skeleton"]["flip_idx"][2] == 3 and card["provenance"]["assets"] == "own CC0 models"
    assert len(card["session"]["config_hash"]) == 16 and card["visibility_flags"]["2"] == "visible"
    # without a provenance note the card says so loudly
    cfg2 = SessionConfig(tmp_path / "ds2", seed=4, target_frames=4, frames_per_scene=4)
    DatasetOrchestrator(MockGameAdapter(), cfg2).run()
    assert "UNSPECIFIED" in json.loads((tmp_path / "ds2" / "DATASET_CARD.json").read_text())["provenance"]["assets"]
