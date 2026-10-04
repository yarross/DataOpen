import threading
from pathlib import Path


from dataopen.adapters.mock import MockGameAdapter
from dataopen.adapters.mock.server import MockServerOptions, serve_mock
from dataopen.adapters.remote import RemoteGameAdapter, RemoteOptions
from dataopen.core.doctor import run_doctor
from dataopen.core.transport import FileMailboxTransport


class Game:
    def __init__(self, directory: Path, **opts):
        self.stop = threading.Event()
        self.t = threading.Thread(target=serve_mock, args=(directory, MockServerOptions(**opts), self.stop), daemon=True)
        self.t.start()

    def close(self):
        self.stop.set()
        self.t.join(3)


def doctor(tmp_path, **opts):
    mb = tmp_path / "mb"
    g = Game(mb, **opts)
    try:
        a = RemoteGameAdapter(FileMailboxTransport(mb, default_timeout_s=10), RemoteOptions())
        return run_doctor(a, tmp_path / "out", frames=6)
    finally:
        g.close()


def status(rep, name):
    return [c.status for c in rep.checks if c.name == name]


def test_healthy_mock_passes_everything_and_writes_overlays(tmp_path):
    rep = doctor(tmp_path)
    assert rep.ok, rep.render()
    for name in ("connect", "projection", "units", "joints", "bone_lengths", "left_right", "negative_frame"):
        assert "FAIL" not in status(rep, name) and status(rep, name), (name, rep.render())
    assert status(rep, "projection") == ["PASS"] and status(rep, "units") == ["PASS"]
    assert status(rep, "left_right") == ["PASS"], rep.render()          # really evaluated, not skipped
    out = tmp_path / "out" / "doctor"
    assert (out / "overlay_0.png").exists() and (out / "doctor_report.json").exists()
    assert "RESULT: ready to collect" in rep.render() and "throughput" in rep.render()


def test_centimeter_units_are_detected(tmp_path):
    rep = doctor(tmp_path, unit_scale=100.0)
    assert status(rep, "units") == ["FAIL"] and not rep.ok
    assert "CENTIMETERS" in rep.render() and "0.01" in rep.render()


def test_flipped_probe_is_reported_with_cause(tmp_path):
    rep = doctor(tmp_path, flip_probe_y=True)
    assert not rep.ok and "Y axis is flipped" in rep.render()


def test_fov_bug_is_reported_with_cause(tmp_path):
    rep = doctor(tmp_path, wrong_probe_fov=1.3)
    assert not rep.ok and "focal length differs" in rep.render()


def test_swapped_left_right_is_detected(tmp_path):
    rep = doctor(tmp_path, swap_lr=True)
    assert status(rep, "left_right") == ["FAIL"] and not rep.ok
    assert "SWAPPED" in rep.render()


def test_no_game_running_gives_actionable_connect_failure(tmp_path):
    a = RemoteGameAdapter(FileMailboxTransport(tmp_path / "empty", default_timeout_s=0.2), RemoteOptions(call_timeout_s=0.2))
    rep = run_doctor(a, tmp_path / "out")
    assert not rep.ok and rep.checks[0].name == "connect" and "mailbox" in rep.checks[0].hint


def test_doctor_works_on_a_direct_in_process_adapter(tmp_path):
    rep = run_doctor(MockGameAdapter(), tmp_path / "out", frames=4)
    assert rep.ok, rep.render()
    assert status(rep, "projection") == [] or status(rep, "projection") == ["PASS"]
