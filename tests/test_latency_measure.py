"""Measurements on the simulation (docs/LATENCY.md): the real C core behind a simulated mouse, the service trace, the stand's analysis."""
import shutil

import numpy as np
import pytest

from dataopen.assist.params import AscConfig
from dataopen.latency import analyze as AN
from dataopen.latency import measure as M
from dataopen.latency import report as RP
from dataopen.runtime.frames import QueueSource
from dataopen.ui.scene import Det, UiSceneBuilder
from dataopen.ui.service import UiService
from dataopen.ui.sim import GEOM_1080P

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")
Z = np.zeros((640, 640, 3), np.uint8)


# ------------------------------------------------------------------------------------------------ the input path
@needs_cc
def test_the_core_adds_no_queueing_to_a_report_on_either_speed():
    for speed in ("FS", "HS"):
        r = M.core_report_delay(speed, 1, n=300)
        assert r["delay_ms"]["n"] == 300 and r["delay_ms"]["max"] == 0.0, speed       # every report answered in the step it arrived in


@needs_cc
def test_the_rig_stamps_the_reports_it_was_given_and_ignores_the_rest():
    r, _, _ = M._rig("steady")
    r.report_delay_us.clear()
    r.move(1, 0)
    r.run(3)                                             # the report is handled in the step it was put in: delay 0
    r.mouse.push_in(r.mouse.motion_ep, r.mouse.pack(0, 1, 0))  # not through move(): no stamp, and still no failure
    r.run(2)
    assert r.report_delay_us == [0, 0]
    r.unplug()
    assert r._enq == {}


def test_two_poll_grids_in_cascade_add_about_half_the_interval_on_average_and_one_at_most():
    for hz, mean in ((125, 4.0), (500, 1.0), (1000, 0.5), (8000, 0.0625)):
        p = M.poll_resample(hz)
        assert p["added_ms"]["mean"] == pytest.approx(mean, rel=0.1), hz
        assert p["added_ms"]["max"] <= 1000.0 / hz + 1e-9 and p["added_ms"]["min"] >= -1e-9, hz
    fast = M.poll_resample(1000, bridge_poll_hz=8000)["added_ms"]["mean"]
    # a bridge that polls the mouse faster than its own interval gives most of it back
    assert fast < M.poll_resample(1000)["added_ms"]["mean"] / 2


@needs_cc
def test_the_tremor_stage_delays_a_deliberate_reach_and_only_for_a_profile_that_has_it():
    runs = {(r["persona"], a): r for r, a in zip(RP._chain_runs(), (0.0, 8.0, 0.0, 0.0))}
    t = runs[("tremor", 0.0)]
    # most at the start of the reach, less later
    assert t["tremor_filter"] and t["delay_ms"][0.1] > t["delay_ms"][0.5] > t["delay_ms"][0.9] > 0.0
    assert 5.0 < t["delay_ms"][0.5] < 40.0 and t["delay_ms"][0.1] < 80.0
    assert 0.0 < t["attenuation"] < 0.3 and t["max_deficit_counts"] > 0
    for who in ("overshooter", "steady"):
        r = runs[(who, 0.0)]
        assert not r["tremor_filter"] and all(v == 0.0 for v in r["delay_ms"].values()) and r["attenuation"] == 0.0


@needs_cc
def test_a_scene_keeps_its_help_until_the_core_ttl_and_then_gives_none():
    runs = {r["age_ms"]: r for r in RP._scene_runs()}
    fresh, fifty, ninety, old = runs[0], runs[50], runs[90], runs[110]
    assert fresh["min_k"] < 0.5 and fresh["ticks_braked"] > 100                        # the target is braked for
    # a 50 ms old scene helps as much (the pointer's path is subtracted)
    assert fifty["min_k"] == pytest.approx(fresh["min_k"], abs=0.05)
    assert fifty["end_counts"] == pytest.approx(fresh["end_counts"], abs=15)
    assert fresh["min_k"] <= ninety["min_k"] < 0.6 and ninety["ticks_braked"] > 100    # near the TTL: a little weaker, still helps
    assert old["min_k"] == 1.0 and old["ticks_braked"] == 0 and old["end_counts"] == 700   # older than the TTL: no help at all
    assert fresh["ttl_ms"] == 100
    for r in runs.values():
        # K is slewed: no faster than slew_per_s (7.2 against 6 at the TTL edge)
        assert r["max_fall_per_s"] <= AscConfig().slew_per_s * 1.25
    assert fresh["first_brake_ms"] is not None and old["first_brake_ms"] is None


# ------------------------------------------------------------------------------------------------ the service trace
class Stub:
    def __init__(self):
        self.seen = []

    def detect(self, f):
        self.seen.append(f.frame_id)
        return [Det("button", 0.9, (100, 100, 164, 164))]


def _service(**kw):
    src, rel = QueueSource(4), []
    det = Stub()
    svc = UiService(src, det, UiSceneBuilder(GEOM_1080P), lambda s, f: None, clock=lambda: 5000, **kw)
    for i in range(4):
        src.push(Z, i, i * 1000, {"ready_us": i * 1000 + 100}, on_release=lambda i=i: rel.append(i))
    return svc, det, rel


def test_by_default_the_service_takes_the_oldest_frame_as_before():
    svc, det, rel = _service()
    assert svc.step(0.0) and det.seen == [0] and rel == [0]
    assert svc.trace.age_ms == [5.0] and svc.trace.ready_ms == [4.9]


def test_latest_only_takes_the_newest_and_lets_go_of_the_rest():
    svc, det, rel = _service(latest_only=True)
    assert svc.step(0.0) and det.seen == [3] and sorted(rel) == [0, 1, 2, 3]
    assert svc.trace.age_ms == [2.0] and svc.frames == 1 and svc.errors == 0


def test_a_frame_older_than_the_limit_is_dropped_unseen_and_counted():
    svc, det, rel = _service(max_age_ms=4.0)
    assert svc.step(0.0) and svc.step(0.0)               # frames 0 (5 ms old) and 1 (4 ms old, not above the limit)
    assert det.seen == [1] and svc.trace.stale_dropped == 1 and rel == [0, 1]
    s = svc.trace.summary()
    assert s["age_ms"]["n"] == 1 and s["detect_ms"]["n"] == 1 and s["update_ms"]["n"] == 1 and s["age_ms"]["max"] == 4.0


# ------------------------------------------------------------------------------------------------ the stand's analysis
def test_a_capture_with_a_known_delay_is_recovered(tmp_path):
    f = tmp_path / "c.csv"
    AN.synth_capture(f, n=300, mean_ms=12.0, jitter_ms=3.0, seed=2)
    r = AN.analyze_file(f, unit="ms")
    assert r["pairs"] == 300 and r["missed"] == 0 and r["orphan_responses"] == 0
    assert r["mean"] == pytest.approx(12.0, abs=0.6) and 9.0 <= r["min"] and r["max"] <= 15.0 and r["p99"] <= 15.0
    assert sum(r["histogram_1ms"].values()) == 300


def test_an_unanswered_stimulus_is_a_miss_and_a_late_answer_is_not_paired(tmp_path):
    f = tmp_path / "c.csv"
    AN.synth_capture(f, n=100, mean_ms=10.0, jitter_ms=1.0, miss=0.2, seed=3)
    r = AN.analyze_file(f, unit="ms")
    assert r["missed"] > 5 and r["pairs"] + r["missed"] == 100
    d, missed, orphans = AN.pair_edges([0.0, 100.0], [250.0], window_ms=100.0)
    assert d == [] and missed == 2 and orphans == 1          # 250 ms is later than the window after 100 ms and after 0 ms
    d, missed, orphans = AN.pair_edges([0.0, 100.0], [5.0, 105.0, 110.0])
    assert d == [5.0, 5.0] and missed == 0 and orphans == 1  # the second answer to the same stimulus belongs to nobody


def test_a_camera_capture_is_in_frames_and_carries_its_resolution(tmp_path):
    f = tmp_path / "cam.csv"
    f.write_text("channel,t\nstim,10\nresp,22\nstim,100\nresp,113\n", encoding="utf-8")
    r = AN.analyze_file(f, unit="frames", fps=1000.0)
    assert r["pairs"] == 2 and r["mean"] == pytest.approx(12.5) and r["resolution_ms"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        AN.analyze_file(f, unit="frames")                    # frames without a rate mean nothing


def test_a_malformed_capture_is_refused_with_the_line(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("channel,t\nstim,1\nwho,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.csv:3"):
        AN.read_edges(f, "ms")
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="channel,t"):
        AN.read_edges(f, "ms")
    with pytest.raises(ValueError):
        AN.read_edges(f, "years")
