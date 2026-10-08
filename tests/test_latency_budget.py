"""The latency budget (docs/LATENCY.md): every number has a basis, the totals add up, and the numbers follow the code that owns them."""
import shutil

import pytest

from dataopen.assist.params import AscConfig
from dataopen.latency import budget as B
from dataopen.latency import measure as M
from dataopen.runtime.frames import QueueSource
from dataopen.ui.scene import SceneConfig

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")
SC = B.SCENARIOS


def ids(sc):
    return {s.id: s for s in B.stages(sc)}


def test_every_stage_of_every_scenario_names_its_basis_and_its_range_is_ordered():
    for sc in SC:
        for s in B.stages(sc):
            assert s.basis in B.BASES and s.floor in B.FLOORS and s.group in B.GROUPS, (sc.key, s.id)
            assert s.lo <= s.typ <= s.hi, (sc.key, s.id)
            assert s.ref and s.name, (sc.key, s.id)
    with pytest.raises(ValueError):
        B.Stage("X", "A", "bad", 2.0, 1.0, 3.0, "calc")
    with pytest.raises(ValueError):
        B.Stage("X", "A", "bad", 1.0, 1.0, 1.0, "guess")


def test_what_is_not_measured_is_not_called_measured():
    for sc in SC:
        s = ids(sc)
        assert {i for i, x in s.items() if x.basis == "sim"} <= {"A3", "A4"}  # only what measure.py really runs
        assert s["B7"].basis == "unmeasured"                                 # the SPI clock and the master's turn are defined nowhere
        assert s["B4"].basis == ("unmeasured" if sc.detector == "npu" else "host")
        assert s["A1"].basis == "estimate"                                   # no Cortex-M number exists


def test_the_totals_are_the_sums_of_the_stages():
    for sc in SC:
        for g in "ABC":
            ss = [s for s in B.stages(sc) if s.group == g]
            lo, typ, hi = B.total(sc, g)
            assert (lo, typ, hi) == pytest.approx((sum(s.lo for s in ss), sum(s.typ for s in ss), sum(s.hi for s in ss)))


def test_the_default_confirmation_is_the_trackers_not_a_copy():
    sc = B.BY_KEY["cpu-latest"]
    assert f"confirm_hits={SceneConfig().confirm_hits}" in ids(sc)["B6"].name
    assert "confirm_hits=1" in ids(B.with_changes(sc, confirm_hits=1))["B6"].name
    assert ids(B.with_changes(sc, confirm_hits=1))["B6"].typ == 0.0
    assert ids(B.with_changes(sc, confirm_hits=3))["B6"].typ > ids(sc)["B6"].typ


def test_the_frame_period_and_the_roi_come_from_the_video_timing():
    a, b = ids(B.BY_KEY["npu-latest"]), ids(B.with_changes(B.BY_KEY["npu-latest"], mode="1920x1080@60"))
    assert b["B1"].typ > 2 * a["B1"].typ                                  # 1080p60 scans a line in 14.8 us, 1080p144 in 6.0 us
    full = ids(B.with_changes(B.BY_KEY["npu-latest"], capture="frame"))["B1"]
    assert full.typ > a["B1"].typ                                         # waiting for the whole frame is later than for the 640x640 ROI


def test_the_queue_depth_is_read_from_the_source_class():
    assert B.queue_capacity() == 4 == QueueSource().capacity
    assert f"ёмкость {B.queue_capacity()}" in ids(B.BY_KEY["as-built"])["B3"].ref


@needs_cc
def test_the_scene_ttl_and_the_poll_limits_are_read_from_the_bridge_core():
    from dataopen.bridge.cbridge import CBridge

    c = CBridge().cfg
    cfg = B.bridge_cfg()
    assert cfg["scene_ttl_ms"] == c.scene_ttl_ms and cfg["slow_poll_us"] == c.slow_poll_us and cfg["split_poll_us"] == c.split_poll_us
    assert B.usb_mode(125) == "SLOW" and B.usb_mode(1000) == "DIRECT" and B.usb_mode(8000) == "SPLIT"
    assert B.summary(B.BY_KEY["npu-latest"])["ttl_ms"] == c.scene_ttl_ms


def test_the_input_path_stays_inside_the_target_and_the_tremor_filter_is_shown_apart():
    for sc in SC:
        s = B.summary(sc)
        if sc.mouse_hz >= 1000:
            assert s["A_pipeline"][1] <= 1.0 and s["A_pipeline"][2] <= 2.0, sc.key      # 1 kHz mouse: about half a ms, at most about one
        assert s["A_total"][1] > s["A_pipeline"][1]                                      # the filter is a separate, larger term
    assert B.summary(B.BY_KEY["slow-mouse"])["A_pipeline"][2] >= 8.0                      # a 125 Hz mouse: two 8 ms poll grids in cascade


def test_the_scene_path_misses_the_ttl_as_built_and_meets_it_with_latest_only():
    assert B.summary(B.BY_KEY["as-built"])["ttl_margin_ms"] < 0                           # the finding: FIFO(4) + CPU detector, worst case
    for k in ("cpu-latest", "npu-latest", "npu-latest-c1", "cpu-60", "npu-4k60"):
        assert B.summary(B.BY_KEY[k])["ttl_margin_ms"] > 0, k
    assert B.summary(B.BY_KEY["npu-latest-c1"])["B"][2] <= 12.0                           # the 10-12 ms target is reachable only here
    assert B.summary(B.BY_KEY["npu-latest"])["B"][2] > 12.0 and B.summary(B.BY_KEY["cpu-latest"])["B"][1] > 12.0
    for k in ("cpu-latest", "npu-latest", "npu-latest-c1"):
        assert B.summary(B.BY_KEY[k])["guard_margin_ms"] > 0


def test_the_policy_group_is_read_from_the_asc_configuration():
    a = AscConfig()
    c = ids(B.BY_KEY["as-built"])
    assert c["C1"].typ == a.on_ms and c["C2"].lo == a.t_min_ms
    assert c["C4"].hi == pytest.approx((1 - a.k_floor) / a.slew_per_s * 1000)
    assert c["C2"].typ > a.t_min_ms and c["C2"].typ == pytest.approx(225.3, abs=1.0)    # the 'tremor' persona's T_lo from build_profile


# ------------------------------------------------------------------------------------------------ the queue model
def test_the_queue_model_has_the_rules_of_the_real_source():
    q = QueueSource(4)
    import numpy as np

    z = np.zeros((2, 2, 3), np.uint8)
    pushed = [q.push(z, i, i * 1000) for i in range(6)]
    # a full queue drops the NEW frame; the reader gets the oldest
    assert pushed == [True] * 4 + [False] * 2 and q.get(0.0).frame_id == 0
    # a detector 100x slower than the frames: the queue stays full
    t = M.pipe_timeline(M.Pipe(1.0, 100.0, 0.0, "fifo", 4, 1))
    # the frame taken is always the oldest of four (3 periods old at least)
    assert min(t["taken_age"]) >= 3.0


def test_fifo_makes_the_picture_old_and_latest_only_does_not():
    fifo = M.pipe_timeline(M.Pipe(6.94, 11.0, 0.3, "fifo", 4, 2))
    late = M.pipe_timeline(M.Pipe(6.94, 11.0, 0.3, "latest", 4, 2))
    f, l = sum(fifo["taken_age"]) / len(fifo["taken_age"]), sum(late["taken_age"]) / len(late["taken_age"])
    assert f > 3 * 6.94 and l < 6.94                                               # 3-4 frame periods old against under one
    assert sum(fifo["total"]) / len(fifo["total"]) > 2 * sum(late["total"]) / len(late["total"])


def test_a_fast_detector_never_queues_and_the_confirmation_costs_whole_cycles():
    for hits, extra in ((1, 0.0), (2, 6.94), (3, 13.88)):
        t = M.pipe_timeline(M.Pipe(6.94, 3.0, 0.3, "fifo", 4, hits))
        assert max(t["wait"]) == pytest.approx(0.0) and max(t["confirm"]) == pytest.approx(extra, abs=0.01), hits
        assert max(t["total"]) == pytest.approx(3.3 + extra, abs=0.01)
    with pytest.raises(ValueError):
        M.pipe_timeline(M.Pipe(6.94, 3.0, policy="lifo"))


def test_the_test_points_name_the_interval_each_one_bounds():
    ids_ = [t.id for t in B.TESTPOINTS]
    assert len(set(ids_)) == len(ids_) and {"S0", "L1", "L4", "L5", "U1", "M1", "V1"} <= set(ids_)
    assert all(t.where and t.signal and t.bounds for t in B.TESTPOINTS)
