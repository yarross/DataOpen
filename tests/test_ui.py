"""The assistive UI-element detector: the wall between the two detector worlds, synthetic screens, data preparation, loss and decode, the tracker and the
scene for the ASC / HID bridge, clock sync, the service, export, and the closed loop with a detector-derived scene."""  # noqa: E501

import shutil
import time

import numpy as np
import pytest

from dataopen.assist.params import AscParams
from dataopen.assist.sim_user import PERSONAS, build_profile, run_trial
from dataopen.assist.tremor import TremorParams
from dataopen.bridge.clocksync import ClockSync
from dataopen.bridge.sim import BridgeAssist
from dataopen.runtime.frames import QueueSource
from dataopen.ui import synth
from dataopen.ui.data import SynthUi, prepare
from dataopen.ui.evaluate import average_precision, evaluate
from dataopen.ui.infer import Detection, decode
from dataopen.ui.taxonomy import STRIDES
from dataopen.ui.scene import Det, SceneConfig, UiSceneBuilder
from dataopen.ui.service import UiService
from dataopen.ui.sim import GEOM_1080P, StandInDetector, UiBridgeAssist
from dataopen.ui.taxonomy import ID, NAMES, TARGET_NAMES, NotAUiModel, require_ui_layout
from dataopen.video.prep import Geometry

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


# ---------------------------------------------------------------- the wall between the two worlds
def test_only_interface_classes_exist_in_the_taxonomy_and_a_pose_model_is_refused():
    assert NAMES == ("button", "icon", "text_field", "menu_item", "toggle", "tab", "window_control", "cursor")
    assert TARGET_NAMES == set(NAMES) - {"cursor"}
    require_ui_layout(NAMES)
    require_ui_layout(["button", "icon"])
    for bad in (["player_ct", "player_t"], ["button", "person"], ["enemy"], []):
        with pytest.raises(NotAUiModel):
            require_ui_layout(bad)
    with pytest.raises(NotAUiModel):
        require_ui_layout(NAMES, n_keypoints=12)  # classes fine, but it has keypoints: that is the pose model


def test_the_scene_builder_refuses_a_model_that_is_not_a_ui_model():
    with pytest.raises(NotAUiModel):
        UiSceneBuilder(GEOM_1080P, class_names=("player_ct", "player_t"))
    with pytest.raises(NotAUiModel):
        UiSceneBuilder(GEOM_1080P, n_keypoints=12)


def test_a_detection_of_a_class_the_model_does_not_have_is_an_error_not_a_silent_target():
    b = UiSceneBuilder(GEOM_1080P, class_names=("button", "cursor"))
    with pytest.raises(ValueError):
        b.update(0, [Det("icon", 0.9, (0, 0, 50, 50))])


# ---------------------------------------------------------------- synthetic screens
def test_screens_are_deterministic_labelled_inside_the_image_and_varied():
    a = synth.render_screen(np.random.default_rng(3), (1920, 1080))
    b = synth.render_screen(np.random.default_rng(3), (1920, 1080))
    assert (a.image == b.image).all() and [e.box for e in a.elements] == [e.box for e in b.elements]
    seen, fams, n_el, n_cur = set(), set(), 0, 0
    for i in range(60):
        s = synth.render_screen(np.random.default_rng(100 + i), None)
        h, w = s.image.shape[:2]
        for e in s.elements:
            x0, y0, x1, y1 = e.box
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and 0 <= e.cls < len(NAMES)
            assert x1 - x0 >= 6 and y1 - y0 >= 6
            seen.add(e.cls)
        fams.add(s.meta["family"])
        n_el += len(s.elements)
        n_cur += s.cursor is not None
    assert seen == set(range(len(NAMES))) and len(fams) >= 4 and n_el / 60 > 6 and n_cur >= 40


def test_heldout_families_are_really_not_in_the_training_families():
    assert set(synth.HELDOUT_FAMILIES).isdisjoint(synth.TRAIN_FAMILIES)
    fams = {synth.render_screen(np.random.default_rng(i), (640, 360), synth.HELDOUT_FAMILIES).meta["family"] for i in range(10)}
    assert fams <= set(synth.HELDOUT_FAMILIES)


def test_the_renderer_is_fast_enough_to_feed_training():
    t0 = time.time()
    for i in range(6):
        synth.render_screen(np.random.default_rng(i), (1920, 1080))
    assert (time.time() - t0) / 6 < 0.4


@needs_cc
def test_a_sample_goes_through_the_deployment_preparation_and_the_boxes_follow():
    ds = SynthUi(4, seed=1, train=False)
    for i in range(4):
        x, b, cls, g, meta = ds[i]
        assert x.shape == (640, 640, 3) and len(b) == len(cls)
        assert (b[:, 0] >= -1e-3).all() and (b[:, 2] <= 640 + 1e-3).all() and (b[:, 1] >= -1e-3).all() and (b[:, 3] <= 640 + 1e-3).all()
    rng = np.random.default_rng(5)
    scr = synth.render_screen(rng, (1920, 1080), synth.TRAIN_FAMILIES)
    boxes = np.array([e.box for e in scr.elements], np.float32)
    x, b, g, keep = prepare(scr.image, boxes, None)
    big = sorted((i for i in range(len(boxes)) if keep[i]), key=lambda i: -(boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
    assert big
    i = big[0]
    sx, sy = (boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2
    ix, iy = g.to_input(sx, sy)
    assert abs(ix - (b[i, 0] + b[i, 2]) / 2) < 1e-3 and abs(iy - (b[i, 1] + b[i, 3]) / 2) < 1e-3
    xr, br, gr, kr = prepare(scr.image, boxes, (300, 200, 900, 600))  # a ROI: boxes cut by it are clipped, mostly-outside ones dropped
    assert gr.crop_w == 900 and kr.sum() < len(boxes) and (br[kr] >= -1e-3).all()


def test_real_screenshots_in_yolo_format_load_through_the_same_path_and_a_pose_class_list_is_refused(tmp_path):
    from PIL import Image

    from dataopen.ui.data import RealScreens

    rng = np.random.default_rng(2)
    for k in range(3):
        s = synth.render_screen(rng, (1280, 720))
        Image.fromarray(s.image).save(tmp_path / f"s{k}.png")
        h, w = s.image.shape[:2]
        rows = [
            f"{e.cls} {(e.box[0] + e.box[2]) / 2 / w:.6f} {(e.box[1] + e.box[3]) / 2 / h:.6f} {(e.box[2] - e.box[0]) / w:.6f} {(e.box[3] - e.box[1]) / h:.6f}"  # noqa: E501
            for e in s.elements
        ]
        (tmp_path / f"s{k}.txt").write_text("\n".join(rows))
    (tmp_path / "classes.txt").write_text("\n".join(NAMES))
    pytest.importorskip("dataopen.video.prep")
    if shutil.which("gcc") or shutil.which("cc"):
        ds = RealScreens(tmp_path, train=False)
        assert len(ds) == 3
        x, b, cls, g, meta = ds[0]
        assert x.shape == (640, 640, 3) and len(b) == len(cls) > 0
    (tmp_path / "classes.txt").write_text("player_ct\nplayer_t")
    with pytest.raises(NotAUiModel):
        RealScreens(tmp_path)


# ---------------------------------------------------------------- model, loss, decode, metrics
def _planted(n_cls=len(NAMES), plants=()):
    levels = [np.full((n_cls + 4, 640 // s, 640 // s), -9.0, np.float32) for s in STRIDES]
    for lv, cell, c, score, ltrb in plants:
        s = STRIDES[lv]
        gx, gy = cell
        levels[lv][c, gy, gx] = float(np.log(score / (1 - score)))
        levels[lv][n_cls:, gy, gx] = np.log(np.array(ltrb, np.float32) / s)
    return levels


def test_decode_finds_planted_boxes_and_merges_duplicates_only_within_a_class():
    levels = _planted(
        plants=[
            (0, (10, 10), 0, 0.9, (8, 6, 8, 6)),
            (0, (11, 10), 0, 0.8, (16, 6, 0.5 * 8, 6)),
            (0, (10, 11), 3, 0.7, (8, 6, 8, 6)),
            (2, (5, 5), 2, 0.95, (64, 16, 64, 16)),
        ]
    )
    dets = decode(levels, len(NAMES), conf=0.3, nms_iou=0.5)
    by = {(d.cls): d for d in dets}
    assert len(dets) == 3 and set(by) == {0, 2, 3}  # the second button is the same object; the menu_item stays (other class)
    assert by[0].score == pytest.approx(0.9, abs=1e-3)
    assert by[0].box == pytest.approx((80 + 4 - 8, 80 + 4 - 6, 80 + 4 + 8, 80 + 4 + 6), abs=0.01)
    assert by[2].box == pytest.approx((5.5 * 32 - 64, 5.5 * 32 - 16, 5.5 * 32 + 64, 5.5 * 32 + 16), abs=0.01)


def test_average_precision_and_the_report():
    assert average_precision(np.array([0.9, 0.8]), np.array([1, 1]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([0.9, 0.8]), np.array([0, 0]), 2) == 0.0
    assert average_precision(np.array([]), np.array([]), 0) != average_precision(np.array([]), np.array([]), 0)  # NaN: no ground truth
    gts = [(np.array([[10.0, 10, 60, 40], [100, 100, 112, 119]]), np.array([ID["button"], ID["cursor"]]))]
    perfect = [[Detection(ID["button"], 0.9, (10, 10, 60, 40)), Detection(ID["cursor"], 0.8, (101, 100, 112, 119))]]
    r = evaluate(perfect, gts)
    assert r.recall_targets == 1.0 and r.cursor_found == 1.0 and r.cursor_err_px == pytest.approx(1.0) and r.fp_per_screen == 0
    r = evaluate([[Detection(ID["button"], 0.9, (300, 300, 340, 330))]], gts)
    assert r.recall_targets == 0.0 and r.fp_per_screen == 1.0 and r.cursor_found == 0.0
    r = evaluate([[Detection(ID["icon"], 0.9, (10, 10, 60, 40))]], gts)  # the right place, the wrong class
    assert (
        r.recall_targets == 0.0 and r.fp_per_screen == 0.0
    )  # a class-aware miss, but not a false element: for the ASC the place is what counts


@needs_cc
def _det(cls, box, conf=0.9):
    x0, y0 = GEOM_1080P.to_input(box[0], box[1])
    x1, y1 = GEOM_1080P.to_input(box[2], box[3])
    return Det(cls, conf, (x0, y0, x1, y1))


def _cursor(x, y):
    return _det("cursor", (x, y, x + 11, y + 19), 0.9)


def test_an_element_becomes_a_target_after_two_frames_and_a_one_frame_false_alarm_never_does():
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0))
    btn = (800, 400, 900, 450)
    s = b.update(0, [_det("button", btn), _cursor(100, 100)])
    assert s.targets == () and s.cursor == pytest.approx((100, 100), abs=0.1)  # seen once: not yet
    s = b.update(7000, [_det("button", btn), _cursor(110, 100), _det("icon", (300, 300, 340, 340), 0.6)])
    assert [t.cls for t in s.targets] == ["button"]
    s = b.update(14000, [_det("button", btn), _cursor(120, 100)])
    assert [t.cls for t in s.targets] == ["button"] and all(t.cls != "icon" for t in s.targets)  # the icon was seen in one frame only


def test_tracks_survive_a_few_missed_frames_and_are_dropped_after_many_and_ids_are_stable():
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0, drop_misses=3))
    btn = (800, 400, 900, 450)
    for k in range(3):
        s = b.update(k * 7000, [_det("button", btn), _cursor(50, 50)])
    tid = s.targets[0].id
    s = b.update(30000, [_cursor(50, 50)])  # one miss: the track holds (but is not offered after one miss)
    s = b.update(37000, [_det("button", btn), _cursor(50, 50)])
    assert s.targets and s.targets[0].id == tid
    for k in range(5):
        s = b.update(50000 + k * 7000, [_cursor(50, 50)])
    assert s.targets == () and len(b.tracks) == 0
    s = b.update(90000, [_det("button", btn), _cursor(50, 50)])
    s = b.update(97000, [_det("button", btn), _cursor(50, 50)])
    assert s.targets[0].id != tid  # a new track, a new id


def test_appearance_time_is_none_for_what_was_there_from_the_start_and_the_capture_time_for_what_pops_up():
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=3))
    a = (800, 400, 900, 450)
    for k in range(6):
        s = b.update(k * 7000, [_det("button", a), _cursor(50, 50)])
    assert s.targets[0].appeared_us is None
    pop = (1000, 600, 1100, 650)
    for k in range(6, 9):
        s = b.update(k * 7000, [_det("button", a), _det("menu_item", pop), _cursor(50, 50)])
    t = {x.cls: x for x in s.targets}
    assert t["menu_item"].appeared_us == 6 * 7000 and t["button"].appeared_us is None


def test_boxes_are_smoothed_and_small_or_unsure_detections_are_ignored():
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0, smooth=0.5))
    rng = np.random.default_rng(1)
    base = np.array([800.0, 400, 900, 450])
    outs = []
    for k in range(40):
        s = b.update(
            k * 7000,
            [
                _det("button", tuple(base + rng.normal(0, 2.0, 4))),
                _cursor(50, 50),
                _det("icon", (10, 10, 12, 12)),
                _det("tab", (500, 500, 600, 540), 0.2),
            ],
        )
        if s.targets:
            outs.append(s.targets[0].box)
    assert [t.cls for t in s.targets] == ["button"]  # the 2 px "icon" and the 0.2-confidence tab never show up
    sd = np.std(np.array(outs)[10:], axis=0).mean()
    assert sd < 2.0 * 0.9  # steadier than the raw 2 px jitter


def test_the_object_for_the_asc_is_the_nearest_point_of_the_box_with_a_radius_that_follows_the_size():
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0, radius_frac=0.35))
    wide = (800, 400, 1200, 440)  # a text field: 400 x 40
    for k in range(3):
        s = b.update(k * 7000, [_det("text_field", wide), _cursor(500, 420)])
    objs = b.objects(s)
    assert len(objs) == 1
    o = objs[0]
    assert (o.x, o.y) == pytest.approx((800.0, 420.0), abs=0.6)  # the near edge, level with the pointer
    assert o.radius == pytest.approx(0.35 * 40, abs=0.5)
    near = b.nearest(s, 1000, 420)  # pointer inside the box: the object is right under it
    assert (near.x, near.y) == pytest.approx((1000, 420), abs=0.6)


def test_the_scene_for_the_bridge_is_relative_to_the_pointer_in_bridge_time_and_empty_without_a_pointer():
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0))
    boxes = [
        (800, 400, 900, 450),
        (300, 800, 360, 860),
        (1500, 100, 1600, 150),
        (1700, 900, 1760, 960),
        (100, 900, 160, 960),
        (900, 900, 960, 960),
        (1200, 300, 1260, 360),
    ]
    for k in range(3):
        s = b.update(1000 + k, [_det("button", x) for x in boxes] + [_cursor(850, 420)])
    tcap, objs = b.bridge_scene(s, to_bridge_time=lambda t: t + 5_000_000)
    assert tcap == 1002 + 5_000_000 and 1 <= len(objs) <= 5
    d = [np.hypot(o.x, o.y) for o in objs]
    assert (
        d == sorted(d) and objs[0].x == pytest.approx(0.0, abs=1.0) and objs[0].y == pytest.approx(0.0, abs=1.0)
    )  # pointer inside the first button
    s = b.update(2000, [_det("button", boxes[0])])  # the pointer was not found in this frame
    assert b.bridge_scene(s)[1] == []


def test_a_mode_change_resets_the_tracks():
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0))
    for k in range(3):
        b.update(k, [_det("button", (800, 400, 900, 450)), _cursor(0, 0)])
    assert b.tracks
    b.set_geometry(Geometry(0, 0, 2560, 1440, 640, 360, 0, 140, 640, 640))
    assert not b.tracks and b.frames == 0


# ---------------------------------------------------------------- clock sync
def test_clock_sync_recovers_the_offset_and_prefers_the_exchange_with_the_least_round_trip():
    cs = ClockSync()
    off = 123_456
    rng = np.random.default_rng(0)
    t = 10_000_000
    for _ in range(20):
        up, down = int(rng.integers(100, 3000)), int(rng.integers(100, 3000))
        m_send = t
        b_rx = m_send + up + off
        b_tx = b_rx + 50
        m_recv = b_tx - off + down
        cs.add(m_send, b_rx, b_tx, m_recv)
        t += 10_000
    assert abs(cs.offset_us - off) <= cs.uncertainty_us + 1
    assert cs.uncertainty_us < 1500
    assert cs.to_bridge(1_000_000) == pytest.approx(1_000_000 + off, abs=cs.uncertainty_us + 1)
    with pytest.raises(RuntimeError):
        ClockSync().to_bridge(1)


# ---------------------------------------------------------------- service
class _StubDetector:
    def __init__(self, fail_at=()):
        self.n, self.fail_at = 0, set(fail_at)

    def detect(self, frame):
        self.n += 1
        if self.n in self.fail_at:
            raise RuntimeError("boom")
        return [_det("button", (800, 400, 900, 450)), _cursor(850, 420)]


def test_the_service_builds_snapshots_releases_every_frame_and_a_failing_detector_gives_no_snapshot():
    src = QueueSource(8)
    released, snaps = [], []
    for i in range(6):
        src.push(np.zeros((640, 640, 3), np.uint8), i, 1000 * i, {"geometry": GEOM_1080P}, on_release=lambda i=i: released.append(i))
    svc = UiService(
        src,
        _StubDetector(fail_at=(4,)),
        UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0)),
        lambda s, f: snaps.append((f.frame_id, s)),
        latest_only=False,
        max_age_ms=None,  # the frames carry virtual timestamps and every one of them is wanted
    )
    while svc.step(0.01):
        pass
    assert sorted(released) == list(range(6))
    assert svc.frames == 5 and svc.errors == 1 and "boom" in svc.last_error
    assert [i for i, _ in snaps] == [0, 1, 2, 4, 5]
    assert snaps[-1][1].targets and snaps[-1][1].cursor is not None


def test_the_service_follows_a_mode_change_announced_in_the_frame_metadata():
    src = QueueSource(8)
    g2 = Geometry(0, 0, 2560, 1440, 640, 360, 0, 140, 640, 640)
    snaps = []
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0))
    svc = UiService(src, _StubDetector(), b, lambda s, f: snaps.append(s), latest_only=False, max_age_ms=None)
    src.push(np.zeros((640, 640, 3), np.uint8), 0, 0, {"geometry": GEOM_1080P})
    src.push(np.zeros((640, 640, 3), np.uint8), 1, 7000, {"geometry": g2})
    while svc.step(0.01):
        pass
    assert b.g == g2 and len(snaps) == 2


# ---------------------------------------------------------------- export and the ONNX detector
OX, OY = 300.0, 540.0


class _Shift(UiBridgeAssist):
    def tick(self, t_us, dx, dy, px, py, obj):
        return super().tick(t_us, dx, dy, px + OX, py + OY, obj)


def _prof(name):
    v = build_profile(PERSONAS[name], seed=1)
    return AscParams.from_view(v), TremorParams.from_view(v)


@needs_cc
def test_the_detector_derived_scene_gives_the_overshooter_nearly_the_help_the_exact_scene_gives():
    ap, tp = _prof("overshooter")
    box = (600 + OX - 32, OY - 32, 600 + OX + 32, OY + 32)
    n = 12

    def rate(mk):
        tr = [run_trial(PERSONAS["overshooter"], None if mk is None else mk(), np.random.default_rng(i)) for i in range(n)]
        return float(np.mean([t.overshoot_px > 30 for t in tr])), tr

    none, _ = rate(None)
    ideal, _ = rate(lambda: BridgeAssist(ap, tp))
    ui, tr = rate(lambda: _Shift(ap, tp, box, fps=60, latency_ms=20, detector=StandInDetector(GEOM_1080P, 1.5, 0.05, 0.1, 0)))
    assert none > 0.9 and ideal < none - 0.3 and ui < none - 0.3 and ui <= ideal + 0.2
    assert all(t.amp_violations == 0 for t in tr)


@needs_cc
def test_with_no_pointer_found_the_bridge_gets_no_scene_and_gives_no_help():
    ap, tp = _prof("overshooter")
    box = (600 + OX - 32, OY - 32, 600 + OX + 32, OY + 32)
    a = _Shift(ap, tp, box, fps=60, latency_ms=20, detector=StandInDetector(GEOM_1080P, 0.0, 0.0, 0.0, 0))
    a.det.detect = lambda targets, cur: [
        d for d in StandInDetector.detect(a.det, targets, cur) if d.cls != "cursor"
    ]  # the pointer is never found
    tr = [
        run_trial(
            PERSONAS["overshooter"],
            a if i == 0 else _Shift(ap, tp, box, detector=StandInDetector(GEOM_1080P, 0, 0, 0, 0)),
            np.random.default_rng(i),
        )
        for i in range(1)
    ]
    assert a.scenes_sent == 0 and tr[0].k_mean_moving == pytest.approx(1.0, abs=0.02)


def test_the_cli_renders_screens_and_runs_the_scene_simulation(capsys, tmp_path):
    from dataopen.cli import build_parser

    for argv in (["ui", "synth", "--out", str(tmp_path), "-n", "2", "--boxes"],):
        a = build_parser().parse_args(argv)
        assert a.fn(a) == 0
    assert len(list(tmp_path.glob("screen_*.png"))) == 2
    assert "classes:" in capsys.readouterr().out


@needs_cc
def test_frames_to_service_to_scene_frames_to_the_real_bridge_core_brake_the_approach():
    """Video-path style frames -> UiService (stub detector standing in for the network) -> BridgeScenePublisher -> BridgeLink -> bridge core."""  # noqa: E501
    from dataopen.assist.fixed import FixedParams
    from dataopen.assist.tremor_fixed import FixedTremorParams
    from dataopen.bridge.sim import Rig
    from dataopen.bridge.sim_usb import SimMouse
    from dataopen.ui.service import scene_service

    ap, tp = _prof("overshooter")
    r = Rig(SimMouse("m16"), asc=FixedParams.from_params(ap), tremor=FixedTremorParams.from_params(tp))
    assert r.engage_fast()
    base_x, base_y = 300.0, 540.0  # where the pointer starts on the screen
    target = (900.0, 508.0, 964.0, 572.0)

    class Gt:
        def detect(self, frame):
            st = r.b.status(frame.ts_us)
            return [_det("button", target), _cursor(base_x + st.cum_x, base_y + st.cum_y)]

    src = QueueSource(4)
    builder = UiSceneBuilder(GEOM_1080P)  # default warm-up: what is there from the start did not 'appear'  # noqa: E501
    svc, pub = scene_service(src, Gt(), builder, lambda f: r.module.send(r.t, [f]), now=lambda: r.t)  # v1 defaults, virtual time
    ks = []
    carry = 0.0
    for k in range(500):
        tau = min(k / 350.0, 1.0)
        carry += 3.6 * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
        i = int(round(carry))
        carry -= i
        r.move(i, 0)
        r.step()
        if k % 8 == 0:  # a frame every 8 ms (125 fps), captured 5 ms ago
            src.push(np.zeros((640, 640, 3), np.uint8), k, r.t - 5000, {"geometry": GEOM_1080P})
            svc.step(0.0)
        ks.append(r.b.status(r.t).k_q16 / 65536)
    assert pub.sent > 20 and svc.errors == 0
    assert min(ks) < 0.5  # the bridge braked near the button
    s = r.status()
    assert s.invariant_viol == 0 and s.scene_n >= 1
