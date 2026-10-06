"""The video path: timing and link budgets, EDID rules, input autodetect, the streaming frame-preparation core against a numpy reference,
geometry back to screen pixels, delivery to the existing runtime."""

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from dataopen.runtime.backends import ScriptedBackend
from dataopen.runtime.loop import InferenceRuntime
from dataopen.video import detect, edid
from dataopen.video.prep import BT601, BT709, PAD, FramePrep, lib, prep_reference
from dataopen.video.sim import Rect, SyntheticScreen, VideoPath, rgb_to_packed422
from dataopen.video.timing import Roi, cta_mode, cvt_rb, data_age_us, dp_link_capacity_gbps, mode, tap_supported

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


# ---------------------------------------------------------------- timing and link budgets
def test_cta_timings_are_the_standard_ones():
    m = cta_mode(16)
    assert (m.h_total, m.v_total, m.pclk_hz) == (2200, 1125, 148_500_000)
    assert (
        m.refresh_hz == pytest.approx(60.0, abs=1e-6)
        and m.line_time_us == pytest.approx(14.815, abs=1e-3)
        and m.frame_time_us == pytest.approx(16666.7, abs=0.1)
    )
    assert cta_mode(97).pclk_hz == 594_000_000 and mode("3840x2160@60").source == "cta"


def test_cvt_rb_hits_the_requested_refresh_with_a_quarter_mhz_clock_grid():
    for w, h, hz in ((1920, 1080, 144), (1920, 1080, 240), (2560, 1440, 144), (2560, 1440, 165)):
        m = cvt_rb(w, h, hz)
        assert m.h_total == w + 160 and m.pclk_hz % 250_000 == 0
        assert m.refresh_hz == pytest.approx(hz, abs=0.2)
        assert (m.v_total - h) * m.line_time_us >= 459  # the 460 us minimum vertical blanking


def test_the_capture_timeline_is_set_by_where_the_roi_lies_in_the_frame():
    m = mode("1920x1080@144")
    full = data_age_us(m, Roi(0, 0, 1920, 1080))
    centre = data_age_us(m, Roi(640, 220, 640, 640))
    assert full["last_row_us"] == pytest.approx(m.active_time_us)
    assert centre["ready_us"] < full["ready_us"] and centre["age_of_content_us"] < full["age_of_content_us"]
    assert centre["age_of_content_us"] == pytest.approx(
        320 * m.line_time_us + 0.0, rel=1e-6
    )  # the middle of a 640-line ROI is 320 lines old when its last line lands
    top = data_age_us(m, Roi(0, 0, 1920, 270))
    assert top["ready_us"] == pytest.approx(
        270 * m.line_time_us
    )  # a ROI at the top of the screen is ready a quarter of the way through the frame
    assert data_age_us(m, Roi(0, 0, 1920, 1080), tail_us=20)["ready_us"] == pytest.approx(full["ready_us"] + 20)


def test_link_budgets_say_which_modes_a_commodity_tap_can_follow():
    assert tap_supported(mode("1920x1080@144"), "hdmi")[0] and tap_supported(mode("3840x2160@60"), "hdmi")[0]
    assert not tap_supported(mode("3840x2160@60"), "hdmi", bpc=10)[0]  # 742 MHz: FRL / 4:2:0 territory
    assert tap_supported(mode("2560x1440@144"), "dp")[0] and not tap_supported(mode("2560x1440@240"), "dp")[0]
    assert tap_supported(mode("2560x1440@240"), "dp", dp_rate="HBR3")[0]
    assert dp_link_capacity_gbps("HBR2") == pytest.approx(17.28)
    with pytest.raises(ValueError):
        tap_supported(mode("1920x1080@60"), "vga")


# ---------------------------------------------------------------- EDID: byte for byte, nothing invented
def _monitor_edid(tmds=600):
    return edid.build(
        [mode("1920x1080@144"), mode("1920x1080@60")], manufacturer="ACM", product=0x4321, hdmi_max_tmds_mhz=tmds, vics=(16, 97, 63)
    )


def test_edid_roundtrip_and_validation():
    b = _monitor_edid()
    info = edid.parse(b)
    assert (info.manufacturer, info.product_code, info.n_ext) == ("ACM", 0x4321, 1)
    assert info.hdmi and info.max_tmds_hz == 600_000_000 and not info.warnings
    names = {m.name for m in info.modes}
    assert {"1920x1080@60", "3840x2160@60", "1920x1080@120"} <= names and info.preferred.pclk_hz == mode(
        "1920x1080@144"
    ).pclk_hz // 10_000 * 10_000
    assert len(edid.required_tap_modes(info)) >= 4
    bad = bytearray(b)
    bad[20] ^= 1
    with pytest.raises(edid.EdidError):
        edid.parse(bytes(bad))
    with pytest.raises(edid.EdidError):
        edid.parse(b"\x01" + b[1:])
    with pytest.raises(edid.EdidError):
        edid.parse(b[:100])
    ext_bad = bytearray(b)
    ext_bad[130] ^= 0xFF
    assert edid.parse(bytes(ext_bad)).warnings  # a damaged extension is reported, not trusted


def test_the_gpu_reads_exactly_the_monitors_bytes_and_the_device_never_invents_an_edid():
    b = _monitor_edid()
    p = edid.EdidProxy()
    assert p.read(0, 0, 128) is None  # no monitor: HPD low, DDC NACK, nothing synthesized
    p.monitor(1_000, b)
    assert p.read(1_000, 0, 256) == b and p.read(2_000, 128, 128) == b[128:]  # byte for byte, extension included
    assert p.tick(2_000).hpd_to_gpu is True
    other = edid.build([mode("2560x1440@144")], manufacturer="XYZ", product=1)
    p.monitor(5_000_000, other)
    assert p.read(5_050_000, 0, 128) is None and p.tick(5_050_000).hpd_to_gpu is False  # replaced: HPD low for the re-read pulse
    assert p.read(5_100_000, 0, 256) == other  # then the NEW monitor's bytes, again exactly
    p.monitor(9_000_000, None)
    assert p.read(9_000_001, 0, 128) is None


def test_the_tap_must_follow_every_mode_the_edid_lists():
    info = edid.parse(_monitor_edid())
    for m in edid.required_tap_modes(info):
        assert tap_supported(m, "hdmi")[0], m.name  # this monitor is within a TMDS tap's reach
    big = edid.parse(edid.build([mode("2560x1440@144")], hdmi_max_tmds_mhz=600))
    assert not all(
        tap_supported(m, "hdmi")[0] for m in edid.required_tap_modes(big)
    )  # and this one is not: the device must say so, not pretend


# ---------------------------------------------------------------- input autodetect
def test_autodetect_locks_after_a_stable_link_and_ignores_flaps():
    d = detect.InputDetector()
    up, down = detect.Link(True, True), detect.Link(False, False)
    t = 0
    for _ in range(10):  # a flaky cable: up 50 ms, down 50 ms
        d.update(t, up, down)
        d.update(t + 50_000, down, down)
        t += 100_000
    assert d.active is None
    for k in range(20):
        d.update(t + k * 10_000, up, down)
    assert d.active == "hdmi"
    d.update(t + 400_000, down, up)  # switches to DP only when HDMI is gone AND DP is stable
    assert d.active == "hdmi"


def test_autodetect_rides_out_a_mode_switch_but_not_a_lost_cable_and_prefers_the_current_input():
    d = detect.InputDetector()
    up, off = detect.Link(True, True), detect.Link(False, False)
    for k in range(30):
        d.update(k * 10_000, off, up)
    assert d.active == "dp"
    t = 300_000
    d.update(t, off, detect.Link(True, False))  # link retrains for a resolution change
    d.update(t + 300_000, off, detect.Link(True, False))
    assert d.active == "dp"  # < 400 ms: still the same input
    d.update(t + 800_000, off, detect.Link(True, False))
    assert d.active is None  # a lost link is given up
    t2 = t + 1_000_000
    for k in range(30):
        d.update(t2 + k * 10_000, up, up)
    assert d.conflict and d.active in ("hdmi", "dp")
    cur = d.active
    for k in range(30):
        d.update(t2 + 400_000 + k * 10_000, up, up)
    assert d.active == cur  # two live inputs: the current one is kept, no flapping


# ---------------------------------------------------------------- frame preparation
def _scene(w=1920, h=1080, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3)).astype(np.float64)
    img = np.kron(base, np.ones((8, 8, 1)))[:h, :w]  # blocky + smooth enough for 4:2:2
    return img.astype(np.uint8)


@needs_cc
@pytest.mark.parametrize("crop", [None, (320, 220, 1280, 640), (481, 141, 959, 799)])
def test_c_core_matches_the_float_reference_for_rgb(crop):
    src = _scene(seed=1)
    for fmt, limited in (("rgb24", False), ("rgb24", True), ("bgr24", False)):
        fp = FramePrep(1920, 1080, fmt, crop, 640, limited=limited)
        out, _, _ = fp.frame(src)
        ref = prep_reference(src, fp)
        d = np.abs(out.astype(int) - ref.astype(int))
        assert d.max() <= 2 and d.mean() < 0.15, (fmt, limited, crop, d.max())  # Q16 footprint edges


@needs_cc
@pytest.mark.parametrize("fmt", ["uyvy", "yuyv"])
@pytest.mark.parametrize("matrix,limited", [(BT709, True), (BT601, True), (BT709, False), (BT601, False)])
def test_c_core_matches_the_float_reference_for_packed_422(fmt, matrix, limited):
    packed = rgb_to_packed422(_scene(960, 540, seed=2), fmt, limited, matrix)
    fp = FramePrep(960, 540, fmt, (101, 50, 800, 400), 640, limited=limited, matrix=matrix)
    out, _, _ = fp.frame(packed)
    ref = prep_reference(packed, fp)
    d = np.abs(out.astype(int) - ref.astype(int))
    assert d.max() <= 2 and d.mean() < 0.3


@needs_cc
def test_colour_conversion_round_trips_the_pixels_the_bridge_chip_sent():
    rgb = _scene(640, 360, seed=3)
    for fmt, limited, matrix in (("uyvy", True, BT709), ("yuyv", False, BT601)):
        fp = FramePrep(640, 360, fmt, None, 640, limited=limited, matrix=matrix)
        out, _, _ = fp.frame(rgb_to_packed422(rgb, fmt, limited, matrix))
        assert (
            np.abs(out[140 - 140 + 140 : 140 + 360].astype(int) - rgb.astype(int)).mean() < 6
        )  # chroma is subsampled at the source: close, not equal


@needs_cc
def test_a_constant_picture_stays_exactly_constant_and_the_padding_is_the_detectors():
    for size, crop in (((1920, 1080), None), ((1280, 720), (7, 5, 1111, 701)), ((3840, 2160), (0, 0, 3840, 2160))):
        src = np.full((size[1], size[0], 3), (37, 201, 90), np.uint8)
        fp = FramePrep(size[0], size[1], "rgb24", crop, 640)
        out, _, _ = fp.frame(src)
        g = fp.geometry
        inner = out[g.pad_y : g.pad_y + g.content_h, g.pad_x : g.pad_x + g.content_w]
        assert (inner == np.array([37, 201, 90])).all()
        mask = np.ones(out.shape[:2], bool)
        mask[g.pad_y : g.pad_y + g.content_h, g.pad_x : g.pad_x + g.content_w] = False
        assert (out[mask] == PAD).all()


@needs_cc
def test_geometry_is_the_detectors_letterbox_and_never_upscales():
    g = FramePrep(1920, 1080).geometry
    assert (g.content_w, g.content_h, g.pad_x, g.pad_y) == (640, 360, 0, 140)  # what quality.evaluators.decode.letterbox does for 1920x1080
    g = FramePrep(2560, 1440).geometry
    assert (g.content_w, g.content_h, g.pad_y) == (640, 360, 140)
    g = FramePrep(1920, 1080, crop=(800, 400, 320, 240)).geometry  # smaller than the detector input: placed 1:1, padded
    assert (g.content_w, g.content_h, g.pad_x, g.pad_y, g.scale) == (320, 240, 160, 200, 1.0)
    with pytest.raises(ValueError):
        FramePrep(1920, 1080, crop=(1800, 0, 400, 100))  # outside the frame
    with pytest.raises(ValueError):
        FramePrep(1920, 1080, crop=(0, 0, 0, 100))


@needs_cc
def test_streaming_gives_the_same_picture_and_finished_rows_never_change():
    src = _scene(seed=4)
    crop = (200, 100, 1500, 700)
    whole = FramePrep(1920, 1080, "rgb24", crop)
    ref, _, _ = whole.frame(src)
    fp = FramePrep(1920, 1080, "rgb24", crop)
    out = np.zeros((640, 640, 3), np.uint8)
    fp.begin(out)
    snap = None
    for y in range(0, fp.last_row + 1):
        fp.line(y, src[y])
        if y == 100 + 700 * 6 // 10:
            snap = out.copy()
        assert fp.complete == (y == fp.last_row)
    assert (out == ref).all()
    g = fp.geometry
    done = g.pad_y + int(g.content_h * 0.55)
    assert (snap[:done] == ref[:done]).all()  # rows finished mid-frame are already final
    assert not (snap[g.pad_y + g.content_h - 5 : g.pad_y + g.content_h] == ref[g.pad_y + g.content_h - 5 : g.pad_y + g.content_h]).all()


@needs_cc
def test_lines_outside_the_crop_are_never_needed_and_the_state_is_constant_size():
    src = _scene(seed=5)
    fp = FramePrep(1920, 1080, "rgb24", (0, 100, 1920, 400))
    out = np.zeros((640, 640, 3), np.uint8)
    fp.begin(out)
    for y in range(100, 500):  # lines 0..99 and 500..1079 are never delivered
        fp.line(y, src[y])
    assert fp.complete and fp.last_row == 499
    assert lib().vp_sizeof() < 200_000  # accumulators only, never a frame buffer


@needs_cc
def test_the_work_left_after_the_last_needed_line_is_one_line_not_a_frame():
    src = _scene(seed=6)
    fp = FramePrep(1920, 1080, "rgb24")
    out = np.empty((640, 640, 3), np.uint8)
    tot, last = [], []
    for _ in range(15):
        _, t, ln = fp.frame(src, out)
        tot.append(t)
        last.append(ln)
    assert np.median(last) * 100 < np.median(tot)  # one line of ~1080 is under 1/100 of the frame (measured ~0.1%)
    assert np.median(last) < 500_000  # and well under half a millisecond, even on a loaded CI host


# ---------------------------------------------------------------- geometry: pixels on the screen <-> pixels in the detector input
@needs_cc
@pytest.mark.parametrize(
    "w,h,crop",
    [(1920, 1080, None), (1920, 1080, (640, 220, 640, 640)), (2560, 1440, (400, 200, 1700, 900)), (1280, 720, (0, 0, 1280, 720))],
)
def test_a_rectangle_on_the_screen_is_where_the_geometry_says_in_the_detector_input(w, h, crop):
    rects = [Rect(w // 2 + 31, h // 2 - 17, 90, 60, (250, 20, 20)), Rect(w // 2 - 200, h // 2 + 60, 80, 80, (20, 250, 20))]
    img = SyntheticScreen(w, h, rects).render()
    fp = FramePrep(w, h, "rgb24", crop)
    out, _, _ = fp.frame(img)
    g = fp.geometry
    for r, ch in zip(rects, (0, 1)):
        cx, cy = r.center
        if crop and not (crop[0] <= cx < crop[0] + crop[2] and crop[1] <= cy < crop[1] + crop[3]):
            continue
        ix, iy = g.to_input(cx, cy)
        px = out[int(round(iy)), int(round(ix))]
        assert px[ch] > 200 and px[(ch + 1) % 3] < 60  # the object's colour is exactly there
        sx, sy = g.to_screen(ix, iy)
        assert abs(sx - cx) < 1e-6 and abs(sy - cy) < 1e-6


# ---------------------------------------------------------------- into the existing runtime
@needs_cc
def test_prepared_frames_reach_the_runtime_with_the_scanout_time_and_the_geometry():
    m = mode("1920x1080@144")
    crop = (640, 220, 640, 640)
    vp = VideoPath(m, "rgb24", crop)
    rect = Rect(930, 500, 100, 80, (240, 30, 30))
    seen = []

    def fn(frame):
        g = frame.meta["geometry"]
        ix, iy = g.to_input(*rect.center)
        seen.append((frame.frame_id, frame.ts_us, frame.array[int(round(iy)), int(round(ix))].copy(), frame.meta["age_of_content_us"]))
        return []

    rt = InferenceRuntime(vp.source, ScriptedBackend(fn=fn), None, "latest", warmup=0)
    rt.start()
    t0 = 5_000_000
    scr = SyntheticScreen(1920, 1080, [rect])
    for k in range(6):
        vp.push(scr.render(), t0 + int(k * m.frame_time_us))
        time.sleep(0.01)
    vp.source.close()
    assert rt.wait_idle(5)
    rt.stop()
    assert len(seen) >= 3
    for fid, ts, px, age in seen:
        assert px[0] > 200 and px[1] < 60
        assert ts == pytest.approx(t0 + fid * m.frame_time_us + (220 + 320) * m.line_time_us, abs=2)  # the middle of the ROI on the wire
        assert 1500 < age < 2500  # microseconds: ~1.9 ms at 1080p144 for this ROI


@needs_cc
def test_a_full_queue_drops_the_frame_it_never_blocks_the_capture_path():
    vp = VideoPath(mode("1920x1080@60"), "rgb24", (0, 0, 640, 640), capacity=2)
    img = np.zeros((1080, 1920, 3), np.uint8)
    assert [vp.push(img, k * 16667) for k in range(4)] == [True, True, False, False]
    assert vp.source.stats()["producer_dropped"] == 2


# ---------------------------------------------------------------- hygiene
@needs_cc
def test_the_prep_core_is_clean_c99_with_no_libc_beyond_mem(tmp_path):
    cc = shutil.which("gcc") or shutil.which("cc")
    csrc = Path(__file__).resolve().parents[1] / "src" / "dataopen" / "video" / "csrc"
    o = tmp_path / "vp.o"
    r = subprocess.run(
        [
            cc,
            "-std=c99",
            "-pedantic",
            "-Wall",
            "-Wextra",
            "-Wconversion",
            "-Wshadow",
            "-Werror",
            "-O2",
            "-fno-stack-protector",
            "-U_FORTIFY_SOURCE",
            f"-I{csrc}",
            "-c",
            str(csrc / "video_prep.c"),
            "-o",
            str(o),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[:600]
    und = {ln.split()[-1] for ln in subprocess.run(["nm", "-u", str(o)], capture_output=True, text=True).stdout.splitlines() if ln.strip()}
    assert und <= {"memcpy", "memset", "memmove"}


@needs_cc
def test_random_configurations_and_lines_under_the_sanitizers(tmp_path):
    cc = shutil.which("gcc") or shutil.which("cc")
    csrc = Path(__file__).resolve().parents[1] / "src" / "dataopen" / "video" / "csrc"
    exe = tmp_path / "vfuzz"
    r = subprocess.run(
        [
            cc,
            "-O1",
            "-g",
            "-std=gnu99",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            f"-I{csrc}",
            str(Path(__file__).parent / "helpers" / "video_fuzz.c"),
            str(csrc / "video_prep.c"),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 and ("sanitize" in r.stderr or "libasan" in r.stderr):
        pytest.skip("no sanitizer runtime")
    assert r.returncode == 0, r.stderr[:800]
    for seed in (1, 2):
        p = subprocess.run([str(exe), "400", str(seed)], capture_output=True, text=True, timeout=600)
        assert p.returncode == 0 and "fuzz ok" in p.stdout, p.stderr[-800:]


def test_the_cli_prints_the_tables(capsys):
    from dataopen.cli import build_parser

    for argv in (["video", "modes"], ["video", "simulate", "--frames", "2"]):
        a = build_parser().parse_args(argv)
        assert a.fn(a) == 0
    out = capsys.readouterr().out
    assert "HDMI tap" in out and "detector input" in out
