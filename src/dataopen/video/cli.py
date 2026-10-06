"""`dataopen video ...`: timing and tap feasibility of display modes, frame-preparation benchmark, EDID inspection, capture simulation."""

from __future__ import annotations

import json
from pathlib import Path

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4
MODES = ["1920x1080@60", "1920x1080@144", "1920x1080@240", "2560x1440@144", "2560x1440@240", "3840x2160@60"]


def _modes(a) -> int:
    from .timing import Roi, data_age_us, mode, tap_supported

    rows = []
    for name in a.mode or MODES:
        m = mode(name)
        h_ok, h_why = tap_supported(m, "hdmi")
        d_ok, d_why = tap_supported(m, "dp")
        roi = Roi(*(a.roi or (0, 0, m.h_active, m.v_active)))
        age = data_age_us(m, roi, tail_us=a.tail_us)
        rows.append(
            {
                "mode": name,
                "timing": m.source,
                "pclk_mhz": round(m.pclk_hz / 1e6, 2),
                "line_us": round(m.line_time_us, 2),
                "frame_ms": round(m.frame_time_us / 1000, 3),
                "hdmi_tap": h_ok,
                "dp_tap": d_ok,
                "hdmi_note": h_why,
                "dp_note": d_why,
                "ready_ms": round(age["ready_us"] / 1000, 3),
                "content_age_ms": round(age["age_of_content_us"] / 1000, 3),
            }
        )
    if a.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    print(
        f"{'mode':16s} {'pclk MHz':>9s} {'line us':>8s} {'frame ms':>9s}  {'HDMI tap':8s} {'DP tap':6s} {'ready ms':>9s} {'content age ms':>15s}"  # noqa: E501
    )
    for r in rows:
        print(
            f"{r['mode']:16s} {r['pclk_mhz']:9.2f} {r['line_us']:8.2f} {r['frame_ms']:9.3f}  {'yes' if r['hdmi_tap'] else 'NO':8s} "
            f"{'yes' if r['dp_tap'] else 'NO':6s} {r['ready_ms']:9.3f} {r['content_age_ms']:15.3f}"
        )
    print(
        "(ready = the ROI's last line has arrived + processing; content age = how old the middle of the ROI is by then. Timing is an estimate: the EDID's own timing is the truth.)"  # noqa: E501
    )
    return EXIT_OK


def _bench(a) -> int:
    import numpy as np

    from .prep import FramePrep
    from .sim import rgb_to_packed422
    from .timing import mode

    m = mode(a.mode)
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (m.v_active, m.h_active, 3), dtype=np.uint8)
    crop = tuple(a.roi) if a.roi else None
    print(
        f"{a.mode}, crop {crop or 'full frame'} -> {a.out}x{a.out}  (host C, -O2; the SoC would use its scaler: this is the reference core)"
    )
    for fmt in ("rgb24", "uyvy"):
        src = rgb if fmt == "rgb24" else rgb_to_packed422(rgb)
        fp = FramePrep(m.h_active, m.v_active, fmt, crop, a.out, limited=(fmt == "uyvy"))
        out = np.empty((a.out, a.out, 3), np.uint8)
        tot, last = [], []
        for _ in range(a.repeat):
            _, t, ln = fp.frame(src, out)
            tot.append(t)
            last.append(ln)
        print(
            f"  {fmt:6s} whole frame {np.median(tot) / 1e6:7.2f} ms (p50)   last line alone {np.median(last) / 1e3:7.1f} us (p50) {np.percentile(last, 99) / 1e3:7.1f} us (p99)"  # noqa: E501
            f"   geometry {fp.geometry.content_w}x{fp.geometry.content_h} at ({fp.geometry.pad_x},{fp.geometry.pad_y})"
        )
    return EXIT_OK


def _edid(a) -> int:
    from . import edid as E

    data = Path(a.file).read_bytes()
    if len(data) >= 256 and data[:1] in (b"0",) and all(c in b"0123456789abcdefABCDEF \n" for c in data):
        data = bytes.fromhex(data.decode())
    try:
        info = E.parse(data)
    except E.EdidError as e:
        print(f"error: {e}")
        return EXIT_FAILED
    print(
        f"{info.manufacturer} product {info.product_code:#06x}  EDID {info.version[0]}.{info.version[1]}  extensions {info.n_ext}  HDMI {info.hdmi}"  # noqa: E501
        + (f"  max TMDS {info.max_tmds_hz / 1e6:.0f} MHz" if info.max_tmds_hz else "")
    )
    from .timing import tap_supported

    for m in E.required_tap_modes(info):
        ok_h, _ = tap_supported(m, "hdmi")
        ok_d, _ = tap_supported(m, "dp")
        print(
            f"  {m.name:16s} {m.pclk_hz / 1e6:8.2f} MHz  {m.refresh_hz:7.2f} Hz   HDMI tap {'yes' if ok_h else 'NO'}  DP tap {'yes' if ok_d else 'NO'}"  # noqa: E501
        )
    for w in info.warnings:
        print(f"  warning: {w}")
    return EXIT_OK


def _simulate(a) -> int:
    import numpy as np

    from .sim import Rect, SyntheticScreen, VideoPath
    from .timing import mode

    m = mode(a.mode)
    crop = tuple(a.roi) if a.roi else None
    vp = VideoPath(m, "rgb24", crop)
    scr = SyntheticScreen(m.h_active, m.v_active, [Rect(m.h_active // 2 - 60, m.v_active // 2 - 40, 120, 80, (230, 40, 40))])
    img = scr.render()
    t0 = 1_000_000
    for k in range(a.frames):
        scr.rects[0].x += 7
        vp.push(scr.render(), t0 + int(k * m.frame_time_us))
    g = vp.geometry
    f = vp.source.get(0.1)
    cx, cy = scr.rects[0].center
    ix, iy = g.to_input(cx, cy)
    print(f"{a.mode}: crop {crop or 'full'} -> geometry {g}")
    print(
        f"frames delivered {vp.source.stats()}, last frame ts {f.ts_us} us (middle of the ROI on the wire), age of content when ready {f.meta['age_of_content_us'] / 1000:.2f} ms"  # noqa: E501
    )
    print(
        f"the red rectangle's centre ({cx:.0f},{cy:.0f}) on the screen is ({ix:.1f},{iy:.1f}) in the detector input; back to the screen: {tuple(round(v, 1) for v in g.to_screen(ix, iy))}"  # noqa: E501
    )
    del img, np
    return EXIT_OK


def _build(a) -> int:
    from .prep import build_library

    print(build_library(a.out))
    return EXIT_OK


def register(sub) -> None:
    s = sub.add_parser(
        "video", help="video path: timing / tap feasibility, frame preparation bench, EDID, capture simulation (docs/VIDEO.md)"
    )
    ss = s.add_subparsers(dest="video_cmd", required=True)
    a = ss.add_parser("modes", help="timing, link budget and the capture timeline of display modes")
    a.add_argument("mode", nargs="*")
    a.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    a.add_argument("--tail-us", type=float, default=0.0, help="processing after the last needed line")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=_modes)
    b = ss.add_parser("bench", help="time of the C frame-preparation core (host numbers)")
    b.add_argument("--mode", default="1920x1080@144")
    b.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    b.add_argument("--out", type=int, default=640)
    b.add_argument("--repeat", type=int, default=20)
    b.set_defaults(fn=_bench)
    c = ss.add_parser("edid", help="what a monitor's EDID tells the tap it has to follow")
    c.add_argument("file")
    c.set_defaults(fn=_edid)
    d = ss.add_parser("simulate", help="synthetic screen through the preparation pipeline into the runtime's frame source")
    d.add_argument("--mode", default="1920x1080@144")
    d.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    d.add_argument("--frames", type=int, default=4)
    d.set_defaults(fn=_simulate)
    e = ss.add_parser("build", help="compile the C frame-preparation core")
    e.add_argument("--out")
    e.set_defaults(fn=_build)
