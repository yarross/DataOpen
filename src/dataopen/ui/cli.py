"""`dataopen ui ...`: the assistive UI-element detector (synthetic screens, training, evaluation, export, scene simulation). docs/UIDET.md."""  # noqa: E501

from __future__ import annotations

import json
from pathlib import Path

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4


def _synth(a) -> int:
    import numpy as np
    from PIL import Image, ImageDraw

    from . import synth
    from .taxonomy import NAMES

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = [(255, 0, 0), (0, 170, 0), (0, 0, 255), (255, 140, 0), (160, 0, 160), (0, 150, 150), (130, 130, 0), (255, 0, 255)]
    fam = synth.HELDOUT_FAMILIES if a.heldout else synth.TRAIN_FAMILIES
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        s = synth.render_screen(rng, None, fam)
        im = Image.fromarray(s.image)
        if a.boxes:
            d = ImageDraw.Draw(im)
            for e in s.elements:
                d.rectangle(e.box, outline=cols[e.cls], width=2)
        im.save(out / f"screen_{i:03d}.png")
        print(f"screen_{i:03d}.png  {s.meta['family']:9s} {s.meta['scene']:16s} {len(s.elements)} elements")
    print("classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(NAMES)))
    return EXIT_OK


def _train(a) -> int:
    from .train import train

    r = train(
        a.out, a.steps, a.batch, a.window, a.lr, a.width, a.neck, a.workers, a.seed, a.val, a.eval_every, a.real, a.real_share, a.resume
    )
    print(json.dumps(r, indent=2))
    return EXIT_OK


def _eval(a) -> int:
    from . import synth
    from .data import SynthUi
    from .evaluate import evaluate
    from .train import load, predict, report_text

    net = load(a.ckpt)
    for name, fam in (("in-distribution themes", synth.TRAIN_FAMILIES), ("HELD-OUT themes (retro, glass)", synth.HELDOUT_FAMILIES)):
        ds = SynthUi(a.n, a.seed + 777, fam, False)
        r = evaluate(*predict(net, ds, a.n))
        print(report_text(name, r))
    if a.real:
        from .data import RealScreens

        ds = RealScreens(a.real, False)
        r = evaluate(*predict(net, ds, len(ds)))
        print(report_text(f"REAL screenshots ({len(ds)})", r))
    return EXIT_OK


def _export(a) -> int:
    from .export import export_onnx
    from .train import load

    print(json.dumps(export_onnx(load(a.ckpt), a.out, uint8_nhwc=not a.float), indent=2))
    return EXIT_OK


def _bench(a) -> int:
    import time

    import numpy as np

    from ..runtime.frames import Frame
    from .service import OrtUiDetector

    det = OrtUiDetector(a.onnx, threads=a.threads)
    f = Frame(np.random.default_rng(0).integers(0, 256, (640, 640, 3), dtype=np.uint8), 0, 0)
    for _ in range(3):
        det.detect(f)
    t = []
    for _ in range(a.runs):
        t0 = time.perf_counter()
        det.detect(f)
        t.append((time.perf_counter() - t0) * 1000)
    print(
        f"ONNX Runtime CPU, {a.threads} threads, 640x640, run + decode: p50 {np.percentile(t, 50):.1f} ms  p95 {np.percentile(t, 95):.1f} ms (host numbers)"  # noqa: E501
    )
    return EXIT_OK


def _scene_sim(a) -> int:
    import numpy as np

    from ..assist.params import AscParams
    from ..assist.sim_user import PERSONAS, build_profile, run_trial
    from ..assist.tremor import TremorParams
    from ..bridge.sim import BridgeAssist
    from .sim import StandInDetector, GEOM_1080P, UiBridgeAssist

    OX, OY = 300.0, 540.0

    class Shift(UiBridgeAssist):
        def tick(self, t_us, dx, dy, px, py, obj):
            return super().tick(t_us, dx, dy, px + OX, py + OY, obj)

    box = (600 + OX - 32, OY - 32, 600 + OX + 32, OY + 32)
    for name in [a.persona] if a.persona != "all" else ["overshooter", "tremor"]:
        v = build_profile(PERSONAS[name], seed=1)
        ap, tp = AscParams.from_view(v), TremorParams.from_view(v)
        rows = {"no assistance": None, "ideal scene (object known exactly, every tick)": lambda: BridgeAssist(ap, tp)}  # noqa: B023
        for fps, lat in ((144, 8), (60, 20), (30, 40)):
            rows[f"detector scene {fps} fps, {lat} ms late"] = lambda fps=fps, lat=lat: Shift(
                ap, tp, box, fps=fps, latency_ms=lat, detector=StandInDetector(GEOM_1080P, a.jitter, a.miss, 0.1, 0)  # noqa: B023
            )
        print(name)
        for lab, mk in rows.items():
            tr = [run_trial(PERSONAS[name], None if mk is None else mk(), np.random.default_rng(i)) for i in range(a.trials)]
            print(
                f"  {lab:48s} overshoot {np.mean([t.overshoot_px > 30 for t in tr]):.2f}  t_acq {np.nanmean([t.t_acquire_ms for t in tr]):4.0f} ms  "  # noqa: E501
                f"hold {np.mean([t.hold_rms_px for t in tr if t.acquired]):5.1f} px  mean K {np.mean([t.k_mean_moving for t in tr]):.2f}"
            )
    return EXIT_OK


def register(sub) -> None:
    s = sub.add_parser("ui", help="assistive UI-element detector: synth / train / eval / export / bench / scene-sim (docs/UIDET.md)")
    ss = s.add_subparsers(dest="ui_cmd", required=True)
    a = ss.add_parser("synth", help="render labelled synthetic screens (preview)")
    a.add_argument("--out", required=True)
    a.add_argument("-n", type=int, default=8)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--heldout", action="store_true", help="the held-out theme families (retro, glass)")
    a.add_argument("--boxes", action="store_true", help="draw the labels")
    a.set_defaults(fn=_synth)
    t = ss.add_parser("train", help="train UiNet on synthetic screens (CPU-friendly), optionally mixed with real labelled screenshots")
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=2500)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--window", type=int, default=512)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--width", type=float, default=0.5)
    t.add_argument("--neck", type=int, default=48)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--val", type=int, default=96)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--real", help="folder of real screenshots + YOLO .txt labels + classes.txt")
    t.add_argument("--real-share", type=float, default=0.3)
    t.add_argument("--resume", help="a checkpoint to fine-tune from")
    t.set_defaults(fn=_train)
    e = ss.add_parser(
        "eval", help="mAP50 and the assistive numbers on in-distribution and held-out synthetic themes (and real screenshots)"
    )
    e.add_argument("--ckpt", required=True)
    e.add_argument("-n", type=int, default=200)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--real")
    e.set_defaults(fn=_eval)
    x = ss.add_parser("export", help="ONNX export (class list stored in the file)")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--float", action="store_true", help="float NCHW input instead of uint8 NHWC")
    x.set_defaults(fn=_export)
    b = ss.add_parser("bench", help="host timing of an exported model")
    b.add_argument("--onnx", required=True)
    b.add_argument("--runs", type=int, default=30)
    b.add_argument("--threads", type=int, default=2)
    b.set_defaults(fn=_bench)
    c = ss.add_parser(
        "scene-sim",
        help="closed loop: person -> ASC -> HID bridge with the scene from a detector stand-in (jitter, misses, latency, frame rate)",
    )
    c.add_argument("--persona", default="all", choices=["all", "overshooter", "tremor"])
    c.add_argument("--trials", type=int, default=20)
    c.add_argument("--jitter", type=float, default=1.5)
    c.add_argument("--miss", type=float, default=0.05)
    c.set_defaults(fn=_scene_sim)
