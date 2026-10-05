"""`dataopen assist ...`: evaluate the correction on simulated people, benchmark the cores, build the C library."""
from __future__ import annotations

import json
import time

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4


def _cmd_simulate(a) -> int:
    from .sim_user import PERSONAS, compare
    names = list(PERSONAS) if a.persona == "all" else [a.persona]
    out = {}
    for n in names:
        base, asc, view = compare(PERSONAS[n], n=a.trials, seed=a.seed)
        out[n] = {"without": base, "with": asc}
    if a.json:
        print(json.dumps(out, indent=2))
        return EXIT_OK
    cols = ("acquired", "t_acquire_ms", "overshoot_rate", "overshoot_px", "hold_rms_px", "k_mean", "n_sub")
    print(f"{'persona':12s} {'':8s} " + " ".join(f"{c:>14s}" for c in cols))
    for n, r in out.items():
        for lab in ("without", "with"):
            print(f"{n:12s} {lab:8s} " + " ".join(f"{r[lab][c]:14.2f}" for c in cols))
    return EXIT_OK


def _cmd_bench(a) -> int:
    import numpy as np

    from .fixed import FixedAsc, FixedParams
    from .model import AdaptiveSensitivity
    from .params import AscParams
    from .sim_user import PERSONAS, build_profile, run_trial
    from .types import ObjectOfInterest
    per = PERSONAS["overshooter"]
    prm = AscParams.from_view(build_profile(per, seed=1))
    rec: list = []
    run_trial(per, AdaptiveSensitivity(prm), np.random.default_rng(0), record=rec)
    obj = ObjectOfInterest(1, 600.0, 0.0, 30.0, t_appear_us=0)
    cores = {"float (python)": AdaptiveSensitivity(prm), "fixed (python golden)": FixedAsc(FixedParams.from_params(prm))}
    try:
        from .cimpl import CAsc
        cores["fixed (C, through ctypes)"] = CAsc(FixedParams.from_params(prm))
    except (RuntimeError, OSError) as e:
        print(f"C core unavailable: {e}")
    for name, core in cores.items():
        n, t0 = 0, time.perf_counter()
        for _ in range(a.repeat):
            core.reset()
            for (t, dx, dy, px, py) in rec:
                core.tick(t, dx, dy, px, py, obj)
                n += 1
        dt = time.perf_counter() - t0
        print(f"{name:28s} {dt / n * 1e6:8.2f} us/tick  ({n / dt / 1000:.0f} kHz; budget at 1 kHz: 1000 us)")
    return EXIT_OK


def _cmd_build(a) -> int:
    from .cimpl import build_library
    print(build_library(a.out))
    return EXIT_OK


def register(sub) -> None:
    s = sub.add_parser("assist", help="Adaptive Sensitivity Correction: simulate / bench / build (docs/ASSIST.md)")
    ss = s.add_subparsers(dest="assist_cmd", required=True)
    a = ss.add_parser("simulate", help="closed-loop reach task with and without the correction, on simulated people")
    a.add_argument("--persona", default="all", choices=["all", "steady", "overshooter", "tremor"])
    a.add_argument("--trials", type=int, default=40)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=_cmd_simulate)
    b = ss.add_parser("bench", help="time per tick of the float, fixed-point golden and C cores (host numbers)")
    b.add_argument("--repeat", type=int, default=20)
    b.set_defaults(fn=_cmd_bench)
    c = ss.add_parser("build", help="compile the C core with the system compiler")
    c.add_argument("--out")
    c.set_defaults(fn=_cmd_build)
