"""`dataopen assist ...`: evaluate the correction on simulated people, benchmark the cores, build the C library."""
from __future__ import annotations

import json
import time

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4


def _cmd_simulate(a) -> int:
    from .sim_user import PERSONAS, compare, compare_all
    names = list(PERSONAS) if a.persona == "all" else [a.persona]
    out = {}
    for n in names:
        if a.chain:                                     # nothing / ASC / tremor suppression / both
            out[n] = compare_all(PERSONAS[n], n=a.trials, seed=a.seed)
        else:
            base, asc, _ = compare(PERSONAS[n], n=a.trials, seed=a.seed)
            out[n] = {"without": base, "with": asc}
    if a.json:
        print(json.dumps(out, indent=2))
        return EXIT_OK
    cols = ("acquired", "t_acquire_ms", "overshoot_rate", "overshoot_px", "hold_rms_px", "final_err_px", "k_mean")
    print(f"{'persona':12s} {'':8s} " + " ".join(f"{c:>14s}" for c in cols))
    for n, r in out.items():
        for lab, v in r.items():
            print(f"{n:12s} {lab:8s} " + " ".join(f"{v[c]:14.2f}" for c in cols))
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
    from .tremor import TremorParams, TremorSuppressor
    from .tremor_fixed import FixedTremor, FixedTremorParams
    tp = TremorParams(True, 0.0199, 0.0957, 0.00995, 0.9, 2, 1.5)
    raw_in = [(t, dx, dy) for (t, dx, dy, _, _) in rec]
    tcores = {"tremor float (python)": TremorSuppressor(tp), "tremor fixed (python golden)": FixedTremor(FixedTremorParams.from_params(tp))}
    try:
        from .cimpl import CTremor
        tcores["tremor fixed (C, through ctypes)"] = CTremor(FixedTremorParams.from_params(tp))
    except (RuntimeError, OSError):
        pass
    for name, core in tcores.items():
        n, t0 = 0, time.perf_counter()
        for _ in range(a.repeat):
            core.reset()
            for (t, dx, dy) in raw_in:
                core.tick(t, dx, dy)
                n += 1
        dt = time.perf_counter() - t0
        print(f"{name:34s} {dt / n * 1e6:8.2f} us/tick")
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
    a.add_argument("--chain", action="store_true", help="compare nothing / ASC / tremor suppression / the chain of both")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=_cmd_simulate)
    b = ss.add_parser("bench", help="time per tick of the float, fixed-point golden and C cores (host numbers)")
    b.add_argument("--repeat", type=int, default=20)
    b.set_defaults(fn=_cmd_bench)
    c = ss.add_parser("build", help="compile the C core with the system compiler")
    c.add_argument("--out")
    c.set_defaults(fn=_cmd_build)
