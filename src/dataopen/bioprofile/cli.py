"""`dataopen bioprofile ...`: simulate a player and build a profile, show a stored profile, write the C header."""
from __future__ import annotations

import json
from pathlib import Path

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4


def _cmd_simulate(a) -> int:
    from .engine import BioProfileEngine, EngineConfig
    from .profile import ProfileView
    from .sim import SimPlayer, simulate
    from .store import ProfileStore
    p = SimPlayer(t_motor_drift_ms_per_h=a.fatigue_ms_per_h, err_drift_deg_per_h=a.fatigue_err_per_h)
    s = simulate(p, a.minutes, a.seed)
    eng = BioProfileEngine(EngineConfig(deg_per_count=p.dpc))
    for e in s.events:
        if e[0] == "mouse":
            eng.on_mouse(e[1], e[2], e[3])
        else:
            eng.on_target(e[1], e[2])
    eng.advance(int(s.duration_s * 1e6) + 2_000_000)
    state = eng.snapshot(clean=True)
    if a.out:
        gen = ProfileStore(a.out).save(state)
        print(f"saved generation {gen} to {a.out}.a / .b")
    print(json.dumps(ProfileView(state.pack()).as_dict(), indent=2))
    print(f"(simulated player: T_motor {p.t_motor_ms} ms, jitter {p.jitter_hz} Hz / {p.jitter_amp_deg} deg, "
          f"phase lag {p.phase_lag_ms} ms, drift {a.fatigue_ms_per_h} ms/h)")
    return EXIT_OK


def _cmd_show(a) -> int:
    from .profile import ProfileError, ProfileView
    from .store import ProfileStore
    try:
        st = ProfileStore(a.path).load()
    except ProfileError as e:
        print(f"error: {e}")
        return EXIT_FAILED
    if st is None:
        print(f"no valid profile at {a.path}.a / {a.path}.b")
        return EXIT_FAILED
    print(json.dumps(ProfileView(st.pack()).as_dict(), indent=2))
    return EXIT_OK


def _cmd_header(a) -> int:
    from .profile import header_text
    Path(a.out).write_text(header_text())
    print(f"written: {a.out}")
    return EXIT_OK


def register(sub) -> None:
    b = sub.add_parser("bioprofile", help="BioProfile engine: per-player mouse biomechanics profile (docs/BIOPROFILE.md)")
    bs = b.add_subparsers(dest="bio_cmd", required=True)
    s = bs.add_parser("simulate", help="run the engine on a simulated player and print the recovered profile")
    s.add_argument("--minutes", type=float, default=20.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", help="store the profile (A/B slot files <out>.a / <out>.b)")
    s.add_argument("--fatigue-ms-per-h", dest="fatigue_ms_per_h", type=float, default=0.0)
    s.add_argument("--fatigue-err-per-h", dest="fatigue_err_per_h", type=float, default=0.0)
    s.set_defaults(fn=_cmd_simulate)
    w = bs.add_parser("show", help="print a stored profile")
    w.add_argument("path")
    w.set_defaults(fn=_cmd_show)
    h = bs.add_parser("header", help="write the C header of the packed profile")
    h.add_argument("--out", default="bioprofile.h")
    h.set_defaults(fn=_cmd_header)
