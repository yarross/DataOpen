"""`dataopen ctl ...`: simulate the phone-to-bridge chain, print the manifest, write the artifacts the browser client is built from."""
from __future__ import annotations

import json
from pathlib import Path

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4
KEYS = ("assist.on", "assist.strength", "tremor.level", "profile.fill")


def _manifest(a) -> int:
    from .manifest import default_manifest, validate_manifest
    m = default_manifest()
    errs = validate_manifest(m)
    if errs:
        print("invalid:", *errs, sep="\n  ")
        return EXIT_FAILED
    text = json.dumps(m, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"written: {a.out}")
    else:
        print(text)
    return EXIT_OK


def _constants(a) -> int:
    from .protocol import constants_js
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(constants_js(), encoding="utf-8")
    print(f"written: {a.out}")
    return EXIT_OK


def _golden(a) -> int:
    from .protocol import golden
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(golden(), indent=1) + "\n", encoding="utf-8")
    print(f"written: {a.out}")
    return EXIT_OK


def _simulate(a) -> int:
    """A scripted session of a simulated phone against the gateway and the real bridge core: the whole chain, printed."""
    import tempfile

    from ..bridge.cbridge import REASONS, STATE
    from .sim import SimLearner, World, seed_profile
    d = a.dir or tempfile.mkdtemp(prefix="dataopen-ctl-")
    if a.profile != "none":
        seed_profile(d, a.profile)
    w = World(d, learner=SimLearner(minutes=4.0), trial_s=a.trial_s)
    ph = w.phone
    log = []

    def show(what: str) -> None:
        b = w.bridge()
        s = w.gw.status()
        log.append({"t_ms": w.t // 1000, "step": what, "bridge": STATE[b.state], "reason": REASONS[b.reason],
                    "state": {k: v for k, v in w.gw.state_tree().items() if k in KEYS},
                    "trial_left_s": s.trial_left_s})

    ph.connect()
    ph.get_manifest()
    show("connected")
    ph.set("assist.on", True)
    w.run(500)
    show("assistance requested (a trial)")
    ph.confirm(True)
    ph.set("assist.strength", 8)
    w.run(500)
    show("strength 8 (a trial again)")
    w.run((a.trial_s + 1) * 1000)
    show("no confirmation: back to the kept state")
    ph.stop()
    w.run(100)
    show("stop")
    ph.hard_bypass()
    w.run(200)
    show("hardware bypass")
    for row in log:
        line = f"{row['t_ms']:>7} ms  {row['step']:<42} bridge={row['bridge']:<9} reason={row['reason']:<13}"
        print(json.dumps(row, ensure_ascii=False) if a.json else f"{line} {row['state']} trial={row['trial_left_s']}")
    return EXIT_OK


def _serve(a) -> int:
    from .ws import serve_sim
    return serve_sim(a)


def register(sub) -> None:
    s = sub.add_parser("ctl", help="phone control gateway and the PWA client's protocol (docs/PWA.md)")
    ss = s.add_subparsers(dest="ctl_cmd", required=True)
    m = ss.add_parser("manifest", help="print (or write) the default device manifest")
    m.add_argument("--out")
    m.set_defaults(fn=_manifest)
    c = ss.add_parser("constants", help="write pwa/js/constants.js from the protocol module")
    c.add_argument("--out", default="pwa/js/constants.js")
    c.set_defaults(fn=_constants)
    g = ss.add_parser("golden", help="write the cross-language test vectors")
    g.add_argument("--out", default="pwa/tests/golden.json")
    g.set_defaults(fn=_golden)
    r = ss.add_parser("simulate", help="a scripted phone session against the gateway and the real bridge core")
    r.add_argument("--dir", help="gateway directory (default: a temporary one)")
    r.add_argument("--profile", default="tremor", help="simulated person for the starting profile: steady, overshooter, tremor, none")
    r.add_argument("--trial-s", dest="trial_s", type=int, default=5)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=_simulate)
    v = ss.add_parser("serve-sim", help="serve pwa/ and a simulated device over WebSocket on localhost (development and e2e tests only)")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--root", help="the pwa directory (default: the repository's pwa/)")
    v.add_argument("--profile", default="tremor")
    v.add_argument("--dir")
    v.set_defaults(fn=_serve)
