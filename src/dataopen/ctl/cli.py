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


def _identity(a) -> int:
    """A gateway directory's device ID and card (written with --out). Read-only: a directory without keys is reported, not provisioned."""
    from .identity import Card, FileKeyStore, Identity
    store = FileKeyStore(Path(a.dir) / "keys" / "keys.json")
    d = store.load()
    if d is None:
        print(f"no device identity in {a.dir} (it is created the first time the gateway starts)")
        return EXIT_FAILED
    ident = Identity(store, d)
    card = ident.card()
    print(f"device ID  {ident.id}\ncreated    {ident.created}\nexports    {ident.export_seq}")
    if a.out:
        Path(a.out).write_text(json.dumps(card.to_json()), encoding="utf-8")
        print(f"card written: {a.out}")
    assert Card.from_json(card.to_json()).id == ident.id
    return EXIT_OK


def _transfer(a) -> int:
    """A profile moves from one simulated device to another through a sealed file, with the button presses the real flow needs."""
    import tempfile

    from . import protocol as P
    from .sim import World, seed_profile
    root = Path(a.dir or tempfile.mkdtemp(prefix="dataopen-xfer-"))
    seed_profile(root / "a", a.profile)
    wa, wb = World(root / "a"), World(root / "b")
    wa.phone.connect()
    wb.phone.connect()
    fill = lambda w: w.gw.state_tree()["profile.fill"]    # noqa: E731
    print(f"A {wa.gw.identity.id} (profile {fill(wa)} %)    B {wb.gw.identity.id} (profile {fill(wb)} %)")
    card = wb.phone.get_identity()
    r = wa.phone.try_bundle(card)
    print(f"A exports for B without the button: {r.json()['key']} ({r.json()['detail']})")
    wa.gw.physical_press()
    raw = wa.phone.get_bundle(card)
    print(f"A exports for B with the button:    {len(raw)} bytes, starts {raw[:4]!r}, the profile's magic in it: {b'BIOP' in raw}")
    r = wb.phone.put_bundle(raw)
    print(f"B imports from an unknown sender:   {r.json()['key']} ({r.json()['detail']})")
    wb.gw.physical_press()
    r = wb.phone.put_bundle(raw)
    print(f"B imports with its button:          {'accepted' if r.type == P.T_ACK else r.json()}; profile {fill(wb)} %")
    r = wb.phone.put_bundle(raw)
    print(f"the same file again:                {r.json()['key']}")
    wa.gw.physical_press()
    print(f"the file on A itself:               {wa.phone.put_bundle(raw).json()['key']}")
    return EXIT_OK


def _pwa_build(a) -> int:
    from .pwa import build, is_current, pwa_root
    root = Path(a.root) if a.root else pwa_root()
    if a.check:
        ok = is_current(root)
        print("current" if ok else "stale: run `dataopen ctl pwa-build`")
        return EXIT_OK if ok else EXIT_FAILED
    print(f"service worker and js/build.js updated, version {build(root)}")
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
    i = ss.add_parser("identity", help="show a gateway directory's device ID and card (read-only)")
    i.add_argument("--dir", required=True)
    i.add_argument("--out", help="write the card (.docard) here")
    i.set_defaults(fn=_identity)
    x = ss.add_parser("transfer", help="a profile moves between two simulated devices through a sealed file, with the button presses")
    x.add_argument("--dir")
    x.add_argument("--profile", default="tremor")
    x.set_defaults(fn=_transfer)
    b = ss.add_parser("pwa-build", help="write the service worker's file list and the cache version into pwa/")
    b.add_argument("--root")
    b.add_argument("--check", action="store_true", help="only check that the committed files are current")
    b.set_defaults(fn=_pwa_build)
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
    v.add_argument("--trial-s", dest="trial_s", type=int, default=20)
    v.set_defaults(fn=_serve)
