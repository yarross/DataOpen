"""`dataopen update ...`: the two channels, the package format, and building / inspecting / checking packages and models (docs/UPDATES.md)."""
from __future__ import annotations

import json
from pathlib import Path

from . import channels as C
from . import report as RP

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
DOC = Path(__file__).resolve().parents[3] / "docs" / "UPDATES.md"


def _reasons(a) -> int:
    rows = [r for r in C.REASONS if a.channel in (None, r.channel)]
    if a.json:
        print(json.dumps([vars(r) for r in rows], ensure_ascii=False, indent=2))
    else:
        print((RP.reasons_a() if a.channel == "A" else RP.reasons_b()) if a.channel else RP.reasons_a() + "\n\n" + RP.reasons_b())
    return EXIT_OK


def _load_identity(directory: str):
    from ..ctl.identity import FileKeyStore, load_identity
    store = FileKeyStore(Path(directory) / "keys" / "keys.json")
    if store.load() is None:
        raise ValueError(f"no device identity in {directory}")
    return load_identity(store)


def _build(a) -> int:
    from ..ctl.identity import Card
    from . import dev
    from . import models as MD
    from . import package as K
    try:
        card = Card.from_json(json.loads(Path(a.to).read_text(encoding="utf-8")))
        me = dev.sender(a.sender_dir, a.sender_name)
        tuning = tuple(int(x) for x in a.tuning.split(",")) if a.tuning else None
        if tuning is not None and len(tuning) != 2:
            raise ValueError("--tuning is STRENGTH,TREMOR")
        model = None
        if a.model:
            raw = Path(a.model).read_bytes()
            card_m = MD.card_from_onnx(raw, a.model_name or Path(a.model).stem, a.model_version)
            MD.check_model(raw, card_m)                              # a model the device would refuse is not worth sending
            model = (raw, card_m)
        pkg = K.build_package(me, card, a.seq or me.next_seq(), slot=a.slot, tuning=tuning, name=a.name, model=model, min_fw=a.min_fw)
    except (OSError, ValueError, C.UpdateError) as e:
        print(f"error: {getattr(e, 'key', '')} {e}".strip())
        return EXIT_USAGE
    Path(a.out).write_bytes(pkg)
    print(f"written: {a.out} ({len(pkg)} bytes) for {card.id} from {me.id}")
    return EXIT_OK


def _inspect(a) -> int:
    from . import package as K
    try:
        print(json.dumps(K.inspect_header(Path(a.file).read_bytes()), ensure_ascii=False, indent=2))
    except (OSError, C.UpdateError) as e:
        print(f"error: {e}")
        return EXIT_USAGE
    return EXIT_OK


def _verify(a) -> int:
    from . import package as K
    try:
        me = _load_identity(a.device_dir)
        o = K.open_package(Path(a.file).read_bytes(), me, fw_version=a.fw_version)
    except C.UpdateError as e:
        print(f"refused: {e.key}: {C.reason('B', e.key).en}")
        return EXIT_FAILED
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return EXIT_USAGE
    print(json.dumps({"ok": True, "sender": o.sender_id, "self": o.is_self, "seq": o.seq, "slot": o.slot, "kinds": o.kind_names,
                      "model": None if o.model is None else o.model_card}, ensure_ascii=False, indent=2))
    return EXIT_OK


def _check_model(a) -> int:
    from . import models as MD
    try:
        raw = Path(a.file).read_bytes()
        found = MD.check_model(raw, MD.card_from_onnx(raw, Path(a.file).stem))
    except OSError as e:
        print(f"error: {e}")
        return EXIT_USAGE
    except C.UpdateError as e:
        print(f"refused: {e.key}: {e}")
        return EXIT_FAILED
    print(json.dumps({"ok": True, "ops": found["ops"], "opset": found["opset"], "size": len(raw)}, indent=2))
    return EXIT_OK


def _docs(a) -> int:
    path = Path(a.path) if a.path else DOC
    if a.check:
        if not RP.doc_is_current(path):
            print(f"{path} is stale: run `dataopen update docs --write`")
            return EXIT_FAILED
        print("current")
        return EXIT_OK
    path.write_text(RP.render_doc(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"written: {path}")
    return EXIT_OK


def register(sub) -> None:
    p = sub.add_parser("update", help="the two update channels (system / packages for the slots) and the package tools (docs/UPDATES.md)")
    ps = p.add_subparsers(dest="update_cmd", required=True)
    for name, fn, h in (("channels", RP.channels_table, "the two channels side by side"), ("policy", RP.policy_table, "the policy of each channel, row by row"),
                        ("invariants", RP.invariants_table, "what keeps the channels apart"), ("parts", RP.parts_table, "what a package can carry"),
                        ("format-a", RP.format_a_table, "the system image header (DOFW)"), ("format-b", RP.format_b_table, "the package header (DOPK)")):
        x = ps.add_parser(name, help=h)
        x.set_defaults(fn=(lambda f: lambda a: (print(f()), EXIT_OK)[1])(fn))
    r = ps.add_parser("reasons", help="every way an update does not happen: when it is noticed, what is left, what the person sees")
    r.add_argument("--channel", choices=["A", "B"])
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=_reasons)
    b = ps.add_parser("build-package", help="make a package for a device's card (a test sender's keys live in --sender-dir)")
    b.add_argument("--to", required=True, help="the receiving device's card (.docard)")
    b.add_argument("--sender-dir", required=True)
    b.add_argument("--sender-name", default="sender")
    b.add_argument("--out", required=True)
    b.add_argument("--slot", type=int)
    b.add_argument("--tuning", help="STRENGTH,TREMOR (0..10 each)")
    b.add_argument("--name")
    b.add_argument("--model", help="an ONNX UI model; it is checked before it is sealed")
    b.add_argument("--model-name")
    b.add_argument("--model-version", type=int, default=1)
    b.add_argument("--min-fw", type=int, default=0)
    b.add_argument("--seq", type=int)
    b.set_defaults(fn=_build)
    i = ps.add_parser("inspect", help="the plain header of a package (nothing is checked, nothing is decrypted)")
    i.add_argument("file")
    i.set_defaults(fn=_inspect)
    v = ps.add_parser("verify", help="open a package with a simulated device's keys: every check the device makes")
    v.add_argument("file")
    v.add_argument("--device-dir", required=True)
    v.add_argument("--fw-version", type=int, default=0)
    v.set_defaults(fn=_verify)
    m = ps.add_parser("check-model", help="the structural check of a model file")
    m.add_argument("file")
    m.set_defaults(fn=_check_model)
    g = ps.add_parser("docs", help="refresh (or check) the generated tables in docs/UPDATES.md")
    g.add_argument("--path")
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.set_defaults(fn=_docs)
