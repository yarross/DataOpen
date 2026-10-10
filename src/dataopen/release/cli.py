"""`dataopen release ...`: the map of the system, the acceptance scenarios, the Definition of Done and the verdict (docs/V1.md)."""
from __future__ import annotations

import json
from pathlib import Path

from . import check as C
from . import report as RP

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
DOC = Path(__file__).resolve().parents[3] / "docs" / "V1.md"


def _status(a) -> int:
    v = C.verdict()
    if a.json:
        print(json.dumps(v, ensure_ascii=False, indent=2))
    else:
        print(RP.counts())
        print()
        print(RP.verdict())
    return EXIT_OK if v["ready"] else EXIT_FAILED


def _check(a) -> int:
    errs = C.check()
    for e in errs:
        print(f"error: {e}")
    if not errs:
        print("the registry matches the repository")
    return EXIT_FAILED if errs else EXIT_OK


def _docs(a) -> int:
    path = Path(a.path) if a.path else DOC
    if a.check:
        if not RP.doc_is_current(path):
            print(f"{path} is stale: run `dataopen release docs --write`")
            return EXIT_FAILED
        print("current")
        return EXIT_OK
    path.write_text(RP.render_doc(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"written: {path}")
    return EXIT_OK


def _map() -> str:
    parts = (("Implemented", RP.modules_implemented), ("Simulated", RP.modules_simulated), ("Architecture-only", RP.modules_architecture),
             ("Deferred", RP.modules_deferred))
    return "\n\n".join(f"### {title}\n\n{fn()}" for title, fn in parts)


def _print(fn):
    return lambda a: (print(fn()), EXIT_OK)[1]


def register(sub) -> None:
    p = sub.add_parser("release", help="the map of the system, the acceptance scenarios, the Definition of Done for the v1 pilot (docs/V1.md)")
    ps = p.add_subparsers(dest="release_cmd", required=True)
    s = ps.add_parser("status", help="counts and the verdict; exit code 1 while any criterion of the Definition of Done is open")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_status)
    c = ps.add_parser("check", help="hold the registry against the repository (files, tests, records); exit code 1 on any mismatch")
    c.set_defaults(fn=_check)
    for name, fn, h in (("map", _map, "the modules by status"), ("scenarios", RP.scenarios, "the acceptance scenarios"), ("dod", RP.dod, "the Definition of Done"),
                        ("risks", RP.risks, "the main risks"), ("order", RP.order, "the order of work to a pilot")):
        x = ps.add_parser(name, help=h)
        x.set_defaults(fn=_print(fn))
    g = ps.add_parser("docs", help="refresh (or check) the generated tables in docs/V1.md")
    g.add_argument("--path")
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.set_defaults(fn=_docs)
