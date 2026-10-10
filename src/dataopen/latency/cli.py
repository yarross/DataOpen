"""`dataopen latency ...`: the budget, the simulation measurements, the stand's analysis tool, the document (docs/LATENCY.md)."""
from __future__ import annotations

import json
from pathlib import Path

from . import analyze as AN
from . import budget as B
from . import measure as M
from . import report as RP

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
DOC = Path(__file__).resolve().parents[3] / "docs" / "LATENCY.md"


def _scenario(a):
    sc = B.BY_KEY.get(a.scenario)
    if sc is None:
        print(f"error: no such scenario; one of: {', '.join(B.BY_KEY)}")
    return sc


def _stages(a) -> int:
    sc = _scenario(a)
    if sc is None:
        return EXIT_USAGE
    if a.json:
        print(json.dumps([vars(s) for s in B.stages(sc)], ensure_ascii=False, indent=2))
    else:
        print(RP.stages_table(a.scenario))
    return EXIT_OK


def _budget(a) -> int:
    if a.json:
        print(json.dumps([B.summary(sc) for sc in B.SCENARIOS], ensure_ascii=False, indent=2))
    else:
        print(RP.scenarios_table())
    return EXIT_OK


def _measure(a) -> int:
    what = a.what
    out = {}
    if what in ("all", "hid"):
        out["core_report_delay"] = [M.core_report_delay("FS", 1), M.core_report_delay("HS", 1)]
        out["poll_resample"] = [M.poll_resample(hz) for hz in (125, 500, 1000, 8000)]
    if what in ("all", "chain"):
        out["chain_delay"] = [M.chain_delay(p, amp) for p, amp in (("tremor", 0.0), ("tremor", 8.0), ("overshooter", 0.0))]
    if what in ("all", "scene"):
        out["scene_effect"] = [M.scene_effect(age) for age in (0, 50, 90, 110)]
    if what in ("all", "pipe"):
        out["pipe_timeline"] = {f"{p.policy}-{p.infer_ms:g}ms-{p.frame_ms:g}": {k: M.stats(v) for k, v in M.pipe_timeline(p).items()}
                                for p in (M.Pipe(6.94, 11.0), M.Pipe(6.94, 11.0, policy="latest"), M.Pipe(6.94, 3.0, policy="latest"))}
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK
    for key, fn in (("hid", RP.usb_floor_table), ("chain", RP.chain_table), ("scene", RP.scene_table), ("pipe", RP.pipe_table)):
        if what in ("all", key):
            print(fn() + "\n")
    return EXIT_OK


def _analyze(a) -> int:
    try:
        res = AN.analyze_file(a.file, a.unit, a.fps, a.window_ms)
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return EXIT_USAGE
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if a.limit_ms is not None and (res["pairs"] == 0 or res["p99"] > a.limit_ms or res["missed"]):
        print(f"FAIL: p99 {res['p99']:.2f} ms, missed {res['missed']} (limit {a.limit_ms} ms)")
        return EXIT_FAILED
    return EXIT_OK


def _docs(a) -> int:
    path = Path(a.path) if a.path else DOC
    if a.check:
        if not RP.doc_is_current(path):
            print(f"{path} is stale: run `dataopen latency docs --write`")
            return EXIT_FAILED
        print("current")
        return EXIT_OK
    path.write_text(RP.render_doc(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"written: {path}")
    return EXIT_OK


def register(sub) -> None:
    p = sub.add_parser("latency", help="end-to-end latency budget of the assistive path and how to measure it (docs/LATENCY.md)")
    ps = p.add_subparsers(dest="latency_cmd", required=True)
    s = ps.add_parser("stages", help="the stages of one scenario with the basis of each number")
    s.add_argument("scenario", nargs="?", default="legacy-fifo")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_stages)
    b = ps.add_parser("budget", help="all scenarios side by side: input path A, scene path B, margin to the scene TTL")
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=_budget)
    m = ps.add_parser("measure", help="run the simulation measurements (virtual time; takes tens of seconds)")
    m.add_argument("what", nargs="?", default="all", choices=["all", "hid", "chain", "scene", "pipe"])
    m.add_argument("--json", action="store_true")
    m.set_defaults(fn=_measure)
    t = ps.add_parser("testpoints", help="the stand: where to probe and which interval each pair of edges bounds")
    t.set_defaults(fn=lambda a: (print(RP.testpoints_table()), EXIT_OK)[1])
    an = ps.add_parser("analyze", help="pair stimulus and response edges of a stand capture (CSV: channel,t) into percentiles")
    an.add_argument("file")
    an.add_argument("--unit", default="us", choices=["s", "ms", "us", "frames"])
    an.add_argument("--fps", type=float, help="camera frame rate (unit frames)")
    an.add_argument("--window-ms", type=float, default=200.0, help="a response later than this after a stimulus does not belong to it")
    an.add_argument("--limit-ms", type=float, help="exit 1 when p99 is above this (or an answer is missing)")
    an.set_defaults(fn=_analyze)
    g = ps.add_parser("docs", help="refresh (or check) the generated tables in docs/LATENCY.md")
    g.add_argument("--path")
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.set_defaults(fn=_docs)
