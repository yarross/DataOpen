"""`dataopen hw ...`: print the product and hardware specification as tables (docs/HARDWARE.md)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from . import failsafe as F
from . import indication as I
from . import power as W
from . import report as R
from . import spec as S

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 4
DOC = Path(__file__).resolve().parents[3] / "docs" / "HARDWARE.md"


def _out(a, table: str, data) -> int:
    print(json.dumps(data, ensure_ascii=False, indent=2) if a.json else table)
    return EXIT_OK


def _sku(a) -> int:
    data = [{"code": s.code, "name": s.name, "groups": list(s.groups), "retail": S.retail(s), "bom": S.cost(s), "ports": S.ports(s)} for s in S.SKUS]
    return _out(a, R.sku_table() + "\n\n" + R.dnp_table() + "\n\n" + R.ports_table(), data)


def _bom(a) -> int:
    if a.sku:
        try:
            s = S.sku(a.sku)
        except KeyError:
            print(f"error: no such SKU: {a.sku}")
            return EXIT_USAGE
        data = [asdict(p) for p in S.bom(s)]
        table = "\n".join(f"{p.ref:5} x{p.qty}  {p.lo:g}-{p.hi:g}  {p.name}" for p in S.bom(s))
        lo, hi = S.cost(s)
        return _out(a, f"{s.code} {s.name}\n{table}\nBOM {lo:.0f} - {hi:.0f} USD, retail about {S.retail(s)} USD", data)
    return _out(a, R.bom_totals_table() + "\n\n" + R.bom_table(), [asdict(p) for p in S.PARTS])


def _power(a) -> int:
    data = {s.code: {"budget": W.budget(s), "thermal": W.thermal(s)} for s in S.SKUS}
    return _out(a, R.domains_table() + "\n\n" + R.power_table() + "\n\n" + R.budget_table(), data)


def _failsafe(a) -> int:
    data = [asdict(s) for s in F.SCENARIOS]
    return _out(a, R.truth_table() + "\n\n" + R.failsafe_table() + f"\n\nPC re-enumerates in {F.reenumeration_note()}", data)


def _indication(a) -> int:
    return _out(a, R.indication_table(), [asdict(s) for s in I.STATES])


def _diagram(a) -> int:
    print(S.DIAGRAM)
    return EXIT_OK


def _docs(a) -> int:
    path = Path(a.path) if a.path else DOC
    text = path.read_text(encoding="utf-8")
    new = R.render_doc(text)
    if a.check:
        if not R.doc_is_current(path):
            print(f"{path} is stale: run `dataopen hw docs --write`")
            return EXIT_FAILED
        print("current")
        return EXIT_OK
    path.write_text(new, encoding="utf-8")
    print(f"written: {path}")
    return EXIT_OK


def register(sub) -> None:
    s = sub.add_parser("hw", help="product and hardware specification of the adapter: SKUs, BOM, power, fail-safe, indication (docs/HARDWARE.md)")
    ss = s.add_subparsers(dest="hw_cmd", required=True)
    for name, fn, help_ in (("sku", _sku, "SKU matrix, DNP groups and ports"), ("power", _power, "power domains, budget and the passive-cooling estimate"),
                            ("failsafe", _failsafe, "the fail-safe truth table and the scenarios"), ("indication", _indication, "LED and buzzer states")):
        p = ss.add_parser(name, help=help_)
        p.add_argument("--json", action="store_true")
        p.set_defaults(fn=fn)
    b = ss.add_parser("bom", help="the bill of materials (all parts, or one SKU)")
    b.add_argument("--sku")
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=_bom)
    d = ss.add_parser("diagram", help="the block diagram")
    d.set_defaults(fn=_diagram)
    g = ss.add_parser("docs", help="refresh (or check) the generated tables in docs/HARDWARE.md")
    g.add_argument("--path")
    g.add_argument("--check", action="store_true", help="only check that the tables are current")
    g.add_argument("--write", action="store_true", help="refresh the tables (the default)")
    g.set_defaults(fn=_docs)
