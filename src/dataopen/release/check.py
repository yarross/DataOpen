"""Hold the registry against the repository: a claim that cites a file or a test is checked to be there, and a claim that needs hardware or
people can not be marked met by editing a flag (it needs a record file)."""
from __future__ import annotations

import re
from pathlib import Path

from . import registry as R

ROOT = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    try:
        return (ROOT / rel).read_text(encoding="utf-8")
    except OSError:
        return ""


def resolve(ref: str) -> bool:
    """`tests/test_x.py` (the file holds tests), `tests/test_x.py::test_name` (that function), `pwa/tests/x.mjs::title` (a test with that
    title, any substring)."""
    path, _, name = ref.partition("::")
    text = _read(path)
    if not text:
        return False
    if path.endswith(".py"):
        return bool(re.search(rf"^(?:async )?def {re.escape(name)}\b", text, re.M)) if name else bool(re.search(r"^def test_", text, re.M))
    if path.endswith(".mjs"):
        return (name in text) if name else bool(re.search(r"\btest\(|\bcheck\(", text))
    return (ROOT / path).exists()


def check() -> list[str]:
    """Everything that is wrong with the registry; empty means it matches the repository."""
    errs: list[str] = []
    seen: set[str] = set()
    for m in R.MODULES:
        if m.id in seen:
            errs.append(f"module {m.id}: duplicate id")
        seen.add(m.id)
        if m.status not in R.STATUSES:
            errs.append(f"module {m.id}: unknown status {m.status!r}")
        if m.area not in R.AREAS:
            errs.append(f"module {m.id}: unknown area {m.area!r}")
        for p in m.paths:
            if not (ROOT / p).exists():
                errs.append(f"module {m.id}: path {p} does not exist")
        for ref in m.evidence:
            if not resolve(ref):
                errs.append(f"module {m.id}: evidence {ref} not found")
        if m.status in (R.IMPLEMENTED, R.SIMULATED) and not (m.paths and m.evidence):
            errs.append(f"module {m.id}: {m.status} needs code and tests")
        if m.status in (R.ARCHITECTURE, R.DEFERRED) and (m.paths and m.status == R.DEFERRED):
            errs.append(f"module {m.id}: deferred but has code")
        if m.status in (R.IMPLEMENTED, R.SIMULATED, R.ARCHITECTURE) and not m.gaps:
            errs.append(f"module {m.id}: say what is NOT verified")
        if m.doc and not (ROOT / m.doc).exists():
            errs.append(f"module {m.id}: doc {m.doc} does not exist")
    for a in R.AREAS:
        if not any(m.area == a for m in R.MODULES):
            errs.append(f"area {a}: no modules")
    ids = set()
    for s in R.SCENARIOS:
        if s.id in ids:
            errs.append(f"scenario {s.id}: duplicate")
        ids.add(s.id)
        if not (s.pre and s.steps and s.expect and s.covered and s.unverified):
            errs.append(f"scenario {s.id}: preconditions, steps, expectation, coverage and what is unverified must all be given")
        for ref in s.covered:
            if not resolve(ref):
                errs.append(f"scenario {s.id}: evidence {ref} not found")
        if not re.search(rf"^def {s.acceptance}\b", _read("tests/test_acceptance.py"), re.M):
            errs.append(f"scenario {s.id}: acceptance function {s.acceptance} not found")
        if f"tests/test_acceptance.py::test_acc{s.id[-2:]}" not in " ".join(s.covered):
            errs.append(f"scenario {s.id}: its own acceptance test is not among the evidence")
    if not resolve(f"tests/test_acceptance.py::{R.SCENARIO_DAY}"):
        errs.append("the pilot-day test is missing")
    dids = set()
    for d in R.DOD:
        if d.id in dids:
            errs.append(f"dod {d.id}: duplicate")
        dids.add(d.id)
        if d.kind not in R.KINDS:
            errs.append(f"dod {d.id}: unknown kind")
        if d.met:
            if not d.evidence and not d.record:
                errs.append(f"dod {d.id}: marked met without evidence")
            if d.kind in (R.HARDWARE, R.HUMAN) and not (d.record and (ROOT / d.record).exists()):
                errs.append(f"dod {d.id}: a {d.kind} criterion is met only by a sign-off record file that exists (field `record`)")
            if d.kind == R.PROCESS and not (d.record and (ROOT / d.record).exists()):
                errs.append(f"dod {d.id}: a process criterion is met only by a record file that exists")
        for ref in d.evidence:
            if not resolve(ref):
                errs.append(f"dod {d.id}: evidence {ref} not found")
    for st in R.ORDER:
        for x in st.exit:
            if x not in dids:
                errs.append(f"step {st.n}: unknown criterion {x}")
        for dep in st.depends:
            if dep >= st.n or not any(o.n == dep for o in R.ORDER):
                errs.append(f"step {st.n}: depends on {dep}")
    covered = {x for st in R.ORDER for x in st.exit}
    for d in R.DOD:
        if not d.met and d.id not in covered:
            errs.append(f"dod {d.id}: open, and no step of the order of work closes it")
    return errs


def verdict() -> dict:
    """Is it ready for a pilot? Only if every criterion is met."""
    open_ = [d for d in R.DOD if not d.met]
    by_kind = {k: [d.id for d in open_ if d.kind == k] for k in R.KINDS}
    by_status = {s: sum(1 for m in R.MODULES if m.status == s) for s in R.STATUSES}
    return {"ready": not open_, "met": len(R.DOD) - len(open_), "total": len(R.DOD), "open": [d.id for d in open_], "open_by_kind": by_kind,
            "modules": by_status, "hardware_verified": sum(1 for d in R.DOD if d.met and d.kind == R.HARDWARE),
            "human_verified": sum(1 for d in R.DOD if d.met and d.kind == R.HUMAN)}


def counts() -> dict:
    """How much is tested, counted from the files (so the document does not carry a number that ages)."""
    py = {p.name: len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), re.M)) for p in (ROOT / "tests").glob("test_*.py")}
    js = {p.name: len(re.findall(r"^test\(", p.read_text(encoding="utf-8"), re.M)) for p in (ROOT / "pwa" / "tests").glob("*.test.mjs")}
    return {"python_files": len(py), "python_tests": sum(py.values()), "node_files": len(js), "node_tests": sum(js.values()),
            "acceptance_tests": py.get("test_acceptance.py", 0)}
