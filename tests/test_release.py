"""The registry of docs/V1.md against the repository: what the document claims is what the files and tests hold, and 'ready' can not be claimed by editing a flag."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dataopen.cli import build_parser
from dataopen.ctl import firmware as F
from dataopen.ctl.sim import dev_image, dev_vendor
from dataopen.release import check as C
from dataopen.release import policy as PO
from dataopen.release import registry as R
from dataopen.release import report as RP

DOC = Path(__file__).resolve().parents[1] / "docs" / "V1.md"
HW = b"DOHW0001"
PUB = dev_vendor()[1]


def run(*argv):
    a = build_parser().parse_args(["release", *argv])
    return a.fn(a)


def test_the_registry_matches_the_repository():
    assert C.check() == []


def test_the_document_is_generated_from_the_registry_and_is_current():
    assert RP.doc_is_current(DOC), "run `dataopen release docs --write`"
    text = DOC.read_text(encoding="utf-8")
    for h in ("## 1. Краткое резюме", "## 2. Карта системы", "## 3. Сквозные acceptance-сценарии", "## 4. Definition of Done для v1",
              "## 5. Главные оставшиеся риски", "## 6. Что сознательно НЕ входит в v1", "## 7. Файлы и артефакты", "## 8. Рекомендуемый порядок"):
        assert h in text
    for m in R.MODULES:
        assert m.name in text, m.id
    for s in R.SCENARIOS:
        assert f"{s.id}. {s.name}" in text


def test_every_status_is_used_and_nothing_is_called_working_without_code_and_tests():
    for st in R.STATUSES:
        assert any(m.status == st for m in R.MODULES), st
    for m in R.MODULES:
        if m.status in (R.IMPLEMENTED, R.SIMULATED):
            assert m.paths and m.evidence and m.gaps, m.id
        if m.status == R.DEFERRED:
            assert not m.evidence, m.id
    # the things this system most needs and does not have are said to be missing
    for must in ("module-daemon", "live-calibration", "hid-bridge-board", "bootloader", "ble-gatt-server", "video-hardware", "factory-hardware"):
        assert R.BY_ID[must].status == R.ARCHITECTURE and not R.BY_ID[must].evidence, must
    assert R.BY_ID["ui-detector"].status == R.SIMULATED and R.BY_ID["hid-bridge-core"].status == R.SIMULATED


def test_nothing_is_claimed_verified_on_hardware_or_with_people():
    v = C.verdict()
    assert v["hardware_verified"] == 0 and v["human_verified"] == 0 and v["ready"] is False
    for d in R.DOD:
        if d.kind in (R.HARDWARE, R.HUMAN):
            assert not d.met, d.id                                          # until a sign-off record exists
    # a criterion that needs hardware can not be closed by flipping a flag
    flipped = tuple(replace(d, met=True, evidence=R.DOD_BY_ID["S-5"].evidence) if d.id == "S-1" else d for d in R.DOD)
    old = R.DOD
    try:
        R.DOD = flipped
        errs = C.check()
    finally:
        R.DOD = old
    assert any("S-1" in e and "record" in e for e in errs)


def test_the_verdict_is_ready_only_when_every_criterion_is_met():
    v = C.verdict()
    assert v["ready"] == (not v["open"]) and v["met"] + len(v["open"]) == v["total"] == len(R.DOD)
    assert set(v["open"]) == {d.id for d in R.DOD if not d.met}
    assert sum(len(x) for x in v["open_by_kind"].values()) == len(v["open"])
    old = R.DOD
    try:
        R.DOD = tuple(replace(d, met=True) for d in old)
        assert C.verdict()["ready"] is True
    finally:
        R.DOD = old


def test_every_open_criterion_is_closed_by_some_step_and_steps_come_in_order():
    closed = {x for st in R.ORDER for x in st.exit}
    assert {d.id for d in R.DOD if not d.met} <= closed
    for st in R.ORDER:
        assert all(dep < st.n for dep in st.depends)
    assert [st.n for st in R.ORDER] == list(range(1, len(R.ORDER) + 1))


def test_the_nine_scenarios_of_the_brief_are_there_with_all_five_parts():
    assert [s.id for s in R.SCENARIOS] == [f"ACC-0{i}" for i in range(1, 10)]
    names = " ".join(s.name for s in R.SCENARIOS).lower()
    for word in ("первый запуск", "калибровка", "работа", "слот", "пакет", "обновление системы", "panic", "восстановление", "сброс"):
        assert word in names, word
    for s in R.SCENARIOS:
        assert s.pre and s.steps and s.expect and s.covered and s.unverified, s.id


# ---------------------------------------------------------------------------------------------------------------- the release rule
def test_a_release_that_would_outrun_the_golden_image_is_refused():
    golden = dev_image(1, 1, b"golden" * 100)
    ok = dev_image(2, 1, b"update" * 100)
    assert PO.check_release(ok, golden, PUB, HW, floor_now=1) == []
    bad = dev_image(2, 2, b"update" * 100)                                           # min_version 2: the floor would pass the golden image
    problems = PO.check_release(bad, golden, PUB, HW, floor_now=1)
    assert len(problems) == 1 and "golden image" in problems[0] and "floor to 2" in problems[0]
    assert PO.check_release(bad, dev_image(2, 1, b"golden2" * 100), PUB, HW) == []   # with a golden image that meets it the same update is fine
    assert any("below the floor" in p for p in PO.check_release(dev_image(1, 1, b"u" * 100), golden, PUB, HW, floor_now=3))


def test_the_rule_agrees_with_what_the_device_really_does():
    """The reason for the rule: with the floor above the golden image the recovery refuses (checked on the real service code in test_acceptance)."""
    golden = dev_image(1, 1, b"golden" * 100)
    mgr = F.SlotManager(PUB, HW)
    mgr.install_factory(dev_image(1, 1, b"f" * 200))
    mgr.stage(dev_image(2, 2, b"u" * 200))
    mgr.approve()
    mgr.reboot()
    mgr.confirm()
    assert mgr.floor == 2 and F.verify_image(golden, PUB, HW).version < mgr.floor


def test_images_that_are_not_acceptable_are_reported_not_guessed_at():
    from cryptography.hazmat.primitives.asymmetric import ed25519
    other = F.build_image(b"x" * 100, HW, 2, 1, ed25519.Ed25519PrivateKey.generate())             # signed by somebody who is not the manufacturer
    assert "not acceptable" in PO.check_release(other, dev_image(1, 1), PUB, HW)[0]
    assert "golden image is not acceptable" in PO.check_release(dev_image(2, 1), b"junk", PUB, HW)[0]


# ---------------------------------------------------------------------------------------------------------------- the command
def test_the_command_reports_and_exits_nonzero_while_anything_is_open(capsys):
    assert run("status") == 1 and "НЕТ" in capsys.readouterr().out
    assert run("status", "--json") == 1
    v = json.loads(capsys.readouterr().out)
    assert v["ready"] is False and v["hardware_verified"] == 0 and "S-1" in v["open"]
    assert run("check") == 0 and "matches" in capsys.readouterr().out
    for sub in ("map", "scenarios", "dod", "risks", "order"):
        assert run(sub) == 0 and capsys.readouterr().out.count("|") + capsys.readouterr().out.count("ACC") > -1
    assert run("docs", "--check") == 0


def test_a_stale_copy_of_the_document_is_refused(tmp_path, capsys):
    stale = tmp_path / "V1.md"
    stale.write_text(DOC.read_text(encoding="utf-8").replace("| **Готовность", "| **Готовность", 1).replace("**Готовность к пилоту: НЕТ.**", "**Готовность к пилоту: ДА.**", 1),
                     encoding="utf-8")
    assert run("docs", "--check", "--path", str(stale)) == 1 and "stale" in capsys.readouterr().out
    assert C.counts()["acceptance_tests"] >= 10 and C.counts()["python_tests"] > 800


@pytest.mark.parametrize("ref,found", [("tests/test_acceptance.py", True), ("tests/test_acceptance.py::test_acc01_first_run", True), ("tests/test_acceptance.py::test_nothing_like_it", False),
                                       ("tests/no_such_file.py", False), ("pwa/tests/live-packages.test.mjs::a forged", True), ("pwa/tests/live-packages.test.mjs::nope nope", False)])
def test_references_are_resolved_against_the_files(ref, found):
    assert C.resolve(ref) is found
