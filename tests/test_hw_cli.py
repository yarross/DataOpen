"""docs/HARDWARE.md stays equal to the data, and `dataopen hw ...` prints what the tests check."""
import json
from pathlib import Path

import pytest

from dataopen.cli import main
from dataopen.hw import report as R
from dataopen.hw import spec as S

DOC = Path(__file__).resolve().parents[1] / "docs" / "HARDWARE.md"


def test_the_generated_tables_in_the_document_are_current():
    assert R.doc_is_current(DOC), "regenerate: dataopen hw docs --write"


def test_every_table_is_used_in_the_document():
    text = DOC.read_text(encoding="utf-8")
    for name in R.TABLES:
        assert f"<!-- hw:{name} -->" in text and f"<!-- /hw:{name} -->" in text, name


def test_the_document_follows_the_twelve_points_of_the_brief():
    heads = [l for l in DOC.read_text(encoding="utf-8").splitlines() if l.startswith("## ")]
    assert [h.split(".")[0][3:] for h in heads] == [str(i) for i in range(13)]


def run(capsys, *argv):
    with pytest.raises(SystemExit) as e:
        main(list(argv))
    assert e.value.code in (0, None)
    return capsys.readouterr().out


def test_cli_prints_sku_bom_power_failsafe_indication_and_the_diagram(capsys):
    out = run(capsys, "hw", "sku")
    assert "DO-4 Pro" in out and "G_DP" in out
    data = json.loads(run(capsys, "hw", "sku", "--json"))
    assert [d["code"] for d in data] == [s.code for s in S.SKUS] and all(d["retail"] > 0 for d in data)
    assert "BOM" in run(capsys, "hw", "bom", "--sku", "DO-1")
    assert len(json.loads(run(capsys, "hw", "bom", "--json"))) == len(S.PARTS)
    assert "ВЫШЕ" in run(capsys, "hw", "power")
    assert "напрямую" in run(capsys, "hw", "failsafe")
    assert len(json.loads(run(capsys, "hw", "indication", "--json"))) >= 10
    assert "BRIDGE MCU" in run(capsys, "hw", "diagram")


def test_cli_refuses_an_unknown_sku_and_checks_the_document(capsys, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["hw", "bom", "--sku", "DO-9"])
    assert e.value.code == 4
    capsys.readouterr()
    stale = tmp_path / "H.md"
    stale.write_text("<!-- hw:sku -->\nold\n<!-- /hw:sku -->\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(["hw", "docs", "--check", "--path", str(stale)])
    assert e.value.code == 1
    capsys.readouterr()
