"""`dataopen latency ...` and the generated document."""
import json
from pathlib import Path

from dataopen.cli import build_parser
from dataopen.latency import analyze as AN
from dataopen.latency import report as RP

DOC = Path(__file__).resolve().parents[1] / "docs" / "LATENCY.md"


def run(*argv):
    a = build_parser().parse_args(["latency", *argv])
    return a.fn(a)


def test_budget_prints_every_scenario_and_the_json_has_the_margins(capsys):
    assert run("budget") == 0
    out = capsys.readouterr().out
    assert "legacy-fifo" in out and "npu-latest-c1" in out and "помощь пропадает" in out
    assert run("budget", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["scenario"] for r in rows} >= {"legacy-fifo", "cpu-latest", "v1-npu", "npu-latest-c1"}
    assert all({"A_pipeline", "A_total", "B", "ttl_margin_ms", "guard_margin_ms"} <= set(r) for r in rows)


def test_stages_name_the_basis_and_an_unknown_scenario_is_a_usage_error(capsys):
    assert run("stages", "v1-npu") == 0
    out = capsys.readouterr().out
    assert "не измерено" in out and "B7" in out and "A4" in out
    assert run("stages", "v1-npu", "--json") == 0
    st = json.loads(capsys.readouterr().out)
    assert {s["id"] for s in st} >= {"B1", "B4", "B7", "A1", "A3", "A4", "C2"} and all(s["basis"] for s in st)
    assert run("stages", "nope") == 2
    assert "one of" in capsys.readouterr().out


def test_testpoints_are_listed(capsys):
    assert run("testpoints") == 0
    out = capsys.readouterr().out
    assert "фотодиод" in out and "анализатор USB" in out


def test_analyze_reports_percentiles_and_fails_on_a_late_tail_or_a_miss(tmp_path, capsys):
    f = tmp_path / "ok.csv"
    AN.synth_capture(f, n=100, mean_ms=6.0, jitter_ms=1.0)
    assert run("analyze", str(f), "--unit", "ms", "--limit-ms", "8") == 0
    res = json.loads(capsys.readouterr().out)
    assert res["pairs"] == 100 and res["p99"] < 8.0
    assert run("analyze", str(f), "--unit", "ms", "--limit-ms", "5") == 1          # p99 above the limit
    assert "FAIL" in capsys.readouterr().out
    g = tmp_path / "miss.csv"
    AN.synth_capture(g, n=50, mean_ms=6.0, jitter_ms=1.0, miss=0.3, seed=5)
    # an answer that never came is a failure, not a smaller sample
    assert run("analyze", str(g), "--unit", "ms", "--limit-ms", "50") == 1
    assert run("analyze", str(tmp_path / "none.csv")) == 2
    assert run("analyze", str(f), "--unit", "frames") == 2                         # no frame rate


def test_the_document_matches_the_code_it_is_generated_from():
    text = DOC.read_text(encoding="utf-8")
    assert RP.doc_is_current(DOC), "run `dataopen latency docs --write`"
    heads = ("## 0. Честный статус", "## 2. Критический путь", "## 3. Бюджет", "## 5. Физические", "## 6. Методика измерения")
    for h in heads + ("## 7. Главные риски",):
        assert h in text
    for needle in ("умолчания v1", "scene_ttl_ms", "не измерено", "latest_only"):
        assert needle in text


def test_the_docs_command_checks_and_refuses_a_stale_copy(tmp_path, capsys):
    assert run("docs", "--check") == 0
    stale = tmp_path / "L.md"
    stale.write_text(DOC.read_text(encoding="utf-8").replace("| `tremor` | включён |", "| `tremor` | выключен |", 1), encoding="utf-8")
    assert run("docs", "--check", "--path", str(stale)) == 1
    assert "stale" in capsys.readouterr().out
