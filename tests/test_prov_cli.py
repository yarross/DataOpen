"""`dataopen prov ...` and `dataopen recover ...`, and the generated tables of docs/PROVISIONING.md."""
import json
import shutil
from pathlib import Path

import pytest

from dataopen.cli import main
from dataopen.provisioning import recovery as RC
from dataopen.provisioning import report as RP
from dataopen.provisioning import station as ST
from prov_helpers import provisioned

DOC = Path(__file__).resolve().parents[1] / "docs" / "PROVISIONING.md"
needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


def run(capsys, *argv, code=0):
    with pytest.raises(SystemExit) as e:
        main(list(argv))
    assert (e.value.code or 0) == code, capsys.readouterr().out
    return capsys.readouterr().out


def test_the_generated_tables_in_the_document_are_current():
    assert RP.doc_is_current(DOC), "regenerate: dataopen prov docs --write"


def test_every_table_is_used_in_the_document_and_the_sections_follow_the_brief():
    text = DOC.read_text(encoding="utf-8")
    for name in RP.TABLES:
        assert f"<!-- fp:{name} -->" in text and f"<!-- /fp:{name} -->" in text, name
    heads = [l for l in text.splitlines() if l.startswith("## ")]
    assert [h.split(".")[0][3:] for h in heads] == [str(i) for i in range(11)]


def test_the_scenario_table_is_computed_from_the_model_not_typed():
    t = RP.scenarios_table()
    for s in RC.SCENARIOS:
        assert s.event in t
    assert t.count("\n") == len(RC.SCENARIOS) + 1


def test_verify_label_and_the_reference_tables(tmp_path, capsys):
    d, hsm, rep = provisioned(tmp_path)
    out = run(capsys, "prov", "verify", "--dir", str(d))
    assert rep.serial in out and "verified" in out
    label = json.loads(run(capsys, "prov", "label", "--dir", str(d)))
    assert label["serial"] == rep.serial and json.loads(label["qr"])["serial"] == rep.serial
    assert len(json.loads(run(capsys, "prov", "levels", "--json"))) == 4
    assert {i["key"] for i in json.loads(run(capsys, "prov", "storage", "--json"))} >= {"serial", "dak", "owner_keys", "fw_mcu"}
    for name in ("steps", "checks", "identity", "lifecycle"):
        assert run(capsys, "prov", name).strip()
    assert "not provisioned" in run(capsys, "prov", "verify", "--dir", str(tmp_path / "empty"), code=1)


def test_recover_simulate_and_the_full_return_need_a_hand(tmp_path, capsys):
    out = run(capsys, "recover", "simulate", "mcu_dead")
    assert "mouse direct" in out and "computer: not needed" in out
    assert run(capsys, "recover", "simulate", "nope", code=4)
    data = [json.loads(l) for l in run(capsys, "recover", "simulate", "all", "--json").splitlines()]
    assert {d["scenario"] for d in data} == {s.key for s in RC.SCENARIOS}
    d, hsm, rep = provisioned(tmp_path)
    assert "presence" in run(capsys, "recover", "full-return", "--dir", str(d), code=1)
    out = json.loads(run(capsys, "recover", "full-return", "--dir", str(d), "--hand-on-device"))
    assert out["serial"] == rep.serial and out["lifecycle"] == "SHIPPED"


@needs_cc
def test_provision_one_board_and_a_faulty_one_from_the_command_line(tmp_path, capsys):
    d = tmp_path / "board"
    out = run(capsys, "prov", "provision", "--dir", str(d), "--json")
    j = json.loads(out)
    assert not j["quarantined"] and j["label"]["serial"].startswith("DO1-") and j["dir"] == str(d)
    bad = run(capsys, "prov", "provision", "--dir", str(tmp_path / "bad"), "--fault", "relay_stuck", code=1)
    assert "QUARANTINED at selftest" in bad and "FAIL power_cut_bypass" in bad
    assert "no such SKU" in run(capsys, "prov", "provision", "--dir", str(tmp_path / "x"), "--sku", "DO-9", code=4)


def test_docs_check_notices_a_stale_document(tmp_path, capsys):
    stale = tmp_path / "P.md"
    stale.write_text("<!-- fp:steps -->\nold\n<!-- /fp:steps -->\n", encoding="utf-8")
    assert "stale" in run(capsys, "prov", "docs", "--check", "--path", str(stale), code=1)


def test_the_dev_server_starts_with_a_provisioned_device(tmp_path):
    from dataopen.ctl.sim import dev_vendor
    d = tmp_path / "srv"
    ST.provision(d, ST.VendorHsm(dev_vendor()[0]), jig=ST.SkipJig())
    from dataopen.provisioning.device import DeviceAgent
    assert DeviceAgent(d, hw_id=ST.DEFAULT_HW, vendor_pub=dev_vendor()[1]).record.serial.startswith("DO1-")
