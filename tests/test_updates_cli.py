"""`dataopen update ...`, the document it generates, and the phone's texts for every reason the device can give."""
import json
import re
import struct
from pathlib import Path


from dataopen.cli import build_parser
from dataopen.ctl.identity import FileKeyStore, provision
from dataopen.updates import channels as C
from dataopen.updates import package as K
from dataopen.updates import report as RP

import pkg_helpers as H

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "UPDATES.md"
I18N = (ROOT / "pwa" / "js" / "i18n.js").read_text(encoding="utf-8")


def run(*argv):
    a = build_parser().parse_args(["update", *argv])
    return a.fn(a)


def test_the_document_matches_the_code_it_is_generated_from():
    assert RP.doc_is_current(DOC), "run `dataopen update docs --write`"
    text = DOC.read_text(encoding="utf-8")
    for h in ("## 0. Честный статус", "## 1. Два канала", "## 2. Форматы", "## 3. Потоки", "## 4. Поведение при ошибках",
              "## 5. Как это стыкуется",
              "## 7. Что сознательно не сделано"):
        assert h in text
    assert run("docs", "--check") == 0


def test_the_header_table_is_the_struct():
    sizes = [int(m.group(1) or 1) * (1 if m.group(2) == "s" else 0) or struct.calcsize("<" + (m.group(1) or "") + m.group(2))
             for m in re.finditer(r"(\d+)?([sBHIQ])", K.HEAD_FMT)]
    assert [n for _, n, _ in K.FIELDS] == sizes and sum(sizes) == K.HEAD == struct.calcsize(K.HEAD_FMT)
    assert [n for n, _, _ in K.FIELDS][:3] == ["magic", "ver", "suite"]


def test_the_channels_differ_in_everything_that_matters():
    a, b = C.BY_KEY["A"], C.BY_KEY["B"]
    for field in ("signer", "trust_root", "audience", "binding", "confidentiality", "replay", "staging", "transport"):
        assert getattr(a, field) != getattr(b, field), field
    assert "FW_" in a.transport and "PKG_" in b.transport and "производител" in a.signer and "отправител" in b.signer
    assert "нет" in a.confidentiality and "всегда" in b.confidentiality


def test_the_phone_has_words_for_every_reason_in_both_languages():
    ru, en = I18N.split("const STRINGS", 1)[1].split("en: {", 1) if "en: {" in I18N else (None, None)
    assert ru and en
    for r in C.REASONS:
        prefix = "err.pkg_rejected." if r.channel == "B" else "err.fw_rejected."
        if r.channel == "A" and r.key == "interrupted":
            continue                                                   # a power cut: nobody is there to read a message
        for lang, block in (("ru", ru), ("en", en)):
            assert f"'{prefix}{r.key}':" in block, (lang, prefix, r.key)
        if r.channel == "B":                                             # the new texts are generated from the table: the same words
            assert re.escape(r.ru) and r.ru.replace("'", "\\'") in ru and r.en.replace("'", "\\'") in en, r.key
    for key in ("pkg.progress", "pkg.loaded", "pkg.loaded_button", "err.pkg_rejected", "err.physical.model", "err.busy.fw_trial"):
        assert f"'{key}':" in ru and f"'{key}':" in en, key


def test_the_tables_print_and_json_has_every_reason(capsys):
    for sub in ("channels", "policy", "invariants", "parts", "format-a", "format-b"):
        assert run(sub) == 0 and capsys.readouterr().out.count("|") > 6
    assert run("reasons", "--channel", "B", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["key"] for r in rows} == C.REASON_KEYS["B"] and all(r["channel"] == "B" for r in rows)
    assert run("reasons") == 0 and "wrong_channel" in capsys.readouterr().out


def test_build_inspect_verify_check_model_round_trip(tmp_path, capsys):
    dev_dir = tmp_path / "dev"
    dev = provision(FileKeyStore(dev_dir / "keys" / "keys.json"))
    card = tmp_path / "dev.docard"
    card.write_text(json.dumps(dev.card().to_json()), encoding="utf-8")
    model = tmp_path / "m.onnx"
    model.write_bytes(H.tiny_model())
    out = tmp_path / "p.dopk"
    assert run("check-model", str(model)) == 0 and json.loads(capsys.readouterr().out)["ok"]
    assert run("sender", "init", "--dir", str(tmp_path / "s")) == 0
    assert run("build-package", "--to", str(card), "--sender-dir", str(tmp_path / "s"), "--out", str(out), "--slot", "2", "--tuning", "7,3",
               "--name", "Работа", "--model", str(model), "--model-name", "icons", "--min-fw", "1") == 0
    assert "written" in capsys.readouterr().out and out.read_bytes()[:4] == b"DOPK"
    assert run("inspect", str(out)) == 0
    head = json.loads(capsys.readouterr().out)
    assert head["kinds"] == ["tuning", "meta", "model"] and head["min_fw"] == 1 and not head["self"]
    assert run("verify", str(out), "--device-dir", str(dev_dir), "--fw-version", "1") == 0
    v = json.loads(capsys.readouterr().out)
    assert v["ok"] and v["slot"] == 2 and v["model"]["name"] == "icons"
    assert run("verify", str(out), "--device-dir", str(dev_dir), "--fw-version", "0") == 1             # the system is too old for it
    assert "needs_update" in capsys.readouterr().out
    other = tmp_path / "other"
    provision(FileKeyStore(other / "keys" / "keys.json"))
    assert run("verify", str(out), "--device-dir", str(other)) == 1 and "wrong_device" in capsys.readouterr().out
    assert run("verify", str(out), "--device-dir", str(tmp_path / "nowhere")) == 2


def test_the_tools_refuse_what_the_device_would_refuse(tmp_path, capsys):
    dev = provision(FileKeyStore(tmp_path / "dev" / "keys" / "keys.json"))
    card = tmp_path / "dev.docard"
    card.write_text(json.dumps(dev.card().to_json()), encoding="utf-8")
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(H.tiny_model(op="Gemm"))
    assert run("check-model", str(bad)) == 1 and "refused: model_ops" in capsys.readouterr().out
    out = tmp_path / "p.dopk"
    assert run("build-package", "--to", str(card), "--sender-dir", str(tmp_path / "s"), "--out", str(out), "--tuning", "1,1") == 2
    assert "sender init" in capsys.readouterr().out and not out.exists()
    assert run("sender", "init", "--dir", str(tmp_path / "s")) == 0
    assert run("build-package", "--to", str(card), "--sender-dir", str(tmp_path / "s"), "--out", str(out), "--model", str(bad)) == 2
    assert not out.exists()                                                       # a model the device would refuse is not sealed
    assert run("build-package", "--to", str(card), "--sender-dir", str(tmp_path / "s"), "--out", str(out), "--tuning", "1") == 2
    gone = str(tmp_path / "no.docard")
    assert run("build-package", "--to", gone, "--sender-dir", str(tmp_path / "s"), "--out", str(out), "--tuning", "1,1") == 2
    assert run("inspect", str(card)) == 2                                          # a card is not a package
    assert run("check-model", str(tmp_path / "missing.onnx")) == 2
    stale = tmp_path / "U.md"
    stale.write_text(DOC.read_text(encoding="utf-8").replace("| `magic` |", "| `mogic` |", 1), encoding="utf-8")
    assert run("docs", "--check", "--path", str(stale)) == 1 and "stale" in capsys.readouterr().out
