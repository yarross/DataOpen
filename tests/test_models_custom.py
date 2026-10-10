"""A person brings their own interface model (docs/MODELS.md): the rules the device applies, the flow from the owner's side, taking it back, and
what does not change (the model stays on the device; nothing of it is for sale or on show).

What is proved here is the CHECKS and the FLOW, in simulation. It is not proved that a model does not detect people: labels, shapes, the list
of operators, the binding to one device, a trusted sender and a button for every model are barriers, not a measurement of behaviour."""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from dataopen.cli import build_parser
from dataopen.ctl import protocol as P
from dataopen.ctl import residency as RS
from dataopen.ctl import residency_probe as RP
from dataopen.ctl.identity import FileKeyStore, provision
from dataopen.ctl.sim import World, seed_profile
from dataopen.runtime.frames import Frame
from dataopen.ui.taxonomy import NAMES
from dataopen.updates import channels as C
from dataopen.updates import models as MD
from dataopen.updates import package as K
from dataopen.updates import report as RPT
from dataopen.updates import sender as SD

import pkg_helpers as H
from test_ctl_gateway import ok
from test_updates_gateway import apply_, detail, give, leftovers, pending

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "dataopen"
DOC = ROOT / "docs" / "MODELS.md"

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")
MODEL = H.tiny_model()
CARD = H.card_of(MODEL, name="mine", version=1)


def refused(raw, card=None, **kw):
    with pytest.raises(C.UpdateError) as e:
        MD.check_model(raw, card or H.card_of(raw, **kw))
    return e.value.key


def with_extra_nodes(raw, n):
    import onnx
    from onnx import helper
    m = onnx.load_from_string(raw)
    m.graph.node.extend(helper.make_node("Identity", ["chw"], [f"dead{i}"]) for i in range(n))
    return m.SerializeToString()


def with_subgraph(raw):
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load_from_string(raw)
    inner = helper.make_graph([helper.make_node("Identity", ["a"], ["b"])], "inner", [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])])
    m.graph.node.append(helper.make_node("Identity", ["chw"], ["with_body"], body=inner))
    return m.SerializeToString()


# ---------------------------------------------------------------------------------------------------------------- 1. the contract, rule by rule
def test_a_model_that_follows_the_contract_is_accepted_and_the_check_tells_what_it_found():
    found = MD.check_model(MODEL, CARD)
    assert found["outputs"] == ["p3", "p4", "p5"] and found["convention"] == MD.DEFAULT_CONVENTION
    assert 0 < found["macs"] < MD.MAX_MACS and 0 < found["nodes"] < MD.MAX_NODES and set(found["ops"]) <= MD.ALLOWED_OPS
    for opset in (10, 11, 13, 20):                                              # the whole range of opsets the contract allows works
        raw = H.tiny_model(opset=opset)
        assert MD.check_model(raw, H.card_of(raw, opset=opset))["opset"] == opset
    f32 = H.tiny_model(float_input=True)                                          # the second input convention is a contract too
    assert MD.check_model(f32, H.card_of(f32))["convention"] == "float NCHW 0..255"


@pytest.mark.parametrize("op", ["Gemm", "MatMul", "Softmax", "Loop", "If", "Scan", "NonMaxSuppression", "Einsum", "TopK", "Gather", "Sigmoid"])
def test_an_operator_outside_the_list_is_refused(op):
    assert refused(H.tiny_model(op=op)) == "model_ops"


def test_a_model_is_data_not_code_files_domains_and_subgraphs_are_refused():
    assert refused(H.tiny_model(external=True)) == "model_files"                  # a second file to be read from the device's disk
    assert refused(H.tiny_model(domain="com.evil")) == "model_files"              # an operator set from outside
    assert refused(H.tiny_model(extra_domain="com.evil")) == "model_files"
    assert refused(with_subgraph(H.tiny_model())) == "model_files"                # control flow hidden inside a node
    assert refused(H.tiny_model(opset=21), opset=20) == "model_ops"               # a newer operator set than the device knows
    assert refused(H.tiny_model(opset=6), opset=7) in {"model_ops", "model_io", "bad_model"}


@pytest.mark.parametrize("classes", [["person"], ["player_ct", "player_t"], ["button", "pose"], list(NAMES) + ["person"], []])
def test_people_pose_and_player_models_are_not_interface_models(classes):
    card = dict(CARD, classes=classes)
    assert refused(MODEL, card) == "not_ui_model"                                  # by what the card says
    layout = dict(H.LAYOUT_OK, classes=classes, n_cls=len(classes))
    raw = H.tiny_model(layout=layout)                                              # and by what the model says about itself
    assert refused(raw, H.card_of(raw, classes=classes)) in {"not_ui_model", "bad_model"}


def test_keypoints_are_not_interface_models_either():
    assert refused(MODEL, dict(CARD, n_keypoints=17)) == "not_ui_model"
    raw = H.tiny_model(layout=dict(H.LAYOUT_OK, n_keypoints=17))
    assert refused(raw) in {"not_ui_model", "bad_model"}
    coco = H.tiny_model(out_channels=84)                                           # a 80-class head with the box: the shape of a people detector
    assert refused(coco) == "model_io"


@pytest.mark.parametrize("kw", [dict(outputs=2), dict(outputs=4), dict(out_channels=9), dict(out_channels=13), dict(input_size=320),
                                dict(input_size=1280)])
def test_the_input_and_the_outputs_have_to_be_what_the_device_reads(kw):
    assert refused(H.tiny_model(**kw)) == "model_io"


def test_the_head_has_to_have_as_many_channels_as_the_model_says_it_has_classes():
    fewer = list(NAMES)[:7]
    layout = dict(H.LAYOUT_OK, classes=fewer, n_cls=7)
    honest = H.tiny_model(layout=layout)
    assert MD.check_model(honest, H.card_of(honest, classes=fewer))["outputs"] == ["p3", "p4", "p5"]
    lying = H.tiny_model(layout=layout, out_channels=len(NAMES) + 4)                # claims 7 classes, has the head of 8
    assert refused(lying, H.card_of(lying, classes=fewer)) == "model_io"


def test_a_model_that_costs_too_much_or_hides_its_cost_is_refused():
    assert refused(H.tiny_model(conv_channels=64000)) == "model_cost"
    assert refused(with_extra_nodes(H.tiny_model(), MD.MAX_NODES + 1)) == "model_cost"
    assert MD.MAX_MACS == 4_000_000_000 and MD.MAX_NODES == 2000                  # stated limits (docs/MODELS.md: assumptions, not measurements)


def test_size_card_and_format_are_checked_before_anything_else():
    big = bytes(MD.MODEL_MAX + 1)
    assert refused(big) == "too_large"
    assert refused(MODEL, dict(CARD, sha256="0" * 64)) == "bad_model"
    assert refused(MODEL, dict(CARD, size=len(MODEL) + 1)) == "bad_model"
    assert refused(MODEL + b"\0", CARD) == "bad_model"                              # bytes that are not the bytes of the card
    assert refused(b"not onnx", H.card_of(b"not onnx")) == "bad_model"
    nometa = H.tiny_model(layout=None)
    assert refused(nometa) == "bad_model"


def test_every_refusal_has_its_own_words_in_the_table_and_the_rules_name_only_reasons_that_exist():
    for key in ("model_ops", "model_files", "model_io", "model_cost", "not_ui_model", "too_large", "bad_model", "wrong_device", "bad_signature"):
        r = C.reason("B", key)
        assert r.ru and r.en and r.do
    for _, _, reasons in RPT.MODEL_RULES:
        for key in re.findall(r"`(\w+)`", reasons):
            assert key in C.REASON_KEYS["B"] | {"physical"}, key
    assert MD.ABI["version"] == 1 and MD.ABI["classes"] == list(NAMES)
    assert MD.ALLOWED_OPS == frozenset(MD.ABI["ops"])


# ---------------------------------------------------------------------------------------------------------------- 2. the device: bringing a model
@pytest.fixture(scope="module")
def tremor_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("mc-tremor")
    seed_profile(d, "tremor")
    return d


def world(tmp_path, tremor_dir=None, **kw) -> World:
    d = tmp_path / "gw"
    if tremor_dir is not None:
        shutil.copytree(tremor_dir, d)
    w = World(d, **kw)
    w.phone.connect()
    return w


def owner(tmp_path, name="owner"):
    """The owner's own sender: made by the tool, kept outside the device."""
    return SD.init(tmp_path / "owner-keys", name)


@needs_cc
def test_the_owner_brings_a_model_with_no_manufacturer_and_no_store(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    assert w.gw.fw is None                                                       # the device has no manufacturer key configured at all
    me = owner(tmp_path)
    raw = K.build_package(me, w.gw.identity.card(), 1, slot=1, name="Игры", model=(MODEL, H.card_of(MODEL, name="mine", version=1)))
    r = w.phone.pkg_send(raw, piece=1500)
    assert r.type == P.T_ACK
    sm = r.json()["pending"]                                                      # what the PWA shows: accepted, who from, what, how big
    assert sm["from"] == me.id and sm["model"] == {"name": "mine", "version": 1, "size": len(MODEL)} and sm["button"] == "trust"
    assert w.phone.get_state()["pkg.state"] == "pending" and not (w.gw.slotset[1].dir / "model.bin").exists()
    # a new sender: the first button is for trusting THEM
    r = apply_(w)
    assert r.json()["code"] == P.E.PHYSICAL and r.json()["detail"] == f"trust:{me.id}"
    ok(apply_(w, button=True))
    # and the weights need a button of their own, every time, from everybody
    assert w.gw.slotset[1].dir.joinpath("model.bin").exists()
    st = w.phone.get_state()
    assert st["pkg.state"] == "none" and leftovers(w) == []
    ok(w.phone.select_slot(1))
    st = w.phone.get_state()
    assert st["model.state"] == "ok" and st["model.name"] == "mine" and st["model.from"] == me.id


@needs_cc
def test_the_first_model_from_a_new_sender_costs_one_press_the_next_one_costs_one_again(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    me = owner(tmp_path)
    w.phone.pkg_send(K.build_package(me, w.gw.identity.card(), 1, slot=0, model=(MODEL, CARD)))
    r = apply_(w)
    assert r.json()["detail"] == f"trust:{me.id}"
    ok(apply_(w, button=True))                                                    # trust AND weights in one press: it is the same person, same moment
    assert w.phone.get_state()["model.state"] == "ok"
    m2 = H.tiny_model(opset=11)
    w.phone.pkg_send(K.build_package(me, w.gw.identity.card(), 2, slot=0, model=(m2, H.card_of(m2, name="mine", version=2, opset=11))))
    r = apply_(w)
    assert r.json()["code"] == P.E.PHYSICAL and r.json()["detail"] == f"model:{me.id}"       # a trusted sender, but the weights always need the button
    assert w.phone.get_state()["model.version"] == 1                                # the refusal changed nothing
    ok(apply_(w, button=True))
    assert w.phone.get_state()["model.version"] == 2


@needs_cc
@pytest.mark.parametrize("kw,key", [(dict(op="Gemm"), "model_ops"), (dict(external=True), "model_files"), (dict(outputs=2), "model_io"),
                                    (dict(conv_channels=64000), "model_cost"), (dict(layout=dict(H.LAYOUT_OK, classes=["person"])), "bad_model")])
def test_a_refusal_comes_with_its_reason_and_leaves_nothing_behind(tmp_path, tremor_dir, kw, key):
    w = world(tmp_path, tremor_dir)
    me = owner(tmp_path)
    bad = H.tiny_model(**kw)
    r = w.phone.pkg_send(K.build_package(me, w.gw.identity.card(), 1, model=(bad, H.card_of(bad))))
    assert detail(r) == key
    assert pending(w) is None and leftovers(w) == [] and w.phone.get_state()["model.state"] == "none"
    assert w.gw.trust["senders"] == {}                                             # being refused does not make anybody trusted


@needs_cc
def test_a_pose_card_is_refused_even_if_the_file_looks_fine(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    me = owner(tmp_path)
    for classes in (["person"], ["player_ct", "player_t"], list(NAMES) + ["head"]):
        r = w.phone.pkg_send(K.build_package(me, w.gw.identity.card(), 1, model=(MODEL, dict(CARD, classes=classes))))
        assert detail(r) == "not_ui_model" and leftovers(w) == []


@needs_cc
def test_a_package_for_another_device_never_gets_as_far_as_its_content(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    me = owner(tmp_path)
    other = provision(FileKeyStore(tmp_path / "other" / "keys.json"))
    r = w.phone.pkg_send(K.build_package(me, other.card(), 1, model=(MODEL, CARD)))
    assert detail(r) == "wrong_device" and leftovers(w) == []
    assert w.phone.get_state()["model.state"] == "none"


@needs_cc
def test_the_manufacturers_signature_buys_nothing_a_sender_has_to_be_trusted_like_any_other(tmp_path, tremor_dir):
    """The package channel has no manufacturer anywhere in it (no import, no key, no certificate). A package signed by the key that signs
    SYSTEM images is just a package from an unknown sender."""
    for name in ("package.py", "manager.py", "models.py", "sender.py"):
        text = (SRC / "updates" / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+\.{0,2}(ctl\.)?firmware", text, re.M), name
        assert "manufacturer_key" not in text and "vendor" not in text.lower().replace("vendor-neutral", ""), name
    w = world(tmp_path, tremor_dir)
    vendor = provision(FileKeyStore(tmp_path / "vendor" / "keys.json"))              # whoever holds the manufacturer's keys is just a sender here
    w.phone.pkg_send(K.build_package(vendor, w.gw.identity.card(), 1, model=(MODEL, CARD)))
    assert apply_(w).json()["detail"] == f"trust:{vendor.id}"


@needs_cc
def test_a_refused_press_is_not_spent_and_nothing_changes(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    me = owner(tmp_path)
    w.phone.pkg_send(K.build_package(me, w.gw.identity.card(), 1, model=(MODEL, CARD)))
    assert apply_(w).json()["code"] == P.E.PHYSICAL
    assert apply_(w).json()["code"] == P.E.PHYSICAL                                  # asking again does not make the button appear
    assert w.gw.trust["senders"] == {} and w.phone.get_state()["model.state"] == "none" and pending(w) is not None
    ok(w.phone.act("pkg.discard", True))                                              # "no": the package goes, nobody becomes trusted
    assert pending(w) is None and w.gw.trust["senders"] == {}


# ---------------------------------------------------------------------------------------------------------------- 3. back, off, and forget
@needs_cc
def test_the_previous_model_comes_back_and_goes_again(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    give(w, tmp_path, slot=0, model=(MODEL, CARD))
    m2 = H.tiny_model(opset=11)
    give(w, tmp_path, slot=0, model=(m2, H.card_of(m2, name="mine", version=2, opset=11)))
    assert w.phone.get_state()["model.version"] == 2
    ok(w.phone.act("pkg.revert", True))
    assert w.phone.get_state()["model.version"] == 1
    ok(w.phone.act("pkg.revert", True))
    assert w.phone.get_state()["model.version"] == 2


@needs_cc
def test_taking_the_model_out_leaves_the_profile_and_the_other_slots_and_needs_no_button(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    give(w, tmp_path, slot=0, model=(MODEL, CARD), tuning=(6, 4), name="Работа")
    m2 = H.tiny_model(opset=11)
    give(w, tmp_path, slot=1, model=(m2, H.card_of(m2, name="other", version=1, opset=11)), name="Игры")
    m3 = H.tiny_model(opset=12)
    give(w, tmp_path, slot=0, model=(m3, H.card_of(m3, name="mine", version=2, opset=12)))          # slot 0 now also has a previous model
    r = w.phone.act("model.clear", False)                                              # two steps: the first one only asks, nothing is touched
    assert detail(r, P.E.NOT_ALLOWED) == "confirm" and w.phone.get_state()["model.state"] == "ok"
    assert w.gw.physical_until < w.t                                                   # no press is waiting: none is needed
    ok(w.phone.act("model.clear", True))
    st = w.phone.get_state()
    assert st["model.state"] == "none" and st["model.name"] == "" and st["model.from"] == ""
    s0, s1 = w.gw.slotset[0], w.gw.slotset[1]
    assert (s0.name, s0.strength, s0.tremor) == ("Работа", 6, 4)                       # the rest of the slot is as it was
    for n in ("model.bin", "model.json", "model.prev.bin", "model.prev.json"):
        assert not (s0.dir / n).exists(), n                                            # the previous copy went too
    assert (s1.dir / "model.bin").exists()                                             # another slot's model was not touched
    assert detail(w.phone.act("model.clear", True)) == "no_model"                       # nothing left to take out


@needs_cc
def test_clearing_the_slot_takes_its_model_with_it_and_the_other_slots_keep_theirs(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    give(w, tmp_path, slot=0, model=(MODEL, CARD))
    m2 = H.tiny_model(opset=11)
    give(w, tmp_path, slot=1, model=(m2, H.card_of(m2, name="other", version=1, opset=11)))
    w.gw.physical_press(w.t)
    ok(w.phone.act("slot.clear", True))
    assert w.phone.get_state()["model.state"] == "none" and not (w.gw.slotset[0].dir / "model.bin").exists()
    assert (w.gw.slotset[1].dir / "model.bin").exists()


@needs_cc
def test_forgetting_the_senders_needs_the_button_and_the_next_package_asks_again(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    me = owner(tmp_path)
    w.phone.pkg_send(K.build_package(me, w.gw.identity.card(), 1, model=(MODEL, CARD)))
    ok(apply_(w, button=True))
    assert w.phone.get_state()["trusted.count"] == 1
    r = w.phone.act("trust.clear", True)
    assert r.json()["code"] == P.E.PHYSICAL and r.json()["detail"] == "trust.clear"
    assert w.phone.get_state()["trusted.count"] == 1                                    # not without the button
    w.gw.physical_press(w.t)
    ok(w.phone.act("trust.clear", True))
    assert w.phone.get_state()["trusted.count"] == 0 and w.phone.get_state()["model.state"] == "ok"      # the model stays: removing it is another action
    m2 = H.tiny_model(opset=11)
    w.phone.pkg_send(K.build_package(me, w.gw.identity.card(), 2, model=(m2, H.card_of(m2, name="mine", version=2, opset=11))))
    assert apply_(w).json()["detail"] == f"trust:{me.id}"                                # trusted again only by the button


@needs_cc
def test_an_empty_slot_and_an_empty_trust_list_are_said_plainly(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    assert detail(w.phone.act("model.clear", True)) == "no_model"
    w.gw.physical_press(w.t)
    assert detail(w.phone.act("trust.clear", True)) == "no_senders"


@needs_cc
def test_the_resets_leave_no_model_and_no_sender(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    give(w, tmp_path, slot=0, model=(MODEL, CARD))
    w.gw.physical_press(w.t)
    ok(w.phone.act("erase.profile", True))
    st = w.phone.get_state()
    assert st["model.state"] == "none" and st["trusted.count"] == 0
    assert not list(w.gw.dir.rglob("model*"))                                          # nothing of the model is left on the disk


# ---------------------------------------------------------------------------------------------------------------- 4. the model stays on the device
@needs_cc
def test_a_custom_model_is_resident_like_any_other(tmp_path):
    profile, model = RP.make_profile(), H.tiny_model(pad=4000, seed=11)
    w = World(tmp_path / "dev")
    w.phone.connect()
    RP.put_resident(w, tmp_path, profile, model)
    can = RP.canary_for(profile, model, w.gw.identity, tmp_path / "dev")
    asked = RP.ask_everything(w)
    w.run(200)
    wire = bytes(w.wire) + w.gw.read_status() + w.gw.read_info()
    assert len(asked) > 10 and can.hits(model) and not can.hits(wire)
    pk = w.phone.get_packages()                                                         # the only thing said about it is its public card
    assert set(pk["models"][0]["model"]) <= set(RS.MODEL_CARD_PUBLIC)
    for f in (tmp_path / "dev").rglob("*"):
        if f.is_file() and "senders" not in f.parts and f.suffix != ".tmp":
            assert not [h for h in can.hits(f.read_bytes()) if not h.startswith("keyfile")], f
    for kind in ("model", "weights", "export", "download"):
        assert w.phone.call(P.T_GET, {"what": kind}).json()["code"] == P.E.RESIDENT


def test_the_surface_for_models_is_what_the_document_says():
    assert {"model.clear", "trust.clear", "pkg.apply", "pkg.revert", "pkg.discard"} <= set(RS.ACT_KEYS)
    assert "model.from" in RS.STATE_KEYS
    assert not any(k in RS.ACT_KEYS + RS.SET_KEYS for k in ("model.export", "model.get", "model.download", "model.store", "model.catalog"))
    assert P.FILE_OPS_IN == ("bundle_put", "fw_put", "pkg_put") and P.FILE_OPS_OUT == ("card_get",)       # no file goes out but the card


def test_there_is_exactly_one_reader_of_the_weights_and_it_builds_the_detector_in_memory():
    calls = [p.relative_to(ROOT).as_posix() for p in SRC.rglob("*.py")
             for line in p.read_text(encoding="utf-8").splitlines() if re.search(r"\.weights\(\)", line) and not line.lstrip().startswith(("#", '"""'))]
    assert calls == ["src/dataopen/ui/resident.py"]
    text = (SRC / "ui" / "resident.py").read_text(encoding="utf-8")
    assert "write_bytes" not in text and "open(" not in text and "tempfile" not in text           # no file of the clear weights is made


@needs_cc
def test_the_resident_model_becomes_a_working_detector_without_a_file(tmp_path, tremor_dir):
    pytest.importorskip("onnxruntime")
    from dataopen.ui.resident import detector_for_slot
    w = world(tmp_path, tremor_dir)
    assert detector_for_slot(w.gw, 0) is None                                           # nothing to run
    give(w, tmp_path, slot=0, model=(MODEL, CARD))
    before = {p: p.stat().st_mtime_ns for p in (tmp_path).rglob("*") if p.is_file()}
    det = detector_for_slot(w.gw, 0)
    assert det is not None and det.classes == tuple(NAMES)
    out = det.detect(Frame(np.random.default_rng(0).integers(0, 256, (640, 640, 3), dtype=np.uint8), 0, 0))
    assert isinstance(out, list)                                                        # the tiny graph has no real detections: the point is that it runs
    after = {p: p.stat().st_mtime_ns for p in (tmp_path).rglob("*") if p.is_file()}
    assert after == before and not list(tmp_path.rglob("*.onnx"))                       # running it wrote nothing
    # a model that needs a newer system is not run; a file on the disk that no longer matches the card is refused
    meta = w.gw.slotset[0].dir / "model.json"
    assert meta.exists()
    w.gw._fw_version = lambda: -1
    assert detector_for_slot(w.gw, 0) is None


@needs_cc
def test_a_model_changed_on_the_disk_is_not_run(tmp_path, tremor_dir):
    pytest.importorskip("onnxruntime")
    from dataopen.ctl.vault import VaultError  # noqa: F401
    from dataopen.ui.resident import detector_for_slot
    from dataopen.updates.manager import ModelStore
    w = world(tmp_path, tremor_dir)
    give(w, tmp_path, slot=0, model=(MODEL, CARD))
    other = H.tiny_model(opset=12)
    slot = w.gw.slotset[0]
    (slot.dir / "model.bin").write_bytes(slot.vault.seal("model.bin", other))                # a changed file, sealed under the right key
    assert ModelStore(slot).weights() == other
    with pytest.raises(C.UpdateError) as e:
        detector_for_slot(w.gw, 0)
    assert e.value.key == "bad_model"


# ---------------------------------------------------------------------------------------------------------------- 5. the sender's tools
def run(*argv):
    a = build_parser().parse_args(["update", *argv])
    return a.fn(a)


def test_the_owner_makes_an_identity_once_and_it_is_not_replaced(tmp_path, capsys):
    d = str(tmp_path / "keys")
    assert run("sender", "show", "--dir", d) == 2 and "sender init" in capsys.readouterr().out
    assert run("sender", "init", "--dir", d) == 0
    first = json.loads(capsys.readouterr().out.split("\n}\n")[0] + "\n}")
    assert re.fullmatch(r"[0-9A-Z]{4}(-[0-9A-Z]{4}){3}", first["id"]) and first["what_the_device_shows"] == f"trust:{first['id']}"
    assert run("sender", "init", "--dir", d) == 2 and "already" in capsys.readouterr().out          # a second init does not change the ID
    assert run("sender", "show", "--dir", d) == 0
    assert json.loads(capsys.readouterr().out)["id"] == first["id"]
    key_file = tmp_path / "keys" / "sender.json"
    assert (key_file.stat().st_mode & 0o777) == 0o600 and (key_file.parent.stat().st_mode & 0o777) == 0o700
    assert SD.load(d).id == first["id"]


@needs_cc
def test_the_tools_and_the_device_agree_on_the_number_and_on_what_is_refused(tmp_path, tremor_dir, capsys):
    w = world(tmp_path, tremor_dir)
    d = str(tmp_path / "keys")
    card = tmp_path / "dev.docard"
    card.write_text(json.dumps(w.phone.get_identity()), encoding="utf-8")
    model = tmp_path / "m.onnx"
    model.write_bytes(MODEL)
    assert run("sender", "init", "--dir", d) == 0
    mine = SD.load(d).id
    capsys.readouterr()
    out = tmp_path / "p.dopk"
    assert run("build-package", "--to", str(card), "--sender-dir", d, "--out", str(out), "--slot", "1", "--model", str(model),
               "--model-name", "mine") == 0
    said = capsys.readouterr().out
    assert f"trust:{mine}" in said and f"model:{mine}" in said                                   # the tool says what the device will ask
    r = w.phone.pkg_send(out.read_bytes())
    assert r.type == P.T_ACK and r.json()["pending"]["from"] == mine
    assert apply_(w).json()["detail"] == f"trust:{mine}"                                         # the device asks for the same number
    ok(apply_(w, button=True))
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(H.tiny_model(op="Loop"))
    assert run("check-model", str(bad)) == 1
    text = capsys.readouterr().out
    assert "model_ops" in text and "to do" in text
    assert run("build-package", "--to", str(card), "--sender-dir", d, "--out", str(tmp_path / "x.dopk"), "--model", str(bad)) == 2
    assert not (tmp_path / "x.dopk").exists()                                                    # the tool will not seal what the device would refuse


def test_the_tools_print_the_contract_and_the_numbers(tmp_path, capsys):
    assert run("abi") == 0 and "ABI v1" in capsys.readouterr().out
    assert run("abi", "--json") == 0
    j = json.loads(capsys.readouterr().out)
    assert j == json.loads(json.dumps(MD.ABI))
    m = tmp_path / "m.onnx"
    m.write_bytes(MODEL)
    assert run("check-model", str(m)) == 0
    found = json.loads(capsys.readouterr().out)
    assert found["ok"] and found["outputs"] == ["p3", "p4", "p5"] and found["limits"]["max_gmacs"] == 4.0 and found["gmacs"] < 4


def test_the_document_and_its_tables_are_current_and_say_what_is_not_done():
    assert RPT.doc_is_current(DOC), "run `dataopen update docs --write`"
    text = DOC.read_text(encoding="utf-8")
    for h in ("## 0. Честный статус", "## 1. Зачем и для кого", "## 2. Что можно и что нельзя", "## 3. Как это выглядит для человека",
              "## 4. Что проверяет устройство", "## 5. Стыковка", "## 6. Что сознательно не сделано"):
        assert h in text, h
    for honest in ("не доказывает", "RKNN", "демон", "допущение"):
        assert honest in text, honest
    assert DOC.name == "MODELS.md" and RPT.expected_tables(DOC) == {"abi", "model_rules"}


# ---------------------------------------------------------------------------------------------------------------- 6. the phone stays dumb
def test_the_phone_has_words_for_every_new_refusal_and_action_in_both_languages():
    i18n = (ROOT / "pwa" / "js" / "i18n.js").read_text(encoding="utf-8")
    ru, en = i18n.split("en: {", 1)
    for key in ("model_ops", "model_files", "model_io", "model_cost", "no_model", "no_senders"):
        assert f"'err.pkg_rejected.{key}':" in ru and f"'err.pkg_rejected.{key}':" in en, key
    assert "'err.physical.trust.clear':" in ru and "'err.physical.trust.clear':" in en


def test_the_client_has_no_parser_for_models_and_does_not_look_inside_a_package():
    for p in (ROOT / "pwa" / "js").glob("*.js"):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"onnx|protobuf|ModelProto|GraphProto|initializer|\.onnx\b", text, re.I), p.name
