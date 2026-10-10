"""A model that was really exported by `dataopen ui export` goes the whole way a custom model goes (docs/MODELS.md): the device's check, a package
from the owner's own sender, the button, the slot, a detector built from memory. Run in the `detector` CI job (PyTorch)."""
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from dataopen.ctl import protocol as P  # noqa: E402
from dataopen.ctl.sim import World, seed_profile  # noqa: E402
from dataopen.runtime.frames import Frame  # noqa: E402
from dataopen.ui.export import export_onnx  # noqa: E402
from dataopen.ui.model import UiConfig, UiNet  # noqa: E402
from dataopen.ui.resident import detector_for_slot  # noqa: E402
from dataopen.ui.taxonomy import NAMES  # noqa: E402
from dataopen.updates import models as MD  # noqa: E402
from dataopen.updates import package as K  # noqa: E402
from dataopen.updates import sender as SD  # noqa: E402

from test_ctl_gateway import ok  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


def exported(tmp_path, **cfg) -> bytes:
    export_onnx(UiNet(UiConfig(**cfg)), tmp_path / "m.onnx")
    return (tmp_path / "m.onnx").read_bytes()


def test_the_real_exported_model_follows_the_contract_and_costs_what_the_limit_assumes(tmp_path):
    raw = exported(tmp_path)
    found = MD.check_model(raw, MD.card_from_onnx(raw, "real"))
    assert found["outputs"] == ["p3", "p4", "p5"] and found["convention"] == "uint8 NHWC RGB"
    from dataopen.ui import policy as PL
    assert found["macs"] == pytest.approx(PL.REF_MACS, rel=0.01)       # the calibration constant is this network, counted by the device's own counter
    assert found["est_infer_ms"]["p95"] == pytest.approx(PL.INFER_MS["cpu"][1], abs=0.2) and found["est_infer_ms"]["fits"]
    assert found["nodes"] < MD.MAX_NODES and len(raw) < MD.MODEL_MAX


def test_a_small_and_a_wide_real_model_both_pass_and_the_cost_follows_the_width(tmp_path):
    small = MD.check_model(*(lambda r: (r, MD.card_from_onnx(r, "s")))(exported(tmp_path, width=0.25, neck=16)))
    wide = MD.check_model(*(lambda r: (r, MD.card_from_onnx(r, "w")))(exported(tmp_path, width=1.0, neck=96)))
    assert small["macs"] < wide["macs"]


def test_the_real_model_goes_from_the_owner_through_the_button_to_a_working_detector(tmp_path):
    d = tmp_path / "gw"
    seed_profile(tmp_path / "seed", "tremor")
    shutil.copytree(tmp_path / "seed", d)
    w = World(d)
    w.phone.connect()
    raw = exported(tmp_path)
    card = MD.card_from_onnx(raw, "real", 1)
    me = SD.init(tmp_path / "keys")
    r = w.phone.pkg_send(K.build_package(me, w.gw.identity.card(), 1, slot=0, model=(raw, card)), piece=4000)
    assert r.type == P.T_ACK and r.json()["pending"]["model"]["name"] == "real"
    refused = w.phone.act("pkg.apply", True)
    assert refused.json()["code"] == P.E.PHYSICAL and refused.json()["detail"] == f"trust:{me.id}"
    w.gw.physical_press(w.t)
    ok(w.phone.act("pkg.apply", True))
    st = w.phone.get_state()
    assert st["model.state"] == "ok" and st["model.from"] == me.id
    det = detector_for_slot(w.gw, 0, conf=0.01)
    assert det is not None and det.classes == tuple(NAMES)
    out = det.detect(Frame(np.random.default_rng(0).integers(0, 256, (640, 640, 3), dtype=np.uint8), 0, 0))
    assert all(x.cls in NAMES for x in out)
    ok(w.phone.act("model.clear", True))
    assert detector_for_slot(w.gw, 0) is None and w.phone.get_state()["model.state"] == "none"
