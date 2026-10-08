"""Fixtures for simulation and tests of the update channels (NOT used by the device): a sender with its own keys, tiny ONNX graphs, and the
packages the browser tests ask the simulated device's server for (`/sim/package/...`). The `onnx` package is needed for the graphs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..ctl.identity import FileKeyStore, provision
from ..ui.taxonomy import NAMES
from . import models as MD
from . import package as K

LAYOUT_OK = {"kind": "ui-elements", "classes": list(NAMES), "n_cls": len(NAMES), "strides": [8, 16, 32], "input_size": 640, "n_keypoints": 0}


def sender(directory: str | Path, name: str = "sender"):
    """Another person's (or another device's) identity: its own keys, its own card. Created once, then reused from the directory."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    store = FileKeyStore(d / f"{name}.json")
    from ..ctl.identity import load_identity
    return load_identity(store) if store.load() is not None else provision(store)


def tiny_model(op: str = "Identity", domain: str = "", layout: Optional[dict] = LAYOUT_OK, external: bool = False, opset: int = 13,
               extra_domain: Optional[str] = None, pad: int = 0) -> bytes:
    """A few-hundred-byte ONNX graph: one operator, the UI layout in its metadata. Valid for the structural check, useless for detection.
    `pad`: that many float weights are added (the graph ignores them) to make the file bigger."""
    import onnx  # noqa: F401
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("frame", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("p3", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(op, ["frame"], ["p3"], domain=domain)
    inits = []
    if external:
        t = helper.make_tensor("w", TensorProto.FLOAT, [1], [0.0])
        t.data_location = TensorProto.EXTERNAL
        e = t.external_data.add()
        e.key, e.value = "location", "weights.bin"
        inits.append(t)
    if pad:
        import numpy as np
        from onnx import numpy_helper
        inits.append(numpy_helper.from_array((np.arange(pad) % 251).astype(np.float32), "padding"))
    g = helper.make_graph([node], "g", [x], [y], initializer=inits)
    imports = [helper.make_opsetid(domain, opset)] if domain else [helper.make_opsetid("", opset)]
    if extra_domain:
        imports.append(helper.make_opsetid(extra_domain, 1))
    m = helper.make_model(g, opset_imports=imports)
    m.ir_version = 8
    if layout is not None:
        p = m.metadata_props.add()
        p.key, p.value = "ui_layout", json.dumps(layout)
    return m.SerializeToString()


def card_of(raw: bytes, classes=NAMES, **kw) -> dict:
    return MD.card_for(raw, classes, **kw)


def sim_package(kind: str, recipient_card, directory: str | Path, seq: int, slot: Optional[int] = None) -> bytes:
    """What the browser tests fetch from the dev server: kinds are 'tuning', 'model', 'big' (a model of about 160 KB), 'forged' (a bit of the
    signed header changed), 'other' (made for a different device), 'tampered' (a byte of the body changed)."""
    me = sender(directory)
    big = tiny_model(pad=40000)
    if kind == "tuning":
        return K.build_package(me, recipient_card, seq, slot=slot, tuning=(8, 3), name="Из приложения")
    if kind == "model":
        m = tiny_model()
        return K.build_package(me, recipient_card, seq, slot=slot, model=(m, card_of(m, name="icons", version=seq)))
    raw = K.build_package(me, recipient_card if kind != "other" else sender(directory, "stranger").card(), seq, slot=slot,
                          model=(big, card_of(big, name="big", version=seq)))
    if kind == "forged":
        raw = bytes(raw[:30]) + bytes([raw[30] ^ 1]) + bytes(raw[31:])
    elif kind == "tampered":
        i = K.PREFIX + len(raw) // 2
        raw = bytes(raw[:i]) + bytes([raw[i] ^ 1]) + bytes(raw[i + 1 :])
    elif kind not in ("big", "other"):
        raise ValueError(kind)
    return raw
