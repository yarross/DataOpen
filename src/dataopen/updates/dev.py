"""Fixtures for simulation and tests of the update channels (NOT used by the device): a sender with its own keys, tiny ONNX graphs, and the
packages the browser tests ask the simulated device's server for (`/sim/package/...`). The `onnx` package is needed for the graphs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..ui.taxonomy import NAMES
from . import models as MD
from . import package as K

LAYOUT_OK = {"kind": "ui-elements", "classes": list(NAMES), "n_cls": len(NAMES), "strides": [8, 16, 32], "input_size": 640, "n_keypoints": 0}


def sender(directory: str | Path, name: str = "sender"):
    """Another person's (or another device's) identity: its own keys, its own card. Created once, then reused from the directory."""
    from . import sender as SD
    return SD.load(directory, name) or SD.init(directory, name)


def tiny_model(op: str = "Identity", domain: str = "", layout: Optional[dict] = LAYOUT_OK, external: bool = False, opset: int = 13,
               extra_domain: Optional[str] = None, pad: int = 0, seed: Optional[int] = None, float_input: bool = False,
               outputs: int = 3, out_channels: Optional[int] = None, input_size: int = 640, conv_channels: int = 3) -> bytes:
    """A few-hundred-byte ONNX detector that follows the contract of docs/MODELS.md: a uint8 NHWC frame (or float NCHW) in, three outputs
    [1, n_classes + 4, 80|40|20, 80|40|20] out, made of a 1x1 Conv and three Resizes. Valid for the checks, useless for detection.
    `op`/`domain`/`extra_domain`/`external`: ONE more node or tensor of that kind is added (for the negative cases); `pad`: that many float
    weights are added (the graph ignores them) to make the file bigger; with `seed` they are random, so that no two models share a run of
    bytes (the residency tests look for such runs in everything the device says). `outputs`, `out_channels`, `input_size`, `float_input`:
    break the contract in one way. `conv_channels`: the width of the Conv (its cost grows with it)."""
    import numpy as np
    import onnx  # noqa: F401
    from onnx import TensorProto, helper, numpy_helper

    classes = (layout or LAYOUT_OK)["classes"]
    ch = out_channels if out_channels is not None else len(classes) + 4
    S = input_size
    if float_input:
        x = helper.make_tensor_value_info("frame", TensorProto.FLOAT, [1, 3, S, S])
        pre = [helper.make_node("Identity", ["frame"], ["chw"])]
    else:
        x = helper.make_tensor_value_info("frame", TensorProto.UINT8, [1, S, S, 3])
        pre = [helper.make_node("Cast", ["frame"], ["f"], to=TensorProto.FLOAT), helper.make_node("Transpose", ["f"], ["chw"], perm=[0, 3, 1, 2])]
    rng = np.random.RandomState(7)
    inits = [numpy_helper.from_array((rng.standard_normal((conv_channels, 3, 1, 1)) * 0.1).astype(np.float32), "w"),
             numpy_helper.from_array(np.zeros(conv_channels, np.float32), "b")]
    nodes = pre + [helper.make_node("Conv", ["chw", "w", "b"], ["feat"], kernel_shape=[1, 1])]
    outs = []
    for i, stride in enumerate((8, 16, 32)[: max(outputs, 0)] + (8,) * max(outputs - 3, 0)):
        inits.append(numpy_helper.from_array(np.array([1, 1, 1.0 / stride, 1.0 / stride], np.float32), f"sc{i}"))
        if conv_channels != ch:
            inits.append(numpy_helper.from_array((rng.standard_normal((ch, conv_channels, 1, 1)) * 0.1).astype(np.float32), f"hw{i}"))
            nodes.append(helper.make_node("Conv", ["feat", f"hw{i}"], [f"h{i}"], kernel_shape=[1, 1]))
            src = f"h{i}"
        else:
            src = "feat"
        # Resize takes (X, scales) in opset 10, (X, roi, scales) from 11 on; an empty roi is legal from opset 13 only
        if opset >= 11 and not any(t.name == "roi" for t in inits):
            inits.append(numpy_helper.from_array(np.zeros(0, np.float32), "roi"))
        ins = [src, "roi", f"sc{i}"] if opset >= 11 else [src, f"sc{i}"]
        nodes.append(helper.make_node("Resize", ins, [f"p{i + 3}"], mode="nearest"))
        outs.append(helper.make_tensor_value_info(f"p{i + 3}", TensorProto.FLOAT, [1, ch, S // stride, S // stride]))
    if op:
        nodes.append(helper.make_node(op, ["chw"], ["extra"], domain=domain))
    if external:
        t = helper.make_tensor("w_ext", TensorProto.FLOAT, [1], [0.0])
        t.data_location = TensorProto.EXTERNAL
        e = t.external_data.add()
        e.key, e.value = "location", "weights.bin"
        inits.append(t)
    if pad:
        w = (np.arange(pad) % 251).astype(np.float32) if seed is None else np.random.RandomState(seed).standard_normal(pad).astype(np.float32)
        inits.append(numpy_helper.from_array(w, "padding"))
    g = helper.make_graph(nodes, "g", [x], outs, initializer=inits)
    imports = [helper.make_opsetid(domain, opset)] if domain else [helper.make_opsetid("", opset)]
    if extra_domain:
        imports.append(helper.make_opsetid(extra_domain, 1))
    m = helper.make_model(g, opset_imports=imports)
    m.ir_version = 8
    if layout is not None:
        p = m.metadata_props.add()
        p.key, p.value = "ui_layout", json.dumps(layout)
    if float_input:
        p = m.metadata_props.add()
        p.key, p.value = "input_convention", "float NCHW 0..255"
    return m.SerializeToString()


def card_of(raw: bytes, classes=NAMES, **kw) -> dict:
    return MD.card_for(raw, classes, **kw)


def sim_package(kind: str, recipient_card, directory: str | Path, seq: int, slot: Optional[int] = None) -> bytes:
    """What the browser tests fetch from the dev server: kinds are 'tuning', 'model', 'big' (a model of about 160 KB), 'forged' (a bit of the
    signed header changed), 'other' (made for a different device), 'tampered' (a byte of the body changed). Models the device must refuse, with
    the reason: 'ops' (model_ops), 'files' (model_files), 'io' (model_io), 'heavy' (model_cost), 'pose' (not_ui_model)."""
    me = sender(directory)
    big = tiny_model(pad=40000)
    if kind == "tuning":
        return K.build_package(me, recipient_card, seq, slot=slot, tuning=(8, 3), name="Из приложения")
    if kind == "model":
        m = tiny_model()
        return K.build_package(me, recipient_card, seq, slot=slot, model=(m, card_of(m, name="icons", version=seq)))
    refused = {"ops": dict(op="Gemm"), "files": dict(external=True), "io": dict(outputs=2), "heavy": dict(conv_channels=64000)}
    if kind in refused or kind == "pose":
        m = tiny_model(**refused.get(kind, {}))
        c = card_of(m, name=kind, version=seq)
        if kind == "pose":
            c = dict(c, classes=["person"])
        return K.build_package(me, recipient_card, seq, slot=slot, model=(m, c))
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


_SIM_PROFILES: dict = {}


def _sim_profile(seed: int):
    if seed not in _SIM_PROFILES:
        from ..assist.sim_user import PERSONAS, build_profile
        _SIM_PROFILES[seed] = build_profile(PERSONAS["tremor"], minutes=6.0, seed=seed)._state
    return _SIM_PROFILES[seed]


def sim_bundle(kind: str, recipient_card, directory: str | Path, seq: int) -> bytes:
    """What the browser tests fetch from the dev server (`/sim/bundle/...`): a small settings file (DOBS) from a test sender, for THIS
    device. The device makes no files of its own (docs/RESIDENCY.md), so this is the only way a test gets one. Kinds: 'profile', 'slots'
    (slots 0 and 2, with names), 'other' (made for a different device)."""
    from ..ctl import seal as SL
    me = sender(directory, "clinic")                 # not the sender of the packages: the two are different people to the device
    st = _sim_profile(3)
    if kind == "profile":
        return SL.seal(me, recipient_card, seq, profile=st, tuning=(8, 3))
    if kind == "slots":
        return SL.seal(me, recipient_card, seq, slots=[SL.SlotData(0, st, (8, 5), None, "Работа"), SL.SlotData(2, _sim_profile(4), (3, 5), None, "Браузер")])
    if kind == "other":
        return SL.seal(me, sender(directory, "stranger").card(), seq, profile=st, tuning=(8, 3))
    raise ValueError(kind)
