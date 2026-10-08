"""The weights of a light UI model as a channel B part: what the device checks before it keeps them (docs/UPDATES.md section 3).

Weights are DATA, never code: the file is an ONNX graph restricted to the plain operators that `dataopen ui export` produces, with no custom
operator domain and no external data file, and its own metadata must say it is a UI-element model (the class list is a subset of the UI
taxonomy, and the pointer is a class of its own: the same guard `UiSceneBuilder` applies, `ui.taxonomy.require_ui_layout`). A pose / people
model is refused here, before it is stored. The check fails closed: without the `onnx` package the device can not inspect a graph, so it keeps
no weights.

This is the structural check. It does not say the model is GOOD (that is a measurement, `dataopen ui eval`); it says the model is the kind of
thing the UI path is allowed to run, and nothing more."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from ..ui.taxonomy import NotAUiModel, require_ui_layout
from .channels import UpdateError

MODEL_MAX = 16 * 1024 * 1024
INPUT_SIZE = 640                      # what the video path hands the detector (video.prep, `out`)
MAX_OPSET = 20
ALLOWED_OPS = frozenset({"Conv", "Relu", "Add", "Resize", "Concat", "Constant", "Cast", "Transpose", "Mul", "Identity"})  # == ui.export.ALLOWED_OPS
ALLOWED_DOMAINS = ("", "ai.onnx")
FORMATS = ("onnx",)


def card_for(raw: bytes, classes, name: str = "", version: int = 1, input_size: int = INPUT_SIZE, opset: int = 13) -> dict:
    """The part's description that travels in the signed, sealed package manifest."""
    return {"format": "onnx", "taxonomy": "ui-v1", "classes": list(classes), "n_keypoints": 0, "input_size": input_size, "opset": opset,
            "name": name[:40], "version": int(version), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _bad(msg: str) -> UpdateError:
    return UpdateError("B", "bad_model", msg)


def check_card(card, raw: bytes) -> dict:
    if not isinstance(card, dict) or card.get("format") not in FORMATS or card.get("taxonomy") != "ui-v1":
        raise _bad("card")
    classes, nk = card.get("classes"), card.get("n_keypoints", 0)
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes) or isinstance(nk, bool) or not isinstance(nk, int):
        raise _bad("card classes")
    try:
        require_ui_layout(classes, nk)
    except NotAUiModel as e:
        raise UpdateError("B", "not_ui_model", str(e)) from None
    if card.get("input_size") != INPUT_SIZE:
        raise _bad(f"input size {card.get('input_size')} (the video path gives {INPUT_SIZE})")
    if not isinstance(card.get("opset"), int) or isinstance(card.get("opset"), bool) or not 7 <= card["opset"] <= MAX_OPSET:
        raise _bad("opset")
    if not isinstance(card.get("version"), int) or isinstance(card.get("version"), bool) or not 0 <= card["version"] < 1 << 31:
        raise _bad("version")
    if not isinstance(card.get("name", ""), str) or len(card.get("name", "")) > 40 or any(ord(c) < 32 for c in card.get("name", "")):
        raise _bad("name")
    if len(raw) > MODEL_MAX or card.get("size") != len(raw) or card.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise _bad("size or hash does not match the card")
    return card


def inspect_graph(raw: bytes, card: dict) -> dict:
    """Parse the ONNX file without running it. Returns what was found. Raises `UpdateError` for anything the UI path may not run."""
    try:
        import onnx
    except ImportError:
        raise _bad("this device can not inspect a model graph") from None
    try:
        m = onnx.load_model_from_string(raw)
    except Exception:
        raise _bad("not an ONNX file") from None
    for init in m.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL or init.external_data:
            raise _bad("external data files are not accepted")
    for opset in m.opset_import:
        if opset.domain not in ALLOWED_DOMAINS or opset.version > MAX_OPSET:
            raise _bad(f"operator set {opset.domain!r} v{opset.version}")
    ops = {n.op_type for n in m.graph.node}
    for n in m.graph.node:
        if n.domain not in ALLOWED_DOMAINS or any(a.type == onnx.AttributeProto.GRAPH or a.type == onnx.AttributeProto.GRAPHS for a in n.attribute):
            raise _bad(f"node {n.op_type} in domain {n.domain!r} or with a subgraph")
    extra = ops - ALLOWED_OPS
    if extra:
        raise _bad("operators outside the allowed list: " + ", ".join(sorted(extra)))
    meta = {p.key: p.value for p in m.metadata_props}
    try:
        lay = json.loads(meta["ui_layout"])
        if list(lay["classes"]) != list(card["classes"]) or int(lay.get("n_keypoints", 0)) != 0 or int(lay["input_size"]) != card["input_size"]:
            raise ValueError("layout")
        require_ui_layout(lay["classes"], 0)
    except NotAUiModel as e:
        raise UpdateError("B", "not_ui_model", str(e)) from None
    except (KeyError, ValueError, TypeError):
        raise _bad("the model's own layout is missing or does not match its card") from None
    if len(m.graph.input) != 1 or len(m.graph.output) < 1:
        raise _bad("one input and at least one output were expected")
    return {"ops": sorted(ops), "opset": max((o.version for o in m.opset_import), default=0), "inputs": [i.name for i in m.graph.input],
            "outputs": [o.name for o in m.graph.output]}


def check_model(raw: bytes, card, inspector: Optional[bool] = True) -> dict:
    """The whole check; `inspector=False` is for tests of the card alone and is never used by the device."""
    check_card(card, raw)
    found = inspect_graph(raw, card) if inspector else {}
    return {"card": card, **found}


def card_from_onnx(raw: bytes, name: str = "", version: int = 1) -> dict:
    """A card made from the model's own metadata (the CLI's way of describing a file it was handed). Not trusted: `check_model` judges it."""
    try:
        import onnx
        m = onnx.load_model_from_string(raw)
        lay = json.loads({p.key: p.value for p in m.metadata_props}["ui_layout"])
        return card_for(raw, lay["classes"], name, version, int(lay["input_size"]), max((o.version for o in m.opset_import), default=13))
    except Exception:
        raise _bad("the file has no readable UI layout in its metadata") from None
