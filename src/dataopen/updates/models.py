"""The weights of a light UI model as a channel B part: what the device checks before it keeps them (docs/UPDATES.md section 3).

Weights are DATA, never code: the file is an ONNX graph restricted to the plain operators that `dataopen ui export` produces, with no custom
operator domain and no external data file, and its own metadata must say it is a UI-element model (the class list is a subset of the UI
taxonomy, and the pointer is a class of its own: the same guard `UiSceneBuilder` applies, `ui.taxonomy.require_ui_layout`). A pose / people
model is refused here, before it is stored. The check fails closed: without the `onnx` package the device can not inspect a graph, so it keeps
no weights.

This is the structural check. It does not say the model is GOOD (that is a measurement, `dataopen ui eval`); it says the model is the kind of
thing the UI path is allowed to run, and nothing more. It also does not PROVE what the model detects: labels, shapes and operators are
checked, behaviour is not (docs/MODELS.md). The check is STATIC: the graph is read, never run.

The contract ("ABI v1", `ABI` below) is the one place the device, the sender's tools and docs/MODELS.md all read: one input of a known
form, exactly three outputs of the shape the host decoder reads, a known operator list, and limits on size, nodes and computation."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from ..ui.policy import ACTIVE_BACKEND, V1, est_infer_ms
from ..ui.taxonomy import NAMES, STRIDES, NotAUiModel, require_ui_layout
from .channels import UpdateError

MODEL_MAX = 16 * 1024 * 1024
INPUT_SIZE = 640                      # what the video path hands the detector (video.prep, `out`)
MIN_OPSET, MAX_OPSET = 7, 20
MAX_NODES = 2000
MAX_MACS = 4_000_000_000              # multiply-adds of all Conv nodes, from static shapes. An ASSUMPTION (unmeasured): UiNet is about 1.0 G
ALLOWED_OPS = frozenset({"Conv", "Relu", "Add", "Resize", "Concat", "Constant", "Cast", "Transpose", "Mul", "Identity"})  # == ui.export.ALLOWED_OPS
ALLOWED_DOMAINS = ("", "ai.onnx")
FORMATS = ("onnx",)
CONVENTIONS = {"uint8 NHWC RGB": ("uint8", "nhwc"), "float NCHW 0..255": ("float", "nchw")}
DEFAULT_CONVENTION = "uint8 NHWC RGB"
N_OUT = len(STRIDES)                  # one output per pyramid level
TAXONOMY = "ui-v1"
ABI_VERSION = 1

ABI = {
    "version": ABI_VERSION, "taxonomy": TAXONOMY, "classes": list(NAMES), "input_size": INPUT_SIZE, "conventions": list(CONVENTIONS),
    "outputs": f"{N_OUT}, each [N, n_classes + 4, {INPUT_SIZE}/s, {INPUT_SIZE}/s] for s in {list(STRIDES)}", "opset": [MIN_OPSET, MAX_OPSET],
    "ops": sorted(ALLOWED_OPS), "max_nodes": MAX_NODES, "max_macs": MAX_MACS, "max_bytes": MODEL_MAX,
}


def card_for(raw: bytes, classes, name: str = "", version: int = 1, input_size: int = INPUT_SIZE, opset: int = 13) -> dict:
    """The part's description that travels in the signed, sealed package manifest."""
    return {"format": "onnx", "taxonomy": TAXONOMY, "classes": list(classes), "n_keypoints": 0, "input_size": input_size, "opset": opset,
            "name": name[:40], "version": int(version), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _bad(msg: str, key: str = "bad_model") -> UpdateError:
    return UpdateError("B", key, msg)


def check_card(card, raw: bytes) -> dict:
    if not isinstance(card, dict) or card.get("format") not in FORMATS or card.get("taxonomy") != TAXONOMY:
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
    if not isinstance(card.get("opset"), int) or isinstance(card.get("opset"), bool) or not MIN_OPSET <= card["opset"] <= MAX_OPSET:
        raise _bad("opset")
    if not isinstance(card.get("version"), int) or isinstance(card.get("version"), bool) or not 0 <= card["version"] < 1 << 31:
        raise _bad("version")
    if not isinstance(card.get("name", ""), str) or len(card.get("name", "")) > 40 or any(ord(c) < 32 for c in card.get("name", "")):
        raise _bad("name")
    if len(raw) > MODEL_MAX:
        raise UpdateError("B", "too_large", f"{len(raw)} bytes (at most {MODEL_MAX})")
    if card.get("size") != len(raw) or card.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise _bad("size or hash does not match the card")
    return card


def _dim(d):
    """A dimension as an int, or None when it is symbolic or missing."""
    return d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None


def _shape(vi):
    t = vi.type.tensor_type
    return t.elem_type, [_dim(d) for d in t.shape.dim] if t.HasField("shape") else None


def _constants(onnx, m) -> dict:
    """The shapes of every tensor whose value is in the file (initializers and Constant nodes): a Conv weight comes from here."""
    shapes = {i.name: list(i.dims) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
                    shapes[n.output[0]] = list(a.t.dims)
    return shapes


def _macs(m, shapes: dict) -> int:
    """Multiply-adds of all Conv nodes from the inferred shapes (batch 1). Anything that can not be worked out is an error: a model that hides
    its cost is refused."""
    value = {v.name: v for v in list(m.graph.value_info) + list(m.graph.output)}
    total = 0
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        out, w = value.get(n.output[0]), shapes.get(n.input[1])
        if out is None or w is None or len(w) != 4:
            raise _bad(f"the cost of {n.output[0]!r} can not be worked out", "model_cost")
        _, shp = _shape(out)
        if shp is None or len(shp) != 4 or any(x is None for x in shp[1:]):
            raise _bad(f"the shape of {n.output[0]!r} is not fixed", "model_cost")
        total += shp[1] * shp[2] * shp[3] * w[1] * w[2] * w[3]
    return total


def inspect_graph(raw: bytes, card: dict) -> dict:
    """Parse the ONNX file without running it. Returns what was found. Raises `UpdateError` for anything the UI path may not run.

    Order (cheapest and most telling first): files and domains, operators, node count, the model's own layout and taxonomy, a valid graph,
    the input and output contract, the cost."""
    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        raise _bad("this device can not inspect a model graph") from None
    try:
        m = onnx.load_model_from_string(raw)
    except Exception:
        raise _bad("not an ONNX file") from None
    for init in m.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL or init.external_data:
            raise _bad("external data files are not accepted", "model_files")
    for opset in m.opset_import:
        if opset.domain not in ALLOWED_DOMAINS:
            raise _bad(f"operator set {opset.domain!r} is not the standard one", "model_files")
        if not MIN_OPSET <= opset.version <= MAX_OPSET:
            raise _bad(f"operator set v{opset.version} is outside {MIN_OPSET}..{MAX_OPSET}", "model_ops")
    for n in m.graph.node:
        if n.domain not in ALLOWED_DOMAINS or any(a.type == onnx.AttributeProto.GRAPH or a.type == onnx.AttributeProto.GRAPHS for a in n.attribute):
            raise _bad(f"node {n.op_type} in domain {n.domain!r} or with a subgraph", "model_files")
    ops = {n.op_type for n in m.graph.node}
    extra = ops - ALLOWED_OPS
    if extra:
        raise _bad("operators outside the allowed list: " + ", ".join(sorted(extra)), "model_ops")
    if len(m.graph.node) > MAX_NODES:
        raise _bad(f"{len(m.graph.node)} nodes; at most {MAX_NODES}", "model_cost")
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
    # a valid graph, with the shapes WORKED OUT (strict: a shape the file declares must be the one the graph produces)
    try:
        onnx.checker.check_model(m)
        m = onnx.shape_inference.infer_shapes(m, check_type=True, strict_mode=True)
    except Exception as e:
        raise _bad(f"not a consistent graph: {str(e).splitlines()[0][:60] if str(e) else type(e).__name__}", "model_io") from None
    consts = _constants(onnx, m)
    ins = [i for i in m.graph.input if i.name not in consts]
    if len(ins) != 1 or len(m.graph.output) != N_OUT:
        raise _bad(f"one input and exactly {N_OUT} outputs were expected", "model_io")
    conv = meta.get("input_convention", DEFAULT_CONVENTION)
    if conv not in CONVENTIONS:
        raise _bad(f"input convention {conv!r}", "model_io")
    kind, order = CONVENTIONS[conv]
    et, shp = _shape(ins[0])
    s = card["input_size"]
    want = [s, s, 3] if order == "nhwc" else [3, s, s]
    if et != (TensorProto.UINT8 if kind == "uint8" else TensorProto.FLOAT) or shp is None or len(shp) != 4 or shp[1:] != want:
        raise _bad(f"the input must be {conv} with {want} after the batch", "model_io")
    n_cls = len(card["classes"])
    for o, stride in zip(m.graph.output, STRIDES):
        et, shp = _shape(o)
        if et != TensorProto.FLOAT or shp is None or len(shp) != 4 or shp[1:] != [n_cls + 4, s // stride, s // stride]:
            raise _bad(f"output {o.name!r} must be float [N, {n_cls + 4}, {s // stride}, {s // stride}]", "model_io")
    macs = _macs(m, consts)
    if macs > MAX_MACS:
        raise _bad(f"{macs / 1e9:.1f} G multiply-adds; at most {MAX_MACS / 1e9:.0f} G", "model_cost")
    return {"ops": sorted(ops), "opset": max((o.version for o in m.opset_import), default=0), "inputs": [i.name for i in ins],
            "outputs": [o.name for o in m.graph.output], "macs": macs, "nodes": len(m.graph.node), "convention": conv}


def check_model(raw: bytes, card, inspector: Optional[bool] = True) -> dict:
    """The whole check; `inspector=False` is for tests of the card alone and is never used by the device."""
    check_card(card, raw)
    found = inspect_graph(raw, card) if inspector else {}
    if "macs" in found:                     # what the budget says about it: ROUGH, from the work alone (ui/policy.py); the guard in ui/health.py checks it
        p50, p95 = est_infer_ms(found["macs"])
        found["est_infer_ms"] = {"backend": ACTIVE_BACKEND, "p50": round(p50, 1), "p95": round(p95, 1), "budget_p95": V1.infer_budget_ms,
                                 "fits": p95 <= V1.infer_budget_ms}
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
