"""ONNX export of UiNet (plain ops only: Conv, Relu, Add, Resize, Concat, plus Cast/Transpose for the uint8 NHWC input) with the class list
and head layout stored in the file, so the host decoder and the UI-only guard work without the training code."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import UiNet
from .taxonomy import STRIDES, require_ui_layout

ALLOWED_OPS = {"Conv", "Relu", "Add", "Resize", "Concat", "Constant", "Cast", "Transpose", "Mul", "Identity"}


class Deploy(torch.nn.Module):
    def __init__(self, net: UiNet, uint8_nhwc: bool) -> None:
        super().__init__()
        self.net, self.u8 = net, uint8_nhwc

    def forward(self, x):
        if self.u8:
            x = x.permute(0, 3, 1, 2).float()
        return tuple(self.net(x * (1.0 / 255.0)))


def layout(net: UiNet) -> dict:
    c = net.cfg
    return {
        "kind": "ui-elements",
        "classes": list(c.classes),
        "n_cls": c.n_cls,
        "strides": list(STRIDES),
        "input_size": c.input_size,
        "n_keypoints": 0,
    }


def export_onnx(net: UiNet, out: str | Path, uint8_nhwc: bool = True, opset: int = 13, tol: float = 1e-4) -> dict:
    import onnx
    import onnxruntime as ort

    net = net.eval()
    require_ui_layout(net.cfg.classes)
    s = net.cfg.input_size
    dummy = torch.randint(0, 256, (1, s, s, 3), dtype=torch.uint8) if uint8_nhwc else torch.rand(1, 3, s, s) * 255
    wrapper = Deploy(net, uint8_nhwc).eval()
    out = Path(out)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["frame" if uint8_nhwc else "images"],
        output_names=["p3", "p4", "p5"],
        opset_version=opset,
        dynamo=False,
    )
    m = onnx.load(str(out))
    m.ir_version = 8
    for k, v in (("ui_layout", json.dumps(layout(net))), ("input_convention", "uint8 NHWC RGB" if uint8_nhwc else "float NCHW 0..255")):
        p = m.metadata_props.add()
        p.key, p.value = k, v
    ops = {n.op_type for n in m.graph.node}
    if not ops <= ALLOWED_OPS:
        raise RuntimeError(f"unexpected ops in the exported graph: {sorted(ops - ALLOWED_OPS)}")
    onnx.save(m, str(out))
    Path(str(out) + ".layout.json").write_text(json.dumps(layout(net)))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})
    with torch.no_grad():
        ref = wrapper(dummy)
    diff = max(float(np.abs(g - r.numpy()).max() / (np.abs(r.numpy()).max() + 1e-9)) for g, r in zip(got, ref))
    if diff > tol:
        raise RuntimeError(f"ONNX output differs from torch by {diff:.2e} (relative)")
    return {"path": str(out), "ops": sorted(ops), "max_rel_diff_vs_torch": diff, "size_bytes": out.stat().st_size}
