"""Checkpoint -> deploy ONNX (reparameterized, one-to-one head only, 1/255 folded into the stem).

Two input conventions:
  float  NCHW float32 in 0..255   what rknn-toolkit2 wants (set mean=0, std=1 in `rknn.config`; the NPU quantizes the input itself)
  uint8  NHWC uint8 camera frame  Cast + Transpose inside the graph; for ONNX Runtime / the closed loop
Outputs: p3 / p4 / p5 (one NCHW tensor per level) and `aim` (stride-4 heatmap + offsets) when the model has it; the layout is
stored in the ONNX metadata (`apollo`), so decoding needs no training code.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..core.schema_io import load_schema  # noqa: F401  (re-export for callers building a schema by name)
from .config import ModelConfig
from .layout import HeadLayout
from .model import ApolloDetector, build_model


def load_checkpoint(path: str | Path, which: str = "ema", device: str = "cpu"):
    """-> (model in eval mode, HeadLayout, schema_meta dict, DetectorConfig dict)."""
    c = torch.load(path, map_location=device, weights_only=False)
    mc = c["cfg"]["model"]
    for k in ("stem", "stages", "input_size"):
        if k in mc:
            mc[k] = tuple(tuple(x) if isinstance(x, list) else x for x in mc[k]) if k == "stages" else tuple(mc[k])
    cfg = ModelConfig(**mc)
    meta = c["schema"]
    prim = tuple(meta["keypoints"].index(n) for n in meta["primary"])
    model = build_model(cfg, prim).to(device)
    model.load_state_dict(c[which])
    lay = model.layout(tuple(meta["keypoints"]), tuple(meta["classes"]), tuple(meta["flip_idx"]), meta["name"])
    return model.eval(), lay, meta, c["cfg"]


class DeployWrapper(torch.nn.Module):
    def __init__(self, net: ApolloDetector, uint8_nhwc: bool) -> None:
        super().__init__()
        self.net, self.uint8_nhwc = net, uint8_nhwc

    def forward(self, x: torch.Tensor):
        if self.uint8_nhwc:
            x = x.permute(0, 3, 1, 2).float()
        return self.net(x)


def export_onnx(model: ApolloDetector, layout: HeadLayout, out: str | Path, input: str = "float", opset: int = 13,
                schema_meta: Optional[dict] = None, check: bool = True, tol: float = 2e-3) -> dict:
    if input not in ("float", "uint8"):
        raise ValueError("input must be 'float' or 'uint8'")
    import onnx

    net = copy.deepcopy(model).cpu().fuse()
    wrapper = DeployWrapper(net, input == "uint8").eval()
    w, h = layout.input_size
    if input == "uint8":
        dummy = torch.randint(0, 256, (1, h, w, 3), dtype=torch.uint8)
        in_name = "frame"
    else:
        dummy = torch.randint(0, 256, (1, 3, h, w)).float()
        in_name = "images"
    n = len(layout.strides)
    names = [f"p{i + 3}" for i in range(n)] + (["aim"] if layout.refine_stride else [])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, (dummy,), str(out), input_names=[in_name], output_names=names, opset_version=opset,
                      do_constant_folding=True, dynamo=False)
    m = onnx.load(str(out))
    m.ir_version = 8                                   # old enough for ONNX Runtime and rknn-toolkit2
    for k, v in (("apollo", layout.to_json()), ("schema", json.dumps(schema_meta or {})), ("input_convention", input)):
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(out))
    out.with_suffix(".layout.json").write_text(layout.to_json())     # the .rknn conversion has no metadata: the runtime reads this
    res = {"path": str(out), "outputs": names, "input": in_name, "opset": opset, "size_mb": round(out.stat().st_size / 1e6, 3)}
    if check:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = sess.run(None, {in_name: dummy.numpy()})
        with torch.no_grad():
            ref = [t.numpy() for t in wrapper(dummy)]
        diff = max(float(np.abs(a - b).max()) / max(1.0, float(np.abs(b).max())) for a, b in zip(got, ref))
        res["max_rel_diff_vs_torch"] = diff
        if diff > tol:
            raise RuntimeError(f"ONNX output differs from PyTorch by {diff:.2e} (> {tol}): do not ship this export")
    return res
