"""Quantization-aware fine-tuning for the deploy model (PyTorch).

When post-training quantization costs too much accuracy (the usual suspect after re-parameterization is the activation range of a
few layers), fine-tune the already-fused deploy network with fake quantization in the forward pass:

  weights      int8, symmetric, PER-CHANNEL   (scale = max|w| per output channel, straight-through gradient)
  activations  uint8, asymmetric, PER-TENSOR  (EMA min/max observers on the input of every convolution; the final head convolutions
                                               also quantize their OUTPUT, which is what an INT8 NPU hands back)
This is the numeric model of the target scheme, trained through. `export_qat_onnx` writes QuantizeLinear/DequantizeLinear (QDQ)
ONNX that ONNX Runtime executes like the fake-quant forward (equal up to rounding-boundary differences). It is NOT a guarantee
about rknn-toolkit2's own quantizer: the final check is always the `.rknn` on the board.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ApolloDetector


class FakeQuantConv(nn.Module):
    """Conv2d with per-channel int8 weights and a per-tensor uint8 fake quantizer on its input (and optionally its output)."""

    def __init__(self, conv: nn.Conv2d, quant_output: bool = False, momentum: float = 0.99) -> None:
        super().__init__()
        self.conv, self.quant_output, self.momentum = conv, quant_output, momentum
        for name in ("in", "out"):
            self.register_buffer(f"{name}_min", torch.zeros(()))
            self.register_buffer(f"{name}_max", torch.zeros(()))
            self.register_buffer(f"{name}_seen", torch.zeros(()))
        self.observing = True
        self.frozen = False
        self._const: dict = {}

    @torch.no_grad()
    def freeze(self) -> None:
        """Bake the trained state for export: weights snapped onto their per-channel int8 grid (a later per-channel int8 quantizer,
        RKNN's included, then reproduces them exactly) and activation qparams turned into Python constants (the ONNX exporter
        needs constants)."""
        w = self.conv.weight
        scale = (w.abs().amax(dim=(1, 2, 3)) / 127.0).clamp(min=1e-8)
        q = torch.clamp(torch.round(w / scale[:, None, None, None]), -127, 127)
        self.conv.weight.copy_(q * scale[:, None, None, None])
        for name in ("in", "out"):
            sc, zp = self._qparams(name)
            self._const[name] = (float(sc), int(zp))
        self.frozen = True

    def _observe(self, name: str, x: torch.Tensor) -> None:
        lo, hi = float(x.detach().min()), float(x.detach().max())
        seen = getattr(self, f"{name}_seen")
        cur_lo, cur_hi = getattr(self, f"{name}_min"), getattr(self, f"{name}_max")
        if float(seen) == 0.0:
            cur_lo.fill_(lo)
            cur_hi.fill_(hi)
        else:
            cur_lo.mul_(self.momentum).add_(lo * (1 - self.momentum))
            cur_hi.mul_(self.momentum).add_(hi * (1 - self.momentum))
        seen.fill_(1.0)

    def _qparams(self, name: str):
        lo = torch.clamp(getattr(self, f"{name}_min"), max=0.0)                    # the range must contain zero
        hi = torch.clamp(getattr(self, f"{name}_max"), min=0.0)
        scale = torch.clamp((hi - lo) / 255.0, min=1e-8)
        zp = torch.clamp(torch.round(-lo / scale), 0, 255).to(torch.int32)
        return scale, zp

    def _fq_act(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            sc, zp = self._const[name]
            return torch.fake_quantize_per_tensor_affine(x, sc, zp, 0, 255)
        if self.observing and self.training:
            self._observe(name, x)
        scale, zp = self._qparams(name)
        return torch.fake_quantize_per_tensor_affine(x, scale, zp, 0, 255)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv
        xq = self._fq_act("in", x)
        if self.frozen:
            wq = c.weight                                                              # already on the int8 grid
        else:
            w = c.weight
            scale = (w.detach().abs().amax(dim=(1, 2, 3)) / 127.0).clamp(min=1e-8)
            wq = torch.fake_quantize_per_channel_affine(w, scale, torch.zeros_like(scale, dtype=torch.int32), 0, -127, 127)
        y = F.conv2d(xq, wq, c.bias, c.stride, c.padding, c.dilation, c.groups)
        return self._fq_act("out", y) if self.quant_output else y


def prepare_qat(model: ApolloDetector) -> ApolloDetector:
    """Deep-copy a deploy (fused) detector and wrap every Conv2d in FakeQuantConv; head / aim output convolutions also quantize
    their output."""
    if not getattr(model, "deployed", False):
        raise ValueError("prepare_qat needs a fused deploy model: call model.fuse() first")
    net = copy.deepcopy(model)
    out_convs = [lvl.cls[-1] for lvl in net.head_o2o.levels] + [lvl.box[-1] for lvl in net.head_o2o.levels] + \
                [lvl.kp[-1] for lvl in net.head_o2o.levels]
    if net.refine is not None:
        out_convs.append(net.refine.out)
    ids = {id(c) for c in out_convs}

    def wrap(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(parent, name, FakeQuantConv(child, quant_output=id(child) in ids))
            else:
                wrap(child)

    wrap(net)
    return net


def freeze(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantConv):
            m.freeze()


def set_observers(model: nn.Module, on: bool) -> None:
    for m in model.modules():
        if isinstance(m, FakeQuantConv):
            m.observing = on


@torch.no_grad()
def calibrate(model: nn.Module, batches) -> None:
    """Run a few batches in train mode with observers on (no gradients): initializes every activation range."""
    model.train()
    set_observers(model, True)
    for imgs in batches:
        model(imgs)


def qat_finetune(model: ApolloDetector, loss_fn, loader, epochs: int, lr: float = 1e-4, device: str = "cpu",
                 calib_batches: int = 8, log=print, max_steps: Optional[int] = None) -> tuple[nn.Module, dict]:
    """Fine-tune a prepared QAT model with the one-to-one head loss. `loader` yields collate() dicts."""
    net = prepare_qat(model.to(device)).to(device)
    gen = iter(loader)
    cal = []
    for _ in range(calib_batches):
        try:
            cal.append(next(gen)["images"].to(device).float())
        except StopIteration:
            break
    calibrate(net, cal)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    step, hist = 0, []
    for epoch in range(epochs):
        net.train()
        set_observers(net, epoch == 0)                    # ranges settle in the first epoch, then stay fixed
        tot, n = 0.0, 0
        for batch in loader:
            imgs = batch["images"].to(device).float()
            tg = {k: batch[k].to(device) for k in ("boxes", "cls", "kpts", "valid")}
            outs = net(imgs)
            levels = [o.float() for o in outs[:3]]
            loss, _ = loss_fn.head_loss(levels, tg, loss_fn.cfg.tal_topk_o2o)
            if len(outs) > 3 and loss_fn.primary:
                loss = loss + loss_fn.cfg.w_heat * loss_fn.heat_loss(outs[3].float(), tg)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            tot += float(loss)
            n += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        hist.append(tot / max(1, n))
        log(f"qat epoch {epoch}: loss {hist[-1]:.4f}")
        if max_steps is not None and step >= max_steps:
            break
    set_observers(net, False)
    return net.eval(), {"steps": step, "loss": hist}


def export_qat_onnx(net: nn.Module, layout, out: str | Path, schema_meta: Optional[dict] = None, opset: int = 13,
                    input: str = "float", check: bool = True) -> dict:
    """QDQ ONNX of a trained QAT model (observers frozen); parity-checked against the fake-quant forward in PyTorch."""
    import json

    import onnx
    set_observers(net, False)
    net.eval()
    freeze(net)                              # weights onto the int8 grid, qparams to constants
    w, h = layout.input_size
    dummy = torch.randint(0, 256, (1, 3, h, w)).float()
    n = len(layout.strides)
    names = [f"p{i + 3}" for i in range(n)] + (["aim"] if layout.refine_stride else [])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net.cpu(), (dummy,), str(out), input_names=["images"], output_names=names, opset_version=opset,
                      do_constant_folding=True, dynamo=False)
    m = onnx.load(str(out))
    m.ir_version = 8
    for k, v in (("apollo", layout.to_json()), ("schema", json.dumps(schema_meta or {})), ("input_convention", input),
                 ("quantization", "qat-qdq")):
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(out))
    res = {"path": str(out), "outputs": names, "size_mb": round(out.stat().st_size / 1e6, 3)}
    if check:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"images": dummy.numpy()})
        with torch.no_grad():
            ref = [t.numpy() for t in net(dummy)]
        res["max_rel_diff_vs_torch"] = max(float(np.abs(a - b).max()) / max(1.0, float(np.abs(b).max())) for a, b in zip(got, ref))
    return res
