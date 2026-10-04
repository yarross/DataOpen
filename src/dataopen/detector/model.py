"""ApolloNet-Pose (PyTorch). Everything on the NPU is Conv3x3/Conv1x1 + ReLU + nearest Resize + Concat/Add:

  backbone  RepHG: HGNetV2-style aggregation blocks whose convolutions are RepVGG blocks (3x3 + 1x1 + identity at training,
            ONE 3x3 at deploy), dense convolutions only (no depthwise: poor NPU utilization), ReLU only (no SiLU/GELU/SE)
  neck      RepBiFPN: top-down then bottom-up path with learned non-negative fusion weights, P3/P4/P5
  head      decoupled cls / box / keypoint branches per level; training has TWO heads (one-to-many for rich supervision,
            one-to-one for NMS-free inference, as in YOLOv10); deploy keeps only the one-to-one head
  aim       optional stride-4 heatmap + offset head for the primary keypoint (the head centre) refined on the host
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layout import HeadLayout


def rc(c: int, mult: float, floor: int = 16) -> int:
    """Scaled channel count, multiple of 8 (NPU channel alignment)."""
    return max(floor, int(math.ceil(c * mult / 8.0) * 8))


class ConvBN(nn.Sequential):
    def __init__(self, c1: int, c2: int, k: int, s: int, p: int) -> None:
        super().__init__(nn.Conv2d(c1, c2, k, s, p, bias=False), nn.BatchNorm2d(c2))


def _fuse_bn(w: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return w * t, bn.bias - bn.running_mean * bn.weight / std


class RepConv(nn.Module):
    """RepVGG block. Training: ReLU(BN(conv3x3) + BN(conv1x1) [+ BN(identity)]). Deploy: ReLU(conv3x3 + bias)."""

    def __init__(self, c1: int, c2: int, stride: int = 1, act: bool = True) -> None:
        super().__init__()
        self.c1, self.c2, self.stride, self.deploy = c1, c2, stride, False
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.rbr3 = ConvBN(c1, c2, 3, stride, 1)
        self.rbr1 = ConvBN(c1, c2, 1, stride, 0)
        self.rbr_id = nn.BatchNorm2d(c1) if (c1 == c2 and stride == 1) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.conv(x))
        y = self.rbr3(x) + self.rbr1(x)
        if self.rbr_id is not None:
            y = y + self.rbr_id(x)
        return self.act(y)

    @torch.no_grad()
    def fuse(self) -> None:
        if self.deploy:
            return
        k3, b3 = _fuse_bn(self.rbr3[0].weight, self.rbr3[1])
        k1, b1 = _fuse_bn(self.rbr1[0].weight, self.rbr1[1])
        k, b = k3 + F.pad(k1, [1, 1, 1, 1]), b3 + b1
        if self.rbr_id is not None:
            idk = torch.zeros(self.c1, self.c1, 3, 3, device=k.device, dtype=k.dtype)
            for i in range(self.c1):
                idk[i, i, 1, 1] = 1.0
            ki, bi = _fuse_bn(idk, self.rbr_id)
            k, b = k + ki, b + bi
        self.conv = nn.Conv2d(self.c1, self.c2, 3, self.stride, 1, bias=True).to(k.device)
        self.conv.weight.copy_(k)
        self.conv.bias.copy_(b)
        del self.rbr3, self.rbr1
        self.rbr_id = None
        self.deploy = True


class Conv1x1(nn.Module):
    def __init__(self, c1: int, c2: int, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.deploy = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) if self.deploy else self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self) -> None:
        if self.deploy:
            return
        w, b = _fuse_bn(self.conv.weight, self.bn)
        self.conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, 1, bias=True).to(w.device)
        self.conv.weight.copy_(w)
        self.conv.bias.copy_(b)
        del self.bn
        self.deploy = True


class RepHGBlock(nn.Module):
    """HG aggregation: n stacked RepConvs, every intermediate map is concatenated with the input and squeezed by a 1x1."""

    def __init__(self, c: int, mid: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([RepConv(c if i == 0 else mid, mid) for i in range(layers)])
        self.agg = Conv1x1(c + layers * mid, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, y = [x], x
        for l in self.layers:
            y = l(y)
            feats.append(y)
        return self.agg(torch.cat(feats, dim=1)) + x


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        m = cfg.width_mult
        c0, c1 = rc(cfg.stem[0], m), rc(cfg.stem[1], m)
        self.stem = nn.Sequential(RepConv(3, c0, 2), RepConv(c0, c1, 2))
        self.out_channels = [c1]
        c_prev, stages = c1, []
        for out, mid, blocks, layers in cfg.stages:
            co, cm = rc(out, m), rc(mid, m)
            stages.append(nn.Sequential(RepConv(c_prev, co, 2), *[RepHGBlock(co, cm, layers) for _ in range(blocks)]))
            self.out_channels.append(co)
            c_prev = co
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p2 = self.stem(x)
        outs, y = [p2], p2
        for s in self.stages:
            y = s(y)
            outs.append(y)
        return outs                                   # strides 4, 8, 16, 32


class WeightedAdd(nn.Module):
    """BiFPN fast normalized fusion: sum_i relu(w_i) x_i / (sum relu(w) + eps). Constant at deploy, folds into Mul/Add."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
        self.coef: Optional[list] = None              # baked python floats after fuse(): the graph gets plain scalar Muls

    def fuse(self) -> None:
        with torch.no_grad():
            w = F.relu(self.w)
            self.coef = [float(v) for v in w / (w.sum() + 1e-4)]

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if self.coef is not None:
            c = self.coef
        else:
            w = F.relu(self.w)
            c = w / (w.sum() + 1e-4)
        out = c[0] * xs[0]
        for i in range(1, len(xs)):
            out = out + c[i] * xs[i]
        return out


class RepBiFPN(nn.Module):
    def __init__(self, in_ch: list[int], c: int, repeats: int) -> None:
        super().__init__()
        self.lat = nn.ModuleList([Conv1x1(ci, c) for ci in in_ch])                    # P3, P4, P5
        self.layers = nn.ModuleList()
        for _ in range(repeats):
            self.layers.append(nn.ModuleDict({
                "w_td4": WeightedAdd(2), "td4": RepConv(c, c), "w_td3": WeightedAdd(2), "td3": RepConv(c, c),
                "w_bu4": WeightedAdd(3), "bu4": RepConv(c, c), "w_bu5": WeightedAdd(2), "bu5": RepConv(c, c),
                "dn3": RepConv(c, c, 2), "dn4": RepConv(c, c, 2)}))

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = [l(f) for l, f in zip(self.lat, feats)]
        up = lambda t: F.interpolate(t, scale_factor=2.0, mode="nearest")             # noqa: E731
        for L in self.layers:
            td4 = L["td4"](L["w_td4"](p4, up(p5)))
            n3 = L["td3"](L["w_td3"](p3, up(td4)))
            n4 = L["bu4"](L["w_bu4"](p4, td4, L["dn3"](n3)))
            n5 = L["bu5"](L["w_bu5"](p5, L["dn4"](n4)))
            p3, p4, p5 = n3, n4, n5
        return [p3, p4, p5]


class HeadLevel(nn.Module):
    """Decoupled branches; their 1x1 outputs are concatenated into ONE tensor (cls | box | kp offsets | kp score | kp vis)."""

    def __init__(self, c: int, cfg: ModelConfig) -> None:
        super().__init__()
        K = cfg.n_kpt
        cls_in = RepConv(c, cfg.head_ch) if cfg.cls_k3 else Conv1x1(c, cfg.head_ch)
        self.cls = nn.Sequential(cls_in, nn.Conv2d(cfg.head_ch, cfg.n_cls, 1))
        self.box = nn.Sequential(RepConv(c, cfg.head_ch), nn.Conv2d(cfg.head_ch, 4, 1))
        kp_layers = [RepConv(c, cfg.kp_head_ch)] + ([RepConv(cfg.kp_head_ch, cfg.kp_head_ch)] if cfg.kp_second_conv else [])
        self.kp = nn.Sequential(*kp_layers, nn.Conv2d(cfg.kp_head_ch, 4 * K, 1))
        nn.init.constant_(self.cls[1].bias, -4.6)                 # p = 0.01 at start (focal-style prior)
        nn.init.constant_(self.box[1].bias, 0.0)
        nn.init.normal_(self.kp[-1].weight, std=0.01)
        nn.init.constant_(self.kp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cls(x), self.box(x), self.kp(x)], dim=1)


class Head(nn.Module):
    def __init__(self, c: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.levels = nn.ModuleList([HeadLevel(c, cfg) for _ in range(3)])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [l(f) for l, f in zip(self.levels, feats)]


class AimRefine(nn.Module):
    """Stride-4 heatmap + sub-cell offsets for the primary keypoints, from the stride-4 backbone map and the neck's P3."""

    def __init__(self, c_p2: int, c_p3: int, cfg: ModelConfig, n_primary: int) -> None:
        super().__init__()
        self.p2 = Conv1x1(c_p2, cfg.refine_ch)
        self.p3 = Conv1x1(c_p3, cfg.refine_ch)
        self.mix = RepConv(cfg.refine_ch, cfg.refine_ch)
        self.out = nn.Conv2d(cfg.refine_ch, 3 * n_primary, 1)
        nn.init.constant_(self.out.bias, 0.0)
        with torch.no_grad():
            self.out.bias[:n_primary] = -4.6

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        return self.out(self.mix(self.p2(p2) + F.interpolate(self.p3(p3), scale_factor=2.0, mode="nearest")))


class ApolloDetector(nn.Module):
    """Input: float NCHW in [0, 255] (cast a uint8 frame; the 1/255 scale is part of the model and folds into the stem)."""

    def __init__(self, cfg: ModelConfig, primary: tuple = ()) -> None:
        super().__init__()
        self.cfg, self.primary = cfg, tuple(primary)
        self.backbone = Backbone(cfg)
        ch = self.backbone.out_channels                      # [s4, s8, s16, s32]
        self.neck = RepBiFPN(ch[1:], cfg.neck_ch, cfg.neck_repeats)
        self.head_o2o = Head(cfg.neck_ch, cfg)
        self.head_o2m = Head(cfg.neck_ch, cfg)
        self.refine = AimRefine(ch[0], cfg.neck_ch, cfg, len(self.primary)) if (cfg.aim_refine and self.primary) else None
        self.register_buffer("input_scale", torch.tensor(1.0 / 255.0))
        self.deployed = False

    @property
    def strides(self) -> tuple[int, int, int]:
        return (8, 16, 32)

    def forward(self, x: torch.Tensor):
        x = x * self.input_scale
        p2, p3, p4, p5 = self.backbone(x)
        feats = self.neck([p3, p4, p5])
        ref = self.refine(p2, feats[0]) if self.refine is not None else None
        if self.training and not self.deployed:
            return {"o2o": self.head_o2o(feats), "o2m": self.head_o2m(feats), "refine": ref}
        outs = self.head_o2o(feats)
        return (*outs, ref) if ref is not None else tuple(outs)

    @torch.no_grad()
    def fuse(self) -> "ApolloDetector":
        """Reparameterize every RepConv / BN, fold the input scale into the stem, drop the one-to-many head."""
        for m in list(self.modules()):
            if isinstance(m, (RepConv, Conv1x1, WeightedAdd)):
                m.fuse()
        stem = self.backbone.stem[0]
        stem.conv.weight.mul_(self.input_scale)
        self.input_scale.fill_(1.0)
        if hasattr(self, "head_o2m"):
            del self.head_o2m
        self.deployed = True
        return self.eval()

    def layout(self, keypoints: tuple = (), classes: tuple = (), flip_idx: tuple = (), schema: str = "") -> HeadLayout:
        c = self.cfg
        return HeadLayout(n_cls=c.n_cls, n_kpt=c.n_kpt, strides=self.strides, offset_scale=c.offset_scale,
                          refine_stride=4 if self.refine is not None else 0, primary=self.primary, flip_idx=tuple(flip_idx),
                          input_size=tuple(c.input_size), classes=tuple(classes), keypoints=tuple(keypoints), schema=schema)


def build_model(cfg: ModelConfig, primary: tuple = ()) -> ApolloDetector:
    return ApolloDetector(cfg, primary)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, size: tuple[int, int] = (640, 640)) -> dict:
    """Multiply-accumulates of all convolutions for one image, per top-level block."""
    import copy
    model = copy.deepcopy(model)
    if isinstance(model, ApolloDetector):
        model.fuse()                                  # a RepConv costs ONE 3x3 at deploy, not 3x3 + 1x1 + identity
    macs: dict[str, int] = {}
    hooks = []

    def hook_for(name):
        def hook(m, inp, out):
            k = m.kernel_size[0] * m.kernel_size[1]
            macs[name] = macs.get(name, 0) + int(out.shape[2] * out.shape[3] * m.out_channels * (m.in_channels // m.groups) * k)
        return hook

    for top, mod in model.named_children():
        for m in mod.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(hook_for(top)))
    model.eval()
    x = torch.zeros(1, 3, size[1], size[0], device=next(model.parameters()).device)
    model(x)
    for h in hooks:
        h.remove()
    macs["total"] = sum(macs.values())
    return macs
