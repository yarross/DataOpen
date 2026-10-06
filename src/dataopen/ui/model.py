"""UiNet: a small boxes-only detector for interface elements.

Plain convolutions (BatchNorm folds away at export, ReLU only) so it quantizes and runs on an NPU; decoding (exp, sigmoid, NMS) is on the
host. Three scales (strides 8, 16, 32) with a light top-down neck; per cell a class-logit vector and the distances (l, t, r, b) to the box
edges, in units of the stride (exp). It is deliberately a different, much smaller thing than the pose network: no keypoints, no people
classes, no shared weights, no shared code path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .taxonomy import NAMES



@dataclass
class UiConfig:
    n_cls: int = len(NAMES)
    width: float = 1.0
    neck: int = 64
    input_size: int = 640
    classes: tuple = NAMES


def cbr(i: int, o: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UiNet(nn.Module):
    def __init__(self, cfg: UiConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UiConfig()
        w = lambda c: max(8, int(round(c * self.cfg.width)))  # noqa: E731
        n = self.cfg.neck
        self.stem = nn.Sequential(cbr(3, w(16), 3, 2), cbr(w(16), w(24), 3, 2))  # stride 4
        self.s8 = nn.Sequential(cbr(w(24), w(48), 3, 2), cbr(w(48), w(48)), cbr(w(48), w(48)))
        self.s16 = nn.Sequential(cbr(w(48), w(96), 3, 2), cbr(w(96), w(96)), cbr(w(96), w(96)))
        self.s32 = nn.Sequential(cbr(w(96), w(160), 3, 2), cbr(w(160), w(160)), cbr(w(160), w(160)))
        self.lat = nn.ModuleList([cbr(w(48), n, 1), cbr(w(96), n, 1), cbr(w(160), n, 1)])
        self.smooth = nn.ModuleList([cbr(n, n, 3) for _ in range(3)])
        self.cls_head = nn.ModuleList([nn.Sequential(cbr(n, 48, 3), cbr(48, 48, 3), nn.Conv2d(48, self.cfg.n_cls, 1)) for _ in range(3)])
        self.box_head = nn.ModuleList([nn.Sequential(cbr(n, 32, 3), cbr(32, 32, 3), nn.Conv2d(32, 4, 1)) for _ in range(3)])
        for h in self.cls_head:
            nn.init.constant_(h[-1].bias, -4.6)  # prior probability ~1%
        for h in self.box_head:
            nn.init.constant_(h[-1].bias, 0.5)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, 3, H, W) float in 0..1. Returns one (B, n_cls + 4, H/s, W/s) tensor per stride: [class logits | raw ltrb]."""
        x = self.stem(x)
        c8 = self.s8(x)
        c16 = self.s16(c8)
        c32 = self.s32(c16)
        p32 = self.lat[2](c32)
        p16 = self.lat[1](c16) + nn.functional.interpolate(p32, scale_factor=2, mode="nearest")
        p8 = self.lat[0](c8) + nn.functional.interpolate(p16, scale_factor=2, mode="nearest")
        outs = []
        for i, p in enumerate((p8, p16, p32)):
            p = self.smooth[i](p)
            outs.append(torch.cat([self.cls_head[i](p), self.box_head[i](p)], dim=1))
        return outs

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def grid_centers(h: int, w: int, stride: int, device=None) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack([(xs + 0.5) * stride, (ys + 0.5) * stride], dim=-1).float()  # (h, w, 2)


def decode_boxes(raw: torch.Tensor, stride: int) -> torch.Tensor:
    """raw (B, 4, H, W) -> boxes (B, H, W, 4) xyxy in input pixels."""
    b, _, h, w = raw.shape
    c = grid_centers(h, w, stride, raw.device)
    d = torch.exp(raw.clamp(-6, 6)).permute(0, 2, 3, 1) * stride
    return torch.cat([c - d[..., :2], c + d[..., 2:]], dim=-1)
