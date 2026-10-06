"""CPU-friendly training of UiNet on synthetic screens (optionally mixed with real labelled screenshots)."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import synth
from .data import RealScreens, SynthUi, collate
from .evaluate import Report, evaluate
from .infer import decode
from .loss import ui_loss
from .model import UiConfig, UiNet
from .taxonomy import NAMES


class Mix:
    """Synthetic samples with, at ratio `real_share`, real labelled screenshots mixed in."""

    def __init__(self, synth_ds, real_ds=None, real_share: float = 0.0, seed: int = 0) -> None:
        self.s, self.r, self.share, self.seed = synth_ds, real_ds, real_share, seed

    def __len__(self) -> int:
        return len(self.s)

    def __getitem__(self, i: int):
        if self.r is not None and len(self.r) and np.random.default_rng([self.seed, i, 7]).random() < self.share:
            return self.r[i % len(self.r)]
        return self.s[i]


def random_window(x: torch.Tensor, gts, size: int, rng: np.random.Generator):
    """A random size x size window of each 640x640 input (cheaper steps, and objects at all positions); boxes follow."""
    b, _, h, w = x.shape
    if size >= h:
        return x, gts
    outs, ng = [], []
    for i in range(b):
        ox, oy = int(rng.integers(0, w - size + 1)), int(rng.integers(0, h - size + 1))
        outs.append(x[i : i + 1, :, oy : oy + size, ox : ox + size])
        bx, cl = gts[i]
        if len(bx):
            nb = bx.clone()
            nb[:, [0, 2]] = (nb[:, [0, 2]] - ox).clamp(0, size)
            nb[:, [1, 3]] = (nb[:, [1, 3]] - oy).clamp(0, size)
            a0 = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
            a1 = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
            keep = (a1 >= 0.5 * a0) & ((nb[:, 2] - nb[:, 0]) >= 2.5) & ((nb[:, 3] - nb[:, 1]) >= 2.5)
            ng.append((nb[keep], cl[keep]))
        else:
            ng.append((bx, cl))
    return torch.cat(outs), ng


def lr_at(step: int, total: int, base: float, warm: int = 200, floor: float = 0.02) -> float:
    if step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def predict(model: UiNet, ds, n: int, conf: float = 0.05, batch: int = 8, device="cpu"):
    model.eval()
    preds, gts = [], []
    for s in range(0, n, batch):
        items = [ds[i] for i in range(s, min(n, s + batch))]
        x, g, _, _ = collate(items)
        outs = model(x.float().div(255).to(device))
        for k in range(len(items)):
            lv = [o[k].cpu().numpy() for o in outs]
            preds.append(decode(lv, model.cfg.n_cls, conf=conf))
            gts.append((g[k][0].numpy(), g[k][1].numpy()))
    return preds, gts


def train(
    out: str | Path,
    steps: int = 2000,
    batch: int = 16,
    window: int = 512,
    lr: float = 2e-3,
    width: float = 0.5,
    neck: int = 48,
    workers: int = 2,
    seed: int = 0,
    val_n: int = 96,
    eval_every: int = 500,
    real_dir: Optional[str] = None,
    real_share: float = 0.3,
    resume: Optional[str] = None,
    log=print,
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UiNet(UiConfig(width=width, neck=neck))
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model = UiNet(UiConfig(**ck["cfg"]))
        model.load_state_dict(ck["model"])
    ema = UiNet(model.cfg)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    real = RealScreens(real_dir, True, seed) if real_dir else None
    ds = Mix(SynthUi(steps * batch + 1, seed, synth.TRAIN_FAMILIES, True), real, real_share if real else 0.0, seed)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers else None,
    )
    val_in = SynthUi(val_n, 10_000 + seed, synth.TRAIN_FAMILIES, False)
    val_out = SynthUi(val_n, 20_000 + seed, synth.HELDOUT_FAMILIES, False)
    decay, wd = [], []
    for n_, p in model.named_parameters():  # noqa: B007
        (wd if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": wd, "weight_decay": 0.0}], lr=lr)
    best, hist, t0 = -1.0, [], time.time()
    step = 0
    for x, gts, _, _ in dl:
        if step >= steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        model.train()
        x, gts = random_window(x.float().div(255), gts, window, rng)
        outs = model(x)
        loss, parts = ui_loss(outs, gts, model.cfg.n_cls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        d = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                if pe.dtype.is_floating_point:
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                else:
                    pe.copy_(pm)
        step += 1
        if step % 50 == 0:
            log(
                f"step {step}/{steps}  loss {float(loss):.3f}  cls {parts['cls']:.3f}  box {parts['box']:.3f}  pos {parts['pos']}  {time.time() - t0:.0f}s"  # noqa: E501
            )
        if step % eval_every == 0 or step == steps:
            r_in = evaluate(*predict(ema, val_in, val_n))
            r_out = evaluate(*predict(ema, val_out, val_n))
            hist.append({"step": step, "map50_in": r_in.map50, "map50_heldout": r_out.map50})
            log(f"  eval step {step}: mAP50 in-distribution {r_in.map50:.3f}, held-out themes {r_out.map50:.3f}")
            ck = {"model": ema.state_dict(), "cfg": asdict(ema.cfg), "step": step, "classes": list(NAMES)}
            torch.save(ck, out / "last.pt")
            if r_in.map50 > best:
                best = r_in.map50
                torch.save(ck, out / "best.pt")
    return {"steps": step, "best_map50_in": best, "history": hist, "seconds": time.time() - t0}


def load(path: str | Path, device="cpu") -> UiNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["cfg"])
    cfg["classes"] = tuple(cfg.get("classes", NAMES))
    m = UiNet(UiConfig(**cfg))
    m.load_state_dict(ck["model"])
    return m.eval()


def report_text(name: str, r: Report) -> str:
    lines = [
        f"{name}: mAP50 {r.map50:.3f}  recall of interface elements {r.recall_targets:.3f}  false elements/screen {r.fp_per_screen:.2f}  "
        f"pointer found {r.cursor_found:.2f}, median error {r.cursor_err_px:.1f} px (input)"
    ]
    lines.append("   AP50: " + "  ".join(f"{k} {v:.2f}" for k, v in r.ap50.items()))
    lines.append("   recall by size (input px): " + "  ".join(f"{k}: {v:.2f}" for k, v in r.by_size.items()))
    return "\n".join(lines)
