"""Training loop: AdamW, linear warm-up + cosine annealing, EMA, AMP, mosaic/mixup/HSV/flip/scale augmentation,
quality-weighted sampling, periodic validation with the deployed (reparameterized) model, resumable checkpoints."""
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..core.schema import SkeletonSchema
from .config import DetectorConfig
from .data import DataError, PoseDataset, collate, load_schema_from_dataset, read_coco
from .evaluate import evaluate_evaluator
from .infer import TorchEvaluator
from .loss import DetectionLoss
from .model import ApolloDetector, build_model, count_macs, count_params


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float, ramp: float) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.ramp, self.updates = decay, ramp, 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.ramp))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


def lr_at(step: int, total: int, warm: int, base: float, min_ratio: float) -> float:
    if step < warm:
        return base * (0.1 + 0.9 * step / max(1, warm))
    t = (step - warm) / max(1, total - warm)
    return base * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


def param_groups(model: torch.nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or n.endswith(".bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def schema_meta(schema: SkeletonSchema) -> dict:
    return {"name": schema.name, "keypoints": list(schema.keypoints), "edges": [list(e) for e in schema.edges],
            "flip_pairs": [list(p) for p in schema.flip_pairs], "flip_idx": schema.flip_idx(), "sigmas": schema.oks_sigmas(),
            "weights": schema.oks_weights(), "derived": list(schema.derived), "primary": list(schema.primary),
            "groups": {g: list(m) for g, m in schema.groups}, "roles": dict(schema.roles), "classes": list(schema.classes),
            "class_key": schema.class_key}


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loss(cfg: DetectorConfig, schema: SkeletonSchema, model: ApolloDetector) -> DetectionLoss:
    lay = model.layout(tuple(schema.keypoints), tuple(schema.classes), tuple(schema.flip_idx()), schema.name)
    return DetectionLoss(cfg.train, lay, schema.oks_sigmas(), schema.oks_weights(), model.primary)


def save_checkpoint(path: Path, model, ema: ModelEMA, opt, step: int, epoch: int, cfg: DetectorConfig, schema: SkeletonSchema,
                    best: float) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.ema.state_dict(), "ema_updates": ema.updates, "opt": opt.state_dict(),
                "step": step, "epoch": epoch, "best": best, "cfg": asdict(cfg), "schema": schema_meta(schema)}, tmp)
    tmp.replace(path)


def train(cfg: DetectorConfig, data: Sequence[Path], out: Path, device: str = "auto", resume: bool = False,
          max_steps: Optional[int] = None, val_max: Optional[int] = None, log: Callable[[str], None] = print) -> dict:
    t = cfg.train
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    dev = pick_device(device)

    schema = load_schema_from_dataset(Path(data[0]))
    for d in data[1:]:
        s2 = load_schema_from_dataset(Path(d))
        if s2.keypoints != schema.keypoints or s2.classes != schema.classes:
            raise DataError(f"{d} was labeled with a different schema than {data[0]}")
    cfg.model.n_kpt, cfg.model.n_cls = schema.num_keypoints, len(schema.classes)
    train_items = [it for d in data for it in read_coco(Path(d), "train", schema.num_keypoints)]
    val_items = [it for d in data for it in read_coco(Path(d), "val", schema.num_keypoints)] if all(
        (Path(d) / "annotations" / "coco_val.json").exists() for d in data) else []
    if not train_items:
        raise DataError("no training images")
    if not val_items:
        log("warning: no validation split (coco_val.json): validation metrics are skipped")
    log(f"schema {schema.name}: {schema.num_keypoints} keypoints, classes {list(schema.classes)}, primary {list(schema.primary)}; "
        f"{len(train_items)} train / {len(val_items)} val images")

    model = build_model(cfg.model, tuple(schema.primary_idx())).to(dev)
    macs = count_macs(model, cfg.model.input_size)
    log(f"model {cfg.model.variant}: {count_params(model) / 1e6:.2f}M params, {macs['total'] / 1e9:.2f} GMACs at "
        f"{cfg.model.input_size[0]}x{cfg.model.input_size[1]} (deploy)")
    loss_fn = make_loss(cfg, schema, model)
    ema = ModelEMA(model, t.ema_decay, t.ema_ramp)

    ds = PoseDataset(train_items, t, schema.flip_idx(), schema.num_keypoints, train=True, imgsz=t.imgsz)
    sampler = WeightedRandomSampler(ds.sample_weights(), num_samples=len(ds), replacement=True,
                                    generator=torch.Generator().manual_seed(t.seed))
    loader = DataLoader(ds, batch_size=t.batch_size, sampler=sampler, num_workers=t.workers, collate_fn=collate,
                        drop_last=len(ds) >= t.batch_size, persistent_workers=t.workers > 0, pin_memory=dev == "cuda")
    steps_per_epoch = max(1, len(loader))
    total = t.epochs * steps_per_epoch
    warm = int(t.warmup_epochs * steps_per_epoch)
    opt = torch.optim.AdamW(param_groups(model, t.weight_decay), lr=t.lr, betas=(t.beta1, t.beta2))
    use_amp = t.amp and dev == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step = epoch0 = 0
    best = -1.0
    ckpt = out / "last.pt"
    if resume and ckpt.exists():
        c = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"])
        ema.ema.load_state_dict(c["ema"])
        ema.updates = c["ema_updates"]
        opt.load_state_dict(c["opt"])
        step, epoch0, best = c["step"], c["epoch"] + 1, c["best"]
        log(f"resumed from {ckpt} at epoch {epoch0}")

    logf = (out / "train_log.jsonl").open("a", encoding="utf-8")
    skipped = 0
    metrics: dict = {}
    t0 = time.time()
    for epoch in range(epoch0, t.epochs):
        ds.set_epoch(epoch, close_mosaic=epoch >= t.epochs - t.close_mosaic_epochs)
        model.train()
        run = {}
        n_b = 0
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warm, t.lr, t.min_lr_ratio)
            imgs = batch["images"].to(dev, non_blocking=True).float()
            tg = {k: batch[k].to(dev) for k in ("boxes", "cls", "kpts", "valid")}
            with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=amp_dtype, enabled=use_amp):
                out_d = model(imgs)
            out_d = {k: ([o.float() for o in v] if isinstance(v, list) else (v.float() if v is not None else None))
                     for k, v in out_d.items()}
            loss, items = loss_fn(out_d, tg)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                if skipped > 20:
                    raise FloatingPointError("the loss was not finite 20 times: lower the learning rate or check the labels")
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1
            n_b += 1
            for k, v in items.items():
                run[k] = run.get(k, 0.0) + float(v)
            if max_steps is not None and step >= max_steps:
                break
        rec = {"epoch": epoch, "step": step, "lr": opt.param_groups[0]["lr"], "skipped": skipped,
               **{k: round(v / max(1, n_b), 4) for k, v in run.items()}, "minutes": round((time.time() - t0) / 60, 2)}
        last = epoch == t.epochs - 1 or (max_steps is not None and step >= max_steps)
        if val_items and ((epoch + 1) % t.eval_every == 0 or last):
            lay = ema.ema.layout(tuple(schema.keypoints), tuple(schema.classes), tuple(schema.flip_idx()), schema.name)
            ev = TorchEvaluator(ema.ema, lay, dev)
            metrics = evaluate_evaluator(ev, val_items[:val_max] if val_max else val_items, schema)
            rec["val"] = metrics
            score = float(np.nan_to_num(0.5 * metrics.get("map50", 0.0) + 0.5 * metrics.get("kp_ap_oks50", 0.0)))
            if score > best:
                best = score
                save_checkpoint(out / "best.pt", model, ema, opt, step, epoch, cfg, schema, best)
        log(json.dumps(rec))
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        save_checkpoint(ckpt, model, ema, opt, step, epoch, cfg, schema, best)
        if last:
            break
    logf.close()
    if best < 0:                                                      # no validation: the EMA at the end is the result
        save_checkpoint(out / "best.pt", model, ema, opt, step, t.epochs, cfg, schema, best)
    return {"steps": step, "best_score": best, "val": metrics, "skipped_steps": skipped, "out": str(out)}
