"""Model / training configuration (plain dataclasses, TOML on disk; no torch import)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    import tomli as tomllib          # type: ignore[no-redef]


class ConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    variant: str = "s"
    width_mult: float = 0.75
    n_cls: int = 2
    n_kpt: int = 12
    stem: tuple = (24, 48)                       # channels after the stride-2 and stride-4 convs (before width_mult)
    stages: tuple = ((96, 32, 1, 3), (192, 64, 2, 3), (384, 96, 1, 3))   # strides 8/16/32: (out, mid, blocks, layers)
    neck_ch: int = 64
    neck_repeats: int = 1
    head_ch: int = 40
    kp_head_ch: int = 48
    cls_k3: bool = True                          # 3x3 (True) or 1x1 (False) first conv in the class branch
    kp_second_conv: bool = True                  # a second 3x3 in the keypoint branch
    aim_refine: bool = True
    refine_ch: int = 24
    offset_scale: float = 4.0
    input_size: tuple = (640, 640)


# Named presets. `t` is for tests; n/s/m trade accuracy for NPU time (measure MACs with `dataopen detector bench --macs`).
VARIANTS: dict[str, dict[str, Any]] = {
    "t": dict(width_mult=0.25, stem=(16, 24), stages=((96, 32, 1, 2), (192, 64, 1, 2), (384, 96, 1, 2)), neck_ch=32,
              head_ch=24, kp_head_ch=24, cls_k3=False, refine_ch=16, input_size=(128, 128)),
    # ~1.2 GMACs at 640x640: the candidate for < 4 ms
    "n": dict(width_mult=0.5, stem=(16, 24), stages=((96, 24, 1, 2), (192, 48, 1, 2), (384, 80, 1, 2)), neck_ch=48,
              head_ch=32, kp_head_ch=32, cls_k3=False, kp_second_conv=True, refine_ch=16),
    # ~2.3 GMACs: the accuracy candidate (runs at the throughput target when the 3 NPU cores work on separate frames)
    "s": dict(width_mult=0.75, stem=(16, 24), stages=((96, 24, 1, 3), (192, 48, 2, 3), (384, 80, 1, 3)), neck_ch=56,
              head_ch=40, kp_head_ch=40, cls_k3=False, kp_second_conv=True, refine_ch=24),
    # reference for the accuracy ceiling, NOT for the NPU latency target
    "m": dict(width_mult=1.0, stem=(24, 32), stages=((96, 32, 1, 3), (192, 64, 2, 3), (384, 96, 1, 3)), neck_ch=80,
              head_ch=64, kp_head_ch=64, neck_repeats=2, cls_k3=True, kp_second_conv=True, refine_ch=32),
}


def model_config(variant: str = "s", **overrides: Any) -> ModelConfig:
    if variant not in VARIANTS:
        raise ConfigError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    return ModelConfig(variant=variant, **{**VARIANTS[variant], **overrides})


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 64
    imgsz: int = 640
    lr: float = 2e-3
    min_lr_ratio: float = 0.01
    weight_decay: float = 0.025
    warmup_epochs: float = 3.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 10.0
    ema_decay: float = 0.9998
    ema_ramp: float = 2000.0
    amp: bool = True
    workers: int = 8
    seed: int = 0
    eval_every: int = 2
    save_every: int = 10
    # augmentation
    mosaic: float = 1.0
    mixup: float = 0.1
    close_mosaic_epochs: int = 15
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flip: float = 0.5
    scale: float = 0.5                           # random scale in [1-scale, 1+scale]
    translate: float = 0.1
    degrees: float = 4.0
    smoke_flash: float = 0.2                     # photometric smoke / flash overlays (labels unchanged)
    hard_sampling: float = 1.0                   # sampling weight = image `weight` ** hard_sampling (the closed loop's weight)
    # loss
    tal_topk_o2m: int = 10
    tal_topk_o2o: int = 1
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    w_cls: float = 1.0
    w_box: float = 7.5
    w_kp: float = 12.0
    w_kp_score: float = 1.0
    w_vis: float = 1.0
    w_heat: float = 2.0
    w_o2o: float = 1.0


@dataclass
class DetectorConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _fill(cls, d: dict, where: str):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ConfigError(f"[{where}] unknown keys {unknown}; known: {sorted(names)}")
    out = {}
    for k, v in d.items():
        out[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
    return out


def load_config(path: str | Path) -> DetectorConfig:
    try:
        d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path}: {e}") from e
    m = dict(d.get("model", {}))
    variant = m.pop("variant", "s")
    model = model_config(variant, **_fill(ModelConfig, m, "model"))
    train = TrainConfig(**_fill(TrainConfig, d.get("train", {}), "train"))
    return DetectorConfig(model, train)
