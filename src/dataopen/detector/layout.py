"""The contract between the network's raw outputs and the host post-processing (numpy only).

Deploy model outputs, one NCHW tensor per pyramid level (and one optional aim-refinement tensor):

    level tensor   (1, n_cls + 4 + 4*K, H_l, W_l), channels  [ cls logits (n_cls) | box (4) | kp offsets (2K) | kp score (K) | kp vis (K) ]
    refine tensor  (1, 3*P, H/4, W/4),            channels  [ heatmap logits (P) | offset x (P) | offset y (P) ]

    anchor of cell (gx, gy) at stride s: ((gx + .5) s, (gy + .5) s)
    box     l, t, r, b  = exp(clip(raw, -6, 6)) * s                    distances from the anchor
    keypoint (x, y)     = anchor + raw * offset_scale * s              offsets in units of offset_scale strides
    scores              = sigmoid(logit)

Everything is plain convolutions on the NPU; exp / sigmoid / top-k run on the host (they are cheap and quantize badly).
The layout is stored in the ONNX file (`custom_metadata_map["apollo"]`) so any consumer decodes it without the training code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class HeadLayout:
    n_cls: int = 2
    n_kpt: int = 12
    strides: tuple[int, ...] = (8, 16, 32)
    offset_scale: float = 4.0            # keypoint offsets are predicted in units of offset_scale * stride
    refine_stride: int = 0               # 0 = no aim-refinement tensor
    primary: tuple[int, ...] = ()        # keypoint indices the refinement tensor refines (schema.primary_idx())
    max_det: int = 20
    flip_idx: tuple[int, ...] = ()
    input_size: tuple[int, int] = (640, 640)
    classes: tuple[str, ...] = ("player_ct", "player_t")
    keypoints: tuple[str, ...] = ()
    schema: str = ""

    @property
    def channels(self) -> int:
        return self.n_cls + 4 + 4 * self.n_kpt

    @property
    def slices(self) -> dict[str, slice]:
        c, k = self.n_cls, self.n_kpt
        return {"cls": slice(0, c), "box": slice(c, c + 4), "kp": slice(c + 4, c + 4 + 2 * k),
                "kp_score": slice(c + 4 + 2 * k, c + 4 + 3 * k), "kp_vis": slice(c + 4 + 3 * k, c + 4 + 4 * k)}

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> "HeadLayout":
        d = json.loads(text)
        for k in ("strides", "primary", "flip_idx", "input_size", "classes", "keypoints"):
            if k in d:
                d[k] = tuple(d[k])
        return HeadLayout(**d)


@dataclass
class DecodeConfig:
    conf_thr: float = 0.25
    max_det: int = 20
    nms_iou: Optional[float] = None      # None = NMS-free (the one-to-one head is trained not to duplicate)
    refine: bool = True
    refine_radius_px: float = 6.0        # the refined point must stay this close to the regressed one (input pixels)
    extras: dict = field(default_factory=dict)
