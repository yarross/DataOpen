"""Run a trained detector inside the closed loop / evaluation as an `IModelEvaluator` (PyTorch)."""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
import torch

from ..core.models import Annotation
from ..quality.evaluators.decode import letterbox
from ..quality.interfaces import IModelEvaluator
from ..quality.types import Prediction
from .layout import DecodeConfig, HeadLayout
from .postprocess import decode_dense, to_predictions


class TorchEvaluator(IModelEvaluator):
    has_keypoints = True

    def __init__(self, model: torch.nn.Module, layout: HeadLayout, device: str = "cpu", conf_thr: float = 0.25,
                 decode: Optional[DecodeConfig] = None) -> None:
        net = model if getattr(model, "deployed", False) else copy.deepcopy(model).fuse()
        self.net, self.lay, self.device = net.to(device).eval(), layout, device
        self.cfg = decode or DecodeConfig(conf_thr=conf_thr, max_det=layout.max_det)
        self.name = "apollo-torch"

    @torch.no_grad()
    def raw(self, rgb_batch: np.ndarray) -> list[np.ndarray]:
        x = torch.from_numpy(np.ascontiguousarray(rgb_batch.transpose(0, 3, 1, 2))).to(self.device).float()
        outs = self.net(x)
        return [o.cpu().numpy() for o in outs]

    def predict(self, images: Sequence[np.ndarray], hints: Optional[Sequence[Sequence[Annotation]]] = None
                ) -> list[list[Prediction]]:
        w, h = self.lay.input_size
        result = []
        for img in images:
            lb_img, lb = letterbox(img, w, h)
            outs = self.raw(lb_img[None])
            n = len(self.lay.strides)
            dets = decode_dense([o[0] for o in outs[:n]], self.lay, self.cfg, outs[n][0] if len(outs) > n else None)
            result.append(to_predictions(dets, lb, self.lay))
        return result
