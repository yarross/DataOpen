"""RKNN (Rockchip NPU) evaluator for YOLOv8-pose exported with rknn-toolkit.

UNTESTED in this repository (no NPU, no rknn-toolkit-lite2 here): it reuses the verified pre/post-processing from
`decode.py` and only adds the runtime calls. Validate with a known image before trusting it.
"""
from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np

from ...core.models import Annotation
from ..interfaces import IModelEvaluator
from ..types import Prediction
from .decode import KeypointMap, decode_yolov8_pose, letterbox


class RknnEvaluator(IModelEvaluator):
    has_keypoints = True

    def __init__(self, model_path: str, input_size: tuple[int, int] = (640, 640), conf_thr: float = 0.05,
                 iou_thr: float = 0.7, keypoint_map: Optional[KeypointMap] = None) -> None:
        from rknnlite.api import RKNNLite  # type: ignore[import-not-found]
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0 or self.rknn.init_runtime() != 0:
            raise RuntimeError(f"cannot load {model_path} on the NPU")
        self.name = f"rknn:{model_path}"
        self.in_w, self.in_h, self.conf_thr, self.iou_thr, self.kmap = input_size[0], input_size[1], conf_thr, iou_thr, keypoint_map
        self._lock = threading.Lock()

    def predict(self, images: Sequence[np.ndarray], hints: Optional[Sequence[Sequence[Annotation]]] = None
                ) -> list[list[Prediction]]:
        out = []
        for img in images:
            lb_img, lb = letterbox(img, self.in_w, self.in_h)
            with self._lock:
                raw = self.rknn.inference(inputs=[lb_img[None]])      # NHWC uint8, quantization handled by the model
            out.append(decode_yolov8_pose(np.asarray(raw[0]), lb, self.conf_thr, self.iou_thr, self.kmap))
        return out

    def close(self) -> None:
        self.rknn.release()
