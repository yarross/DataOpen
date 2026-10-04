"""ONNX Runtime evaluator: runs a YOLOv8-pose or D-FINE model. TensorRT is used through ONNX Runtime's TensorRT
execution provider (`device="tensorrt"`), so one backend covers CPU, CUDA and TensorRT."""
from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np

from ...core.models import Annotation
from ..interfaces import IModelEvaluator
from ..types import Prediction
from .decode import KeypointMap, Letterbox, decode_dfine, decode_yolov8_pose, letterbox, resize_bilinear, to_chw_float

PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}


class OnnxEvaluator(IModelEvaluator):
    def __init__(self, model_path: str, fmt: str = "yolov8_pose", input_size: tuple[int, int] = (640, 640),
                 device: str = "cpu", conf_thr: float = 0.05, iou_thr: float = 0.7,
                 keypoint_map: Optional[KeypointMap] = None, threads: int = 0, session=None) -> None:
        if fmt not in ("yolov8_pose", "dfine"):
            raise ValueError("fmt must be 'yolov8_pose' or 'dfine'")
        self.fmt, self.in_w, self.in_h = fmt, int(input_size[0]), int(input_size[1])
        self.conf_thr, self.iou_thr, self.kmap = conf_thr, iou_thr, keypoint_map
        self.name = f"onnx:{fmt}:{model_path.split('/')[-1].split(chr(92))[-1]}"
        self.has_keypoints = fmt == "yolov8_pose"
        self._lock = threading.Lock()                # one session, serialized: ORT sessions are thread-safe but we batch
        if session is not None:
            self.session = session
        else:
            import onnxruntime as ort
            wanted = PROVIDERS.get(device)
            if wanted is None:
                raise ValueError(f"device must be one of {sorted(PROVIDERS)}")
            avail = set(ort.get_available_providers())
            providers = [p for p in wanted if p in avail]
            if providers[0] != wanted[0]:
                import logging
                logging.getLogger("dataopen").warning("requested %s but only %s is available: running on %s",
                                                      wanted[0], sorted(avail), providers[0])
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.inputs = [i.name for i in self.session.get_inputs()]

    def warmup(self) -> None:
        self.predict([np.zeros((self.in_h, self.in_w, 3), dtype=np.uint8)])

    def predict(self, images: Sequence[np.ndarray], hints: Optional[Sequence[Sequence[Annotation]]] = None
                ) -> list[list[Prediction]]:
        out: list[list[Prediction]] = []
        for img in images:
            out.append(self._predict_one(img))
        return out

    def _predict_one(self, img: np.ndarray) -> list[Prediction]:
        h, w, _ = img.shape
        if self.fmt == "yolov8_pose":
            lb_img, lb = letterbox(img, self.in_w, self.in_h)
            feed = {self.inputs[0]: to_chw_float(lb_img)[None]}
        else:
            lb = Letterbox(1.0, 0, 0, self.in_w, self.in_h, w, h)
            resized = img if (w, h) == (self.in_w, self.in_h) else resize_bilinear(img, self.in_w, self.in_h)
            feed = {self.inputs[0]: to_chw_float(resized)[None]}
            if len(self.inputs) > 1:                  # D-FINE export takes orig_target_sizes = [[w, h]]
                feed[self.inputs[1]] = np.array([[w, h]], dtype=np.int64)
        with self._lock:
            outputs = self.session.run(None, feed)
        if self.fmt == "yolov8_pose":
            return decode_yolov8_pose(outputs[0], lb, self.conf_thr, self.iou_thr, self.kmap)
        return decode_dfine(outputs[0], outputs[1], outputs[2], self.conf_thr)
