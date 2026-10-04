"""ONNX Runtime evaluator: runs a YOLOv8-pose or D-FINE model. TensorRT is used through ONNX Runtime's TensorRT
execution provider (`device="tensorrt"`), so one backend covers CPU, CUDA and TensorRT."""
from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np

from ...core.models import Annotation
from ..interfaces import IModelEvaluator
from ..types import Prediction
from .decode import (KeypointMap, Letterbox, decode_dfine, decode_table, decode_yolov8_pose, letterbox, parse_layout,
                     resize_bilinear, to_chw_float)

PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}


class OnnxEvaluator(IModelEvaluator):
    def __init__(self, model_path: str, fmt: str = "yolov8_pose", input_size: tuple[int, int] = (640, 640),
                 device: str = "cpu", conf_thr: float = 0.05, iou_thr: float = 0.7,
                 keypoint_map: Optional[KeypointMap] = None, threads: int = 0, session=None,
                 layout: Optional[Sequence[str]] = None, coords: str = "pixels",
                 class_map: Optional[Sequence[Optional[int]]] = None, input_dtype: str = "auto",
                 input_layout: str = "auto", max_det: int = 20, nms: bool = False) -> None:
        """fmt: yolov8_pose | dfine | table. `table` is a fixed-size set-prediction head (<= max_det rows) whose columns
        `layout` describes (decode.parse_layout), e.g. ["xyxy", "score", "class", "kp:12", "vis:12"]."""
        if fmt not in ("yolov8_pose", "dfine", "table", "apollo"):
            raise ValueError("fmt must be 'yolov8_pose', 'dfine', 'table' or 'apollo'")
        if fmt == "table":
            if not layout:
                raise ValueError("fmt 'table' needs `layout` (column order of the detection table)")
            parse_layout(layout)                         # fail early on a bad vocabulary
        if input_dtype not in ("auto", "float32", "uint8") or input_layout not in ("auto", "nchw", "nhwc"):
            raise ValueError("input_dtype: auto|float32|uint8; input_layout: auto|nchw|nhwc")
        self.fmt, self.in_w, self.in_h = fmt, int(input_size[0]), int(input_size[1])
        self.conf_thr, self.iou_thr, self.kmap = conf_thr, iou_thr, keypoint_map
        self.layout, self.coords, self.class_map, self.max_det, self.use_nms = layout, coords, class_map, max_det, nms
        self.name = f"onnx:{fmt}:{model_path.split('/')[-1].split(chr(92))[-1]}"
        self.has_keypoints = fmt in ("yolov8_pose", "apollo") or (fmt == "table" and any(t.startswith("kp") for t in (layout or [])))
        self.apollo_layout = None
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
        if fmt == "apollo":                          # raw per-level outputs + the layout stored in the ONNX metadata
            from ...detector.layout import HeadLayout
            meta = {}
            try:
                meta = self.session.get_modelmeta().custom_metadata_map
            except Exception:
                pass
            if "apollo" not in meta:
                raise ValueError("this ONNX file has no 'apollo' layout metadata (export it with `dataopen detector export`)")
            self.apollo_layout = HeadLayout.from_json(meta["apollo"])
            self.in_w, self.in_h = self.apollo_layout.input_size
            self.keypoint_dim = self.apollo_layout.n_kpt
        ins = self.session.get_inputs()
        self.inputs = [i.name for i in ins]
        first = ins[0]
        typ = str(getattr(first, "type", "") or "")
        shape = list(getattr(first, "shape", []) or [])
        self.in_dtype = ("uint8" if "uint8" in typ else "float32") if input_dtype == "auto" else input_dtype
        self.in_layout = input_layout if input_layout != "auto" else (
            "nhwc" if len(shape) == 4 and shape[-1] == 3 and shape[1] != 3 else "nchw")

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
        if self.fmt in ("yolov8_pose", "table", "apollo"):
            lb_img, lb = letterbox(img, self.in_w, self.in_h)
            feed = {self.inputs[0]: self._tensor(lb_img)}
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
        if self.fmt == "apollo":
            from ...detector.layout import DecodeConfig
            from ...detector.postprocess import decode_dense, to_predictions
            lay, n = self.apollo_layout, len(self.apollo_layout.strides)
            cfg = DecodeConfig(conf_thr=self.conf_thr, max_det=min(self.max_det, lay.max_det))
            dets = decode_dense([o for o in outputs[:n]], lay, cfg, outputs[n] if len(outputs) > n else None)
            return to_predictions(dets, lb, lay)
        if self.fmt == "table":
            return decode_table(outputs[0], self.layout or [], lb, self.conf_thr, self.coords, self.kmap, self.class_map,
                                self.max_det, self.iou_thr if self.use_nms else None)
        return decode_dfine(outputs[0], outputs[1], outputs[2], self.conf_thr)

    def _tensor(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 -> the model's input tensor: dtype (float32 0..1 or raw uint8) and layout (NCHW / NHWC)."""
        if self.in_dtype == "uint8":
            t = np.ascontiguousarray(rgb if self.in_layout == "nhwc" else rgb.transpose(2, 0, 1))
        elif self.fmt == "apollo":                    # the 1/255 scale is inside the model: it takes floats in 0..255
            t = np.ascontiguousarray(rgb if self.in_layout == "nhwc" else rgb.transpose(2, 0, 1), dtype=np.float32)
        else:
            t = to_chw_float(rgb) if self.in_layout == "nchw" else np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
        return t[None]
