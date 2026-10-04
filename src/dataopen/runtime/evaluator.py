"""`RuntimeEvaluator`: the production runtime (frame source -> policy -> backend -> quantized KeypointArray) as an `IModelEvaluator`.

Why: the closed loop should judge the model THE WAY IT SHIPS. With this evaluator the quality pipeline's frames go through the same
reader / buffer policy / worker / post-processor / Q12.4 packing that the board runs, so a post-processing or quantization-of-the-
output regression shows up as lower OKS in the loop, not first in the field. (The reverse direction, `EvaluatorBackend`, puts any
closed-loop evaluator, the mock included, inside the runtime.)

Differences from the live path, on purpose: frames are submitted with back-pressure and a queue large enough to never drop (an
evaluation must score every frame), a failed inference raises instead of publishing an empty result (an error must not read as
"the model saw nothing"), and arbitrary image sizes are letterboxed in and mapped back out.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

import numpy as np

from ..core.models import Annotation
from ..detector import structs
from ..quality.evaluators.decode import letterbox
from ..quality.interfaces import IModelEvaluator
from ..quality.types import Prediction
from .backends import ModelBackend
from .frames import QueueSource
from .loop import FLAG_EMPTY_ERROR, InferenceRuntime


class RuntimeEvaluator(IModelEvaluator):
    has_keypoints = True

    def __init__(self, backend: ModelBackend, window: int = 8, timeout_s: float = 60.0, input_size: tuple[int, int] = (640, 640),
                 n_kpt: int = 12, workers: int = 1, backend_factory=None) -> None:
        lay = getattr(backend, "layout", None)
        if lay is not None and getattr(lay, "input_size", None):
            input_size = tuple(lay.input_size)
        self.backend, self.window, self.timeout_s, self.in_w, self.in_h, self.n_kpt = backend, window, timeout_s, *input_size[:2], n_kpt
        self.name = f"runtime:{backend.name}"
        self.wants_hints = bool(getattr(getattr(backend, "ev", None), "wants_hints", False))
        backends = [backend] + [backend_factory() for _ in range(workers - 1)] if workers > 1 and backend_factory else [backend]
        self._src = QueueSource(capacity=window)
        self._cv = threading.Condition()
        self._results: dict[int, object] = {}
        self._fid = 0
        self._lock = threading.Lock()                    # one predict() at a time: results are matched by frame id
        self.rt = InferenceRuntime(self._src, backends, None, policy=f"queue:{window}:oldest", warmup=0, n_kpt=n_kpt,
                                   on_result=self._collect)
        self.rt.start()

    def _collect(self, arr, frame) -> None:
        with self._cv:
            self._results[frame.meta["eval_id"]] = arr
            self._cv.notify_all()

    def predict(self, images: Sequence[np.ndarray], hints: Optional[Sequence[Sequence[Annotation]]] = None
                ) -> list[list[Prediction]]:
        out: list[list[Prediction]] = []
        with self._lock:
            for lo in range(0, len(images), self.window):
                chunk = range(lo, min(lo + self.window, len(images)))
                boxes, ids = {}, {}
                for i in chunk:
                    img = np.asarray(images[i])
                    if self.wants_hints or img.shape[:2] == (self.in_h, self.in_w):
                        arr, lb = np.ascontiguousarray(img), None       # hint-driven test doubles work in the image's own pixels
                    else:
                        arr, lb = letterbox(img, self.in_w, self.in_h)
                    self._fid += 1
                    ids[i], boxes[i] = self._fid, lb
                    meta = {"eval_id": self._fid}
                    if lb is not None:
                        meta["letterbox"] = lb
                    if hints is not None:
                        meta["hints"] = hints[i]
                    if not self._src.push(arr, self._fid, meta=meta):
                        raise RuntimeError("runtime input queue full: window larger than the queue capacity")
                end = time.monotonic() + self.timeout_s
                with self._cv:
                    while any(ids[i] not in self._results for i in chunk):
                        left = end - time.monotonic()
                        if left <= 0:
                            raise TimeoutError(f"{self.name}: no result within {self.timeout_s}s (stalled backend?)")
                        self._cv.wait(left)
                    got = {i: self._results.pop(ids[i]) for i in chunk}
                for i in chunk:
                    a = got[i]
                    if a.flags & FLAG_EMPTY_ERROR:
                        raise RuntimeError(f"{self.name}: inference failed: {self.rt.metrics.last_error}")
                    out.append(self._to_predictions(a, boxes[i]))
        return out

    def _to_predictions(self, a, lb) -> list[Prediction]:
        preds = []
        for d in structs.unpack(a, self.n_kpt):
            box, kxy = np.array(d["box"], dtype=np.float64), np.array(d["kxy"], dtype=np.float64)
            if lb is not None:
                box = box.reshape(2, 2)
                box = ((box - [lb.pad_x, lb.pad_y]) / lb.scale).reshape(4)
                kxy = (kxy - [lb.pad_x, lb.pad_y]) / lb.scale
            kp = np.concatenate([kxy, np.asarray(d["kscore"], dtype=np.float64)[:, None]], axis=1)
            preds.append(Prediction(bbox=(float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])),
                                    score=float(d["score"]), keypoints=kp, class_id=int(d["cls"]),
                                    visibility=np.asarray(d["kvis"], dtype=np.float64)))
        return preds

    def warmup(self) -> None:
        self.backend.warmup(2)

    def runtime_metrics(self) -> dict:
        return self.rt.metrics.snapshot()

    def close(self) -> None:
        self.rt.stop(close=False)
        self.backend.close()
