"""Model backends: `infer(frame) -> KeypointArray` (detections in the 640x640 model-input pixels of that frame).

  OrtBackend          ONNX Runtime: CPU, CUDA, TensorRT (through ORT's execution providers), DirectML. The deploy ONNX carries its own
                      layout, so no training code is needed. uint8 NHWC models take the frame buffer as it is (zero copy).
  RknnLiteBackend     Rockchip NPU through rknn-toolkit-lite2 (`rknnlite`). Written to the library's documented API, NOT run here.
  EvaluatorBackend    ANY `IModelEvaluator` of the closed loop (SimulatedEvaluator, CallableEvaluator, OnnxEvaluator, TorchEvaluator...):
                      the mock model for tests and the research-path model behind the production runtime.
  ScriptedBackend     deterministic detections with a configurable latency / failure pattern: tests and load experiments.
  FallbackBackend     a primary backend with a secondary one to switch to after repeated failures (NPU hiccup -> CPU), reported as
                      `degraded` (flag bit in every result).
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..detector import structs
from ..detector.layout import DecodeConfig, HeadLayout
from ..quality.interfaces import IModelEvaluator
from .frames import Frame
from .postproc import make_postprocessor

log = logging.getLogger("dataopen.runtime")

PROVIDERS = {"cpu": ["CPUExecutionProvider"], "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
             "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
             "directml": ["DmlExecutionProvider", "CPUExecutionProvider"]}


class ModelBackend(ABC):
    name = "backend"
    degraded = False
    layout: Optional[HeadLayout] = None

    def warmup(self, n: int = 3) -> None:
        """Run a few throw-away inferences (first-call costs: kernel compile, allocator growth) before real frames arrive."""
        dummy = Frame(np.zeros((640, 640, 3), np.uint8), -1, 0)
        for _ in range(n):
            self.infer(dummy)

    @abstractmethod
    def infer(self, frame: Frame):
        """-> KeypointArray (ctypes), detections in model-input pixels."""

    def close(self) -> None:
        pass

    def info(self) -> dict[str, Any]:
        return {"backend": self.name}


def _check_input(frame: Frame, size: tuple[int, int]) -> None:
    a = frame.array
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or (a.shape[1], a.shape[0]) != tuple(size):
        raise ValueError(f"frame must be {size[0]}x{size[1]}x3 uint8 RGB, got {a.shape} {a.dtype}")


class OrtBackend(ModelBackend):
    def __init__(self, model_path: str | Path, provider: str | Sequence[str] = "cpu", conf_thr: float = 0.25, max_det: int = 20,
                 post: str = "auto", threads: int = 0, layout: Optional[HeadLayout] = None) -> None:
        import onnxruntime as ort
        self.path = str(model_path)
        wanted = PROVIDERS.get(provider, [provider]) if isinstance(provider, str) else list(provider)
        avail = set(ort.get_available_providers())
        chosen = [p for p in wanted if p in avail]
        if not chosen:
            raise RuntimeError(f"none of {wanted} is available in this onnxruntime (has {sorted(avail)})")
        if chosen[0] != wanted[0]:
            log.warning("requested %s but only %s is available: running on %s", wanted[0], sorted(avail), chosen[0])
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(self.path, sess_options=so, providers=chosen)
        self.provider_used = self.sess.get_providers()[0]
        meta = self.sess.get_modelmeta().custom_metadata_map
        if layout is None:
            if "apollo" not in meta:
                raise ValueError(f"{self.path} has no 'apollo' layout metadata: export it with `dataopen detector export`")
            layout = HeadLayout.from_json(meta["apollo"])
        self.layout = layout
        inp = self.sess.get_inputs()[0]
        self.in_name, self.uint8 = inp.name, "uint8" in inp.type
        shape = list(inp.shape)
        self.nhwc = len(shape) == 4 and shape[-1] == 3 and shape[1] != 3
        self.size = tuple(layout.input_size)
        self.post = make_postprocessor(layout, DecodeConfig(conf_thr=conf_thr, max_det=min(max_det, layout.max_det)), post)
        self.name = f"ort:{self.provider_used}:{Path(self.path).name}"

    def _tensor(self, a: np.ndarray) -> np.ndarray:
        if self.uint8 and self.nhwc:
            return a[None]                                                     # zero copy: the model reads the frame as it is
        t = a if self.nhwc else a.transpose(2, 0, 1)
        return np.ascontiguousarray(t[None], dtype=np.uint8 if self.uint8 else np.float32)

    def infer(self, frame: Frame):
        _check_input(frame, self.size)
        outs = self.sess.run(None, {self.in_name: self._tensor(frame.array)})
        n = len(self.layout.strides)
        return self.post.decode(outs[:n], outs[n] if len(outs) > n else None)

    def info(self) -> dict[str, Any]:
        return {"backend": "onnxruntime", "provider": self.provider_used, "model": self.path, "post": self.post.name,
                "input": ("uint8" if self.uint8 else "float32") + (" NHWC" if self.nhwc else " NCHW")}


class RknnLiteBackend(ModelBackend):
    """Rockchip NPU via `rknnlite`. `core` is "auto", "0", "1", "2", "0_1" or "all" (core fusion across the three RK3588 cores).
    The .rknn file carries no layout: pass `layout` or keep the `<model>.layout.json` that `dataopen detector export` writes next to the
    ONNX file beside it. NOT executed in the dataopen repository (no Rockchip hardware or toolkit here)."""

    def __init__(self, model_path: str | Path, core: str = "auto", conf_thr: float = 0.25, max_det: int = 20, post: str = "auto",
                 layout: Optional[HeadLayout] = None) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as e:
            raise RuntimeError("rknnlite is not installed (Rockchip rknn-toolkit-lite2, on the board)") from e
        path = Path(model_path)
        if layout is None:
            side = path.with_suffix(".layout.json")
            if not side.exists():
                raise ValueError(f"no layout for {path.name}: pass layout=... or put {side.name} next to it")
            layout = HeadLayout.from_json(side.read_text())
        self.layout = layout
        masks = {"auto": RKNNLite.NPU_CORE_AUTO, "0": RKNNLite.NPU_CORE_0, "1": RKNNLite.NPU_CORE_1, "2": RKNNLite.NPU_CORE_2,
                 "0_1": RKNNLite.NPU_CORE_0_1, "all": RKNNLite.NPU_CORE_0_1_2}
        if core not in masks:
            raise ValueError(f"core must be one of {sorted(masks)}")
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(str(path)) != 0 or self.rknn.init_runtime(core_mask=masks[core]) != 0:
            raise RuntimeError(f"cannot load/initialize {path} on NPU core {core}")
        self.size = tuple(layout.input_size)
        self.post = make_postprocessor(layout, DecodeConfig(conf_thr=conf_thr, max_det=min(max_det, layout.max_det)), post)
        self.name = f"rknn:{core}:{path.name}"

    def infer(self, frame: Frame):
        _check_input(frame, self.size)
        outs = self.rknn.inference(inputs=[frame.array[None]], data_format="nhwc")
        n = len(self.layout.strides)
        return self.post.decode([np.asarray(o, dtype=np.float32) for o in outs[:n]],
                                np.asarray(outs[n], dtype=np.float32) if len(outs) > n else None)

    def close(self) -> None:
        self.rknn.release()

    def info(self) -> dict[str, Any]:
        return {"backend": "rknnlite", "model": self.name, "post": self.post.name}


def predictions_to_array(preds, layout: Optional[HeadLayout] = None, n_kpt: int = 12):
    """Closed-loop `Prediction`s (pixels of the frame) -> KeypointArray detections."""
    dets = []
    for p in preds:
        if p.keypoints is None:
            continue
        kp = np.asarray(p.keypoints, dtype=np.float64)
        if kp.shape[0] != n_kpt:
            raise ValueError(f"the model predicts {kp.shape[0]} keypoints, the KeypointArray carries {n_kpt}")
        x, y, w, h = p.bbox
        vis = np.asarray(p.visibility, dtype=np.float64) if p.visibility is not None else (kp[:, 2] > 0.5).astype(np.float64)
        dets.append({"cls": int(p.class_id or 0), "score": float(p.score), "box": np.array([x, y, x + w, y + h]),
                     "kxy": kp[:, :2], "kscore": kp[:, 2], "kvis": vis})
    kw = dict(keypoints=layout.keypoints, classes=layout.classes) if layout is not None else {}
    return structs.pack(dets, 0, 0, 0, n_kpt=n_kpt, **kw)


class EvaluatorBackend(ModelBackend):
    """Any closed-loop `IModelEvaluator` as a runtime backend. `hints_key` names the frame.meta entry holding the ground truth for
    evaluators that need it (SimulatedEvaluator is a test double that peeks at the labels)."""

    def __init__(self, evaluator: IModelEvaluator, layout: Optional[HeadLayout] = None, hints_key: str = "hints",
                 n_kpt: int = 12) -> None:
        self.ev, self.layout, self.hints_key, self.n_kpt = evaluator, layout, hints_key, n_kpt
        self.name = f"evaluator:{getattr(evaluator, 'name', type(evaluator).__name__)}"
        self._lock = threading.Lock()            # evaluators are not required to be thread-safe: serialize per backend instance

    def infer(self, frame: Frame):
        hints = None
        if getattr(self.ev, "wants_hints", False):
            h = frame.meta.get(self.hints_key)
            if h is None:
                raise ValueError(f"{self.name} needs ground-truth hints in frame.meta[{self.hints_key!r}]")
            hints = [h]
        with self._lock:
            preds = self.ev.predict([frame.array], hints)[0]
        return predictions_to_array(preds, self.layout, self.n_kpt)

    def warmup(self, n: int = 3) -> None:
        self.ev.warmup()                         # not a dummy frame: hint-driven doubles cannot score one

    def close(self) -> None:
        self.ev.close()

    def info(self) -> dict[str, Any]:
        return {"backend": "evaluator", "evaluator": self.name}


class ScriptedBackend(ModelBackend):
    """Deterministic fake model. `latency_ms` is slept (like waiting on an NPU, the GIL is released); `fail_every=N` raises on every
    N-th call; `fn(frame) -> list[det dict]` overrides the default single moving detection."""

    def __init__(self, latency_ms: float = 0.0, jitter_ms: float = 0.0, fail_every: int = 0, fn: Optional[Callable] = None,
                 seed: int = 0, stall_at: Optional[tuple[int, float]] = None) -> None:
        self.latency, self.jitter, self.fail_every, self.fn = latency_ms / 1000.0, jitter_ms / 1000.0, fail_every, fn
        self.rng = random.Random(seed)
        self.calls = 0
        self.stall_at = stall_at                 # (call number, seconds): one long stall (an NPU hang), for failure tests
        self.name = f"scripted:{latency_ms}ms"
        self._lock = threading.Lock()
        self.layout = HeadLayout(keypoints=tuple(f"k{i}" for i in range(12)), classes=("player_ct", "player_t"))

    def infer(self, frame: Frame):
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.fail_every and n % self.fail_every == 0:
            raise RuntimeError(f"scripted failure on call {n}")
        d = self.latency + (self.rng.uniform(0, self.jitter) if self.jitter else 0.0)
        if self.stall_at and n == self.stall_at[0]:
            d += self.stall_at[1]
        if d > 0:
            time.sleep(d)
        dets = self.fn(frame) if self.fn else [self._default(frame)]
        return structs.pack(dets, 0, 0, 0, n_kpt=12)

    @staticmethod
    def _default(frame: Frame) -> dict:
        x = 100.0 + (frame.frame_id % 200)
        k = np.stack([np.full(12, x) + np.arange(12), np.full(12, 200.0) + 3 * np.arange(12)], axis=1)
        return {"cls": frame.frame_id % 2, "score": 0.9, "box": np.array([x - 30, 150.0, x + 30, 350.0]), "kxy": k,
                "kscore": np.full(12, 0.8), "kvis": np.ones(12)}


class FallbackBackend(ModelBackend):
    """Use `primary`; after `max_errors` consecutive failures switch to `secondary` (flagging `degraded`) and probe the primary
    again every `retry_s` seconds."""

    def __init__(self, primary: ModelBackend, secondary: ModelBackend, max_errors: int = 3, retry_s: float = 30.0) -> None:
        self.primary, self.secondary, self.max_errors, self.retry_s = primary, secondary, max_errors, retry_s
        self.layout = primary.layout
        self.name = f"fallback:{primary.name}->{secondary.name}"
        self._errors, self._since, self.degraded, self.switches = 0, 0.0, False, 0
        self._lock = threading.Lock()

    def infer(self, frame: Frame):
        with self._lock:
            use_primary = not self.degraded or (time.monotonic() - self._since) > self.retry_s
        if use_primary:
            try:
                out = self.primary.infer(frame)
                with self._lock:
                    if self.degraded:
                        log.warning("primary backend %s recovered", self.primary.name)
                    self._errors, self.degraded = 0, False
                return out
            except Exception as e:                                       # any backend failure counts
                with self._lock:
                    self._errors += 1
                    self._since = time.monotonic()
                    if self._errors >= self.max_errors and not self.degraded:
                        self.degraded = True
                        self.switches += 1
                        log.error("primary backend %s failed %d times (%s): switching to %s", self.primary.name, self._errors, e,
                                  self.secondary.name)
                if not self.degraded:
                    raise
        return self.secondary.infer(frame)

    def warmup(self, n: int = 3) -> None:
        self.primary.warmup(n)
        self.secondary.warmup(1)

    def close(self) -> None:
        self.primary.close()
        self.secondary.close()

    def info(self) -> dict[str, Any]:
        return {"backend": "fallback", "primary": self.primary.info(), "secondary": self.secondary.info(),
                "degraded": self.degraded, "switches": self.switches}


def write_layout_sidecar(layout: HeadLayout, path: str | Path) -> Path:
    """`<model>.layout.json`: what RknnLiteBackend needs because a .rknn file has nowhere to keep it."""
    p = Path(path)
    p.write_text(json.dumps(json.loads(layout.to_json()), indent=2))
    return p
