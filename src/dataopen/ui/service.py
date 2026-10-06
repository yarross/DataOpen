"""The assistive UI detector as a service next to (and independent of) the pose runtime.

    FrameSource (the video path's ring or queue) -> detector -> UiSceneBuilder (tracks, pointer, objects) -> snapshot callback

Policy: latest frame only (an old picture is worthless for a pointer that moves at 1 kHz), the buffer is always released, and a failing
detector produces no snapshot (so no scene, so the bridge gives no help) rather than a wrong one. Nothing here knows about the pose runtime,
its result struct or its models."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..runtime.frames import Frame, FrameSource
from .infer import decode
from .taxonomy import STRIDES
from .scene import Det, Snapshot, UiSceneBuilder
from .taxonomy import NAMES, require_ui_layout  # noqa: F401


class OrtUiDetector:
    """The exported ONNX model. Refuses a file whose class list is not the UI taxonomy or that declares keypoints."""

    def __init__(self, path: str | Path, conf: float = 0.35, threads: int = 2) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        if "ui_layout" not in meta:
            raise ValueError(f"{path}: not a UI-element model (no ui_layout metadata)")
        self.layout = json.loads(meta["ui_layout"])
        require_ui_layout(self.layout["classes"], int(self.layout.get("n_keypoints", 0)))
        self.classes = tuple(self.layout["classes"])
        self.conf, self.in_name = conf, self.sess.get_inputs()[0].name
        self.u8 = "uint8" in meta.get("input_convention", "uint8")
        self.size = int(self.layout["input_size"])

    def detect(self, frame: Frame) -> list[Det]:
        a = frame.array
        if a.shape[:2] != (self.size, self.size) or a.dtype != np.uint8:
            raise ValueError(f"frame must be {self.size}x{self.size} uint8 RGB, got {a.shape} {a.dtype}")
        x = a[None] if self.u8 else np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)
        outs = self.sess.run(None, {self.in_name: x})
        dets = decode([o[0] for o in outs], len(self.classes), conf=self.conf, strides=STRIDES)
        return [Det(self.classes[d.cls], d.score, d.box) for d in dets]


class TorchUiDetector:
    def __init__(self, net, conf: float = 0.35) -> None:
        import torch

        require_ui_layout(net.cfg.classes)
        self.net, self.conf, self.torch, self.classes = net.eval(), conf, torch, tuple(net.cfg.classes)

    def detect(self, frame: Frame) -> list[Det]:
        t = self.torch
        with t.no_grad():
            outs = self.net(t.from_numpy(frame.array).permute(2, 0, 1)[None].float().div(255))
        dets = decode([o[0].numpy() for o in outs], len(self.classes), conf=self.conf)
        return [Det(self.classes[d.cls], d.score, d.box) for d in dets]


class UiService:
    def __init__(self, source: FrameSource, detector, builder: UiSceneBuilder, on_snapshot: Callable[[Snapshot, Frame], None]) -> None:
        self.source, self.detector, self.builder, self.on_snapshot = source, detector, builder, on_snapshot
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames = self.errors = 0
        self.latency_ms: list[float] = []
        self.last_error = ""

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ui-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)

    def step(self, timeout: float = 0.1) -> bool:
        f = self.source.get(timeout)
        if f is None:
            return False
        try:
            g = f.meta.get("geometry")
            if g is not None and g != self.builder.g:
                self.builder.set_geometry(g)  # a mode change: tracks from the old geometry mean nothing now
            t0 = time.perf_counter()
            dets = self.detector.detect(f)
            snap = self.builder.update(f.ts_us, dets)
            self.latency_ms.append((time.perf_counter() - t0) * 1000)
            self.frames += 1
            self.on_snapshot(snap, f)
        except Exception as e:  # no snapshot = no scene = no help; never a wrong one
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"
        finally:
            f.release()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            self.step(0.1)
