"""The assistive UI detector as a service next to (and independent of) the pose runtime.

    FrameSource (the video path's ring or queue) -> detector -> UiSceneBuilder (tracks, pointer, objects) -> snapshot callback

Policy: latest frame only (an old picture is worthless for a pointer that moves at 1 kHz), the buffer is always released, and a failing
detector produces no snapshot (so no scene, so the bridge gives no help) rather than a wrong one. Nothing here knows about the pose runtime,
its result struct or its models."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..runtime.frames import Frame, FrameSource, now_us
from .health import SceneHealth
from .infer import decode
from .policy import V1, ScenePolicy
from .scene import BridgeScenePublisher, Det, Snapshot, UiSceneBuilder
from .taxonomy import STRIDES
from .taxonomy import NAMES, require_ui_layout  # noqa: F401


class OrtUiDetector:
    """The exported ONNX model. Refuses a file whose class list is not the UI taxonomy or that declares keypoints. `path` may also be the
    model's BYTES: a model that lives on the device in a slot is built from memory, never written back in the clear (ui/resident.py)."""

    def __init__(self, path: str | Path | bytes, conf: float = 0.35, threads: int = 2) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path if isinstance(path, bytes) else str(path), so,
                                        providers=["CPUExecutionProvider"])
        path = "<memory>" if isinstance(path, bytes) else path
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


def _pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(q / 100.0 * len(s)))] if s else 0.0


@dataclass
class LatencyTrace:
    """Where the time went, per frame that reached the detector (docs/LATENCY.md). `age_ms`: how old the picture was WHEN THE DETECTOR
    TOOK IT (`now - ts_us`: capture time of the middle of the ROI, so it already contains the queue wait); `ready_ms`: how long the frame
    had been ready (`now - meta['ready_us']`, when the video path gives it); `detect_ms` and `update_ms`: inference + decode, and
    the tracker."""

    age_ms: list[float] = field(default_factory=list)
    ready_ms: list[float] = field(default_factory=list)
    detect_ms: list[float] = field(default_factory=list)
    update_ms: list[float] = field(default_factory=list)
    stale_dropped: int = 0

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k in ("age_ms", "ready_ms", "detect_ms", "update_ms"):
            v = getattr(self, k)
            out[k] = {"n": len(v), "p50": _pct(v, 50), "p95": _pct(v, 95), "p99": _pct(v, 99), "max": max(v) if v else 0.0}
        return out


class UiService:
    """The defaults are the safe ones for v1 (`ui.policy.V1`, docs/LATENCY.md):

    * `latest_only=True`: when the source holds several frames, take the newest and release the older ones unseen. A FIFO of 4 behind an
      11 ms detector makes the picture ~44 ms old and pushes the worst case past the bridge's 100 ms TTL.
    * `max_age_ms=35`: a frame older than that when it is taken is dropped unseen; the pipeline is behind and catches up instead of
      working on the past.

    Pass `latest_only=False` / `max_age_ms=None` for the original behaviour (tests of the old pipeline do). `clock` (microseconds) and
    `perf` (seconds) make the trace testable in virtual time. `health`: the scene path's state machine (ui/health.py) is told about dropped
    frames and inference times."""

    def __init__(self, source: FrameSource, detector, builder: UiSceneBuilder, on_snapshot: Callable[[Snapshot, Frame], None],
                 max_age_ms: Optional[float] = V1.max_age_ms, latest_only: bool = V1.latest_only, clock: Callable[[], int] = now_us,
                 health: Optional[SceneHealth] = None, perf: Callable[[], float] = time.perf_counter) -> None:
        self.source, self.detector, self.builder, self.on_snapshot = source, detector, builder, on_snapshot
        self.max_age_ms, self.latest_only, self.clock, self.health, self.perf = max_age_ms, latest_only, clock, health, perf
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames = self.errors = 0
        self.latency_ms: list[float] = []
        self.trace = LatencyTrace()
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
        while self.latest_only:  # whatever else is waiting is newer: the older picture is released unseen (worthless for a moving pointer)
            g = self.source.get(0.0)
            if g is None:
                break
            f.release()
            f = g
        now = self.clock()
        age_ms = (now - f.ts_us) / 1000.0
        if self.max_age_ms is not None and age_ms > self.max_age_ms:
            self.trace.stale_dropped += 1
            if self.health is not None:
                self.health.stale_frame(now / 1000.0)
            f.release()
            return True
        try:
            self.trace.age_ms.append(age_ms)
            if "ready_us" in f.meta:
                self.trace.ready_ms.append((now - f.meta["ready_us"]) / 1000.0)
            g = f.meta.get("geometry")
            if g is not None and g != self.builder.g:
                self.builder.set_geometry(g)  # a mode change: tracks from the old geometry mean nothing now
            t0 = self.perf()
            dets = self.detector.detect(f)
            t1 = self.perf()
            snap = self.builder.update(f.ts_us, dets)
            t2 = self.perf()
            self.latency_ms.append((t2 - t0) * 1000)
            self.trace.detect_ms.append((t1 - t0) * 1000)
            self.trace.update_ms.append((t2 - t1) * 1000)
            if self.health is not None:
                self.health.detect((t1 - t0) * 1000)
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


def scene_service(source: FrameSource, detector, builder: UiSceneBuilder, send, clock=None, now: Callable[[], int] = now_us,
                  policy: ScenePolicy = V1) -> tuple[UiService, BridgeScenePublisher]:
    """The scene path as v1 ships it: the service and the publisher share ONE `SceneHealth`, so a pipeline that is behind (frames dropped
    for their age, a slow detector, scenes ready too late) switches the scene help off in one place and back on with hysteresis.
    `send` writes one BridgeLink frame; `clock` is the module/bridge `ClockSync` (or None); `now` is the module's clock in microseconds."""
    health = SceneHealth(policy)
    svc = UiService(source, detector, builder, lambda snap, frame: pub(snap, frame), max_age_ms=policy.max_age_ms,
                    latest_only=policy.latest_only, clock=now, health=health)
    pub = BridgeScenePublisher(builder, send, clock, health=health, now_us=now)
    return svc, pub
