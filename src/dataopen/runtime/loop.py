"""The inference runtime: the main loop.

    FrameSource --reader thread--> BufferPolicy --worker thread(s)--> ModelBackend --> header fill --> ResultPublisher
         (DMA ring / queue)         (latest|queue)        (infer + C post-process)      (UDS stream + shm mailbox)

Non-blocking for everyone else: the reader only moves a reference into the policy (O(1), never waits on the model), a slow model costs
dropped frames (counted, with the reason), never a stalled producer or publisher. Every frame is released back to its producer exactly
once, whatever happens to it (computed, superseded, stale, error, shutdown).

Pseudocode of a worker (the real one is `_work`):

    while running:
        frame = policy.get()                      # blocks only this worker
        if frame.age > max_age: drop(stale); continue
        try:    arr = backend.infer(frame)        # model + post-processing
        except: arr = empty(flag EMPTY_ERROR); metrics.error()      # consumers must never keep acting on an old answer
        if a newer frame was already published: drop(late); continue  # several workers finish out of order
        fill header (frame_id, capture ts, brightness, flags); publish; metrics
        frame.release()
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np

from ..detector import structs
from .backends import ModelBackend
from .channel import CallbackPublisher, ResultPublisher
from .frames import Frame, FrameSource, now_us
from .metrics import MetricsLogger, RuntimeMetrics
from .policy import BufferPolicy, make_policy

log = logging.getLogger("dataopen.runtime")

FLAG_DEGRADED = 1          # a fallback backend produced this result
FLAG_STALE = 2             # the frame was older than `stale_flag_ms` when published
FLAG_DROPPED_BEFORE = 4    # frames were dropped between the previous published result and this one
FLAG_EMPTY_ERROR = 8       # inference failed: no detections, do not trust "nothing there"


class InferenceRuntime:
    def __init__(self, source: FrameSource, backend: ModelBackend | Sequence[ModelBackend],
                 publisher: Optional[ResultPublisher] = None, policy: str | BufferPolicy = "latest",
                 max_age_ms: Optional[float] = None, stale_flag_ms: Optional[float] = None, hang_ms: float = 500.0,
                 metrics: Optional[RuntimeMetrics] = None, log_every_s: float = 0.0, metrics_jsonl: Optional[str] = None,
                 on_result: Optional[Callable[[object, Frame], None]] = None, publish_errors: bool = True,
                 warmup: int = 3, n_kpt: int = 12) -> None:
        self.source = source
        self.backends = [backend] if isinstance(backend, ModelBackend) else list(backend)
        if not self.backends:
            raise ValueError("at least one backend")
        self.publisher = publisher or CallbackPublisher(lambda b: None)
        self.policy = make_policy(policy) if isinstance(policy, str) else policy
        self.max_age_ms, self.stale_flag_ms, self.hang_ms = max_age_ms, stale_flag_ms, hang_ms
        cap = getattr(self.policy, "capacity", 1)
        self.metrics = metrics or RuntimeMetrics(queue_capacity=cap)
        self.on_result, self.publish_errors, self.warmup, self.n_kpt = on_result, publish_errors, warmup, n_kpt
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._seq = 0                                    # dequeue order
        self._published_seq = -1
        self._dropped_since_deq = 0
        self._inflight: dict[int, float] = {}
        self._logger = MetricsLogger(self.metrics, log_every_s, metrics_jsonl) if log_every_s > 0 else None
        self._started = False
        self._source_done = threading.Event()

    # ---- lifecycle ----
    def start(self) -> "InferenceRuntime":
        if self._started:
            raise RuntimeError("already started")
        self._started = True
        for b in self.backends:
            if self.warmup:
                b.warmup(self.warmup)
        self._threads = [threading.Thread(target=self._read, name="rt-reader", daemon=True)]
        self._threads += [threading.Thread(target=self._work, args=(i, b), name=f"rt-worker{i}", daemon=True)
                          for i, b in enumerate(self.backends)]
        for t in self._threads:
            t.start()
        if self._logger:
            self._logger.start()
        return self

    def stop(self, timeout: float = 5.0, close: bool = True) -> None:
        self._stop.set()
        for f, _ in self.policy.close():
            self._drop(f, "shutdown")
        for t in self._threads:
            t.join(timeout)
        if self._logger:
            self._logger.stop()
        if close:
            self.source.close()
            for b in self.backends:
                b.close()
            self.publisher.close()

    def __enter__(self) -> "InferenceRuntime":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._started and not self._stop.is_set()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Until the source is exhausted (or idle) and nothing is queued or in flight. For tests and finite sources."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.policy.depth() == 0 and not self._inflight and (self._source_done.is_set()):
                return True
            time.sleep(0.002)
        return False

    def health(self) -> dict:
        now = time.monotonic()
        with self._lock:
            stalled = [i for i, t in self._inflight.items() if (now - t) * 1000 > self.hang_ms]
        m = self.metrics
        return {"ok": not stalled and m.consecutive_errors < 5, "stalled_workers": stalled,
                "consecutive_errors": m.consecutive_errors, "degraded": m.degraded}

    # ---- internals ----
    def _drop(self, frame: Frame, reason: str) -> None:
        self.metrics.drop(reason)
        with self._lock:
            self._dropped_since_deq += 1
        frame.release()

    def _read(self) -> None:
        pol, m = self.policy, self.metrics
        while not self._stop.is_set():
            try:
                f = self.source.get(0.05)
            except Exception as e:                       # a broken source must not kill the loop silently
                m.error(f"source: {e}")
                log.exception("frame source failed")
                time.sleep(0.05)
                continue
            if f is None:
                if getattr(self.source, "exhausted", False):
                    self._source_done.set()
                continue
            f.arrived_us = now_us()
            m.frame_in()
            for dropped, reason in pol.put(f):
                self._drop(dropped, reason)
            m.set_depth(pol.depth())
        self._source_done.set()

    def _source_exhausted(self) -> bool:
        try:
            return bool(self.source.stats().get("exhausted"))
        except Exception:
            return False

    def _brightness(self, a: np.ndarray) -> int:
        return int(a[::32, ::32].mean())                 # 400 samples: ~microseconds, no copy

    def _work(self, wid: int, backend: ModelBackend) -> None:
        pol, m = self.policy, self.metrics
        while not self._stop.is_set():
            f = pol.get(0.05)
            if f is None:
                continue
            with self._lock:                             # visible as "in flight" from the moment it leaves the policy
                self._seq += 1
                seq = self._seq
                dropped, self._dropped_since_deq = self._dropped_since_deq, 0   # skipped between the previous frame and this one
                self._inflight[wid] = time.monotonic()
            m.set_depth(pol.depth())
            t0 = now_us()
            m.lat["frame_age_ms"].add((t0 - f.ts_us) / 1000.0)
            m.lat["queue_wait_ms"].add((t0 - f.arrived_us) / 1000.0)
            if self.max_age_ms is not None and (t0 - f.ts_us) / 1000.0 > self.max_age_ms:
                self._drop(f, "stale")
                self._done(wid)
                continue
            try:
                self._process(backend, f, seq, dropped, t0)
            except Exception:                            # a bug in the loop itself must not kill the worker
                log.exception("worker failed on frame %d", f.frame_id)
                f.release()
            finally:
                self._done(wid)

    def _done(self, wid: int) -> None:
        with self._lock:
            self._inflight.pop(wid, None)

    def _process(self, backend: ModelBackend, f: Frame, seq: int, dropped: int, t0: int) -> None:
        m = self.metrics
        flags, arr = 0, None
        try:
            arr = backend.infer(f)
            m.ok()
        except Exception as e:                       # any backend failure: report, publish an explicit empty-error result
            m.error(f"{type(e).__name__}: {e}")
            log.error("inference failed on frame %d: %s", f.frame_id, e)
            flags |= FLAG_EMPTY_ERROR
            arr = structs.pack([], 0, 0, 0, n_kpt=self.n_kpt)
        t1 = now_us()
        m.lat["infer_ms"].add((t1 - t0) / 1000.0)
        with self._lock:
            late = seq < self._published_seq
            if not late:
                self._published_seq = seq
        if late:
            m.drop("late")
            f.release()
            return
        if (flags & FLAG_EMPTY_ERROR) and not self.publish_errors:
            m.drop("error")
            f.release()
            return
        m.degraded = bool(backend.degraded) or m.consecutive_errors >= 5
        if backend.degraded:
            flags |= FLAG_DEGRADED
        if dropped:
            flags |= FLAG_DROPPED_BEFORE
        if self.stale_flag_ms is not None and (t1 - f.ts_us) / 1000.0 > self.stale_flag_ms:
            flags |= FLAG_STALE
        lb = f.meta.get("letterbox")
        arr.frame_id, arr.timestamp_us, arr.flags = f.frame_id & 0xFFFFFFFF, f.ts_us & 0xFFFFFFFF, flags | arr.flags
        arr.avg_scene_brightness = self._brightness(f.array)
        if lb is not None:
            arr.letterbox_scale, arr.pad_x, arr.pad_y = float(lb.scale), int(lb.pad_x), int(lb.pad_y)
            arr.src_w, arr.src_h = int(lb.orig_w), int(lb.orig_h)
        else:
            arr.letterbox_scale, arr.src_w, arr.src_h = 1.0, f.array.shape[1], f.array.shape[0]
        try:
            self.publisher.publish(bytes(arr))
        except Exception as e:
            m.error(f"publish: {e}")
            log.error("publish failed: %s", e)
        t2 = now_us()
        m.lat["publish_ms"].add((t2 - t1) / 1000.0)
        m.lat["e2e_ms"].add((t2 - f.ts_us) / 1000.0)
        m.published_one()
        if self.on_result is not None:
            try:
                self.on_result(arr, f)
            except Exception:
                log.exception("on_result callback failed")
        f.release()
