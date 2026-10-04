"""Runtime metrics: the minimum a deployed detector must be able to answer.

  latency   end-to-end (capture timestamp -> result published), frame age when inference starts, queue wait, inference
            (backend incl. post-processing), publish: p50 / p95 / p99 / max over a sliding window
  FPS       frames in, frames published (windowed)
  queue     current depth, maximum since start, capacity
  drops     by reason: superseded (latest-only), queue_full_oldest/newest, stale, late (out-of-order result), producer (the source's
            own drops: the ring was full), shutdown
  health    errors, consecutive errors, degraded (fallback backend), last error
Everything is thread-safe, cheap (no allocation on the hot path beyond a deque append) and snapshot-able as a dict, a log line, or
Prometheus text.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("dataopen.runtime")


class LatencyWindow:
    def __init__(self, size: int = 4096) -> None:
        self._v: deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()
        self.total = 0

    def add(self, ms: float) -> None:
        with self._lock:
            self._v.append(ms)
            self.total += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            a = np.fromiter(self._v, dtype=np.float64, count=len(self._v))
            total = self.total
        if not len(a):
            return {"count": total, "n": 0}
        p = np.percentile(a, [50, 95, 99])
        return {"count": total, "n": int(len(a)), "mean": round(float(a.mean()), 3), "p50": round(float(p[0]), 3),
                "p95": round(float(p[1]), 3), "p99": round(float(p[2]), 3), "max": round(float(a.max()), 3)}


class Rate:
    """Events per second over a sliding window."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window, self._t, self._lock, self.total = window_s, deque(), threading.Lock(), 0

    def mark(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            self._t.append(now)
            self.total += 1
            self._trim(now)

    def _trim(self, now: float) -> None:
        while self._t and now - self._t[0] > self.window:
            self._t.popleft()

    def per_second(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        with self._lock:
            self._trim(now)
            if len(self._t) < 2:
                return 0.0
            span = max(now - self._t[0], 1e-6) if now - self._t[-1] < self.window else max(self._t[-1] - self._t[0], 1e-6)
            return (len(self._t) - 1) / span if span > 0 else 0.0


DROP_REASONS = ("superseded", "queue_full_oldest", "queue_full_newest", "stale", "late", "shutdown", "error")


class RuntimeMetrics:
    def __init__(self, queue_capacity: int = 1, window: int = 4096) -> None:
        self.started = time.monotonic()
        self.queue_capacity = queue_capacity
        self.lat = {k: LatencyWindow(window) for k in ("e2e_ms", "frame_age_ms", "queue_wait_ms", "infer_ms", "publish_ms")}
        self.fps_in, self.fps_out = Rate(), Rate()
        self._lock = threading.Lock()
        self.drops = {k: 0 for k in DROP_REASONS}
        self.producer_drops = 0
        self.errors = self.consecutive_errors = self.published = self.processed = 0
        self.depth = self.depth_max = 0
        self.degraded = False
        self.last_error = ""
        self.extra: dict[str, Any] = {}

    # hot-path hooks
    def frame_in(self) -> None:
        self.fps_in.mark()

    def drop(self, reason: str, n: int = 1) -> None:
        with self._lock:
            self.drops[reason] = self.drops.get(reason, 0) + n

    def set_depth(self, d: int) -> None:
        self.depth = d
        if d > self.depth_max:
            self.depth_max = d

    def ok(self) -> None:
        with self._lock:
            self.consecutive_errors = 0

    def error(self, msg: str) -> None:
        with self._lock:
            self.errors += 1
            self.consecutive_errors += 1
            self.last_error = msg[:300]

    def published_one(self) -> None:
        self.fps_out.mark()
        with self._lock:
            self.published += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            drops = dict(self.drops)
            drops["producer"] = self.producer_drops
        up = time.monotonic() - self.started
        n_in = self.fps_in.total
        return {"uptime_s": round(up, 2), "frames_in": n_in, "published": self.published,
                "fps_in": round(self.fps_in.per_second(), 1), "fps_out": round(self.fps_out.per_second(), 1),
                "queue": {"depth": self.depth, "max": self.depth_max, "capacity": self.queue_capacity},
                "latency_ms": {k: w.stats() for k, w in self.lat.items()}, "dropped": drops,
                "dropped_total": sum(drops.values()),
                "drop_rate": round(sum(drops.values()) / max(1, n_in + self.producer_drops), 4),
                "errors": {"total": self.errors, "consecutive": self.consecutive_errors, "last": self.last_error},
                "degraded": self.degraded, **self.extra}

    def line(self) -> str:
        s = self.snapshot()
        e2e, inf = s["latency_ms"]["e2e_ms"], s["latency_ms"]["infer_ms"]
        return (f"fps in/out {s['fps_in']}/{s['fps_out']}  e2e p50/p99 {e2e.get('p50', '-')}/{e2e.get('p99', '-')} ms  "
                f"infer p50/p99 {inf.get('p50', '-')}/{inf.get('p99', '-')} ms  queue {s['queue']['depth']}/{s['queue']['max']}"
                f"  dropped {s['dropped_total']} ({s['drop_rate']:.1%})  errors {s['errors']['total']}"
                + ("  DEGRADED" if s["degraded"] else ""))

    def prometheus(self, prefix: str = "apollo_runtime") -> str:
        s = self.snapshot()
        out = [f"{prefix}_frames_in_total {s['frames_in']}", f"{prefix}_published_total {s['published']}",
               f"{prefix}_fps_in {s['fps_in']}", f"{prefix}_fps_out {s['fps_out']}", f"{prefix}_queue_depth {s['queue']['depth']}",
               f"{prefix}_queue_depth_max {s['queue']['max']}", f"{prefix}_errors_total {s['errors']['total']}",
               f"{prefix}_degraded {int(s['degraded'])}"]
        for reason, n in s["dropped"].items():
            out.append(f'{prefix}_dropped_total{{reason="{reason}"}} {n}')
        for name, st in s["latency_ms"].items():
            for q in ("p50", "p95", "p99", "max"):
                if q in st:
                    out.append(f'{prefix}_latency_ms{{stage="{name[:-3]}",quantile="{q}"}} {st[q]}')
        return "\n".join(out) + "\n"


class MetricsLogger(threading.Thread):
    """Logs a one-line summary every `every_s` seconds and optionally appends the full snapshot to a JSONL file."""

    def __init__(self, metrics: RuntimeMetrics, every_s: float = 5.0, jsonl: Optional[str | Path] = None) -> None:
        super().__init__(name="runtime-metrics", daemon=True)
        self.m, self.every, self.path, self._halt = metrics, every_s, Path(jsonl) if jsonl else None, threading.Event()

    def run(self) -> None:
        while not self._halt.wait(self.every):
            log.info("%s", self.m.line())
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": time.time(), **self.m.snapshot()}) + "\n")

    def stop(self) -> None:
        self._halt.set()
