"""Frames and frame sources.

A `Frame` is a 640x640 RGB uint8 view over memory somebody else owns (a shared-memory slot, a DMA-BUF mapping, a numpy array).
It must be `release()`d when the runtime is done with it, which hands the buffer back to the producer. Sources never block the
producer: if the runtime is slow the producer's frames are dropped (and counted) rather than the capture path stalling.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


def now_us() -> int:
    """Monotonic microseconds (CLOCK_MONOTONIC: comparable across processes on one machine)."""
    return time.monotonic_ns() // 1000


@dataclass(eq=False)
class Frame:
    array: np.ndarray                    # (H, W, 3) uint8, RGB; may be a read-only view of shared memory
    frame_id: int
    ts_us: int                           # capture time, monotonic microseconds
    meta: dict[str, Any] = field(default_factory=dict)
    _release: Optional[Callable[[], None]] = None
    arrived_us: int = 0                  # set by the runtime when the frame reaches it

    def release(self) -> None:
        """Hand the buffer back to the producer. Idempotent. After this, `array` must not be touched."""
        rel, self._release = self._release, None
        if rel is not None:
            rel()

    def age_ms(self, now: Optional[int] = None) -> float:
        return ((now if now is not None else now_us()) - self.ts_us) / 1000.0


class FrameSource(ABC):
    """Where frames come from. `get` returns the next frame or None after `timeout` seconds (and None forever after `close`)."""

    @abstractmethod
    def get(self, timeout: float = 0.1) -> Optional[Frame]: ...

    def close(self) -> None:
        pass

    @property
    def exhausted(self) -> bool:
        """True once a finite source will never produce another frame (live sources stay False)."""
        return False

    def stats(self) -> dict[str, Any]:
        return {}


class QueueSource(FrameSource):
    """In-process source: `push()` never blocks and returns False (a producer-side drop) when `capacity` frames are waiting."""

    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self._q: deque[Frame] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self.pushed = self.dropped = 0

    def push(self, array: np.ndarray, frame_id: int, ts_us: Optional[int] = None, meta: Optional[dict] = None,
             on_release: Optional[Callable[[], None]] = None) -> bool:
        with self._cv:
            if self._closed or len(self._q) >= self.capacity:
                self.dropped += 1
                return False
            self._q.append(Frame(array, frame_id, ts_us if ts_us is not None else now_us(), meta or {}, on_release))
            self.pushed += 1
            self._cv.notify()
            return True

    @property
    def exhausted(self) -> bool:
        return self._closed and not self._q

    def get(self, timeout: float = 0.1) -> Optional[Frame]:
        with self._cv:
            if not self._q and not self._closed:
                self._cv.wait(timeout)
            return self._q.popleft() if self._q else None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            leftovers, self._q = list(self._q), deque()
            self._cv.notify_all()
        for f in leftovers:
            f.release()

    def stats(self) -> dict[str, Any]:
        with self._cv:
            return {"source": "queue", "pushed": self.pushed, "producer_dropped": self.dropped, "waiting": len(self._q)}


class DirSource(FrameSource):
    """Replays image files (sorted) at `fps` (0 = as fast as the runtime asks). For bench and offline checks."""

    def __init__(self, directory: str | Path, fps: float = 0.0, loop: bool = False, size: int = 640) -> None:
        from ..detector.data import letterbox, load_image
        self._lb, self._load = letterbox, load_image
        self.files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not self.files:
            raise ValueError(f"no images in {directory}")
        self.fps, self.loop, self.size, self._i, self._next = fps, loop, size, 0, 0.0

    @property
    def exhausted(self) -> bool:
        return not self.loop and self._i >= len(self.files)

    def get(self, timeout: float = 0.1) -> Optional[Frame]:
        if self._i >= len(self.files):
            if not self.loop:
                return None
            self._i = 0
        if self.fps > 0:
            wait = self._next - time.monotonic()
            if wait > timeout:
                time.sleep(timeout)
                return None
            if wait > 0:
                time.sleep(wait)
            self._next = max(self._next, time.monotonic()) + 1.0 / self.fps
        img, _ = self._lb(self._load(self.files[self._i]), self.size)
        fid = self._i
        self._i += 1
        return Frame(img, fid, now_us(), {"path": str(self.files[fid % len(self.files)])})


class SyntheticSource(FrameSource):
    """Generated frames at a fixed rate (a camera stand-in). `hints_fn(frame_id) -> annotations` feeds the closed loop's
    SimulatedEvaluator, which needs the ground truth."""

    def __init__(self, fps: float = 60.0, size: int = 640, hints_fn: Optional[Callable[[int], Any]] = None,
                 frames: Optional[int] = None, seed: int = 0) -> None:
        self.fps, self.size, self.hints_fn, self.limit = fps, size, hints_fn, frames
        self._i = 0
        self._next = time.monotonic()
        rng = np.random.default_rng(seed)
        self._base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self._i >= self.limit

    def get(self, timeout: float = 0.1) -> Optional[Frame]:
        if self.limit is not None and self._i >= self.limit:
            return None
        wait = self._next - time.monotonic()
        if wait > timeout:
            time.sleep(timeout)
            return None
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next + 1.0 / self.fps, time.monotonic() - 1.0 / self.fps)
        fid = self._i
        self._i += 1
        arr = np.roll(self._base, fid % self.size, axis=1)
        meta = {"hints": self.hints_fn(fid)} if self.hints_fn else {}
        return Frame(arr, fid, now_us(), meta)
