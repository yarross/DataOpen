"""What to do when a new frame arrives while the previous one is still being computed.

  LatestOnly      keep ONE pending frame; a newer arrival replaces (drops) the older. Lowest latency, bounded staleness: the right
                  default for anything that acts on the result (aiming), where an old answer is worth less than none.
  BoundedQueue    FIFO of `capacity` frames; when full, drop the OLDEST (default: still prefers fresh) or the NEWEST (keep the
                  backlog intact, e.g. for offline analysis where every frame matters but memory must stay bounded).
All policies are non-blocking for the producer and report every drop with a reason. Unbounded queues do not exist: they turn a slow
model into growing latency and finally an out-of-memory kill.
A frame older than `max_age_ms` when it is dequeued is dropped as stale: computing an answer nobody can use only delays the next one.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from .frames import Frame

Dropped = list[tuple[Frame, str]]


class BufferPolicy(ABC):
    name = "policy"

    @abstractmethod
    def put(self, frame: Frame) -> Dropped:
        """Offer a frame; returns the frames this displaced (with the reason). Never blocks."""

    @abstractmethod
    def get(self, timeout: float) -> Optional[Frame]: ...

    @abstractmethod
    def depth(self) -> int: ...

    @abstractmethod
    def close(self) -> Dropped:
        """Stop; returns what was still waiting (the runtime releases it)."""


class LatestOnly(BufferPolicy):
    name = "latest"

    def __init__(self) -> None:
        self._f: Optional[Frame] = None
        self._cv = threading.Condition()
        self._closed = False

    def put(self, frame: Frame) -> Dropped:
        with self._cv:
            old, self._f = self._f, frame
            self._cv.notify()
        return [(old, "superseded")] if old is not None else []

    def get(self, timeout: float) -> Optional[Frame]:
        with self._cv:
            if self._f is None and not self._closed:
                self._cv.wait(timeout)
            f, self._f = self._f, None
            return f

    def depth(self) -> int:
        return 0 if self._f is None else 1

    def close(self) -> Dropped:
        with self._cv:
            self._closed = True
            f, self._f = self._f, None
            self._cv.notify_all()
        return [(f, "shutdown")] if f is not None else []


class BoundedQueue(BufferPolicy):
    def __init__(self, capacity: int = 4, drop: str = "oldest") -> None:
        if capacity < 1 or drop not in ("oldest", "newest"):
            raise ValueError("capacity >= 1 and drop = 'oldest' | 'newest'")
        self.capacity, self.drop = capacity, drop
        self.name = f"queue:{capacity}:{drop}"
        self._q: deque[Frame] = deque()
        self._cv = threading.Condition()
        self._closed = False

    def put(self, frame: Frame) -> Dropped:
        out: Dropped = []
        with self._cv:
            if len(self._q) >= self.capacity:
                if self.drop == "newest":
                    return [(frame, "queue_full_newest")]
                out.append((self._q.popleft(), "queue_full_oldest"))
            self._q.append(frame)
            self._cv.notify()
        return out

    def get(self, timeout: float) -> Optional[Frame]:
        with self._cv:
            if not self._q and not self._closed:
                self._cv.wait(timeout)
            return self._q.popleft() if self._q else None

    def depth(self) -> int:
        return len(self._q)

    def close(self) -> Dropped:
        with self._cv:
            self._closed = True
            left, self._q = list(self._q), deque()
            self._cv.notify_all()
        return [(f, "shutdown") for f in left]


def make_policy(spec: str) -> BufferPolicy:
    """'latest' | 'queue:N' | 'queue:N:oldest' | 'queue:N:newest'."""
    parts = spec.split(":")
    if parts[0] == "latest" and len(parts) == 1:
        return LatestOnly()
    if parts[0] == "queue":
        try:
            return BoundedQueue(int(parts[1]) if len(parts) > 1 else 4, parts[2] if len(parts) > 2 else "oldest")
        except (ValueError, IndexError) as e:
            raise ValueError(f"bad policy {spec!r}: {e}") from e
    raise ValueError(f"unknown policy {spec!r}; use 'latest' or 'queue:N[:oldest|newest]'")
