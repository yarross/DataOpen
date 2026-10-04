"""Shared-memory frame ring: lets a game process hand pixels to the core without a disk or socket round trip.

The core creates N named segments; for each `peek` the mod writes raw RGB24 into the segment it was told to use, and
the core reads it as a numpy VIEW (zero copy). Slots have an explicit lifetime (acquire -> release) so a slow
evaluator back-pressures frame capture instead of letting frames pile up in memory.
"""
from __future__ import annotations

import os
import queue
import secrets
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np


@dataclass
class ShmSlot:
    name: str
    shm: shared_memory.SharedMemory
    capacity: int

    def view(self, width: int, height: int) -> np.ndarray:
        need = width * height * 3
        if need > self.capacity:
            raise ValueError(f"frame needs {need} bytes but the slot holds {self.capacity}")
        return np.ndarray((height, width, 3), dtype=np.uint8, buffer=self.shm.buf)


def attach(name: str) -> shared_memory.SharedMemory:
    """Writer side (the process that is NOT the owner, e.g. the mock game server in tests): attach by name without
    taking over the lifetime of the segment."""
    try:
        return shared_memory.SharedMemory(name=name, track=False)      # Python 3.13+
    except TypeError:
        shm = shared_memory.SharedMemory(name=name)
        try:                                                           # <3.13 registers attached segments with the resource
            from multiprocessing import resource_tracker               # tracker, which would unlink them behind our back
            resource_tracker.unregister(shm._name, "shared_memory")   # type: ignore[attr-defined]
        except Exception:
            pass
        return shm


class ShmRing:
    def __init__(self, slots: int = 4, capacity_bytes: int = 1920 * 1080 * 3) -> None:
        self.capacity = capacity_bytes
        self._free: "queue.Queue[ShmSlot]" = queue.Queue()
        self._all: list[ShmSlot] = []
        self._lock = threading.Lock()
        self._tag = f"dopen_{os.getpid()}_{secrets.token_hex(3)}"
        for i in range(slots):
            self._add(i)

    def _add(self, i: int) -> None:
        name = f"{self._tag}_{i}"
        slot = ShmSlot(name, shared_memory.SharedMemory(name=name, create=True, size=self.capacity), self.capacity)
        self._all.append(slot)
        self._free.put(slot)

    def acquire(self, timeout_s: float = 60.0) -> ShmSlot:
        try:
            return self._free.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError("no free shared-memory slot: the evaluator is not keeping up (or a slot leaked)") from None

    def release(self, slot: ShmSlot) -> None:
        self._free.put(slot)

    @property
    def free_slots(self) -> int:
        return self._free.qsize()

    def grow(self, new_capacity: int) -> None:
        """A larger frame arrived (resolution change): recreate every FREE slot with the new capacity."""
        with self._lock:
            if new_capacity <= self.capacity:
                return
            drained = []
            while True:
                try:
                    drained.append(self._free.get_nowait())
                except queue.Empty:
                    break
            for s in drained:
                self._destroy(s)
                self._all.remove(s)
            self.capacity = new_capacity
            for i in range(len(drained)):
                self._add(1000 + len(self._all) + i)

    @staticmethod
    def _destroy(slot: ShmSlot) -> None:
        try:
            slot.shm.close()
            slot.shm.unlink()
        except (FileNotFoundError, BufferError):
            pass

    def close(self) -> None:
        for s in self._all:
            self._destroy(s)
        self._all.clear()
