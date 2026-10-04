"""The IPC channel `KeypointArray`s are published on, so any other module can consume them without touching the runtime.

Two complementary transports (use either or both; `ResultChannel` offers both):

  UdsPublisher / UdsSubscriber   every result, in order, to every connected subscriber (SOCK_SEQPACKET: message boundaries kept,
                                 reliable while connected). Per-subscriber SMALL send buffer: a subscriber that cannot keep up
                                 misses messages (counted for it) instead of slowing the runtime or everyone else.
  ShmLatest                      the newest result only, in shared memory, for modules that poll at their own rate (an aiming loop
                                 reading at 1 kHz). Single writer, many readers; each read is validated by a sequence counter AND a
                                 CRC32 of the payload, so a torn read is detected whatever the CPU's memory ordering.
Neither transport ever blocks the publisher.
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time
import zlib
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable, Optional

from ..quality.shm import attach

_HDR = struct.Struct("<QIIQ")            # seq, length, crc32, publish timestamp (us)


class ResultPublisher:
    def publish(self, data: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def stats(self) -> dict[str, Any]:
        return {}


class CallbackPublisher(ResultPublisher):
    """In-process: hands every result to `fn(bytes)`."""

    def __init__(self, fn: Callable[[bytes], None]) -> None:
        self.fn, self.n = fn, 0

    def publish(self, data: bytes) -> None:
        self.n += 1
        self.fn(data)

    def stats(self) -> dict[str, Any]:
        return {"publisher": "callback", "published": self.n}


class UdsPublisher(ResultPublisher):
    def __init__(self, path: str | Path, sndbuf: int = 16384) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.path):
            os.unlink(self.path)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.server.bind(self.path)
        self.server.listen(16)
        self.server.setblocking(False)
        self.sndbuf = sndbuf
        self._subs: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.published = 0

    def _accept(self) -> None:
        while True:
            try:
                c, _ = self.server.accept()
            except BlockingIOError:
                return
            c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf)
            c.setblocking(False)
            self._subs.append({"sock": c, "sent": 0, "dropped": 0})

    def publish(self, data: bytes) -> None:
        with self._lock:
            self._accept()
            alive = []
            for s in self._subs:
                try:
                    s["sock"].send(data, socket.MSG_DONTWAIT)
                    s["sent"] += 1
                except BlockingIOError:
                    s["dropped"] += 1                              # slow subscriber: it misses this one, nobody else is affected
                except OSError:
                    s["sock"].close()
                    continue                                       # gone
                alive.append(s)
            self._subs = alive
            self.published += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"publisher": "uds", "published": self.published, "subscribers": len(self._subs),
                    "subscriber_dropped": [s["dropped"] for s in self._subs]}

    def close(self) -> None:
        with self._lock:
            for s in self._subs:
                s["sock"].close()
            self._subs = []
        self.server.close()
        if os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass


class UdsSubscriber:
    def __init__(self, path: str | Path, timeout: float = 5.0, rcvbuf: int = 16384) -> None:
        deadline = time.monotonic() + timeout
        while True:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
            try:
                self.sock.connect(str(path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                self.sock.close()
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)

    def recv(self, timeout: float = 1.0) -> Optional[bytes]:
        """Next message, or None on timeout; b'' means the publisher closed."""
        self.sock.settimeout(timeout)
        try:
            return self.sock.recv(65536)
        except (socket.timeout, BlockingIOError):
            return None

    def latest(self, timeout: float = 1.0) -> Optional[bytes]:
        """The NEWEST waiting message (older ones are skipped): for a consumer that only cares about the present."""
        msg = self.recv(timeout)
        if not msg:
            return msg
        self.sock.setblocking(False)
        while True:
            try:
                nxt = self.sock.recv(65536)
            except BlockingIOError:
                return msg
            if not nxt:
                return msg
            msg = nxt

    def close(self) -> None:
        self.sock.close()


class ShmLatest(ResultPublisher):
    """Newest-value mailbox in shared memory (single writer, any number of readers)."""

    def __init__(self, name: str, max_bytes: int = 4096, create: bool = True) -> None:
        self.name, self.max = name, max_bytes
        if create:
            try:
                old = shared_memory.SharedMemory(name=name)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=_HDR.size + max_bytes)
            self.shm.buf[:_HDR.size] = bytes(_HDR.size)
        else:
            self.shm = attach(name)
            self.max = self.shm.size - _HDR.size
        self._seq = 0
        self._owner = create

    # ---- writer ----
    def publish(self, data: bytes) -> None:
        if len(data) > self.max:
            raise ValueError(f"message of {len(data)} bytes does not fit the {self.max}-byte mailbox")
        buf = self.shm.buf
        self._seq += 1                                                    # odd: write in progress
        struct.pack_into("<Q", buf, 0, self._seq)
        buf[_HDR.size:_HDR.size + len(data)] = data
        struct.pack_into("<IIQ", buf, 8, len(data), zlib.crc32(data) & 0xFFFFFFFF, time.monotonic_ns() // 1000)
        self._seq += 1                                                    # even: stable
        struct.pack_into("<Q", buf, 0, self._seq)

    # ---- readers ----
    def read(self, retries: int = 100) -> Optional[tuple[int, bytes, int]]:
        """(sequence, payload, publish_ts_us) of the newest complete message, or None if nothing was published yet / the writer
        kept overwriting during every retry."""
        buf = self.shm.buf
        for _ in range(retries):
            s1 = struct.unpack_from("<Q", buf, 0)[0]
            if s1 == 0:
                return None
            if s1 & 1:
                continue
            _, n, crc, ts = _HDR.unpack_from(buf, 0)
            if n > self.max:
                continue
            data = bytes(buf[_HDR.size:_HDR.size + n])
            s2 = struct.unpack_from("<Q", buf, 0)[0]
            if s1 == s2 and zlib.crc32(data) & 0xFFFFFFFF == crc:
                return s1 // 2, data, ts
        return None

    def wait_new(self, after_seq: int, timeout: float = 1.0, poll_s: float = 0.0002) -> Optional[tuple[int, bytes, int]]:
        deadline = time.monotonic() + timeout
        while True:
            r = self.read()
            if r is not None and r[0] > after_seq:
                return r
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_s)

    def stats(self) -> dict[str, Any]:
        return {"publisher": "shm-latest", "name": self.name, "published": self._seq // 2}

    def close(self) -> None:
        try:
            self.shm.close()
        finally:
            if self._owner:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass


class ResultChannel(ResultPublisher):
    """UDS stream + shared-memory mailbox behind one `publish`."""

    def __init__(self, uds_path: Optional[str] = None, shm_name: Optional[str] = None, max_bytes: int = 4096) -> None:
        self.parts: list[ResultPublisher] = []
        if uds_path:
            self.parts.append(UdsPublisher(uds_path))
        if shm_name:
            self.parts.append(ShmLatest(shm_name, max_bytes))

    def publish(self, data: bytes) -> None:
        for p in self.parts:
            p.publish(data)

    def stats(self) -> dict[str, Any]:
        return {"channel": [p.stats() for p in self.parts]}

    def close(self) -> None:
        for p in self.parts:
            p.close()
