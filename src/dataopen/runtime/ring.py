"""A frame ring between a capture process and the runtime, passed the way DMA buffers are passed on Linux: the producer creates N
buffer slots and sends their FILE DESCRIPTORS over a Unix socket (SCM_RIGHTS); both sides mmap them; afterwards only tiny messages
travel ("slot k holds frame f" / "slot k is free again"). Pixels never move.

  producer  `RingProducer`   memfd slots (stand-in for dma-buf fds: on the board the capture/RGA path hands over its dma-buf fds
                             instead, and `dmabuf=True` makes the consumer bracket CPU reads with DMA_BUF_IOCTL_SYNC)
  consumer  `RingConsumer`   a `FrameSource`: listens on the socket, maps the slots, yields zero-copy read-only frames

Properties: the producer NEVER blocks (no free slot = the frame is dropped and counted); a slot is owned by the consumer from the
FRAME message until `Frame.release()`; a vanished peer is detected and the consumer goes back to waiting for a new producer.
"""
from __future__ import annotations

import fcntl
import json
import mmap
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .frames import Frame, FrameSource, now_us

_FRAME = struct.Struct("<4sIIQ")          # tag, slot, frame_id, ts_us
_RELEASE = struct.Struct("<4sI")
_TAG_FRAME, _TAG_RELEASE = b"FRME", b"RLSE"
DMA_BUF_IOCTL_SYNC = 0x40086200           # _IOW('b', 0, struct dma_buf_sync)
_SYNC_READ, _SYNC_START, _SYNC_END = 1, 0, 4


class RingError(RuntimeError):
    pass


def _sync(fd: int, end: bool) -> None:
    fcntl.ioctl(fd, DMA_BUF_IOCTL_SYNC, struct.pack("<Q", _SYNC_READ | (_SYNC_END if end else _SYNC_START)))


class RingProducer:
    def __init__(self, path: str | Path, width: int = 640, height: int = 640, n_slots: int = 4, dmabuf: bool = False,
                 connect_timeout: float = 5.0) -> None:
        self.path, self.w, self.h, self.n = str(path), width, height, n_slots
        self.slot_bytes = width * height * 3
        self.dmabuf = dmabuf
        self.fds: list[int] = []
        self.maps: list[mmap.mmap] = []
        self.views: list[np.ndarray] = []
        for i in range(n_slots):
            fd = os.memfd_create(f"apollo-frame-{i}")
            os.ftruncate(fd, self.slot_bytes)
            self.fds.append(fd)
            mm = mmap.mmap(fd, self.slot_bytes)
            self.maps.append(mm)
            self.views.append(np.frombuffer(mm, dtype=np.uint8).reshape(height, width, 3))
        self.free = set(range(n_slots))
        self.submitted = self.dropped_no_slot = self.dropped_send = 0
        self.connected = False
        self.sock: Optional[socket.socket] = None
        self._connect(connect_timeout)

    def _connect(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        err: Optional[Exception] = None
        while time.monotonic() < deadline:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                s.connect(self.path)
                s.settimeout(5.0)
                hello = json.dumps({"v": 1, "w": self.w, "h": self.h, "fmt": "rgb8", "n_slots": self.n,
                                    "slot_bytes": self.slot_bytes, "dmabuf": self.dmabuf}).encode()
                socket.send_fds(s, [hello], self.fds)
                reply = s.recv(64)
                if reply != b"OK":
                    s.close()
                    raise RingError(f"consumer refused the ring: {reply.decode(errors='replace')}")
                s.setblocking(False)
                self.sock, self.connected = s, True
                self.free = set(range(self.n))
                return
            except (FileNotFoundError, ConnectionRefusedError) as e:
                s.close()
                err = e
                time.sleep(0.05)
        raise RingError(f"no consumer listening at {self.path}: {err}")

    def _drain(self) -> None:
        """Read RELEASE messages (non-blocking) and mark those slots free."""
        if self.sock is None:
            return
        while True:
            try:
                msg = self.sock.recv(64)
            except BlockingIOError:
                return
            except OSError:
                self.connected = False
                return
            if not msg:
                self.connected = False
                return
            if len(msg) == _RELEASE.size:
                tag, slot = _RELEASE.unpack(msg)
                if tag == _TAG_RELEASE and slot < self.n:
                    self.free.add(slot)

    def acquire(self) -> Optional[tuple[int, np.ndarray]]:
        """A free slot and a writable (H, W, 3) view over it (write the frame straight into it: zero copy), or None."""
        self._drain()
        if not self.connected or not self.free:
            self.dropped_no_slot += 1
            return None
        slot = min(self.free)
        self.free.discard(slot)
        return slot, self.views[slot]

    def commit(self, slot: int, frame_id: int, ts_us: Optional[int] = None) -> bool:
        try:
            assert self.sock is not None
            self.sock.send(_FRAME.pack(_TAG_FRAME, slot, frame_id & 0xFFFFFFFF, ts_us if ts_us is not None else now_us()))
        except BlockingIOError:                                    # the consumer's socket buffer is full: drop, never wait
            self.free.add(slot)
            self.dropped_send += 1
            return False
        except OSError:                                            # the consumer went away
            self.free.add(slot)
            self.dropped_send += 1
            self.connected = False
            return False
        self.submitted += 1
        return True

    def submit(self, array: np.ndarray, frame_id: int, ts_us: Optional[int] = None) -> bool:
        got = self.acquire()
        if got is None:
            return False
        slot, view = got
        view[...] = array
        return self.commit(slot, frame_id, ts_us)

    def stats(self) -> dict[str, Any]:
        return {"submitted": self.submitted, "dropped_no_slot": self.dropped_no_slot, "dropped_send": self.dropped_send,
                "free_slots": len(self.free), "connected": self.connected}

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock, self.connected = None, False
        self.views.clear()
        for mm in self.maps:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds, self.maps = [], []


class RingConsumer(FrameSource):
    """Listens at `path`, adopts the slot buffers a producer sends, and yields zero-copy read-only frames."""

    def __init__(self, path: str | Path, expect: tuple[int, int] = (640, 640)) -> None:
        self.path = str(path)
        if len(self.path.encode()) > 100:
            raise RingError("socket path too long for AF_UNIX (max ~100 bytes)")
        self.expect = expect
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.path):
            os.unlink(self.path)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.server.bind(self.path)
        self.server.listen(1)
        self.server.setblocking(False)
        self.conn: Optional[socket.socket] = None
        self.maps: list[mmap.mmap] = []
        self.fds: list[int] = []
        self.views: list[np.ndarray] = []
        self.dmabuf = False
        self.received = self.released = 0
        self.rejected = 0
        self._pending_release: list[int] = []
        self._gen = 0                          # connection generation: releases of an old connection are ignored
        self._closed = False

    # ---- connection ----
    def _accept(self) -> None:
        try:
            conn, _ = self.server.accept()
        except BlockingIOError:
            return
        try:
            conn.settimeout(5.0)
            data, fds, _flags, _addr = socket.recv_fds(conn, 4096, 64)
            hello = json.loads(data.decode())
            w, h = hello["w"], hello["h"]
            if (w, h) != tuple(self.expect) or hello.get("fmt") != "rgb8" or len(fds) != hello["n_slots"]:
                for fd in fds:
                    os.close(fd)
                conn.send(f"ERR:expected {self.expect[0]}x{self.expect[1]} rgb8, got {w}x{h} {hello.get('fmt')}".encode())
                conn.close()
                self.rejected += 1
                return
            self._drop_ring()
            self.fds, self.dmabuf = list(fds), bool(hello.get("dmabuf"))
            for fd in fds:
                mm = mmap.mmap(fd, hello["slot_bytes"], prot=mmap.PROT_READ)
                self.maps.append(mm)
                self.views.append(np.frombuffer(mm, dtype=np.uint8).reshape(h, w, 3))
            conn.send(b"OK")
            conn.setblocking(False)
            self.conn = conn
            self._gen += 1
            self._pending_release = []
        except (OSError, ValueError, KeyError) as e:
            conn.close()
            self.rejected += 1
            raise RingError(f"bad producer handshake: {e}") from e

    def _drop_ring(self) -> None:
        self.views.clear()
        for mm in self.maps:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass                                   # a frame still references it: freed when that frame is released
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.maps, self.fds = [], []

    def _disconnect(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.conn = None

    # ---- FrameSource ----
    def _flush_releases(self) -> None:
        while self._pending_release and self.conn is not None:
            try:
                self.conn.send(_RELEASE.pack(_TAG_RELEASE, self._pending_release[0]))
            except BlockingIOError:
                return
            except OSError:
                self._disconnect()
                return
            self._pending_release.pop(0)

    def _release_slot(self, slot: int, gen: int) -> None:
        if gen != self._gen or self.conn is None:
            return
        if self.dmabuf:
            try:
                _sync(self.fds[slot], end=True)
            except OSError:
                pass
        self.released += 1
        self._pending_release.append(slot)
        self._flush_releases()

    def get(self, timeout: float = 0.1) -> Optional[Frame]:
        if self._closed:
            return None
        deadline = time.monotonic() + timeout
        while True:
            if self.conn is None:
                try:
                    self._accept()
                except RingError:
                    pass
            if self.conn is not None:
                self._flush_releases()
                try:
                    msg = self.conn.recv(64) if self.conn is not None else b""
                except BlockingIOError:
                    msg = None
                except OSError:
                    msg = b""
                if msg == b"":
                    self._disconnect()
                elif msg and len(msg) == _FRAME.size:
                    tag, slot, fid, ts = _FRAME.unpack(msg)
                    if tag == _TAG_FRAME and slot < len(self.views):
                        if self.dmabuf:
                            try:
                                _sync(self.fds[slot], end=False)
                            except OSError:
                                pass
                        gen = self._gen
                        self.received += 1
                        return Frame(self.views[slot], fid, ts, {"slot": slot}, lambda s=slot, g=gen: self._release_slot(s, g))
                    continue
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.0005)

    def stats(self) -> dict[str, Any]:
        return {"source": "ring", "connected": self.conn is not None, "received": self.received, "released": self.released,
                "rejected_handshakes": self.rejected, "slots": len(self.views)}

    def close(self) -> None:
        self._closed = True
        self._disconnect()
        try:
            self.server.close()
        finally:
            if os.path.exists(self.path):
                try:
                    os.unlink(self.path)
                except OSError:
                    pass
        self._drop_ring()
