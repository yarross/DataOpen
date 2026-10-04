"""Single-slot file mailbox transport.

Why files: the lowest common denominator. Lua mods (Garry's Mod, UE4SS, Cyber Engine Tweaks) have no
sockets but can read/write files in their data folder; C# mods can do both. One fixed request file and
one fixed response file means the game never has to list directories or rename files:

    core  --atomic write-->  <dir>/req.json   {"v":1,"id":N,"method":"...","params":{...}}
    game  --write-------->   <dir>/res.json   {"id":N,"result":{...}}  or  {"id":N,"error":{"message":...}}

The game processes a request when its id differs from the last one it handled. The core ignores any
response whose id does not match and retries on a half-written (unparseable) response, so the game
does not need atomic writes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .interfaces import AdapterError
from .protocol import PROTOCOL_VERSION


class TransportError(AdapterError):
    pass


class TransportTimeout(TransportError):
    pass


class RemoteError(AdapterError):
    """The game mod answered with an error."""

    def __init__(self, method: str, message: str, kind: str = "") -> None:
        super().__init__(f"{method}: {message}" + (f" [{kind}]" if kind else ""))
        self.method, self.remote_message, self.kind = method, message, kind


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(50):  # Windows: replace fails while the other side has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.002 * (attempt + 1))
    raise TransportError(f"could not replace {path}")


class FileMailboxTransport:
    def __init__(self, directory: Path | str, poll_s: float = 0.004, default_timeout_s: float = 60.0) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.req, self.res = self.dir / "req.json", self.dir / "res.json"
        self.poll_s, self.default_timeout_s = poll_s, default_timeout_s
        self._id = int(time.time()) % 2_000_000_000  # differs from the previous run's last id
        self._lock = threading.Lock()

    def call(self, method: str, params: Optional[dict[str, Any]] = None,
             timeout_s: Optional[float] = None) -> dict[str, Any]:
        timeout_s = self.default_timeout_s if timeout_s is None else timeout_s
        with self._lock:
            self._id += 1
            rid = self._id
            _atomic_write(self.req, json.dumps({"v": PROTOCOL_VERSION, "id": rid, "method": method,
                                                 "params": params or {}}, separators=(",", ":")))
            deadline = time.monotonic() + timeout_s
            while True:
                msg = self._try_read(rid)
                if msg is not None:
                    if "error" in msg:
                        e = msg["error"] if isinstance(msg["error"], dict) else {"message": str(msg["error"])}
                        raise RemoteError(method, str(e.get("message", "")), str(e.get("type", "")))
                    return msg.get("result") or {}
                if time.monotonic() > deadline:
                    raise TransportTimeout(
                        f"no response to {method!r} within {timeout_s:.0f}s (is the game running with the "
                        f"DataOpen mod loaded and pointed at {self.dir}?)")
                time.sleep(self.poll_s)

    def _try_read(self, rid: int) -> Optional[dict[str, Any]]:
        try:
            raw = self.res.read_bytes()
        except OSError:
            return None
        try:
            msg = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None  # half-written by the game; try again
        return msg if isinstance(msg, dict) and msg.get("id") == rid else None

    def clear(self) -> None:
        for p in (self.req, self.res):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


Handler = Callable[[str, dict[str, Any]], dict[str, Any]]


class MailboxServer:
    """Reference game side in Python: used by the mock server and by tests."""

    def __init__(self, directory: Path | str, handler: Handler, poll_s: float = 0.002) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.req, self.res = self.dir / "req.json", self.dir / "res.json"
        self.handler, self.poll_s = handler, poll_s
        self._last_id: Any = None

    def poll_once(self) -> bool:
        try:
            msg = json.loads(self.req.read_bytes().decode("utf-8"))
        except (OSError, ValueError):
            return False
        rid = msg.get("id")
        if rid is None or rid == self._last_id:
            return False
        self._last_id = rid
        try:
            if msg.get("v") != PROTOCOL_VERSION:
                raise ValueError(f"protocol version {msg.get('v')} != {PROTOCOL_VERSION}")
            reply: dict[str, Any] = {"id": rid, "result": self.handler(msg["method"], msg.get("params") or {})}
        except Exception as e:  # report to the core instead of dying
            reply = {"id": rid, "error": {"message": str(e), "type": type(e).__name__}}
        _atomic_write(self.res, json.dumps(reply, separators=(",", ":")))
        return True

    def serve_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if not self.poll_once():
                time.sleep(self.poll_s)
