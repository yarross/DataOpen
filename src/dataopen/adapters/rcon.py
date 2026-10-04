"""Server-side control over RCON for games with a client/server split (Rust).

Environment (time of day, weather) lives on the SERVER; the client plugin only renders and reports bones.
`CompositeAdapter` applies the scene on the server first and then lets the client adapter observe it.
Commands are templates from the profile (`[server] scene_begin = [...]`), so they can be fixed without code.
Own local server only.
"""
from __future__ import annotations

import base64
import hashlib
import itertools
import json
import os
import socket
import struct
import time
from typing import Any, Optional, Sequence

from ..core.interfaces import AdapterError, IGameAdapter
from ..core.models import SceneSpec

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class RconError(AdapterError):
    pass


class WebSocketClient:
    """Minimal RFC 6455 client (text frames, ping/pong, close). Enough for Rust's WebRCON."""

    def __init__(self, host: str, port: int, path: str, timeout_s: float = 10.0) -> None:
        self.host, self.port, self.path, self.timeout_s = host, port, path, timeout_s
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        except OSError as e:
            raise RconError(f"cannot connect to {self.host}:{self.port}: {e}. Is the server running with RCON "
                            f"enabled (rcon.port / rcon.web 1)?") from e
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise RconError("server closed the connection during the WebSocket handshake (wrong RCON password?)")
            head += chunk
        status = head.split(b"\r\n", 1)[0].decode(errors="replace")
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        if " 101 " not in status or accept.lower() not in head.decode(errors="replace").lower():
            raise RconError(f"WebSocket handshake failed: {status!r} (wrong RCON password or port?)")

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RconError("connection closed by the server")
            buf += chunk
        return buf

    def send_text(self, text: str, opcode: int = 0x1) -> None:
        payload = text.encode()
        n = len(payload)
        header = bytes([0x80 | opcode])
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def recv_text(self) -> str:
        message = b""
        while True:
            b1, b2 = self._recv_exact(2)
            fin, opcode = b1 & 0x80, b1 & 0x0F
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv_exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if b2 & 0x80 else None
            data = self._recv_exact(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode == 0x9:        # ping -> pong
                self.send_text(data.decode(errors="replace"), opcode=0xA)
                continue
            if opcode == 0x8:
                raise RconError("server closed the RCON connection")
            if opcode in (0x1, 0x0):
                message += data
                if fin:
                    return message.decode(errors="replace")

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None


class RconClient:
    def __init__(self, address: str, password: str, timeout_s: float = 10.0) -> None:
        host, _, port = address.rpartition(":")
        self.ws = WebSocketClient(host or "127.0.0.1", int(port), "/" + password, timeout_s)
        self.timeout_s = timeout_s
        self._ids = itertools.count(1000)

    def connect(self) -> None:
        self.ws.connect()

    def command(self, cmd: str) -> str:
        ident = next(self._ids)
        self.ws.send_text(json.dumps({"Identifier": ident, "Message": cmd, "Name": "DataOpen"}))
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                msg = json.loads(self.ws.recv_text())
            except socket.timeout:
                break
            if msg.get("Identifier") == ident:   # other identifiers are the server's log stream
                return str(msg.get("Message", ""))
        raise RconError(f"no reply to RCON command {cmd!r} within {self.timeout_s:.0f}s")

    def close(self) -> None:
        self.ws.close()


def derive_rust_env(env: dict[str, Any]) -> dict[str, Any]:
    """Core environment parameters -> the placeholders used by the command templates."""
    weather = env.get("weather", "clear")
    fog_density = float(env.get("fog_density", 0.0))
    return {**env,
            "rain": 0.8 if weather == "rain" else 0.0,
            "fog": min(1.0, fog_density * 15.0),
            "cloud_cover": float(env.get("cloud_cover", 0.3))}


class RconController:
    def __init__(self, client: RconClient, scene_begin: Sequence[str], scene_end: Sequence[str] = (),
                 settle_s: float = 1.0) -> None:
        self.client, self.begin, self.end, self.settle_s = client, list(scene_begin), list(scene_end), settle_s
        # fail at construction, not mid-session, if a template uses an unknown placeholder
        probe = derive_rust_env({"time_of_day": 12.0, "weather": "clear", "fog_density": 0.001, "cloud_cover": 0.3})
        for t in self.begin + self.end:
            try:
                t.format(**probe)
            except (KeyError, IndexError, ValueError) as e:
                raise RconError(f"bad command template {t!r}: {e!r}; available placeholders: {sorted(probe)}") from e

    def apply_scene(self, scene: SceneSpec) -> None:
        env = derive_rust_env(scene.environment)
        for t in self.begin:
            self.client.command(t.format(**env))
        if self.settle_s:
            time.sleep(self.settle_s)

    def end_scene(self) -> None:
        for t in self.end:
            self.client.command(t)


class CompositeAdapter(IGameAdapter):
    """Server control (RCON) + client observation (mod). Presents the normal adapter interface."""

    def __init__(self, client_adapter, controller: RconController) -> None:
        self.inner, self.ctl = client_adapter, controller
        self._spawner = _ServerSpawner(self)

    info = property(lambda self: self.inner.info)
    environment = property(lambda self: self.inner.environment)
    extractor = property(lambda self: self.inner.extractor)
    capture = property(lambda self: self.inner.capture)
    spawner = property(lambda self: self._spawner)

    def parameter_space(self):
        return self.inner.parameter_space()

    def connect(self) -> None:
        self.ctl.client.connect()
        self.inner.connect()

    def close(self) -> None:
        self.inner.close()
        self.ctl.client.close()

    def restart(self) -> None:
        self.ctl.client.close()
        self.ctl.client.connect()
        self.inner.restart()

    def __getattr__(self, name: str):  # hello, health, selftest, capture_mode ... of the client adapter
        return getattr(self.inner, name)

    def selftest(self) -> list[dict[str, Any]]:
        checks = list(self.inner.selftest())
        try:
            reply = self.ctl.client.command("serverinfo")
            checks.append({"name": "rcon", "ok": True, "detail": f"server answered: {reply[:80]!r}"})
        except AdapterError as e:
            checks.append({"name": "rcon", "ok": False, "detail": str(e),
                           "hint": "start your own local server with RCON enabled and set the password env var"})
        return checks


class _ServerSpawner:
    def __init__(self, outer: CompositeAdapter) -> None:
        self.o = outer

    def spawn(self, scene):
        self.o.ctl.apply_scene(scene)
        return self.o.inner.spawner.spawn(scene)

    def update_actors(self, handles, frame):
        return self.o.inner.spawner.update_actors(handles, frame)

    def set_active(self, handles, active):
        return self.o.inner.spawner.set_active(handles, active)

    def despawn_all(self):
        try:
            self.o.inner.spawner.despawn_all()
        finally:
            self.o.ctl.end_scene()
