import base64
import hashlib
import json
import socket
import struct
import threading

import pytest

from dataopen.adapters.rcon import (_WS_GUID, CompositeAdapter, RconClient, RconController, RconError,
                                    derive_rust_env)
from dataopen.core.models import SceneSpec


class FakeRustServer:
    """A tiny WebSocket server speaking Rust's WebRCON JSON, for tests."""

    def __init__(self, password="secret", noise=True):
        self.password, self.noise = password, noise
        self.commands: list[str] = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _frame(self, text: str, opcode=0x1) -> bytes:
        p = text.encode()
        h = bytes([0x80 | opcode]) + (bytes([len(p)]) if len(p) < 126 else bytes([126]) + struct.pack(">H", len(p)))
        return h + p

    def _read_frame(self, c):
        def exact(n):
            b = b""
            while len(b) < n:
                chunk = c.recv(n - len(b))
                if not chunk:
                    raise ConnectionError
                b += chunk
            return b
        b1, b2 = exact(2)
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", exact(2))[0]
        mask = exact(4)
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(exact(n)))
        return b1 & 0x0F, data.decode()

    def _serve(self):
        c, _ = self.sock.accept()
        head = b""
        while b"\r\n\r\n" not in head:
            head += c.recv(1024)
        lines = head.decode().split("\r\n")
        path = lines[0].split()[1]
        key = next(ln.split(": ")[1] for ln in lines if ln.lower().startswith("sec-websocket-key"))
        if path != "/" + self.password:
            c.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            c.close()
            return
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        c.sendall(f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode())
        try:
            while True:
                op, text = self._read_frame(c)
                if op == 0x8:
                    return
                if op == 0xA:      # pong for our ping
                    continue
                msg = json.loads(text)
                self.commands.append(msg["Message"])
                if self.noise:  # unrelated log line first, as the real server does
                    c.sendall(self._frame(json.dumps({"Identifier": 0, "Message": "log noise"})))
                    c.sendall(self._frame("x", opcode=0x9))
                big = "x" * 300 if msg["Message"] == "big" else f"ok:{msg['Message']}"
                c.sendall(self._frame(json.dumps({"Identifier": msg["Identifier"], "Message": big})))
        except (ConnectionError, OSError):
            return


def test_command_roundtrip_with_noise_ping_and_long_reply():
    srv = FakeRustServer()
    cli = RconClient(f"127.0.0.1:{srv.port}", "secret", timeout_s=5)
    cli.connect()
    assert cli.command("env.time 12") == "ok:env.time 12"
    assert cli.command("serverinfo") == "ok:serverinfo"
    assert len(cli.command("big")) == 300
    assert srv.commands == ["env.time 12", "serverinfo", "big"]
    cli.close()


def test_wrong_password_and_no_server_are_clear_errors():
    srv = FakeRustServer(password="right")
    with pytest.raises(RconError, match="handshake|closed"):
        RconClient(f"127.0.0.1:{srv.port}", "wrong", timeout_s=3).connect()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(RconError, match="cannot connect"):
        RconClient(f"127.0.0.1:{port}", "x", timeout_s=1).connect()


def test_templates_are_validated_up_front_and_filled_from_the_scene():
    srv = FakeRustServer()
    cli = RconClient(f"127.0.0.1:{srv.port}", "secret", timeout_s=5)
    with pytest.raises(RconError, match="bad command template"):
        RconController(cli, ["weather.snow {snowiness}"])
    cli.connect()
    ctl = RconController(cli, ["env.time {time_of_day:.1f}", "weather.rain {rain:.2f}", "weather.fog {fog:.2f}"], settle_s=0)
    ctl.apply_scene(SceneSpec(0, 1, "train", {"time_of_day": 18.25, "weather": "rain", "fog_density": 0.02}, []))
    assert srv.commands == ["env.time 18.2", "weather.rain 0.80", "weather.fog 0.30"]


def test_derived_environment_values():
    d = derive_rust_env({"weather": "fog", "fog_density": 0.2})
    assert d["fog"] == 1.0 and d["rain"] == 0.0 and d["cloud_cover"] == 0.3


class FakeInner:
    """Stands in for the client-side RemoteGameAdapter."""

    def __init__(self):
        self.calls = []
        self.spawner = self
        self.info, self.environment, self.extractor, self.capture = "info", "env", "extractor", "capture"
        self.hello = {"game": "rust"}

    def parameter_space(self): return "space"
    def connect(self): self.calls.append("connect")
    def close(self): self.calls.append("close")
    def restart(self): self.calls.append("restart")
    def spawn(self, scene):
        self.calls.append("spawn")
        return ["h"]
    def update_actors(self, h, f): self.calls.append("update")
    def set_active(self, h, a): self.calls.append(f"active={a}")
    def despawn_all(self): self.calls.append("despawn")
    def selftest(self): return [{"name": "client", "ok": True}]


def test_composite_orders_server_before_client_and_cleans_up_in_reverse():
    srv = FakeRustServer()
    inner = FakeInner()
    cli = RconClient(f"127.0.0.1:{srv.port}", "secret", timeout_s=5)
    comp = CompositeAdapter(inner, RconController(cli, ["env.time {time_of_day:.0f}"], ["env.progresstime true"], 0))
    comp.connect()
    assert comp.spawner.spawn(SceneSpec(0, 1, "train", {"time_of_day": 6.0}, [])) == ["h"]
    assert srv.commands == ["env.time 6"] and inner.calls == ["connect", "spawn"]
    comp.spawner.despawn_all()
    assert srv.commands[-1] == "env.progresstime true" and inner.calls[-1] == "despawn"
    assert comp.info == "info" and comp.parameter_space() == "space" and comp.hello == {"game": "rust"}
    names = {c["name"]: c for c in comp.selftest()}
    assert names["client"]["ok"] and names["rcon"]["ok"]
    comp.close()
    assert inner.calls[-1] == "close"


def test_factory_requires_the_password_env_var(monkeypatch, tmp_path):
    from dataopen.factory import build_adapter
    from dataopen.profiles import ProfileError, load_profile
    monkeypatch.delenv("RUST_RCON_PASSWORD", raising=False)
    with pytest.raises(ProfileError, match="RUST_RCON_PASSWORD"):
        build_adapter(load_profile("rust"), mailbox=str(tmp_path), connect=False)
    monkeypatch.setenv("RUST_RCON_PASSWORD", "pw")
    comp = build_adapter(load_profile("rust"), mailbox=str(tmp_path), connect=False)
    assert isinstance(comp, CompositeAdapter) and len(comp.ctl.begin) == 5
