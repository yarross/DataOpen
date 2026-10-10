"""A development server: the PWA's static files plus a simulated device (the real bridge core behind the gateway) over WebSocket.

For developing the browser client and for the end-to-end tests. It is NOT how a device serves the page: it binds to localhost only, is
stdlib asyncio and implements just enough of HTTP and RFC 6455 for one browser. The WebSocket carries the same CtlLink chunks as the BLE
characteristics, one chunk per binary frame, first byte = channel:  0 ctl (both ways) | 1 status (device -> client) | 2 info (at connect).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import struct
import tempfile
import time
from pathlib import Path
from typing import Optional

from ..bridge.cbridge import REASONS, STATE

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
CH_CTL, CH_STATUS, CH_INFO = 0, 1, 2
mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("text/javascript", ".js")


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def ws_frame(payload: bytes, opcode: int = 0x2) -> bytes:
    n = len(payload)
    if n < 126:
        size = bytes([n])
    elif n < 65536:
        size = b"\x7e" + struct.pack(">H", n)
    else:
        size = b"\x7f" + struct.pack(">Q", n)
    return bytes([0x80 | opcode]) + size + payload


async def read_frame(r: asyncio.StreamReader) -> Optional[tuple[int, bytes]]:
    """One (opcode, payload) from a client; None at EOF. Fragmented messages are not supported (a browser never splits ours)."""
    try:
        b0, b1 = await r.readexactly(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", await r.readexactly(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", await r.readexactly(8))[0]
        if n > 1 << 20:
            return None
        mask = await r.readexactly(4) if b1 & 0x80 else b"\0\0\0\0"
        data = bytearray(await r.readexactly(n))
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    for i in range(len(data)):
        data[i] ^= mask[i & 3]
    return b0 & 0x0F, bytes(data)


class SimServer:
    def __init__(self, root: Path, world) -> None:
        self.root, self.world = root.resolve(), world
        self.clients: list[asyncio.StreamWriter] = []
        self.bind(world.gw)
        world.restart_hooks.append(self.rebind)

    def bind(self, gw) -> None:
        gw.notify = lambda c: self.broadcast(CH_CTL, c)
        gw.notify_status = lambda b: self.broadcast(CH_STATUS, b)

    def rebind(self, gw) -> None:
        """The gateway was swapped for a fresh one (an update was applied): the old links are gone, as on a device that rebooted."""
        self.bind(gw)
        for w in list(self.clients):
            try:
                w.close()
            except (ConnectionError, RuntimeError):
                pass
        self.clients.clear()

    def broadcast(self, ch: int, data: bytes) -> None:
        for w in list(self.clients):
            try:
                w.write(ws_frame(bytes([ch]) + data))
            except (ConnectionError, RuntimeError):
                self.clients.remove(w)

    async def clock(self) -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.005)
            now = time.monotonic()
            self.world.run(min((now - last) * 1000.0, 100.0))
            last = now

    async def handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            line = await r.readline()
            parts = line.decode("latin-1").split()
            if len(parts) < 2:
                return
            method, target = parts[0], parts[1].split("?")[0]
            headers = {}
            while True:
                h = (await r.readline()).decode("latin-1").strip()
                if not h:
                    break
                k, _, v = h.partition(":")
                headers[k.strip().lower()] = v.strip()
            if headers.get("upgrade", "").lower() == "websocket" and target == "/ws":
                await self.websocket(r, w, headers)
            elif target.startswith("/sim/"):
                await self.sim(w, method, target, await r.read(0))
            else:
                self.static(w, target)
            await w.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if w in self.clients:
                self.clients.remove(w)
            w.close()

    def static(self, w: asyncio.StreamWriter, target: str) -> None:
        rel = target.lstrip("/") or "index.html"
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root or not p.is_file():
            w.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        data = p.read_bytes()
        mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        w.write(f"HTTP/1.1 200 OK\r\nContent-Type: {mt}\r\nContent-Length: {len(data)}\r\nCache-Control: no-cache\r\n"
                f"Connection: close\r\n\r\n".encode() + data)

    async def sim(self, w: asyncio.StreamWriter, method: str, target: str, _body: bytes) -> None:
        """Test hooks of the simulated rig: what the bridge is doing, and the hand's panic button."""
        wd = self.world
        if target == "/sim/bridge":
            b, gw = wd.bridge(), wd.gw
            out = {"state": STATE[b.state], "reason": REASONS[b.reason], "params_rejected": b.params_rejected,
                   "invariant_viol": b.invariant_viol, "assist_wanted": gw.settings.assist_wanted,
                   "strength": gw.settings.strength, "tremor": gw.settings.tremor,
                   "trial": gw.trial is not None, "calibrating": gw.calibrating,
                   "slot": gw.active, "slots": [s.has for s in gw.slotset], "names": [s.name for s in gw.slotset],
                   "leds": [int(x) for x in gw.leds()], "led_mode": gw.led_mode(),
                   "fw": None if gw.fw is None else {"running": gw.fw.running, "active": gw.fw.active, "trial": gw.fw.trial,
                                                      "boots": gw.fw.boots, "version": gw._fw_version(), "reboots": list(wd.reboots)}}
        elif target.startswith("/sim/panic/"):
            wd.rig.panic(True)
            wd.run(int(target.rsplit("/", 1)[1]))
            wd.rig.panic(False)
            out = {"ok": True}
        elif target == "/sim/button":
            wd.gw.physical_press()
            out = {"ok": True}
        elif target.startswith("/sim/press/"):                    # /sim/press/<slot|confirm>[/<ms>]: the front panel, held for a while
            parts = target.split("/")
            ms = float(parts[4]) if len(parts) > 4 else 100.0
            wd.press(parts[3], ms)
            out = {"ok": True}
        elif target.startswith("/sim/package/"):
            # /sim/package/<kind>/<seq>[/<slot>]: a package for THIS device from a test sender
            # (kinds: tuning, model, big, forged, other, tampered, and the refused models: ops, files, io, heavy, pose)

            from ..updates.dev import sim_package
            parts = target.split("/")
            slot = int(parts[5]) if len(parts) > 5 else None
            data = sim_package(parts[3], wd.gw.identity.card(), Path(wd.dir) / "sim-senders", int(parts[4]), slot)
            head = f"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n"
            w.write(head.encode() + data)
            return
        elif target.startswith("/sim/bundle/"):
            # /sim/bundle/<kind>/<seq>: a small settings file for THIS device from a test sender (kinds: profile, slots, other)
            from ..updates.dev import sim_bundle
            parts = target.split("/")
            data = sim_bundle(parts[3], wd.gw.identity.card(), Path(wd.dir) / "sim-senders", int(parts[4]))
            head = f"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n"
            w.write(head.encode() + data)
            return
        elif target.startswith("/sim/image/"):
            # /sim/image/<version>[/<min_version>]: an update image signed by the simulation's TEST manufacturer
            from .sim import dev_image
            parts = target.split("/")
            data = dev_image(int(parts[3]), int(parts[4]) if len(parts) > 4 else None)
            head = f"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n"
            w.write(head.encode() + data)
            return
        else:
            w.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        body = json.dumps(out).encode()
        head = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        w.write(head.encode() + body)

    async def websocket(self, r: asyncio.StreamReader, w: asyncio.StreamWriter, headers: dict) -> None:
        w.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                 f"Sec-WebSocket-Accept: {accept_key(headers.get('sec-websocket-key', ''))}\r\n\r\n").encode())
        self.clients.append(w)
        gw = self.world.gw
        gw.on_connect()
        w.write(ws_frame(bytes([CH_INFO]) + gw.read_info()))
        w.write(ws_frame(bytes([CH_STATUS]) + gw.read_status()))
        await w.drain()
        while True:
            fr = await read_frame(r)
            if fr is None or fr[0] == 0x8:
                break
            op, data = fr
            if op == 0x9:
                w.write(ws_frame(data, 0xA))
            elif op == 0x2 and data and data[0] == CH_CTL:
                gw.on_write(data[1:])
            await w.drain()
        gw.on_disconnect()


def serve_sim(a) -> int:
    from .sim import SimLearner, World, dev_image, dev_vendor, seed_profile
    root = Path(a.root) if a.root else Path(__file__).resolve().parents[3] / "pwa"
    if not (root / "index.html").exists():
        print(f"error: no PWA at {root} (use --root)")
        return 1
    d = a.dir or tempfile.mkdtemp(prefix="dataopen-ctl-")
    # a provisioned device (serial, device key, attestation) made by the TEST manufacturer
    if not (Path(d) / "otp" / "otp.json").exists():
        from ..provisioning import station as ST
        ST.provision(d, ST.VendorHsm(dev_vendor()[0]), jig=ST.SkipJig())
    if a.profile != "none":
        seed_profile(d, a.profile)
    world = World(d, learner=SimLearner(minutes=5.0, seed=2, speed=240.0), trial_s=getattr(a, "trial_s", 20) or 20,
                  vendor_pub=dev_vendor()[1], factory_image=dev_image(1, 1), fw_confirm_s=4)
    srv = SimServer(root, world)

    async def main() -> None:
        server = await asyncio.start_server(srv.handle, "127.0.0.1", a.port)
        asyncio.get_running_loop().create_task(srv.clock())
        port = server.sockets[0].getsockname()[1]
        print(f"serving {root} and a simulated device on http://127.0.0.1:{port}/  (state in {d}); Ctrl-C to stop", flush=True)
        async with server:
            await server.serve_forever()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    return 0
