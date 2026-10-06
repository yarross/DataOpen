"""Simulated USB peripherals for testing the bridge: real-looking mouse descriptors (boot mouse, a Logitech-style composite with a
vendor HID++ channel and a keyboard interface, a high-rate mouse), a host that enumerates them, and report packing helpers.
Nothing here is a model of any particular vendor's firmware: the descriptors are shaped like the real ones and the control requests
behave as the USB/HID specs say, which is what the proxy has to be transparent to."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

STALL, OK = 1, 0


# ---------------------------------------------------------------- HID descriptor item encoders
def _item(kind: int, tag: int, v: Optional[int] = None, signed: bool = False) -> bytes:
    if v is None:
        return bytes([(tag << 4) | (kind << 2)])
    if signed:
        n = 1 if -128 <= v <= 127 else 2 if -32768 <= v <= 32767 else 4
        raw = v.to_bytes(n, "little", signed=True)
    else:
        n = 1 if v < 256 else 2 if v < 65536 else 4
        raw = v.to_bytes(n, "little")
    return bytes([(tag << 4) | (kind << 2) | (3 if n == 4 else n)]) + raw


def usage_page(v):
    return _item(1, 0, v)


def usage(v):
    return _item(2, 0, v)


def umin(v):
    return _item(2, 1, v)


def umax(v):
    return _item(2, 2, v)


def lmin(v):
    return _item(1, 1, v, signed=True)


def lmax(v):
    return _item(1, 2, v, signed=True)


def rsize(v):
    return _item(1, 7, v)


def rcount(v):
    return _item(1, 9, v)


def rid(v):
    return _item(1, 8, v)


def push():
    return _item(1, 10)


def pop():
    return _item(1, 11)


def collection(t):
    return _item(0, 10, t)


def end():
    return _item(0, 12)


def inp(f):
    return _item(0, 8, f)


def outp(f):
    return _item(0, 9, f)


def feat(f):
    return _item(0, 11, f)


def _buttons(n: int, total_bits: int) -> bytes:
    d = usage_page(9) + umin(1) + umax(n) + lmin(0) + lmax(1) + rsize(1) + rcount(n) + inp(0x02)
    if total_bits > n:
        d += rcount(1) + rsize(total_bits - n) + inp(0x01)
    return d


def _xy(bits: int, rel: bool = True, extra: bytes = b"") -> bytes:
    lim = (1 << (bits - 1)) - 1
    return usage_page(1) + usage(0x30) + usage(0x31) + extra + lmin(-lim) + lmax(lim) + rsize(bits) + rcount(2) + inp(0x06 if rel else 0x02)


def boot_mouse_rd() -> bytes:
    return usage_page(1) + usage(2) + collection(1) + usage(1) + collection(0) + _buttons(3, 8) + _xy(8) + end() + end()


def mouse16_rd(wheel: bool = True) -> bytes:
    d = usage_page(1) + usage(2) + collection(1) + usage(1) + collection(0) + _buttons(5, 8) + _xy(16)
    if wheel:
        d += usage_page(1) + usage(0x38) + lmin(-127) + lmax(127) + rsize(8) + rcount(1) + inp(0x06)
        d += usage_page(0x0C) + _item(2, 0, 0x0238) + lmin(-127) + lmax(127) + rsize(8) + rcount(1) + inp(0x06)
    return d + end() + end()


def logi_mouse_rd(buttons: int = 16, xy_bits: int = 12) -> bytes:
    """A gaming-mouse shaped descriptor: report id 1, many buttons, 12-bit X/Y, wheel, AC Pan. Also a vendor input report id 2."""
    d = usage_page(1) + usage(2) + collection(1) + rid(1) + usage(1) + collection(0) + _buttons(buttons, 16) + _xy(xy_bits)
    d += usage_page(1) + usage(0x38) + lmin(-127) + lmax(127) + rsize(8) + rcount(1) + inp(0x06)
    d += usage_page(0x0C) + _item(2, 0, 0x0238) + lmin(-127) + lmax(127) + rsize(8) + rcount(1) + inp(0x06) + end()
    d += (
        usage_page(0xFF00)
        + usage(0x02)
        + collection(2)
        + rid(2)
        + rsize(8)
        + rcount(4)
        + lmin(0)
        + lmax(255)
        + usage(3)
        + inp(0x02)
        + end()
    )
    return d + end()


def hidpp_rd() -> bytes:
    """Vendor channel with report ids 0x10 (short) 0x11 (long), input and output, and a feature report (HID++ shaped)."""
    d = usage_page(0xFF00) + usage(1) + collection(1)
    for i, n in ((0x10, 6), (0x11, 19)):
        d += rid(i) + rsize(8) + rcount(n) + lmin(0) + lmax(255) + usage(1) + inp(0x00) + usage(2) + outp(0x00)
    d += rid(0x20) + rsize(8) + rcount(3) + usage(3) + feat(0x02)
    return d + end()


def kbd_rd() -> bytes:
    return (
        usage_page(1)
        + usage(6)
        + collection(1)
        + usage_page(7)
        + umin(0xE0)
        + umax(0xE7)
        + lmin(0)
        + lmax(1)
        + rsize(1)
        + rcount(8)
        + inp(0x02)
        + rcount(1)
        + rsize(8)
        + inp(0x01)
        + rcount(6)
        + rsize(8)
        + lmin(0)
        + lmax(101)
        + usage_page(7)
        + umin(0)
        + umax(101)
        + inp(0x00)
        + end()
    )


def abs_pointer_rd() -> bytes:
    return (
        usage_page(1)
        + usage(2)
        + collection(1)
        + usage(1)
        + collection(0)
        + _buttons(2, 8)
        + usage_page(1)
        + usage(0x30)
        + usage(0x31)
        + lmin(0)
        + lmax(4095)
        + rsize(16)
        + rcount(2)
        + inp(0x02)
        + end()
        + end()
    )


# ---------------------------------------------------------------- report packing
def put_bits(buf: bytearray, off: int, size: int, v: int) -> None:
    v &= (1 << size) - 1
    for k in range(size):
        i = off + k
        if (v >> k) & 1:
            buf[i >> 3] |= 1 << (i & 7)
        else:
            buf[i >> 3] &= ~(1 << (i & 7)) & 0xFF


def get_bits(buf: bytes, off: int, size: int, signed: bool = False) -> int:
    v = 0
    for k in range(size):
        i = off + k
        v |= ((buf[i >> 3] >> (i & 7)) & 1) << k
    if signed and v >= 1 << (size - 1):
        v -= 1 << size
    return v


def pack_boot(buttons: int, dx: int, dy: int, wheel: int = 0) -> bytes:
    return struct.pack("<Bbb", buttons & 7, dx, dy)


def pack_m16(buttons: int, dx: int, dy: int, wheel: int = 0, pan: int = 0) -> bytes:
    return struct.pack("<Bhhbb", buttons & 0x1F, dx, dy, wheel, pan)


def pack_logi(buttons: int, dx: int, dy: int, wheel: int = 0, pan: int = 0, xy_bits: int = 12) -> bytes:
    n = 8 + 16 + 2 * xy_bits + 16
    b = bytearray((n + 7) // 8)
    b[0] = 1
    put_bits(b, 8, 16, buttons)
    put_bits(b, 24, xy_bits, dx)
    put_bits(b, 24 + xy_bits, xy_bits, dy)
    put_bits(b, 24 + 2 * xy_bits, 8, wheel)
    put_bits(b, 32 + 2 * xy_bits, 8, pan)
    return bytes(b)


# ---------------------------------------------------------------- descriptors
def device_desc(vid: int, pid: int, bcd: int = 0x0100, mps0: int = 64, n_conf: int = 1, cls: int = 0, bcd_usb: int = 0x0200) -> bytes:
    return struct.pack("<BBHBBBBHHHBBBB", 18, 1, bcd_usb, cls, 0, 0, mps0, vid, pid, bcd, 1, 2, 3, n_conf)


@dataclass
class Eps:
    addr: int
    attr: int = 3
    mps: int = 8
    interval: int = 1


@dataclass
class IfaceSpec:
    num: int
    cls: int = 3
    sub: int = 1
    proto: int = 2
    rd: bytes = b""
    eps: list = field(default_factory=list)
    alt: int = 0


def config_desc(ifaces: list[IfaceSpec], max_power: int = 50, attrs: int = 0xA0, value: int = 1) -> bytes:
    body = b""
    for f in ifaces:
        body += struct.pack("<BBBBBBBBB", 9, 4, f.num, f.alt, len(f.eps), f.cls, f.sub, f.proto, 0)
        if f.cls == 3:
            body += struct.pack("<BBHBBBH", 9, 0x21, 0x0111, 0, 1, 0x22, len(f.rd))
        for e in f.eps:
            body += struct.pack("<BBBBHB", 7, 5, e.addr, e.attr, e.mps, e.interval)
    return struct.pack("<BBHBBBBB", 9, 2, 9 + len(body), len(ifaces), value, 0, attrs, max_power) + body


def string_desc(s: str) -> bytes:
    raw = s.encode("utf-16-le")
    return bytes([2 + len(raw), 3]) + raw


class SimMouse:
    """A USB mouse (possibly composite) as seen by the bridge's host port: descriptors, control-request behavior, interrupt endpoints."""

    def __init__(
        self,
        kind: str = "boot",
        vid: int = 0x046D,
        pid: int = 0xC077,
        strings: Optional[dict] = None,
        interval: int = 1,
        speed: str = "FS",
        hidpp: bool = False,
        kbd: bool = False,
        bulk: bool = False,
        buttons: int = 16,
        xy_bits: int = 12,
        n_conf: int = 1,
        alt: bool = False,
        bcd: int = 0x0100,
        mps0: int = 64,
    ) -> None:
        self.kind, self.speed, self.xy_bits, self.interval = kind, speed, xy_bits, interval
        rd = {
            "boot": boot_mouse_rd(),
            "m16": mouse16_rd(),
            "logi": logi_mouse_rd(buttons, xy_bits),
            "abs": abs_pointer_rd(),
            "vendor": hidpp_rd(),
        }[kind]
        sub, proto = (1, 2) if kind in ("boot", "m16") else (0, 0)
        ifaces = [IfaceSpec(0, 3, sub, proto, rd, [Eps(0x81, 3, 16 if kind == "logi" else 8, interval)])]
        if hidpp:
            ifaces.append(IfaceSpec(1, 3, 0, 0, hidpp_rd(), [Eps(0x82, 3, 20, 1), Eps(0x02, 3, 20, 1)]))
        if kbd:
            ifaces.append(IfaceSpec(len(ifaces), 3, 1, 1, kbd_rd(), [Eps(0x80 | (len(ifaces) + 1), 3, 8, 8)]))
        if bulk:
            ifaces.append(IfaceSpec(len(ifaces), 8, 6, 0x50, b"", [Eps(0x81 + len(ifaces) + 1, 2, 64, 0), Eps(len(ifaces) + 3, 2, 64, 0)]))
        if alt:
            ifaces.append(IfaceSpec(0, 3, 0, 0, b"", [], alt=1))
        self.ifaces = ifaces
        self.rds = {f.num: f.rd for f in ifaces if f.cls == 3 and f.alt == 0}
        self.dev = device_desc(vid, pid, bcd, mps0, n_conf)
        self.cfg = config_desc(ifaces)
        self.langs = (0x0409, 0x0419)
        self.strings = strings if strings is not None else {1: "Logitech", 2: "USB Receiver", 3: "A1B2C3"}
        self.config = 0
        self.protocol = {f.num: 1 for f in ifaces}
        self.idle = {f.num: 0 for f in ifaces}
        self.reports: dict[tuple[int, int, int], bytes] = {
            (0, 1, 3): b"\x01\x02",
            (1, 0x20, 3): b"\x00\x11\x22",
        }  # (iface, id, type) -> data
        self.vendor_log: list = []
        self.out_log: list = []
        self.in_q: dict[int, list[bytes]] = {}
        self.fail_control = False  # when True every control request times out (the proxy has to survive that)
        self.control_log: list = []
        self._factory = dict(kind=kind, vid=vid, pid=pid)

    def reset_state(self) -> None:
        """Back to the state of a freshly plugged mouse (what a new enumeration sees)."""
        self.config = 0
        self.protocol = {f.num: 1 for f in self.ifaces}
        self.idle = {f.num: 0 for f in self.ifaces}
        self.reports = {(0, 1, 3): b"\x01\x02", (1, 0x20, 3): b"\x00\x11\x22"}
        self.vendor_log, self.out_log, self.control_log = [], [], []
        self.in_q = {}

    # -- control endpoint
    def control(self, bm: int, req: int, value: int, index: int, length: int, data: bytes = b"") -> tuple[int, bytes]:
        if self.fail_control:
            raise TimeoutError("mouse did not answer")
        r = self._control(bm, req, value, index, length, data)
        self.control_log.append((bm, req, value, index, length, data, r))
        return r

    def _control(self, bm: int, req: int, value: int, index: int, length: int, data: bytes) -> tuple[int, bytes]:
        typ, idx = value >> 8, value & 0xFF
        if bm == 0x80 and req == 6:
            if typ == 1:
                return OK, self.dev[:length]
            if typ == 2 and idx == 0:
                return OK, self.cfg[:length]
            if typ == 3:
                if idx == 0:
                    return OK, (bytes([2 + 2 * len(self.langs), 3]) + b"".join(struct.pack("<H", x) for x in self.langs))[:length]
                if index in self.langs and idx in self.strings:
                    return OK, string_desc(self.strings[idx])[:length]
            return STALL, b""
        if bm == 0x81 and req == 6:
            if typ == 0x22 and (index & 0xFF) in self.rds:
                return OK, self.rds[index & 0xFF][:length]
            if typ == 0x21:
                off = self._hid_offset(index & 0xFF)
                if off is not None:
                    return OK, self.cfg[off : off + 9][:length]
            return STALL, b""
        if bm == 0x80 and req == 0:
            return OK, b"\x00\x00"[:length]
        if bm == 0x80 and req == 8:
            return OK, bytes([self.config])[:length]
        if bm == 0x00 and req == 9:
            self.config = value & 0xFF
            for k in self.protocol:
                self.protocol[k] = 1
            return OK, b""
        if bm == 0x00 and req == 3:
            return OK, b""
        if (bm & 0x60) == 0x20:  # HID class requests to an interface
            i = index & 0xFF
            if i not in self.protocol:
                return STALL, b""
            if bm == 0xA1 and req == 1:
                d = self.reports.get((i, value & 0xFF, value >> 8))
                return (OK, d[:length]) if d is not None else (STALL, b"")
            if bm == 0x21 and req == 9:
                self.reports[(i, value & 0xFF, value >> 8)] = bytes(data)
                return OK, b""
            if bm == 0xA1 and req == 2:
                return OK, bytes([self.idle[i]])[:length]
            if bm == 0x21 and req == 0x0A:
                self.idle[i] = value >> 8
                return OK, b""
            sub = next(f.sub for f in self.ifaces if f.num == i and f.alt == 0)
            if bm == 0xA1 and req == 3 and sub == 1:
                return OK, bytes([self.protocol[i]])[:length]
            if bm == 0x21 and req == 0x0B and sub == 1:
                self.protocol[i] = value & 1
                return OK, b""
            return STALL, b""
        if bm == 0xC0:  # vendor request, device to host
            return OK, bytes((k * 7 + req + value) & 0xFF for k in range(length))
        if bm == 0x40:
            self.vendor_log.append((req, value, index, bytes(data)))
            return OK, b""
        return STALL, b""

    def _hid_offset(self, ifnum: int) -> Optional[int]:
        i = 9
        cur = None
        while i + 2 <= len(self.cfg):
            n, t = self.cfg[i], self.cfg[i + 1]
            if t == 4:
                cur = self.cfg[i + 2]
            if t == 0x21 and cur == ifnum:
                return i
            i += n
        return None

    # -- interrupt endpoints
    def push_in(self, ep: int, report: bytes) -> None:
        self.in_q.setdefault(ep, []).append(bytes(report))

    def pop_in(self, ep: int) -> Optional[bytes]:
        q = self.in_q.get(ep)
        return q.pop(0) if q else None

    def interrupt_out(self, ep: int, data: bytes) -> None:
        self.out_log.append((ep, bytes(data)))
        if ep == 0x02 and data and data[0] in (0x10, 0x11):  # HID++-shaped request: answer with the same report, marked
            self.push_in(0x82, bytes([data[0], data[1], data[2] ^ 0xFF]) + bytes(data[3:]))

    # -- the person's motion, in this mouse's report layout
    def pack(self, buttons: int, dx: int, dy: int, wheel: int = 0, pan: int = 0) -> bytes:
        if self.kind == "boot":
            return pack_boot(buttons, dx, dy)
        if self.kind == "m16":
            return pack_m16(buttons, dx, dy, wheel, pan)
        if self.kind == "logi":
            return pack_logi(buttons, dx, dy, wheel, pan, self.xy_bits)
        raise ValueError(self.kind)

    @property
    def motion_ep(self) -> int:
        return 0x81

    @property
    def max_count(self) -> int:
        bits = {"boot": 8, "m16": 16, "logi": self.xy_bits}[self.kind]
        return (1 << (bits - 1)) - 1


# ---------------------------------------------------------------- the PC's enumeration
class SimPC:
    """Enumerates whatever answers control requests (a mouse directly, or the bridge in front of it) and records the transcript."""

    @staticmethod
    def enumerate(dev) -> list[tuple]:
        t: list[tuple] = []

        def ctl(bm, req, value, index, length, data=b""):
            try:
                st, d = dev.control(bm, req, value, index, length, data)
            except TimeoutError:
                st, d = 2, b""
            t.append((bm, req, value, index, length, bytes(data), st, bytes(d)))
            return st, d

        _, d8 = ctl(0x80, 6, 0x0100, 0, 8)
        _, d = ctl(0x80, 6, 0x0100, 0, 18)
        ctl(0x80, 6, 0x0200, 0, 9)
        _, c9 = ctl(0x80, 6, 0x0200, 0, 9)
        total = struct.unpack_from("<H", c9, 2)[0] if len(c9) >= 4 else 0
        _, cfg = ctl(0x80, 6, 0x0200, 0, total)
        ctl(0x80, 6, 0x0300, 0, 255)
        for lang in (0x0409, 0x0419, 0x0407):
            for idx in (1, 2, 3, 4, 0xEE):
                ctl(0x80, 6, 0x0300 | idx, lang, 255)
        ctl(0x80, 6, 0x0600, 0, 10)  # device qualifier
        ctl(0x80, 6, 0x0F00, 0, 5)  # BOS
        ctl(0x80, 0, 0, 0, 2)
        ifaces = []
        i = 9
        cur = None
        while i + 2 <= len(cfg):
            n, ty = cfg[i], cfg[i + 1]
            if n < 2:
                break
            if ty == 4:
                cur = cfg[i + 2]
                alt = cfg[i + 3]
                ifaces.append([cur, cfg[i + 5], cfg[i + 6], cfg[i + 7], 0, alt])
            if ty == 0x21 and ifaces:
                ifaces[-1][4] = struct.unpack_from("<H", cfg, i + 7)[0]
            i += n
        for num, cls, _sub, _proto, rdlen, alt in ifaces:
            if cls == 3 and alt == 0:
                ctl(0x81, 6, 0x2100, num, 9)
                ctl(0x81, 6, 0x2200, num, rdlen)
        ctl(0x00, 9, 1, 0, 0)
        for num, cls, _sub, _proto, _rdlen, alt in ifaces:
            if cls == 3 and alt == 0:
                ctl(0x21, 0x0A, 0, num, 0)
                ctl(0xA1, 2, 0, num, 1)
                ctl(0xA1, 3, 0, num, 1)
                ctl(0x21, 0x0B, 0, num, 0)  # boot protocol (BIOS style), then back
                ctl(0xA1, 3, 0, num, 1)
                ctl(0x21, 0x0B, 1, num, 0)
                ctl(0xA1, 1, 0x0301, num, 8)  # get feature report 1 (a mouse may stall)
                ctl(0xA1, 1, 0x0320, num, 8)  # feature report 0x20 (HID++ shaped)
                ctl(0x21, 9, 0x0220, num, 3, b"\x01\x02\x03")
                ctl(0x21, 9, 0x0210, num, 6, b"\x10\x01\x00\x00\x00\x00")
        ctl(0xC0, 0x01, 0x0005, 0, 16)  # vendor requests both ways
        ctl(0x40, 0x02, 0x1234, 0x0001, 4, b"\xde\xad\xbe\xef")
        ctl(0x80, 8, 0, 0, 1)
        return t
