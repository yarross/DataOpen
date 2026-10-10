"""CtlLink v1: the byte protocol between the phone client and the control gateway on the compute module (docs/PWA.md).

Transport-independent. Over BLE every message is cut into GATT-sized chunks (`ctl_in` write, `ctl_out` notify); over a WebSocket the same
chunks travel as binary frames. Three layers:

  message   ver u8 | type u8 | req u16 | len u16 | flags u16 | body | crc32 u32 (zlib, over everything before it), little endian
  chunk     1 header byte (0x80 first, 0x40 last, low 6 bits a sequence number) + up to `chunk - 1` message bytes
  status    a FIXED 20-byte snapshot (the `status` characteristic) that fits the default 23-byte ATT MTU, so the safety shell can always
            read it without any negotiation and without the manifest

`dataopen ctl constants` writes pwa/js/constants.js from this module and `dataopen ctl golden` writes the cross-language test vectors,
so the browser client and this module cannot drift apart silently.
"""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

from ..bridge.cbridge import REASONS

VER = 1
HDR = 8
MAX_BODY = 16384
MAX_MSG = HDR + MAX_BODY + 4
CHUNK_MIN, CHUNK_MAX, CHUNK_DEFAULT = 20, 244, 20
REASM_TIMEOUT_US = 5_000_000

# GATT. Private random 128-bit UUIDs (nothing is registered with the Bluetooth SIG; placeholders to regenerate per product line).
UUID_SERVICE = "d0a7e100-0001-4d0e-9c3a-5e1f00a51570"
UUID_INFO = "d0a7e100-0002-4d0e-9c3a-5e1f00a51570"
UUID_STATUS = "d0a7e100-0003-4d0e-9c3a-5e1f00a51570"
UUID_CTL_IN = "d0a7e100-0004-4d0e-9c3a-5e1f00a51570"
UUID_CTL_OUT = "d0a7e100-0005-4d0e-9c3a-5e1f00a51570"

# message types
T_HELLO, T_HELLO_R = 0x01, 0x02
T_GET, T_DATA = 0x10, 0x11
T_SET, T_ACT, T_CONFIRM, T_BUNDLE_PUT = 0x20, 0x21, 0x22, 0x23
T_STOP, T_HARD_BYPASS = 0x24, 0x25
T_FW_BEGIN, T_FW_CHUNK, T_FW_END = 0x26, 0x27, 0x28
T_PKG_BEGIN, T_PKG_CHUNK, T_PKG_END = 0x29, 0x2A, 0x2B      # channel B: sealed user packages (docs/UPDATES.md); never mixed with FW_*
T_ACK, T_ERR = 0x30, 0x31
T_EVENT = 0x40
T_PING, T_PONG = 0x7E, 0x7F
TYPES = {k: v for k, v in dict(HELLO=T_HELLO, HELLO_R=T_HELLO_R, GET=T_GET, DATA=T_DATA, SET=T_SET, ACT=T_ACT, CONFIRM=T_CONFIRM,
                               BUNDLE_PUT=T_BUNDLE_PUT, STOP=T_STOP, HARD_BYPASS=T_HARD_BYPASS, FW_BEGIN=T_FW_BEGIN, FW_CHUNK=T_FW_CHUNK,
                               FW_END=T_FW_END, PKG_BEGIN=T_PKG_BEGIN, PKG_CHUNK=T_PKG_CHUNK, PKG_END=T_PKG_END,
                               ACK=T_ACK, ERR=T_ERR, EVENT=T_EVENT, PING=T_PING, PONG=T_PONG).items()}
# what may be sent before HELLO: the safety actions must never wait for anything
NO_SESSION_OK = (T_STOP, T_HARD_BYPASS, T_PING, T_HELLO)

# error codes carried by T_ERR; every code has an i18n key `err.<name>` in the client
ERRORS = dict(BAD_MSG=1, BAD_KEY=2, BAD_VALUE=3, BUSY=4, NOT_ALLOWED=5, BAD_BUNDLE=6, NO_PROFILE=7, TOO_BIG=8, NO_SESSION=9, UNSUPPORTED=10,
              PHYSICAL=11, BAD_VERSION=12, WRONG_DEVICE=13, BAD_SIGNATURE=14, REPLAY=15, PLAIN_REFUSED=16, FW_REJECTED=17, PKG_REJECTED=18,
              RESIDENT=19)
E = type("E", (), ERRORS)

# the `get` targets
GET_MANIFEST, GET_STATE, GET_IDENTITY, GET_SLOTS, GET_FIRMWARE = "manifest", "state", "identity", "slots", "firmware"
GET_PACKAGES = "packages"
# The closed list of things the phone can ask the device to SAY (docs/RESIDENCY.md). There is no 'bundle', no 'profile', no 'model': the
# profile of the hand and the weights of the model never leave the device, so there is nothing of that kind to ask for.
GET_KINDS = (GET_MANIFEST, GET_STATE, GET_IDENTITY, GET_SLOTS, GET_FIRMWARE, GET_PACKAGES)
# what a file control of the manifest can do: carry a file TO the device, or save the device's public card. Nothing else leaves the device.
FILE_OPS_IN = ("bundle_put", "fw_put", "pkg_put")
FILE_OPS_OUT = ("card_get",)
FILE_OPS = FILE_OPS_IN + FILE_OPS_OUT
# hardware slots (docs/SLOTS.md); the status carries the active one and a bit per slot that holds a profile
SLOT_COUNT = 4
# data bytes per FW_CHUNK message (the message body limit is 16 KB; BLE cuts it into small writes anyway)
FW_CHUNK_MAX = 8192
PKG_CHUNK_MAX = 8192

# status flags (u16)
SF_LATCH_SOFT, SF_LATCH_HW, SF_ASSIST_WANTED, SF_CALIBRATING = 1, 2, 4, 8
SF_PARAMS_OK, SF_TRIAL, SF_PC_ATTACHED, SF_LINK_OK = 16, 32, 64, 128
SF_HEALTHY, SF_BRIDGE_SEEN = 256, 512
STATUS_FLAGS = dict(LATCH_SOFT=SF_LATCH_SOFT, LATCH_HW=SF_LATCH_HW, ASSIST_WANTED=SF_ASSIST_WANTED, CALIBRATING=SF_CALIBRATING,
                    PARAMS_OK=SF_PARAMS_OK, TRIAL=SF_TRIAL, PC_ATTACHED=SF_PC_ATTACHED, LINK_OK=SF_LINK_OK, HEALTHY=SF_HEALTHY,
                    BRIDGE_SEEN=SF_BRIDGE_SEEN)
# ready bits (u8)
RB_ASC, RB_TREMOR, RB_TREMOR_NOT_NEEDED = 1, 2, 4
MODES = {0: "HW_BYPASS", 1: "PROBE", 2: "PASSTHRU", 3: "ASSIST"}   # the bridge's fail-safe states
MODE_UNKNOWN = 255
LEVEL_UNKNOWN = 255

STATUS_FMT = "<BBBBHBBHHHHBBBB"
STATUS_SIZE = struct.calcsize(STATUS_FMT)
INFO_FMT = "<BB4s4sHHH"
INFO_SIZE = struct.calcsize(INFO_FMT)
assert STATUS_SIZE == 20 and INFO_SIZE == 16, (STATUS_SIZE, INFO_SIZE)


def crc32(b: bytes) -> int:
    return zlib.crc32(b) & 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------------------------- messages
@dataclass
class Message:
    type: int
    req: int = 0
    body: bytes = b""
    flags: int = 0

    def json(self) -> dict:
        v = json.loads(self.body.decode("utf-8")) if self.body else {}
        if not isinstance(v, dict):
            raise ValueError("a JSON object was expected")
        return v


def pack_message(type_: int, req: int = 0, body: bytes = b"", flags: int = 0) -> bytes:
    if len(body) > MAX_BODY:
        raise ValueError("message body too long")
    head = struct.pack("<BBHHH", VER, type_ & 0xFF, req & 0xFFFF, len(body), flags & 0xFFFF)
    raw = head + body
    return raw + struct.pack("<I", crc32(raw))


def pack_json(type_: int, req: int = 0, obj: Optional[dict] = None) -> bytes:
    return pack_message(type_, req, json.dumps(obj or {}, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def unpack_message(raw: bytes) -> Optional[Message]:
    """None for anything that is not a valid message (size, version, length or CRC wrong)."""
    if len(raw) < HDR + 4 or len(raw) > MAX_MSG:
        return None
    ver, typ, req, n, flags = struct.unpack_from("<BBHHH", raw)
    if ver != VER or n > MAX_BODY or len(raw) != HDR + n + 4:
        return None
    if struct.unpack_from("<I", raw, HDR + n)[0] != crc32(raw[: HDR + n]):
        return None
    return Message(typ, req, bytes(raw[HDR : HDR + n]), flags)


# ---------------------------------------------------------------------------------------------------------------- chunks
def chunk_message(raw: bytes, size: int = CHUNK_DEFAULT, seq0: int = 0) -> tuple[list[bytes], int]:
    """Cut a message into chunks of at most `size` bytes each (header included). Returns the chunks and the next sequence number."""
    if not CHUNK_MIN <= size <= CHUNK_MAX:
        raise ValueError(f"chunk size must be {CHUNK_MIN}..{CHUNK_MAX}")
    room = size - 1
    parts = [raw[i : i + room] for i in range(0, len(raw), room)] or [b""]
    out, seq = [], seq0 & 0x3F
    for i, p in enumerate(parts):
        out.append(bytes([(0x80 if i == 0 else 0) | (0x40 if i == len(parts) - 1 else 0) | seq]) + p)
        seq = (seq + 1) & 0x3F
    return out, seq


class Reassembler:
    """Chunks in, whole (not yet CRC-checked) messages out. Bounded memory, bounded wait: a lost chunk can never wedge the link."""

    def __init__(self, timeout_us: int = REASM_TIMEOUT_US) -> None:
        self.timeout_us = timeout_us
        self.buf = bytearray()
        self.active = False
        self.expect = 0
        self.last_us = 0
        self.dropped = 0

    def reset(self) -> None:
        self.buf.clear()
        self.active = False

    def feed(self, chunk: bytes, now_us: int = 0) -> Optional[bytes]:
        if not chunk:
            self.dropped += 1
            return None
        if self.active and now_us - self.last_us > self.timeout_us:
            self.reset()
            self.dropped += 1
        h = chunk[0]
        first, last, seq = bool(h & 0x80), bool(h & 0x40), h & 0x3F
        if first:
            if self.active:
                self.dropped += 1
            self.buf.clear()
            self.active = True
        elif not self.active or seq != self.expect:
            self.reset()
            self.dropped += 1
            return None
        self.buf += chunk[1:]
        self.expect = (seq + 1) & 0x3F
        self.last_us = now_us
        if len(self.buf) > MAX_MSG:
            self.reset()
            self.dropped += 1
            return None
        if last:
            out = bytes(self.buf)
            self.reset()
            return out
        return None


# ---------------------------------------------------------------------------------------------------------------- status / info
@dataclass
class StatusSnapshot:
    bridge: int = 0                    # 0 the gateway has not heard the bridge yet, 1 it has
    mode: int = MODE_UNKNOWN           # the bridge's fail-safe state (MODES) or MODE_UNKNOWN
    reason: int = 0                    # index into REASONS
    flags: int = 0
    fill: int = 0                      # profile progress, 0..100
    ready: int = 0                     # RB_* bits
    state_rev: int = 0
    manifest_rev: int = 0
    uptime_min: int = 0                # minutes since the gateway started (u16: 45 days; a status is a glance, not a log)
    trial_left_s: int = 0
    strength: int = LEVEL_UNKNOWN
    tremor: int = LEVEL_UNKNOWN
    slot: int = 0                      # the active hardware slot, 0..SLOT_COUNT-1
    slot_mask: int = 0                 # bit k set: slot k holds a profile

    def pack(self) -> bytes:
        return struct.pack(STATUS_FMT, VER, self.bridge & 0xFF, self.mode & 0xFF, self.reason & 0xFF, self.flags & 0xFFFF, self.fill & 0xFF,
                           self.ready & 0xFF, self.state_rev & 0xFFFF, self.manifest_rev & 0xFFFF, min(self.uptime_min, 0xFFFF),
                           min(self.trial_left_s, 0xFFFF), self.strength & 0xFF, self.tremor & 0xFF, self.slot & 0xFF,
                           self.slot_mask & 0xFF)

    @staticmethod
    def unpack(b: bytes) -> Optional["StatusSnapshot"]:
        if len(b) != STATUS_SIZE or b[0] != VER:
            return None
        _, br, mode, reason, flags, fill, ready, rev, mrev, up, trial, st, tr, slot, mask = struct.unpack(STATUS_FMT, b)
        return StatusSnapshot(br, mode, reason, flags, fill, ready, rev, mrev, up, trial, st, tr, slot, mask)

    @property
    def reason_name(self) -> str:
        return REASONS[self.reason] if self.reason < len(REASONS) else f"R{self.reason}"


@dataclass
class Info:
    caps: int = 0
    device_id: bytes = b"\0\0\0\0"
    manifest_hash: bytes = b"\0\0\0\0"
    manifest_rev: int = 0
    chunk_hint: int = CHUNK_DEFAULT

    def pack(self) -> bytes:
        return struct.pack(INFO_FMT, VER, self.caps, self.device_id, self.manifest_hash, self.manifest_rev, self.chunk_hint, 0)

    @staticmethod
    def unpack(b: bytes) -> Optional["Info"]:
        if len(b) != INFO_SIZE or b[0] != VER:
            return None
        _, caps, did, mh, rev, hint, _ = struct.unpack(INFO_FMT, b)
        return Info(caps, did, mh, rev, hint)


# ---------------------------------------------------------------------------------------------------------------- generated artifacts
def constants_js() -> str:
    """pwa/js/constants.js: every number and name the browser client shares with this module."""
    def obj(d: dict) -> str:
        return "{ " + ", ".join(f"{k}: {json.dumps(v)}" for k, v in d.items()) + " }"
    lines = [
        "// Generated by `dataopen ctl constants` from src/dataopen/ctl/protocol.py: do not edit.",
        f"export const VER = {VER};",
        f"export const HDR = {HDR};",
        f"export const MAX_BODY = {MAX_BODY};",
        f"export const MAX_MSG = {MAX_MSG};",
        f"export const CHUNK_MIN = {CHUNK_MIN};",
        f"export const CHUNK_MAX = {CHUNK_MAX};",
        f"export const CHUNK_DEFAULT = {CHUNK_DEFAULT};",
        f"export const REASM_TIMEOUT_MS = {REASM_TIMEOUT_US // 1000};",
        f"export const SLOT_COUNT = {SLOT_COUNT};",
        f"export const FW_CHUNK_MAX = {FW_CHUNK_MAX};",
        f"export const PKG_CHUNK_MAX = {PKG_CHUNK_MAX};",
        f"export const STATUS_SIZE = {STATUS_SIZE};",
        f"export const INFO_SIZE = {INFO_SIZE};",
        f"export const MODE_UNKNOWN = {MODE_UNKNOWN};",
        f"export const LEVEL_UNKNOWN = {LEVEL_UNKNOWN};",
        "export const UUID = " + obj(dict(service=UUID_SERVICE, info=UUID_INFO, status=UUID_STATUS,
                                          ctlIn=UUID_CTL_IN, ctlOut=UUID_CTL_OUT)) + ";",
        f"export const T = {obj(TYPES)};",
        f"export const NO_SESSION_OK = {json.dumps(list(NO_SESSION_OK))};",
        f"export const ERR = {obj(ERRORS)};",
        f"export const GET_KINDS = {json.dumps(list(GET_KINDS))};",
        f"export const FILE_OPS = {json.dumps(list(FILE_OPS))};",
        f"export const SF = {obj(STATUS_FLAGS)};",
        f"export const RB = {obj(dict(ASC=RB_ASC, TREMOR=RB_TREMOR, TREMOR_NOT_NEEDED=RB_TREMOR_NOT_NEEDED))};",
        f"export const MODES = {json.dumps(MODES)};",
        f"export const REASONS = {json.dumps(REASONS)};",
        "",
    ]
    return "\n".join(lines)


def golden() -> dict:
    """Cross-language test vectors (pwa/tests/golden.json): the browser client must produce and parse exactly these bytes."""
    body_big = bytes((i * 7 + 3) & 0xFF for i in range(700))
    msgs = [
        ("hello", pack_json(T_HELLO, 1, {"v": 1, "chunk": 100, "lang": "ru"})),
        ("set", pack_json(T_SET, 7, {"key": "assist.strength", "value": 7})),
        ("stop", pack_message(T_STOP, 2)),
        ("hard_bypass", pack_message(T_HARD_BYPASS, 3)),
        ("slot", pack_json(T_SET, 8, {"key": "slot.active", "value": 2})),
        ("fw_chunk", pack_message(T_FW_CHUNK, 10, struct.pack("<I", 4096) + bytes(range(64)))),
        ("pkg_chunk", pack_message(T_PKG_CHUNK, 11, struct.pack("<I", 8192) + bytes(range(64)))),
        ("utf8", pack_json(T_EVENT, 0, {"msg": "Помощь выключена"})),
        ("big", pack_message(T_DATA, 9, body_big)),
    ]
    vectors = (b"", b"123456789", bytes(range(256)))
    out = {"crc32": [{"data": d.hex(), "crc": crc32(d)} for d in vectors], "messages": [], "status": [], "info": []}
    for name, raw in msgs:
        m = unpack_message(raw)
        assert m is not None
        entry = {"name": name, "hex": raw.hex(), "type": m.type, "req": m.req, "body": m.body.hex(), "chunks": {}}
        for size in (20, 100, 244):
            ch, _ = chunk_message(raw, size, 5)
            entry["chunks"][str(size)] = [c.hex() for c in ch]
        out["messages"].append(entry)
    flags = SF_ASSIST_WANTED | SF_LINK_OK | SF_HEALTHY
    samples = (StatusSnapshot(), StatusSnapshot(1, 3, 0, flags, 70, RB_ASC | RB_TREMOR, 513, 4, 1440, 12, 6, 4, 2, 0b0111),
               StatusSnapshot(1, 2, 8, SF_LATCH_SOFT | SF_PC_ATTACHED, 100, RB_ASC | RB_TREMOR_NOT_NEEDED, 65535, 65535, 0xFFFF,
                              20, 0, 10, 3, 0b1000))
    for s in samples:
        out["status"].append({"hex": s.pack().hex(), **{k: getattr(s, k) for k in s.__dataclass_fields__}})
    out["info"].append({"hex": Info(3, b"\x01\x02\x03\x04", b"\xde\xad\xbe\xef", 4, 100).pack().hex(), "caps": 3, "device_id": "01020304",
                        "manifest_hash": "deadbeef", "manifest_rev": 4, "chunk_hint": 100})
    return out
