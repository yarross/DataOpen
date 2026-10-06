"""BridgeLink v1, the Python side: 128-byte SPI frames, parameter blobs, scene, commands, telemetry and status parsing.
Mirrors csrc/link.h; tests compare both implementations (CRC, round trips, bit flips)."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, fields
from typing import Iterable, Optional

from ..assist.fixed import FixedParams
from ..assist.tremor_fixed import FixedTremorParams

FRAME, PAYLOAD, SYNC, VER = 128, 112, 0xB7, 1
LK_NOP, LK_TELEM, LK_STATUS, LK_TSYNC_REPLY = 0, 1, 2, 3
LK_HELLO, LK_PARAMS_ASC, LK_PARAMS_TREMOR, LK_SCENE, LK_CMD, LK_TSYNC = 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
CMD_PASSTHRU, CMD_ASSIST, CMD_BYPASS = 0, 1, 2
ASC_BLOB, TRM_BLOB = 16 + 152 + 4, 16 + 60 + 4
ONE = 65536


@dataclass
class Frame:
    kind: int
    payload: bytes = b""
    seq: int = 0
    ack: int = 0
    flags: int = 0
    frag_idx: int = 0
    frag_cnt: int = 1


def pack_frame(f: Frame) -> bytes:
    if len(f.payload) > PAYLOAD:
        raise ValueError("payload too long")
    head = struct.pack("<BBBBHHHBB", SYNC, VER, f.kind, f.flags, f.seq, f.ack, len(f.payload), f.frag_idx, f.frag_cnt)
    body = head + f.payload.ljust(PAYLOAD, b"\0")
    return body + struct.pack("<I", zlib.crc32(body))


def unpack_frame(b: bytes) -> Optional[Frame]:
    """None for anything that is not a valid frame (wrong size, sync, version, CRC or length)."""
    if len(b) != FRAME or b[0] != SYNC or b[1] != VER:
        return None
    if struct.unpack_from("<I", b, 124)[0] != zlib.crc32(b[:124]):
        return None
    _, _, kind, flags, seq, ack, n, fi, fc = struct.unpack_from("<BBBBHHHBB", b)
    if n > PAYLOAD:
        return None
    return Frame(kind, bytes(b[12 : 12 + n]), seq, ack, flags, fi, fc)


def asc_blob_bytes(p: FixedParams) -> bytes:
    return struct.pack("<38i", *[getattr(p, f.name) for f in fields(FixedParams)])


def tremor_blob_bytes(p: FixedTremorParams) -> bytes:
    return struct.pack("<15i", *[getattr(p, f.name) for f in fields(FixedTremorParams)])


def make_blob(generation: int, profile_id: int, ppc_q16: int, data: bytes) -> bytes:
    body = struct.pack("<IIII", generation & 0xFFFFFFFF, profile_id & 0xFFFFFFFF, ppc_q16 & 0xFFFFFFFF, 0) + data
    return body + struct.pack("<I", zlib.crc32(body))


def blob_frames(kind: int, blob: bytes, serial: int, seq0: int = 0) -> list[Frame]:
    chunks = [blob[i : i + PAYLOAD] for i in range(0, len(blob), PAYLOAD)]
    return [Frame(kind, c, seq=(seq0 + i) & 0xFFFF, flags=serial & 15, frag_idx=i, frag_cnt=len(chunks)) for i, c in enumerate(chunks)]


def asc_frames(p: FixedParams, generation: int, profile_id: int = 1, ppc: float = 1.0, serial: int = 0, seq0: int = 0) -> list[Frame]:
    return blob_frames(LK_PARAMS_ASC, make_blob(generation, profile_id, int(round(ppc * ONE)), asc_blob_bytes(p)), serial, seq0)


def tremor_frames(p: FixedTremorParams, generation: int, profile_id: int = 1, serial: int = 0, seq0: int = 0) -> list[Frame]:
    return blob_frames(LK_PARAMS_TREMOR, make_blob(generation, profile_id, 0, tremor_blob_bytes(p)), serial, seq0)


@dataclass(frozen=True)
class SceneObject:
    id: int
    x: float  # px relative to the cursor at capture time
    y: float
    radius: float = 0.0
    t_appear_us: Optional[int] = None


def _q(v: float) -> int:
    """Q16.16 saturated to int32 (an object further than +-32767 px from the cursor is simply 'very far')."""
    return max(-2147483647, min(2147483647, int(round(v * ONE))))


def scene_frame(t_capture_us: int, objs: Iterable[SceneObject], seq: int = 0) -> Frame:
    objs = list(objs)
    body = struct.pack("<IB3x", t_capture_us & 0xFFFFFFFF, len(objs))
    for o in objs:
        body += struct.pack(
            "<HBxiiiI",
            o.id & 0xFFFF,
            int(o.t_appear_us is not None),
            _q(o.x),
            _q(o.y),
            _q(max(o.radius, 0.0)),
            (o.t_appear_us or 0) & 0xFFFFFFFF,
        )
    return Frame(LK_SCENE, body, seq=seq)


def cmd_frame(cmd: int, seq: int = 0) -> Frame:
    return Frame(LK_CMD, struct.pack("<B3x", cmd), seq=seq)


def tsync_frame(m_t_us: int, seq: int = 0) -> Frame:
    return Frame(LK_TSYNC, struct.pack("<Q", m_t_us), seq=seq)


@dataclass(frozen=True)
class Sample:
    t_us: int
    raw_dx: int
    raw_dy: int
    out_dx: int
    out_dy: int
    k: float
    buttons: int
    flags: int

    @property
    def gap_before(self) -> bool:
        return bool(self.flags & 0x80)


def parse_telem(f: Frame) -> tuple[list[Sample], int]:
    n = f.payload[0]
    dropped = struct.unpack_from("<H", f.payload, 2)[0]
    out = []
    for i in range(n):
        t, a, b, c, d, k, bt, fl = struct.unpack_from("<I4hHBB", f.payload, 4 + 16 * i)
        out.append(Sample(t, a, b, c, d, k / 32768.0, bt, fl))
    return out, dropped


def parse_status(f: Frame) -> dict:
    p = f.payload
    (
        state,
        reason,
        hw,
        mode,
        flags,
        img,
        nmi,
        gen_asc,
        gen_trm,
        motion,
        other,
        inv,
        lock,
        over,
        rej,
        rxok,
        rxbad,
        drop,
        tmax,
        tavg,
        sage,
        usb,
        vid,
        pid,
        crashes,
        guard,
        t,
    ) = _status_fields(p)
    return {
        "state": state,
        "reason": reason,
        "hw": hw,
        "mode": mode,
        "latch_soft": bool(flags & 1),
        "latch_hw": bool(flags & 2),
        "attach": bool(flags & 4),
        "params_asc_ok": bool(flags & 8),
        "params_trm_ok": bool(flags & 16),
        "link_ok": bool(flags & 32),
        "scene": bool(flags & 64),
        "lockin_hold": bool(flags & 128),
        "chord": bool(flags & 256),
        "healthy": bool(flags & 512),
        "img_code": img,
        "n_motion_if": nmi,
        "gen_asc": gen_asc,
        "gen_tremor": gen_trm,
        "motion_reports": motion,
        "other_reports": other,
        "invariant_viol": inv,
        "lockin_trips": lock,
        "overruns": over,
        "params_rejected": rej,
        "link_rx_ok": rxok,
        "link_rx_bad": rxbad,
        "telem_dropped": drop,
        "tick_max_us": tmax,
        "tick_avg_us": tavg,
        "scene_age_ms": None if sage == 0xFFFF else sage,
        "usb_errors": usb,
        "vid": vid,
        "pid": pid,
        "crashes": crashes,
        "guard": guard,
        "t_us": t,
    }


def _status_fields(p: bytes):
    state, reason, hw, mode = p[0], p[1], p[2], p[3]
    flags = struct.unpack_from("<H", p, 4)[0]
    img, nmi = p[6], p[7]
    (gen_asc, gen_trm, motion, other, inv, lock, over, rej, rxok, rxbad, drop) = struct.unpack_from("<11I", p, 8)
    tmax, tavg, sage, usb, vid, pid, crashes = struct.unpack_from("<7H", p, 52)
    guard = p[66]
    t = struct.unpack_from("<I", p, 68)[0]
    return (
        state,
        reason,
        hw,
        mode,
        flags,
        img,
        nmi,
        gen_asc,
        gen_trm,
        motion,
        other,
        inv,
        lock,
        over,
        rej,
        rxok,
        rxbad,
        drop,
        tmax,
        tavg,
        sage,
        usb,
        vid,
        pid,
        crashes,
        guard,
        t,
    )


def parse_tsync_reply(f: Frame) -> dict:
    m, brx, btx = struct.unpack_from("<QQQ", f.payload)
    return {"m_t": m, "bridge_rx": brx, "bridge_tx": btx}
