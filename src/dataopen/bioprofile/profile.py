"""The profile: what is stored, published and read by other modules.

`BioProfileV1` is a packed little-endian struct of 98 bytes (< 100): magic, version, size, CRC32. Every field is quantized to 8/16/32
bits with a documented unit; `ProfileState` is the convenient Python form, `ProfileView` the read-only accessor other modules use
(it validates magic / version / size / CRC before it hands out a single number).

Field groups (units in `STATS`):
  stats      nine metrics, each (median, sigma, n): T_motor (normal / low visibility), V_max, D_brake, overshoot, phase lag,
             jitter frequency and amplitude, final error
  fatigue    session length, slopes of T_motor and final error over the session (Theil-Sen), their z-scores, baseline T_motor
  counts     episodes seen: flick, tracking, surprise, low-visibility, anticipations (reactions faster than a human can react)
  rates      how often each error scenario happens: overshoot, undershoot, wrong initial direction, miss, lapse (no reaction)
"""
from __future__ import annotations

import ctypes
import math
import zlib
from dataclasses import dataclass, field
from typing import Optional

MAGIC = 0x504F4942                  # 'BIOP' little endian
VERSION = 1


class ProfileError(ValueError):
    pass


class ProfileVersionError(ProfileError):
    """The data is valid but written by a newer version: refuse to read it and never overwrite it."""


@dataclass(frozen=True)
class StatSpec:
    name: str
    unit: str
    scale: float           # stored integer = round(value / scale)
    signed: bool
    doc: str


STATS: tuple[StatSpec, ...] = (
    StatSpec("t_motor", "ms", 0.1, False, "time from target appearance to the first mouse movement (surprise episodes)"),
    StatSpec("t_motor_lowvis", "ms", 0.1, False, "the same under low contrast / smoke / noise"),
    StatSpec("v_max", "deg/s", 0.1, False, "peak angular speed of a wide flick"),
    StatSpec("d_brake", "fraction of the flick", 1e-4, False, "remaining distance when braking starts / flick distance"),
    StatSpec("overshoot", "fraction of the flick", 1e-4, False, "how far past the target the first movement went / flick distance"),
    StatSpec("phase_lag", "ms", 0.1, True, "delay of the mouse velocity behind the target velocity while tracking"),
    StatSpec("jitter_hz", "Hz", 0.01, False, "frequency of micro-corrections while tracking"),
    StatSpec("jitter_amp", "deg", 1e-3, False, "amplitude of micro-corrections while tracking"),
    StatSpec("err_final", "deg", 1e-3, False, "distance from the target when the movement settled"),
)
STAT_INDEX = {s.name: i for i, s in enumerate(STATS)}
COUNTS = ("flick", "track", "surprise", "lowvis", "anticipation")
# error scenarios, stored as slow-moving rates (EWMA over episodes) scaled to 0..255
RATES = ("overshoot", "undershoot", "direction", "miss", "lapse")


def _fields() -> list:
    f: list = [("magic", ctypes.c_uint32), ("version", ctypes.c_uint8), ("flags", ctypes.c_uint8), ("size", ctypes.c_uint16),
               ("profile_id", ctypes.c_uint32), ("generation", ctypes.c_uint32),
               ("deg_per_count", ctypes.c_float), ("latency_comp_us", ctypes.c_uint16)]
    for s in STATS:
        t = ctypes.c_int16 if s.signed else ctypes.c_uint16
        f += [(f"{s.name}_med", t), (f"{s.name}_sig", ctypes.c_uint16), (f"{s.name}_n", ctypes.c_uint8)]
    f += [("session_s", ctypes.c_uint32), ("slope_t", ctypes.c_int16), ("slope_e", ctypes.c_int16), ("z_t", ctypes.c_int8),
          ("z_e", ctypes.c_int8), ("baseline_t", ctypes.c_uint16)]
    f += [(f"n_{c}", ctypes.c_uint16) for c in COUNTS]
    f += [(f"r_{r}", ctypes.c_uint8) for r in RATES]
    f += [("crc32", ctypes.c_uint32)]
    return f


class BioProfileV1(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = _fields()


SIZE = ctypes.sizeof(BioProfileV1)
assert SIZE < 100, SIZE
FLAG_FATIGUE_VALID = 1
FLAG_CLEAN = 2                       # written at a clean shutdown / checkpoint, not mid-update
SLOPE_T_SCALE = 0.1                  # ms / hour per unit
SLOPE_E_SCALE = 1e-3                 # deg / hour per unit
Z_SCALE = 0.1


@dataclass
class Stat:
    median: float = math.nan
    sigma: float = math.nan
    n: int = 0

    @property
    def valid(self) -> bool:
        return self.n > 0 and math.isfinite(self.median)

    def percentile(self, p: float) -> float:
        """Normal approximation median + z_p * sigma (documented as an approximation: reaction times are right-skewed)."""
        from statistics import NormalDist
        return self.median + NormalDist().inv_cdf(min(max(p, 1e-6), 1 - 1e-6)) * (self.sigma if math.isfinite(self.sigma) else 0.0)


@dataclass
class Fatigue:
    session_s: float = 0.0
    slope_t_ms_per_h: float = math.nan
    slope_err_deg_per_h: float = math.nan
    z_t: float = math.nan
    z_err: float = math.nan
    baseline_t_ms: float = math.nan
    valid: bool = False

    @property
    def fatiguing(self) -> bool:
        """Both reaction time and accuracy drift the wrong way with |z| >= 2: a flag, not a diagnosis."""
        return self.valid and self.z_t >= 2.0 and self.z_err >= 2.0


@dataclass
class ProfileState:
    profile_id: int = 0
    generation: int = 0
    deg_per_count: float = 0.0
    latency_comp_us: int = 0
    stats: dict[str, Stat] = field(default_factory=lambda: {s.name: Stat() for s in STATS})
    fatigue: Fatigue = field(default_factory=Fatigue)
    counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in COUNTS})
    rates: dict[str, float] = field(default_factory=lambda: {r: 0.0 for r in RATES})
    clean: bool = False

    def pack(self) -> bytes:
        b = BioProfileV1()
        b.magic, b.version, b.size = MAGIC, VERSION, SIZE
        b.flags = (FLAG_FATIGUE_VALID if self.fatigue.valid else 0) | (FLAG_CLEAN if self.clean else 0)
        b.profile_id, b.generation = self.profile_id & 0xFFFFFFFF, self.generation & 0xFFFFFFFF
        b.deg_per_count, b.latency_comp_us = float(self.deg_per_count), int(min(max(self.latency_comp_us, 0), 65535))
        for s in STATS:
            st = self.stats[s.name]
            lo, hi = (-32768, 32767) if s.signed else (0, 65535)
            if st.valid:
                setattr(b, f"{s.name}_med", _q(st.median / s.scale, lo, hi))
                setattr(b, f"{s.name}_sig", _q((st.sigma if math.isfinite(st.sigma) else 0.0) / s.scale, 0, 65535))
                setattr(b, f"{s.name}_n", min(st.n, 255))
        f = self.fatigue
        b.session_s = int(min(max(f.session_s, 0), 0xFFFFFFFF))
        if f.valid:
            b.slope_t, b.slope_e = _q(f.slope_t_ms_per_h / SLOPE_T_SCALE, -32768, 32767), _q(f.slope_err_deg_per_h / SLOPE_E_SCALE,
                                                                                             -32768, 32767)
            b.z_t, b.z_e = _q(f.z_t / Z_SCALE, -128, 127), _q(f.z_err / Z_SCALE, -128, 127)
            b.baseline_t = _q(f.baseline_t_ms / 0.1, 0, 65535)
        for c in COUNTS:
            setattr(b, f"n_{c}", min(self.counts.get(c, 0), 65535))
        for r in RATES:
            setattr(b, f"r_{r}", _q(self.rates.get(r, 0.0) * 255.0, 0, 255))
        raw = bytearray(bytes(b))
        raw[-4:] = (zlib.crc32(bytes(raw[:-4])) & 0xFFFFFFFF).to_bytes(4, "little")
        return bytes(raw)

    @staticmethod
    def unpack(buf: bytes) -> "ProfileState":
        b = validate(buf)
        st = ProfileState(profile_id=b.profile_id, generation=b.generation, deg_per_count=float(b.deg_per_count),
                          latency_comp_us=b.latency_comp_us, clean=bool(b.flags & FLAG_CLEAN))
        for s in STATS:
            n = getattr(b, f"{s.name}_n")
            if n:
                st.stats[s.name] = Stat(getattr(b, f"{s.name}_med") * s.scale, getattr(b, f"{s.name}_sig") * s.scale, n)
        if b.flags & FLAG_FATIGUE_VALID:
            st.fatigue = Fatigue(float(b.session_s), b.slope_t * SLOPE_T_SCALE, b.slope_e * SLOPE_E_SCALE, b.z_t * Z_SCALE,
                                 b.z_e * Z_SCALE, b.baseline_t * 0.1, True)
        else:
            st.fatigue.session_s = float(b.session_s)
        st.counts = {c: getattr(b, f"n_{c}") for c in COUNTS}
        st.rates = {r: getattr(b, f"r_{r}") / 255.0 for r in RATES}
        return st


def _q(v: float, lo: int, hi: int) -> int:
    if not math.isfinite(v):
        return 0
    return int(min(max(round(v), lo), hi))


def validate(buf: bytes) -> BioProfileV1:
    """What every reader does first: size, magic, CRC, then version. Raises ProfileError (ProfileVersionError for a newer file)."""
    if len(buf) < 12:
        raise ProfileError("too short")
    magic, version, _, size = int.from_bytes(buf[0:4], "little"), buf[4], buf[5], int.from_bytes(buf[6:8], "little")
    if magic != MAGIC:
        raise ProfileError(f"bad magic {magic:#x}")
    if version > VERSION:
        raise ProfileVersionError(f"profile version {version} is newer than this reader ({VERSION})")
    if version != VERSION or size != SIZE or len(buf) != SIZE:
        raise ProfileError(f"unsupported layout: version {version}, size {size}, got {len(buf)} bytes")
    if zlib.crc32(buf[:-4]) & 0xFFFFFFFF != int.from_bytes(buf[-4:], "little"):
        raise ProfileError("CRC mismatch (corrupt or torn write)")
    return BioProfileV1.from_buffer_copy(buf)


class ProfileView:
    """Read-only accessor for other modules. Construction validates the bytes; every number is returned with its sample count so the
    caller can decide how much to trust it (`confident`)."""

    def __init__(self, buf: bytes, age_ms: Optional[float] = None) -> None:
        self._state = ProfileState.unpack(buf)
        self.age_ms = age_ms

    @property
    def profile_id(self) -> int:
        return self._state.profile_id

    @property
    def generation(self) -> int:
        return self._state.generation

    @property
    def deg_per_count(self) -> float:
        """Mouse sensitivity the profile was measured with (degrees per HID count); 0 when unknown."""
        return self._state.deg_per_count

    @property
    def latency_comp_us(self) -> int:
        return self._state.latency_comp_us

    def stat(self, name: str) -> Stat:
        if name not in STAT_INDEX:
            raise KeyError(f"unknown metric {name!r}; known: {', '.join(STAT_INDEX)}")
        s = self._state.stats[name]
        return Stat(s.median, s.sigma, s.n)

    def confident(self, name: str, min_n: int = 12) -> bool:
        s = self._state.stats[name]
        return s.valid and s.n >= min_n and math.isfinite(s.sigma)

    def percentile(self, name: str, p: float) -> Optional[float]:
        s = self._state.stats[name]
        return s.percentile(p) if s.valid else None

    @property
    def fatigue(self) -> Fatigue:
        return self._state.fatigue

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._state.counts)

    @property
    def error_rates(self) -> dict[str, float]:
        """How often each error scenario happens (0..1): overshoot, undershoot, wrong initial direction, miss, lapse."""
        return dict(self._state.rates)

    def as_dict(self) -> dict:
        out = {"profile_id": self.profile_id, "generation": self.generation, "counts": self.counts,
               "error_rates": {k: round(v, 3) for k, v in self.error_rates.items()}, "metrics": {}}
        for s in STATS:
            st = self._state.stats[s.name]
            if st.valid:
                out["metrics"][s.name] = {"median": round(st.median, 4), "sigma": round(st.sigma, 4), "n": st.n, "unit": s.unit}
        f = self.fatigue
        if f.valid:
            out["fatigue"] = {"session_min": round(f.session_s / 60, 1), "t_motor_ms_per_h": round(f.slope_t_ms_per_h, 2),
                              "err_deg_per_h": round(f.slope_err_deg_per_h, 4), "z_t": round(f.z_t, 1), "z_err": round(f.z_err, 1),
                              "fatiguing": f.fatiguing}
        return out


def header_text() -> str:
    """C header so non-Python modules (the SoC side) read the same bytes."""
    names = {ctypes.c_uint32: "uint32_t", ctypes.c_uint16: "uint16_t", ctypes.c_uint8: "uint8_t", ctypes.c_int16: "int16_t",
             ctypes.c_int8: "int8_t", ctypes.c_float: "float"}
    body = "\n".join(f"    {names[t]} {n};" for n, t in BioProfileV1._fields_)
    scales = "\n".join(f"/* {s.name}: {s.unit}, stored = value / {s.scale:g}: {s.doc} */" for s in STATS)
    return f"""/* Generated by dataopen.bioprofile.profile: do not edit. Little endian, packed. Read-only for consumers. */
#ifndef BIOPROFILE_H
#define BIOPROFILE_H
#include <stdint.h>
#define BIOPROFILE_MAGIC 0x{MAGIC:08X}u
#define BIOPROFILE_VERSION {VERSION}
{scales}
typedef struct __attribute__((packed)) {{
{body}
}} bio_profile_v1_t;
_Static_assert(sizeof(bio_profile_v1_t) == {SIZE}, "layout drift");
/* Validate before use: size == {SIZE}, magic, version <= BIOPROFILE_VERSION, CRC32 (zlib) over the first {SIZE - 4} bytes == crc32. */
#endif
"""
