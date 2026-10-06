"""The C99 core (csrc/asc_core.c) behind the same interface as `FixedAsc`: compiled on demand, called through ctypes."""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Optional

from .fixed import FixedParams, q
from .tremor_fixed import FixedTremorParams
from .types import Guard, ObjectOfInterest, Reason, TickOut

CSRC = Path(__file__).resolve().parent / "csrc"
PARAM_FIELDS = ["enabled", "v_on", "v_still", "on_us", "still_us", "t_lo_us", "ramp_us", "f_b", "ov_rate", "ov_med", "s_brake",
                "tremor_px", "hold_scale", "v_ref", "v_leave", "ov_zone_gain", "r_min_px", "deep_mult", "back_gain", "hold_div", "lam",
                "mu", "c0", "inv_c", "k_floor", "s_cap", "lead_base", "lead_gain", "away_us", "away_ramp_us", "open_us", "open_ramp_us",
                "w_att", "w_rel", "slew", "v_tau", "vp_tau", "gap_us"]
assert PARAM_FIELDS == [f.name for f in fields(FixedParams)], "C param struct and FixedParams drifted"


TREMOR_FIELDS = ["enabled", "a_lp", "a_band", "a_e", "s_max", "trim_cap", "v_t", "lp_weight", "eps", "r_lo", "inv_r", "big_lo", "inv_big",
                 "reset_us", "reset_ms"]
assert TREMOR_FIELDS == [f.name for f in fields(FixedTremorParams)], "C tremor param struct and FixedTremorParams drifted"


class CTremorParams(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in TREMOR_FIELDS]


class CTremorState(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64 * 2) for n in ("l1", "l2", "b1", "b2")] + [("e_band", ctypes.c_int64), ("e_lp", ctypes.c_int64),
                                                                             ("last_t", ctypes.c_int64), ("carry", ctypes.c_int32 * 2),
                                                                             ("zero_ms", ctypes.c_int32), ("s", ctypes.c_int32),
                                                                             ("r", ctypes.c_int32), ("has_last", ctypes.c_int32)]


class CParams(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in PARAM_FIELDS]


class CState(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ("k", "kd", "vx", "vy", "vpx", "vpy", "pxp", "pyp", "d0", "zone", "obj_id", "on_us",
                                               "still_us", "away_us", "carry_x", "carry_y", "s", "guard", "have_p", "has_last")] + \
        [("last_t", ctypes.c_int64), ("t_move", ctypes.c_int64), ("t_open", ctypes.c_int64)]


class CObj(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int32), ("x", ctypes.c_int32), ("y", ctypes.c_int32), ("radius", ctypes.c_int32),
                ("has_appear", ctypes.c_int32), ("t_appear_us", ctypes.c_int64)]


class COut(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ("k", "dx", "dy", "guard", "reason", "s")]


def build_library(out_dir: Optional[str | Path] = None, cc: Optional[str] = None) -> Path:
    compiler = cc or os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        raise RuntimeError("no C compiler found (set CC, or install gcc/clang)")
    srcs = [CSRC / "asc_core.c", CSRC / "tremor_core.c"]
    h = hashlib.blake2b(b"".join(p.read_bytes() for p in (*srcs, CSRC / "asc_core.h", CSRC / "tremor_core.h")), digest_size=6).hexdigest()
    out = Path(out_dir or Path(tempfile.gettempdir()) / "dataopen-asc")
    out.mkdir(parents=True, exist_ok=True)
    so = out / f"asc_core_{h}.so"
    if not so.exists():
        tmp = out / f"{so.name}.{os.getpid()}.tmp"
        r = subprocess.run([compiler, "-O2", "-std=c99", "-Wall", "-Wextra", "-shared", "-fPIC", f"-I{CSRC}", "-o", str(tmp),
                            *map(str, srcs)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compiling the ASC / tremor C cores failed:\n{r.stderr[:800]}")
        tmp.replace(so)
    return so


class CAsc:
    def __init__(self, params: Optional[FixedParams] = None, library: Optional[str | Path] = None) -> None:
        self.lib = ctypes.CDLL(str(library or build_library()))
        self.lib.asc_tick.argtypes = [ctypes.POINTER(CParams), ctypes.POINTER(CState), ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
                                      ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(CObj), ctypes.POINTER(COut)]
        self.lib.asc_reset.argtypes = [ctypes.POINTER(CState)]
        self.cp = CParams()
        self.st = CState()
        self.set_params(params or FixedParams())
        self.reset()

    def set_params(self, params: FixedParams) -> None:
        for n in PARAM_FIELDS:
            setattr(self.cp, n, getattr(params, n))

    def reset(self) -> None:
        self.lib.asc_reset(ctypes.byref(self.st))

    def tick(self, t_us: int, dx: int, dy: int, px: float, py: float, obj: Optional[ObjectOfInterest]) -> TickOut:
        o, po = COut(), None
        if obj is not None:
            co = CObj(obj.id, q(obj.x), q(obj.y), q(obj.radius), int(obj.t_appear_us is not None), obj.t_appear_us or 0)
            po = ctypes.byref(co)
        self.lib.asc_tick(ctypes.byref(self.cp), ctypes.byref(self.st), t_us, dx, dy, q(px), q(py), po, ctypes.byref(o))
        return TickOut(o.k / 65536, o.dx, o.dy, Guard(o.guard), Reason(o.reason), o.s / 65536)

    @property
    def state_bytes(self) -> int:
        return ctypes.sizeof(CState)


class CTremor:
    """The C tremor suppressor behind the same interface as `FixedTremor`."""

    def __init__(self, params: Optional[FixedTremorParams] = None, library: Optional[str | Path] = None) -> None:
        self.lib = ctypes.CDLL(str(library or build_library()))
        i32p = ctypes.POINTER(ctypes.c_int32)
        self.lib.tremor_tick.argtypes = [ctypes.POINTER(CTremorParams), ctypes.POINTER(CTremorState), ctypes.c_int64, ctypes.c_int32,
                                         ctypes.c_int32, i32p, i32p]
        self.lib.tremor_reset.argtypes = [ctypes.POINTER(CTremorState)]
        self.cp, self.st = CTremorParams(), CTremorState()
        self.set_params(params or FixedTremorParams())
        self.reset()

    def set_params(self, params: FixedTremorParams) -> None:
        for n in TREMOR_FIELDS:
            setattr(self.cp, n, getattr(params, n))

    def reset(self) -> None:
        self.lib.tremor_reset(ctypes.byref(self.st))

    def tick(self, t_us: int, dx: int, dy: int) -> tuple[int, int]:
        ox, oy = ctypes.c_int32(), ctypes.c_int32()
        self.lib.tremor_tick(ctypes.byref(self.cp), ctypes.byref(self.st), t_us, dx, dy, ctypes.byref(ox), ctypes.byref(oy))
        return ox.value, oy.value

    @property
    def state_bytes(self) -> int:
        return ctypes.sizeof(CTremorState)
