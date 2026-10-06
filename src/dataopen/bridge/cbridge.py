"""ctypes wrapper of the bridge C core (compiled on demand), plus the shared C header facts the tests check against."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

CSRC = Path(__file__).resolve().parent / "csrc"
ASC_CSRC = Path(__file__).resolve().parent.parent / "assist" / "csrc"
SOURCES = [CSRC / n for n in ("hid_desc.c", "usb_image.c", "usb_proxy.c", "link.c", "failsafe.c", "bridge.c")] + [
    ASC_CSRC / "asc_core.c",
    ASC_CSRC / "tremor_core.c",
]
HEADERS = sorted(CSRC.glob("*.h")) + sorted(ASC_CSRC.glob("*.h"))

CFG_FIELDS = [
    "scope",
    "chord_mask",
    "chord_ms",
    "panic_long_ms",
    "rearm_ms",
    "engage_hold_ms",
    "probe_ms",
    "pc_cfg_ms",
    "soft_hold_ms",
    "auto_engage",
    "param_ttl_ms",
    "scene_ttl_ms",
    "link_ttl_ms",
    "slow_poll_us",
    "split_poll_us",
    "lockin_ratio_q16",
    "lockin_min_counts",
    "lockin_hold_ms",
    "budget_us",
    "overrun_limit",
    "status_period_ms",
    "domain_counts",
]
STATUS_FIELDS = [
    "state",
    "reason",
    "hw",
    "attach",
    "latch_soft",
    "latch_hw",
    "mode",
    "guard",
    "k_q16",
    "gen_asc",
    "gen_tremor",
    "params_asc_ok",
    "params_trm_ok",
    "link_ok",
    "scene_n",
    "scene_age_ms",
    "lockin_hold",
    "chord_active",
    "healthy",
    "img_code",
    "n_motion_if",
    "motion_reports",
    "other_reports",
    "short_reports",
    "invariant_viol",
    "lockin_trips",
    "overruns",
    "params_rejected",
    "link_rx_ok",
    "link_rx_bad",
    "telem_dropped",
    "telem_pending",
    "tick_max_us",
    "tick_avg_us",
    "usb_errors",
    "crashes",
    "wants_kick",
    "vid",
    "pid",
    "buttons",
    "active",
    "cum_x",
    "cum_y",
]

# state / reason / hw / mode names (the C enums)
STATE = {0: "HW_BYPASS", 1: "PROBE", 2: "PASSTHRU", 3: "ASSIST"}
HW_BYPASS, HW_PROBE, HW_ENGAGED = 0, 1, 2
S_BYPASS, S_PROBE, S_PASSTHRU, S_ASSIST = 0, 1, 2, 3
REASONS = [
    "NONE",
    "POWER_ON",
    "NO_DEVICE",
    "SETTLING",
    "PROBING",
    "IMAGE",
    "PC_TIMEOUT",
    "USB_ERRORS",
    "PANIC",
    "PANIC_LONG",
    "CHORD",
    "CMD_PASSTHRU",
    "CMD_BYPASS",
    "STALE_PARAMS",
    "STALE_LINK",
    "INVARIANT",
    "OVERRUN",
    "SLOW_MOUSE",
    "CRASHLOOP",
    "FATAL",
    "ENGAGE_FAILED",
    "WAIT_PC",
    "IMAGE_TIMEOUT",
]
R = {n: i for i, n in enumerate(REASONS)}
MODE_DIRECT, MODE_SPLIT, MODE_SLOW = 0, 1, 2
SCOPE_TRANSPARENT, SCOPE_MOUSE_ONLY = 0, 1
RESET_POWER, RESET_WATCHDOG, RESET_FAULT, RESET_SOFT = 0, 1, 2, 3
PX_LOCAL, PX_SERVE, PX_FORWARD = 0, 1, 2
IMG = {
    0: "OK",
    -1: "BAD_DEVICE",
    -2: "BAD_CONFIG",
    -3: "TOO_BIG",
    -4: "MULTI_CONFIG",
    -5: "UNSUPPORTED_EP",
    -6: "ALT_SETTING",
    -7: "NO_MOTION",
    -8: "NOT_MOUSE_ONLY",
    -9: "MISSING_RD",
    -10: "TOO_MANY",
}
HID_OK, HID_E_BAD, HID_E_NOXY, HID_E_TOO_BIG, HID_E_LIMIT = 0, -1, -2, -3, -4


class CCfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in CFG_FIELDS]


class CStatus(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in STATUS_FIELDS]


class CSetup(ctypes.Structure):
    _fields_ = [
        ("bmRequestType", ctypes.c_uint8),
        ("bRequest", ctypes.c_uint8),
        ("wValue", ctypes.c_uint16),
        ("wIndex", ctypes.c_uint16),
        ("wLength", ctypes.c_uint16),
    ]


class CMotion(ctypes.Structure):
    _fields_ = [
        ("report_id", ctypes.c_uint8),
        ("size_x", ctypes.c_uint8),
        ("size_y", ctypes.c_uint8),
        ("off_x", ctypes.c_uint16),
        ("off_y", ctypes.c_uint16),
        ("len", ctypes.c_uint16),
        ("btn_off", ctypes.c_uint16),
        ("btn_n", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8),
    ]


class CHidMap(ctypes.Structure):
    _fields_ = [
        ("m", CMotion * 8),
        ("n", ctypes.c_uint8),
        ("uses_ids", ctypes.c_uint8),
        ("kinds", ctypes.c_uint8),
        ("n_app", ctypes.c_uint8),
        ("app", ctypes.c_uint32 * 8),
    ]


def build_library(
    out_dir: Optional[str | Path] = None,
    cc: Optional[str] = None,
    testing: bool = False,
    sanitize: bool = False,
    extra: tuple[str, ...] = (),
) -> Path:
    compiler = cc or os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        raise RuntimeError("no C compiler found (set CC, or install gcc/clang)")
    flags = ["-O2", "-std=c99", "-Wall", "-Wextra", "-shared", "-fPIC", f"-I{CSRC}", f"-I{ASC_CSRC}"]
    if testing:
        flags.append("-DBRIDGE_TESTING")
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-g"]
    flags += list(extra)
    h = hashlib.blake2b(b"".join(p.read_bytes() for p in (*SOURCES, *HEADERS)) + " ".join(flags).encode(), digest_size=6).hexdigest()
    out = Path(out_dir or Path(tempfile.gettempdir()) / "dataopen-bridge")
    out.mkdir(parents=True, exist_ok=True)
    so = out / f"bridge_{h}.so"
    if not so.exists():
        tmp = out / f"{so.name}.{os.getpid()}.tmp"
        r = subprocess.run([compiler, *flags, "-o", str(tmp), *map(str, SOURCES)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compiling the bridge C core failed:\n{r.stderr[:1500]}")
        tmp.replace(so)
    return so


_LIBS: dict[tuple, ctypes.CDLL] = {}


def load(testing: bool = False, sanitize: bool = False) -> ctypes.CDLL:
    key = (testing, sanitize)
    if key not in _LIBS:
        lib = ctypes.CDLL(str(build_library(testing=testing, sanitize=sanitize)))
        u8p, c = ctypes.POINTER(ctypes.c_uint8), ctypes
        vp = c.c_void_p
        lib.bridge_sizeof.restype = c.c_size_t
        lib.bridge_status_sizeof.restype = c.c_size_t
        lib.bridge_cfg_sizeof.restype = c.c_size_t
        lib.bridge_cfg_defaults.argtypes = [c.POINTER(CCfg)]
        lib.bridge_init.argtypes = [vp, c.POINTER(CCfg)]
        lib.bridge_boot.argtypes = [vp, c.c_int64, c.c_int, c.c_uint16]
        lib.bridge_nv_take.argtypes = [vp, c.POINTER(c.c_uint16)]
        lib.bridge_set_speed.argtypes = [vp, c.c_int]
        lib.bridge_dev_present.argtypes = [vp, c.c_int64, c.c_int]
        for n in ("bridge_img_device", "bridge_img_config"):
            getattr(lib, n).argtypes = [vp, u8p, c.c_uint16]
        lib.bridge_img_n_if.argtypes = [vp]
        lib.bridge_img_rd_wanted.argtypes = [vp, c.c_int]
        lib.bridge_img_rd_wanted.restype = c.c_uint16
        lib.bridge_img_report_desc.argtypes = [vp, c.c_int, u8p, c.c_uint16]
        lib.bridge_img_done.argtypes = [vp, c.c_int64]
        lib.bridge_pc_setup.argtypes = [vp, c.POINTER(CSetup), c.POINTER(c.POINTER(c.c_uint8)), c.POINTER(c.c_uint16)]
        lib.bridge_pc_forwarded_ok.argtypes = [vp, c.c_int64, c.POINTER(CSetup)]
        lib.bridge_pc_bus_reset.argtypes = [vp, c.c_int64]
        lib.bridge_usb_error.argtypes = [vp, c.c_int64]
        lib.bridge_mouse_in.argtypes = [vp, c.c_int64, c.c_uint8, u8p, c.c_uint16, u8p]
        lib.bridge_merge.argtypes = [vp, c.c_uint8, u8p, u8p, c.c_uint16]
        lib.bridge_poll.argtypes = [vp, c.c_int64]
        lib.bridge_panic.argtypes = [vp, c.c_int64, c.c_int]
        lib.bridge_note_cost.argtypes = [vp, c.c_int64, c.c_uint32]
        lib.bridge_fatal.argtypes = [vp, c.c_int64]
        lib.bridge_link_rx.argtypes = [vp, c.c_int64, u8p]
        lib.bridge_link_tx.argtypes = [vp, c.c_int64, u8p]
        for n in ("bridge_hw_select", "bridge_attach", "bridge_wants_kick"):
            getattr(lib, n).argtypes = [vp]
        lib.bridge_status.argtypes = [vp, c.c_int64, c.POINTER(CStatus)]
        lib.bridge_asc_params_valid.argtypes = [vp]
        lib.bridge_tremor_params_valid.argtypes = [vp]
        lib.hid_parse.argtypes = [u8p, c.c_uint16, c.POINTER(CHidMap)]
        lib.hid_boot_map.argtypes = [c.POINTER(CHidMap)]
        lib.hid_find.argtypes = [c.POINTER(CHidMap), u8p, c.c_uint16]
        lib.hid_find.restype = c.POINTER(CMotion)
        lib.hid_xy_get.argtypes = [c.POINTER(CMotion), u8p, c.POINTER(c.c_int32), c.POINTER(c.c_int32)]
        lib.hid_xy_set.argtypes = [c.POINTER(CMotion), u8p, c.c_int32, c.c_int32]
        lib.hid_xy_mask.argtypes = [c.POINTER(CMotion), u8p, c.c_uint16]
        lib.hid_merge.argtypes = [c.POINTER(CMotion), u8p, u8p, c.c_uint16]
        lib.hid_buttons.argtypes = [c.POINTER(CMotion), u8p, c.c_uint16]
        lib.hid_buttons.restype = c.c_uint32
        lib.link_crc32.argtypes = [u8p, c.c_uint32]
        lib.link_crc32.restype = c.c_uint32
        if testing:
            lib.bridge_debug_inject.argtypes = [vp, c.c_int]
        _LIBS[key] = lib
    return _LIBS[key]


def _u8(b: bytes | bytearray):
    n = len(b)
    return (ctypes.c_uint8 * max(n, 1)).from_buffer_copy(bytes(b) + (b"\0" if n == 0 else b""))


def crc32(b: bytes, lib: Optional[ctypes.CDLL] = None) -> int:
    lib = lib or load()
    return lib.link_crc32(_u8(b), len(b))


def hid_parse(desc: bytes, lib: Optional[ctypes.CDLL] = None) -> tuple[int, CHidMap]:
    lib = lib or load()
    m = CHidMap()
    return lib.hid_parse(_u8(desc), len(desc), ctypes.byref(m)), m


class CBridge:
    """One bridge core instance (its C memory is owned here)."""

    def __init__(self, cfg: Optional[dict] = None, testing: bool = False, sanitize: bool = False) -> None:
        self.lib = load(testing=testing, sanitize=sanitize)
        assert self.lib.bridge_status_sizeof() == ctypes.sizeof(CStatus), "bridge_status_t drifted from CStatus"
        assert self.lib.bridge_cfg_sizeof() == ctypes.sizeof(CCfg), "bridge_cfg_t drifted from CCfg"
        self.mem = ctypes.create_string_buffer(self.lib.bridge_sizeof())
        self.p = ctypes.cast(self.mem, ctypes.c_void_p)
        c = CCfg()
        self.lib.bridge_cfg_defaults(ctypes.byref(c))
        for k, v in (cfg or {}).items():
            if k not in CFG_FIELDS:
                raise KeyError(k)
            setattr(c, k, v)
        self.cfg = c
        self.lib.bridge_init(self.p, ctypes.byref(c))

    # lifecycle / host side
    def boot(self, t: int, cause: int = RESET_POWER, crashes: int = 0) -> None:
        self.lib.bridge_boot(self.p, t, cause, crashes)

    def nv_take(self) -> Optional[int]:
        v = ctypes.c_uint16()
        return v.value if self.lib.bridge_nv_take(self.p, ctypes.byref(v)) and (v.value is not None) else None

    def set_speed(self, speed: int) -> None:
        self.lib.bridge_set_speed(self.p, speed)

    def dev_present(self, t: int, present: bool) -> None:
        self.lib.bridge_dev_present(self.p, t, int(present))

    def img_device(self, d: bytes) -> int:
        return self.lib.bridge_img_device(self.p, _u8(d), len(d))

    def img_config(self, c: bytes) -> int:
        return self.lib.bridge_img_config(self.p, _u8(c), len(c))

    def img_n_if(self) -> int:
        return self.lib.bridge_img_n_if(self.p)

    def img_rd_wanted(self, i: int) -> int:
        return self.lib.bridge_img_rd_wanted(self.p, i)

    def img_report_desc(self, i: int, rd: bytes) -> int:
        return self.lib.bridge_img_report_desc(self.p, i, _u8(rd), len(rd))

    def img_done(self, t: int) -> int:
        return self.lib.bridge_img_done(self.p, t)

    # device side
    def pc_setup(self, bm: int, req: int, value: int, index: int, length: int) -> tuple[int, bytes]:
        s = CSetup(bm, req, value, index, length)
        data = ctypes.POINTER(ctypes.c_uint8)()
        n = ctypes.c_uint16()
        r = self.lib.bridge_pc_setup(self.p, ctypes.byref(s), ctypes.byref(data), ctypes.byref(n))
        return r, (bytes(data[: n.value]) if r == PX_SERVE else b"")

    def pc_forwarded_ok(self, t: int, bm: int, req: int, value: int, index: int, length: int) -> None:
        self.lib.bridge_pc_forwarded_ok(self.p, t, ctypes.byref(CSetup(bm, req, value, index, length)))

    def pc_bus_reset(self, t: int) -> None:
        self.lib.bridge_pc_bus_reset(self.p, t)

    def usb_error(self, t: int) -> None:
        self.lib.bridge_usb_error(self.p, t)

    def mouse_in(self, t: int, ep: int, report: bytes) -> tuple[bytes, bool]:
        out = (ctypes.c_uint8 * max(len(report), 1))()
        changed = self.lib.bridge_mouse_in(self.p, t, ep, _u8(report), len(report), out)
        return bytes(out[: len(report)]), bool(changed)

    def merge(self, ep: int, a: bytes, b: bytes) -> tuple[bool, bytes]:
        ab = _u8(a)
        ok = self.lib.bridge_merge(self.p, ep, ab, _u8(b), len(a))
        return bool(ok), bytes(ab[: len(a)])

    def poll(self, t: int) -> None:
        self.lib.bridge_poll(self.p, t)

    def panic(self, t: int, pressed: bool) -> None:
        self.lib.bridge_panic(self.p, t, int(pressed))

    def note_cost(self, t: int, us: int) -> None:
        self.lib.bridge_note_cost(self.p, t, us)

    def fatal(self, t: int) -> None:
        self.lib.bridge_fatal(self.p, t)

    def inject(self, mode: int = 1) -> None:
        self.lib.bridge_debug_inject(self.p, mode)

    # link
    def link_rx(self, t: int, frame: bytes) -> None:
        assert len(frame) == 128
        self.lib.bridge_link_rx(self.p, t, _u8(frame))

    def link_tx(self, t: int) -> bytes:
        out = (ctypes.c_uint8 * 128)()
        self.lib.bridge_link_tx(self.p, t, out)
        return bytes(out)

    # outputs
    @property
    def hw_select(self) -> int:
        return self.lib.bridge_hw_select(self.p)

    @property
    def attach(self) -> bool:
        return bool(self.lib.bridge_attach(self.p))

    @property
    def wants_kick(self) -> bool:
        return bool(self.lib.bridge_wants_kick(self.p))

    def status(self, t: int) -> CStatus:
        s = CStatus()
        self.lib.bridge_status(self.p, t, ctypes.byref(s))
        return s

    @property
    def sizeof(self) -> int:
        return len(self.mem)
