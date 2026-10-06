"""The assistive HID bridge core: descriptor parsing and in-place X/Y editing, transparency of the proxy, equivalence with the
correction chain, split mode for high-rate mice, scene compensation, parameter validation, the SPI link, the lock-in backstop."""

import ctypes
import functools
import random
import shutil
import struct
import zlib

import numpy as np
import pytest

from dataopen.assist.chain import AssistChain
from dataopen.assist.fixed import FixedParams
from dataopen.assist.params import AscParams
from dataopen.assist.sim_user import PERSONAS, build_profile, run_trial
from dataopen.assist.tremor import TremorParams
from dataopen.assist.tremor_fixed import FixedTremorParams
from dataopen.bridge import protocol as P
from dataopen.bridge import sim_usb as U
from dataopen.bridge.cbridge import (
    CMotion,
    HID_E_BAD,
    HID_E_NOXY,
    HID_E_TOO_BIG,
    HID_OK,
    MODE_DIRECT,
    MODE_SPLIT,
    MODE_SLOW,
    PX_FORWARD,
    PX_LOCAL,
    PX_SERVE,
    REASONS,
    S_ASSIST,
    S_PASSTHRU,
    STATE,
    CBridge,
    crc32,
    hid_parse,
    load,
)
from dataopen.bridge.sim import BridgeAssist, Rig
from dataopen.bridge.sim_usb import SimMouse, SimPC

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


@functools.lru_cache(maxsize=None)
def persona_params(name="overshooter", seed=1):
    v = build_profile(PERSONAS[name], seed=seed)
    return AscParams.from_view(v), TremorParams.from_view(v)


def fixed_params(name="overshooter"):
    a, t = persona_params(name)
    return FixedParams.from_params(a), FixedTremorParams.from_params(t)


def rig(kind="m16", name="overshooter", mouse=None, **kw):
    asc, trm = fixed_params(name)
    r = Rig(mouse or SimMouse(kind), asc=asc, tremor=trm, **kw)
    assert r.engage_fast(), (STATE[r.status().state], REASONS[r.status().reason])
    return r


def xy(kind, report):
    if kind == "boot":
        return struct.unpack_from("<bb", report, 1)
    if kind == "m16":
        return struct.unpack_from("<hh", report, 1)
    if kind == "logi":
        return U.get_bits(report, 24, 12, True), U.get_bits(report, 36, 12, True)
    raise ValueError(kind)


def ok_axis(o, i):
    return abs(o) <= abs(i) and (o == 0 or (o > 0) == (i > 0)) and (i != 0 or o == 0)


# ------------------------------------------------------------------ descriptor parsing and in-place editing
SHAPES = {
    "boot": (U.boot_mouse_rd(), dict(size=8, off_x=8, off_y=16, length=3, btn_n=3, ids=0)),
    "m16": (U.mouse16_rd(), dict(size=16, off_x=8, off_y=24, length=7, btn_n=5, ids=0)),
    "logi12": (U.logi_mouse_rd(16, 12), dict(size=12, off_x=24, off_y=36, length=8, btn_n=16, ids=1)),
    "logi16": (U.logi_mouse_rd(8, 16), dict(size=16, off_x=24, off_y=40, length=9, btn_n=8, ids=1)),
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_mouse_descriptor_shapes_parse_to_the_right_fields(name):
    rd, want = SHAPES[name]
    r, m = hid_parse(rd)
    assert r == HID_OK and m.n == 1
    f = m.m[0]
    assert (f.size_x, f.size_y, f.off_x, f.off_y, f.len, f.btn_n) == (
        want["size"],
        want["size"],
        want["off_x"],
        want["off_y"],
        want["length"],
        want["btn_n"],
    )
    assert f.report_id == want["ids"] and m.uses_ids == int(want["ids"] != 0)


def test_non_motion_descriptors_are_classified_not_edited():
    for rd, kinds in ((U.hidpp_rd(), 8), (U.kbd_rd(), 2), (U.abs_pointer_rd(), 1)):
        r, m = hid_parse(rd)
        assert r == HID_E_NOXY and m.n == 0 and m.kinds == kinds  # vendor / keyboard / mouse collection with ABSOLUTE axes
    assert hid_parse(b"")[0] == HID_E_BAD
    assert hid_parse(U.boot_mouse_rd()[:-1])[0] == HID_E_BAD  # unbalanced collection
    assert hid_parse(U.boot_mouse_rd()[:7])[0] == HID_E_BAD or hid_parse(U.boot_mouse_rd()[:7])[0] == HID_E_NOXY
    assert hid_parse(bytes(2000))[0] == HID_E_TOO_BIG


def test_unsigned_or_wide_axes_are_not_treated_as_relative_motion():
    wide = (
        U.usage_page(1)
        + U.usage(2)
        + U.collection(1)
        + U.usage(0x30)
        + U.usage(0x31)
        + U.lmin(-1000)
        + U.lmax(1000)
        + U.rsize(32)
        + U.rcount(2)
        + U.inp(6)
        + U.end()
    )
    assert hid_parse(wide)[0] == HID_E_NOXY
    unsigned = (
        U.usage_page(1)
        + U.usage(2)
        + U.collection(1)
        + U.usage(0x30)
        + U.usage(0x31)
        + U.lmin(0)
        + U.lmax(255)
        + U.rsize(8)
        + U.rcount(2)
        + U.inp(6)
        + U.end()
    )
    assert hid_parse(unsigned)[0] == HID_E_NOXY


def cbuf(n, data=None):
    return (ctypes.c_uint8 * max(n, 1)).from_buffer_copy(data if data is not None else bytes(max(n, 1)))


def c_xy_get(f, data):
    x, y = ctypes.c_int32(), ctypes.c_int32()
    load().hid_xy_get(ctypes.byref(f), cbuf(len(data), data), ctypes.byref(x), ctypes.byref(y))
    return x.value, y.value


def c_xy_set(f, data, dx, dy):
    buf = cbuf(len(data), data)
    load().hid_xy_set(ctypes.byref(f), buf, dx, dy)
    return bytes(buf[: len(data)])


def c_mask(f, n):
    buf = cbuf(n)
    load().hid_xy_mask(ctypes.byref(f), buf, n)
    return bytes(buf[:n])


@pytest.mark.parametrize("name", list(SHAPES))
def test_editing_touches_exactly_the_xy_bits(name):
    rd, want = SHAPES[name]
    _, m = hid_parse(rd)
    f = m.m[0]
    n = f.len
    rng = random.Random(1)
    lim = (1 << (f.size_x - 1)) - 1
    mask_b = c_mask(f, n)
    for _ in range(300):
        rep = bytearray(rng.randrange(256) for _ in range(n))
        if m.uses_ids:
            rep[0] = f.report_id
        nx, ny = rng.randint(-lim, lim), rng.randint(-lim, lim)
        out = c_xy_set(f, bytes(rep), nx, ny)
        assert all((a ^ b) & ~mk & 0xFF == 0 for a, b, mk in zip(rep, out, mask_b))  # nothing outside the X/Y bits moved
        assert c_xy_get(f, out) == (nx, ny)
        assert c_xy_get(f, c_xy_set(f, bytes(rep), 10**6, -(10**6))) == (lim, -lim - 1)  # out of range: clamped, not wrapped


def test_xy_get_via_python_bits_agrees_with_the_c_reader():
    lib = load()
    _, m = hid_parse(U.logi_mouse_rd(16, 12))
    f = m.m[0]
    rng = random.Random(2)
    for _ in range(200):
        dx, dy = rng.randint(-2047, 2047), rng.randint(-2047, 2047)
        rep = U.pack_logi(rng.randrange(1 << 16), dx, dy, rng.randint(-127, 127), rng.randint(-127, 127))
        assert c_xy_get(f, rep) == (dx, dy)
        assert lib.hid_buttons(ctypes.byref(f), cbuf(8, rep), 8) == (rep[1] | rep[2] << 8)


def test_merge_only_when_everything_else_is_identical_and_the_sums_fit():
    r = rig("m16")
    a, b = U.pack_m16(1, 10, 5, 0, 0), U.pack_m16(1, 7, -2, 0, 0)
    ok, m = r.b.merge(0x81, a, b)
    assert ok and xy("m16", m) == (17, 3) and m[0] == 1
    assert r.b.merge(0x81, U.pack_m16(1, 1, 1), U.pack_m16(0, 1, 1))[0] is False  # a button changed in between: never merged
    assert r.b.merge(0x81, U.pack_m16(0, 1, 1, 1, 0), U.pack_m16(0, 1, 1, 0, 0))[0] is False  # neither the wheel
    assert r.b.merge(0x81, U.pack_m16(0, 30000, 0), U.pack_m16(0, 30000, 0))[0] is False  # would overflow the 16-bit field
    assert r.b.merge(0x82, a, b)[0] is False


def with_scene(r, ox=600.0, oy=300.0, radius=30.0):
    """An object at a fixed place in the world; the scene the module sends is relative to the cursor, as the bridge expects."""

    def fn(t):
        st = r.b.status(t)
        return [P.SceneObject(1, ox - st.cum_x, oy - st.cum_y, radius)]

    r.module.scene_fn = fn
    r.module.next_scene = 0
    return r


# ------------------------------------------------------------------ the proxy is transparent
MICE = {
    "boot": dict(kind="boot"),
    "m16": dict(kind="m16"),
    "logi": dict(kind="logi"),
    "composite": dict(kind="logi", hidpp=True, kbd=True),
    "hs": dict(kind="m16", speed="HS", interval=4),
}


@pytest.mark.parametrize("name", list(MICE))
def test_enumeration_through_the_bridge_is_identical_to_a_direct_connection(name):
    direct = SimMouse(**MICE[name])
    t_direct = SimPC.enumerate(direct)
    behind = SimMouse(**MICE[name])
    r = rig(mouse=behind)
    behind.reset_state()
    t_bridge = SimPC.enumerate(r.front)
    assert len(t_direct) == len(t_bridge) > 30
    for a, b in zip(t_direct, t_bridge):
        assert a == b  # every request, status and byte
    assert any(x[6] == 1 for x in t_direct)  # stalls were part of the exchange, and were mirrored too


def test_the_cache_serves_descriptors_and_everything_else_is_forwarded():
    r = rig("logi", mouse=SimMouse("logi", hidpp=True))
    b = r.b
    assert b.pc_setup(0x00, 5, 7, 0, 0)[0] == PX_LOCAL  # SET_ADDRESS addresses the bridge's own port
    kind, dev = b.pc_setup(0x80, 6, 0x0100, 0, 64)
    assert kind == PX_SERVE and dev == r.mouse.dev
    assert b.pc_setup(0x80, 6, 0x0100, 0, 8)[1] == r.mouse.dev[:8]  # truncated to wLength
    assert b.pc_setup(0x80, 6, 0x0200, 0, 4096)[1] == r.mouse.cfg
    assert b.pc_setup(0x81, 6, 0x2200, 0, 4096)[1] == r.mouse.rds[0]
    assert b.pc_setup(0x81, 6, 0x2200, 1, 4096)[1] == r.mouse.rds[1]
    for req in (
        (0x80, 6, 0x0302, 0x0409, 255),
        (0x80, 6, 0x03EE, 0, 255),
        (0x80, 6, 0x0F00, 0, 5),
        (0xA1, 1, 0x0320, 1, 8),
        (0x21, 9, 0x0210, 1, 6),
        (0x21, 0x0B, 0, 0, 0),
        (0x40, 2, 0x1234, 1, 4),
        (0xC0, 1, 5, 0, 16),
        (0x80, 6, 0x0200, 0, 4096 * 0 + 9) if False else (0x80, 6, 0x0201, 0, 9),
        (0x80, 0, 0, 0, 2),
        (0x00, 9, 1, 0, 0),
        (0x01, 11, 0, 0, 0),
    ):
        assert b.pc_setup(*req)[0] == PX_FORWARD, req


def test_hidpp_shaped_traffic_passes_in_both_directions_untouched():
    m = SimMouse("logi", hidpp=True, kbd=True)
    r = rig(mouse=m)
    r.run(5)
    req = bytes([0x10, 0x01, 0x80, 0x00, 0x00, 0x00, 0x00]) + bytes(13)
    assert r.pc_out(0x02, req)
    r.run(5)
    assert m.out_log[-1] == (0x02, req)
    got = [x for x in r.pc_reports if x[2] == 0x82]
    assert got and got[-1][3] == got[-1][4] == bytes([0x10, 0x01, 0x80 ^ 0xFF]) + req[3:]
    st = r.status()
    assert st.other_reports >= 1 and st.motion_reports == 0


def test_stalls_and_timeouts_of_the_mouse_are_mirrored_not_swallowed():
    r = rig("m16")
    assert r.front.control(0xA1, 1, 0x0355, 0, 8)[0] == 1  # the mouse stalls: so does the bridge
    r.mouse.fail_control = True
    n = r.status().usb_errors
    with pytest.raises(TimeoutError):
        r.front.control(0xA1, 2, 0, 0, 1)
    assert r.status().usb_errors == n + 1


def test_boot_protocol_switches_the_report_layout_and_back():
    r = with_scene(rig("boot", mouse=SimMouse("m16")))
    r.run(5)
    assert r.front.control(0x21, 0x0B, 0, 0, 0)[0] == 0  # BIOS-style: boot protocol
    r.move(0, 0)
    r.mouse.in_q.clear()
    for _ in range(300):  # boot layout: 3 bytes
        r.mouse.push_in(0x81, U.pack_boot(1, 12, 6))
        r.step()
    outs = [x for x in r.pc_reports if len(x[3]) == 3][-250:]
    assert outs and all(x[3][0] == 1 and ok_axis(xy("boot", x[3])[0], 12) and ok_axis(xy("boot", x[3])[1], 6) for x in outs)
    assert any(xy("boot", x[3]) != (12, 6) for x in outs)  # and the chain really edited them
    assert r.front.control(0x21, 0x0B, 1, 0, 0)[0] == 0  # back to the report protocol
    n0 = len(r.pc_reports)
    for _ in range(50):
        r.move(9, 4)
        r.step()
    new = r.pc_reports[n0:]
    assert len(new) == 50 and all(len(x[3]) == 7 and ok_axis(xy("m16", x[3])[0], 9) for x in new)


# ------------------------------------------------------------------ in-place editing in the running bridge
def run_motion(r, kind, n, rng, scale=20, buttons=True, wheel=True):
    for _ in range(n):
        dx, dy = int(rng.normal(0, scale)), int(rng.normal(0, scale)) if rng.random() < 0.8 else 0
        r.move(
            dx,
            dy,
            rng.integers(0, 32) if buttons else 0,
            int(rng.integers(-3, 4)) if wheel else 0,
            int(rng.integers(-3, 4)) if wheel else 0,
        )
        r.step()


def bridge_reports(r):
    return [x for x in r.pc_reports if x[1] == "bridge" and x[2] == 0x81]


@pytest.mark.parametrize("kind", ["boot", "m16", "logi"])
def test_only_xy_bits_change_and_there_is_one_report_out_for_every_report_in(kind):
    r = with_scene(rig(kind))
    r.pc_reports.clear()
    rng = np.random.default_rng(3)
    for _ in range(600):
        lim = r.mouse.max_count
        dx, dy = int(np.clip(rng.normal(0, 25), -lim, lim)), int(np.clip(rng.normal(0, 25), -lim, lim))
        if kind == "boot":
            r.mouse.push_in(0x81, U.pack_boot(int(rng.integers(0, 8)), dx, dy))
        else:
            r.move(dx, dy, int(rng.integers(0, 32)), int(rng.integers(-3, 4)), int(rng.integers(-3, 4)))
        r.step()
    reps = bridge_reports(r)
    assert len(reps) == 600
    rd = {"boot": U.boot_mouse_rd(), "m16": U.mouse16_rd(), "logi": U.logi_mouse_rd()}[kind]
    _, m = hid_parse(rd)
    f = m.m[0]
    mask_b = c_mask(f, len(reps[0][3]))
    changed = 0
    for _, _, _, out, raw in reps:
        assert len(out) == len(raw)
        assert all((a ^ b) & ~mk & 0xFF == 0 for a, b, mk in zip(raw, out, mask_b))  # buttons, wheel, id, vendor bytes: untouched
        changed += out != raw
        for o, i in zip(xy(kind, out), xy(kind, raw)):
            assert ok_axis(o, i)
    assert changed > 0


def test_in_passthrough_the_output_is_bit_for_bit_the_input():
    r = rig("logi")
    r.panic(True)
    r.pc_reports.clear()
    rng = np.random.default_rng(4)
    run_motion(r, "logi", 500, rng, scale=30)
    reps = bridge_reports(r)
    assert len(reps) == 500 and all(x[3] == x[4] for x in reps)
    assert r.status().state == S_PASSTHRU


def test_extreme_and_adversarial_inputs_keep_the_per_axis_invariants():
    for name in ("steady", "overshooter", "tremor"):
        r = with_scene(rig("m16", name))
        r.pc_reports.clear()
        rng = np.random.default_rng(5)
        lim = r.mouse.max_count
        seq = []
        for k in range(6000):
            mode = (k // 300) % 6
            if mode == 0:
                dx, dy = lim, -lim
            elif mode == 1:
                dx, dy = int(rng.choice([-1, 1])), int(rng.choice([-1, 0, 1]))
            elif mode == 2:
                dx, dy = int(rng.integers(-lim, lim)), int(rng.integers(-lim, lim))
            elif mode == 3:
                dx, dy = (5 if k % 2 else -5), 0
            elif mode == 4:
                dx, dy = 0, 0
            else:
                dx, dy = int(rng.normal(0, 3)), int(rng.normal(0, 3))
            r.move(dx, dy)
            r.step()
            seq.append((dx, dy))
        reps = bridge_reports(r)
        assert len(reps) == 6000
        for (_, _, _, out, raw), (dx, dy) in zip(reps, seq):
            assert xy("m16", raw) == (dx, dy)
            for o, i in zip(xy("m16", out), (dx, dy)):
                assert ok_axis(o, i)
        s = r.status()
        assert s.invariant_viol == 0 and s.state in (S_ASSIST, S_PASSTHRU)


# ------------------------------------------------------------------ the correction path is the validated one
@pytest.mark.parametrize("name,n", [("steady", 25), ("overshooter", 25), ("tremor", 25)])
def test_closed_loop_through_the_bridge_is_identical_to_the_correction_chain(name, n):
    """Report bytes in, report bytes out, the scene over real link frames: the closed-loop trace equals AssistChain(impl="c") tick for tick.
    (The lock-in backstop is switched off here: it is the one deliberate difference, tested separately.)"""
    ap, tp = persona_params(name)
    same = 0
    for i in range(n):
        ref = AssistChain.build(ap, tp, "c")
        r1, r2 = [], []
        run_trial(PERSONAS[name], ref, np.random.default_rng(i), record=r1)
        ba = BridgeAssist(ap, tp, cfg={"lockin_min_counts": 10**9})
        run_trial(PERSONAS[name], ba, np.random.default_rng(i), record=r2)
        same += r1 == r2
        assert ba.b.status(ba.rig.t).invariant_viol == 0
    assert same == n


def test_the_same_chain_through_the_bridge_helps_the_overshooter_and_leaves_the_steady_user_alone():
    ap, tp = persona_params("overshooter")
    base, helped = [], []
    for i in range(20):
        base.append(run_trial(PERSONAS["overshooter"], None, np.random.default_rng(i)))
        helped.append(run_trial(PERSONAS["overshooter"], BridgeAssist(ap, tp), np.random.default_rng(i)))
    assert np.mean([t.overshoot_px > 30 for t in helped]) < np.mean([t.overshoot_px > 30 for t in base]) - 0.2
    ap, tp = persona_params("steady")
    for i in range(10):
        a = run_trial(PERSONAS["steady"], None, np.random.default_rng(i))
        b = run_trial(PERSONAS["steady"], BridgeAssist(ap, tp), np.random.default_rng(i))
        assert b.guard_violations == 0 and b.amp_violations == 0 and abs(a.final_err_px - b.final_err_px) < 3


def test_a_report_with_no_motion_is_not_a_tick_and_never_changes():
    r = rig("logi")
    r.pc_reports.clear()
    for b in (0, 1, 3, 0):
        r.move(0, 0, b)
        r.step()
    assert [x[3] for x in bridge_reports(r)] == [x[4] for x in bridge_reports(r)]


# ------------------------------------------------------------------ high-rate mice (split mode)
def _hs_rig(interval, step_us, speed="HS", name="overshooter"):
    r = with_scene(rig(mouse=SimMouse("m16", speed=speed, interval=interval), name=name, step_us=step_us), 600.0, 0.0)
    return r


def _reach(r, per_ms, ms=600, sub=1):
    """A min-jerk reach toward the object, delivered as `sub` reports per millisecond (integer counts with carry)."""
    carry, t0 = 0.0, r.t
    for k in range(ms * sub):
        tau = min(k / sub / 350.0, 1.0)
        carry += per_ms * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / sub
        i = int(round(carry))
        carry -= i
        r.move(i, 0)
        r.step()
    reps = [x for x in r.pc_reports if x[0] >= t0 and x[1] == "bridge"]
    return reps, sum(xy("m16", x[4])[0] for x in reps), sum(xy("m16", x[3])[0] for x in reps)


def test_the_poll_interval_of_the_mouse_selects_direct_split_or_slow_mode():
    cases = [
        (dict(speed="FS", interval=1), MODE_DIRECT),
        (dict(speed="FS", interval=2), MODE_DIRECT),
        (dict(speed="FS", interval=4), MODE_DIRECT),
        (dict(speed="HS", interval=4), MODE_DIRECT),
        (dict(speed="HS", interval=3), MODE_SPLIT),
        (dict(speed="HS", interval=1), MODE_SPLIT),
        (dict(speed="FS", interval=8), MODE_SLOW),
        (dict(speed="FS", interval=10), MODE_SLOW),
    ]
    for kw, want in cases:
        asc, trm = fixed_params()
        r = Rig(SimMouse("m16", **kw), asc=asc, tremor=trm)
        r.run(6000)
        s = r.status()
        if want == MODE_SLOW:
            assert STATE[s.state] == "PASSTHRU" and REASONS[s.reason] == "SLOW_MOUSE", kw  # a 125 Hz mouse is not helped (and not harmed)
        else:
            assert s.mode == want and STATE[s.state] == "ASSIST", kw


def test_split_mode_keeps_every_report_and_the_sums_and_matches_a_1khz_mouse():
    direct, split = _hs_rig(1, 1000, "FS"), _hs_rig(1, 125)
    assert direct.status().mode == MODE_DIRECT and split.status().mode == MODE_SPLIT
    rd, in_d, out_d = _reach(direct, 3.2)
    rs, in_s, out_s = _reach(split, 3.2, sub=8)
    assert len(rd) == 600 and len(rs) == 4800  # nothing merged, nothing dropped
    assert in_d == in_s
    assert abs(out_s - out_d) <= 0.03 * out_d + 2  # same distance delivered as the 1 kHz mouse (measured: 1 count)
    assert out_s < 0.8 * in_s  # and the chain really helped
    for _, _, _, out, raw in rs:
        assert ok_axis(xy("m16", out)[0], xy("m16", raw)[0]) and ok_axis(xy("m16", out)[1], xy("m16", raw)[1])
    assert split.status().invariant_viol == 0


def test_split_mode_survives_bursts_and_silence():
    r = _hs_rig(1, 125)
    rng = np.random.default_rng(6)
    n_in = 0
    for k in range(20000):
        if (k // 800) % 3 == 1:
            r.step()
            continue
        dx, dy = int(rng.integers(-60, 60)), int(rng.integers(-60, 60))
        r.move(dx, dy)
        n_in += 1
        r.step()
    reps = [x for x in r.pc_reports if x[1] == "bridge"]
    assert len(reps) == n_in
    assert all(ok_axis(xy("m16", x[3])[a], xy("m16", x[4])[a]) for x in reps for a in (0, 1))
    assert r.status().invariant_viol == 0


# ------------------------------------------------------------------ the scene arrives late: the bridge compensates
def _scene_run(scene_ms, latency_ms, comp, ms=700, per_ms=4.0):
    r = rig("m16")
    hist = {}

    def fn(t):
        tc = t - latency_ms * 1000
        c = hist.get(tc // 1000 * 1000, r.b.status(t).cum_x)
        return (tc if comp else t, [P.SceneObject(1, 600 - c, 0, 30)])

    r.module.scene_fn, r.module.scene_us, r.module.next_scene = fn, scene_ms * 1000, 0
    carry, ks = 0.0, []
    for k in range(ms):
        tau = min(k / 450.0, 1.0)
        carry += per_ms * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
        i = int(round(carry))
        carry -= i
        r.move(i, 0)
        r.step()
        st = r.b.status(r.t)
        hist[r.t // 1000 * 1000] = st.cum_x
        ks.append(st.k_q16 / 65536)
    return np.array(ks), r.b.status(r.t).cum_x


def test_a_detector_that_is_20ms_late_and_30hz_gives_nearly_the_same_help_as_a_perfect_one():
    ideal, cum_ideal = _scene_run(1, 0, True)
    assert ideal.min() < 0.3  # there is something to compare
    k, cum = _scene_run(33, 20, True)
    k_raw, cum_raw = _scene_run(33, 20, False)  # the same scene WITHOUT telling the bridge when it was captured
    assert np.abs(k - ideal).mean() < 0.01 and np.abs(k - ideal).max() < 0.05
    assert abs(cum - cum_ideal) <= 3
    assert np.abs(k_raw - ideal).mean() > 10 * np.abs(k - ideal).mean()  # the compensation is what makes it work (measured: 38x)


def test_a_stale_scene_means_no_help_not_wrong_help():
    r = with_scene(rig("m16"), 600.0, 0.0)
    for _ in range(300):
        r.move(5, 0)
        r.step()
    assert r.status().scene_n == 1 and r.status().k_q16 < 65536
    r.module.scene_fn = lambda t: (t - 1_000_000, [P.SceneObject(1, 100, 0, 30)])  # captured a second ago: older than the TTL
    for _ in range(400):
        r.move(5, 0)
        r.step()
    assert r.status().k_q16 > 0.99 * 65536  # the smoothing relaxes back to 1.0 (no object: no brake)


# ------------------------------------------------------------------ the module is untrusted: parameter blobs
def _asc_bytes(**over):
    p = fixed_params()[0]
    vals = {f: getattr(p, f) for f in P.FixedParams.__dataclass_fields__}
    vals.update(over)
    return struct.pack("<38i", *vals.values())


def _trm_bytes(**over):
    p = fixed_params()[1]
    vals = {f: getattr(p, f) for f in P.FixedTremorParams.__dataclass_fields__}
    vals.update(over)
    return struct.pack("<15i", *vals.values())


def _send(b, t, frames):
    for k, f in enumerate(frames):
        f.seq = k + 1
        b.link_rx(t, P.pack_frame(f))


def _bridge_with_link():
    b = CBridge()
    b.boot(0)
    return b


def test_valid_blobs_are_applied_and_hostile_ones_are_not():
    b = _bridge_with_link()
    t = 1_000_000
    _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, P.make_blob(5, 1, 65536, _asc_bytes()), 1))
    s = b.status(t)
    assert s.gen_asc == 5 and s.params_asc_ok and s.params_rejected == 0
    base = s.params_rejected
    # every field, pushed to either extreme of int32, is rejected (the ranges are what keep the cores free of overflow and division by zero)
    for i in range(len(P.FixedParams.__dataclass_fields__)):
        for v in (2**31 - 1, -(2**31)):
            vals = [getattr(fixed_params()[0], f) for f in P.FixedParams.__dataclass_fields__]
            vals[i] = v
            _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, P.make_blob(100 + i, 1, 65536, struct.pack("<38i", *vals)), 2))
    assert b.status(t).params_rejected == base + 38 * 2 and b.status(t).gen_asc == 5
    # a zero where the core divides is rejected even though zero is "small"
    for name in (
        "ramp_us",
        "v_ref",
        "v_leave",
        "r_min_px",
        "deep_mult",
        "hold_div",
        "away_ramp_us",
        "open_ramp_us",
        "gap_us",
        "k_floor",
        "on_us",
        "still_us",
    ):
        n = b.status(t).params_rejected
        _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, P.make_blob(200, 1, 65536, _asc_bytes(**{name: 0})), 3))
        assert b.status(t).params_rejected == n + 1, name
    for name in ("v_t", "eps", "reset_ms", "reset_us"):
        n = b.status(t).params_rejected
        _send(b, t, P.blob_frames(P.LK_PARAMS_TREMOR, P.make_blob(7, 1, 0, _trm_bytes(**{name: 0})), 4))
        assert b.status(t).params_rejected == n + 1, name
    assert b.status(t).params_trm_ok == 0
    # k_floor below 0.1: the bridge never accepts a stronger brake than the documented one
    n = b.status(t).params_rejected
    _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, P.make_blob(300, 1, 65536, _asc_bytes(k_floor=1000)), 5))
    assert b.status(t).params_rejected == n + 1
    # bad px_per_count, bad length, bad CRC
    for blob in (
        P.make_blob(301, 1, 0, _asc_bytes()),
        P.make_blob(301, 1, 70 * 65536, _asc_bytes()),
        P.make_blob(301, 1, 65536, _asc_bytes())[:-8],
        P.make_blob(301, 1, 65536, _asc_bytes())[:-1] + b"\x00",
    ):
        n = b.status(t).params_rejected
        _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, blob, 6))
        assert b.status(t).params_rejected == n + 1


def test_replays_conflicts_refreshes_and_module_restarts():
    b = _bridge_with_link()
    t = 1_000_000
    blob = lambda g, **o: P.make_blob(g, 1, 65536, _asc_bytes(**o))  # noqa: E731
    _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, blob(10), 1))
    assert b.status(t).gen_asc == 10
    n = b.status(t).params_rejected
    _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, blob(9), 2))  # an older generation while the current one is fresh: a replay
    _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, blob(10, f_b=1000), 3))  # the same generation with other content
    assert b.status(t).params_rejected == n + 2 and b.status(t).gen_asc == 10
    _send(b, t + 4_000_000, P.blob_frames(P.LK_PARAMS_ASC, blob(10), 4))  # the same blob again = a keepalive
    assert b.status(t + 8_500_000).params_asc_ok == 1 and b.status(t + 8_500_000).params_rejected == n + 2
    assert b.status(t + 9_200_000).params_asc_ok == 0  # but only 5 s of it
    _send(
        b, t + 20_000_000, P.blob_frames(P.LK_PARAMS_ASC, blob(1), 5)
    )  # a restarted module (counter back at 1) once the old data is stale
    assert b.status(t + 20_000_000).gen_asc == 1
    _send(
        b, t + 20_001_000, P.blob_frames(P.LK_PARAMS_ASC, blob(2**32 - 1), 6)
    )  # generations compare modulo 2**32: this one is older than 1
    assert b.status(t + 20_001_000).gen_asc == 1
    _send(b, t + 20_002_000, P.blob_frames(P.LK_PARAMS_ASC, blob(2), 7))
    assert b.status(t + 20_002_000).gen_asc == 2


def test_fragments_in_any_order_duplicated_or_mixed_are_reassembled_or_discarded_safely():
    b = _bridge_with_link()
    t = 1_000_000
    frames = P.blob_frames(P.LK_PARAMS_ASC, P.make_blob(3, 1, 65536, _asc_bytes()), 1)
    assert len(frames) == 2
    _send(b, t, [frames[1], frames[1], frames[0]])  # out of order and duplicated
    assert b.status(t).gen_asc == 3
    b2 = _bridge_with_link()
    other = P.blob_frames(P.LK_PARAMS_ASC, P.make_blob(4, 1, 65536, _asc_bytes(f_b=30000)), 2)
    _send(b2, t, [frames[0], other[1]])  # halves of two different blobs never combine
    assert b2.status(t).gen_asc == 0
    _send(b2, t + 500_000, [frames[0]])  # a half that never completes is dropped after 100 ms
    _send(b2, t + 900_000, [frames[1]])
    assert b2.status(t + 900_000).gen_asc == 0
    _send(b2, t + 1_000_000, frames)
    assert b2.status(t + 1_000_000).gen_asc == 3
    bad = P.Frame(P.LK_PARAMS_ASC, b"\x00" * 50, flags=1, frag_idx=0, frag_cnt=2)  # a non-final fragment must be full
    n = b2.status(t).params_rejected
    _send(b2, t + 2_000_000, [bad])
    assert b2.status(t).params_rejected == n + 1


def test_random_valid_parameter_sets_never_break_the_output_invariants():
    rng = random.Random(8)
    base = [getattr(fixed_params()[0], f) for f in P.FixedParams.__dataclass_fields__]
    accepted = 0
    for trial in range(40):
        vals = list(base)
        for i in rng.sample(range(len(vals)), 6):  # perturb a few fields at a time, so many sets stay valid
            vals[i] = int(vals[i] * rng.choice([0.25, 0.5, 2, 4])) if vals[i] else vals[i]
        b = _bridge_with_link()
        t = 1_000_000
        _send(b, t, P.blob_frames(P.LK_PARAMS_ASC, P.make_blob(1, 1, 65536, struct.pack("<38i", *vals)), 1))
        if not b.status(t).gen_asc:
            continue
        accepted += 1
        r = Rig(SimMouse("m16"), asc=FixedParams(*vals), tremor=fixed_params()[1])
        assert r.engage_fast()
        with_scene(r, 600.0, 200.0)
        rg = np.random.default_rng(trial)
        seq = []
        for _ in range(3000):
            dx, dy = int(rg.integers(-80, 80)), int(rg.integers(-80, 80))
            seq.append((dx, dy))
            r.move(dx, dy)
            r.step()
        reps = bridge_reports(r)
        for (_, _, _, out, _), (dx, dy) in zip(reps, seq):
            assert ok_axis(xy("m16", out)[0], dx) and ok_axis(xy("m16", out)[1], dy)
        assert r.status().invariant_viol == 0
    assert accepted >= 10


# ------------------------------------------------------------------ the SPI link
def test_crc32_in_c_matches_zlib():
    rng = random.Random(9)
    for n in list(range(0, 130)) + [255, 256, 1000]:
        data = bytes(rng.randrange(256) for _ in range(n))
        assert crc32(data) == zlib.crc32(data)


def test_frames_roundtrip_and_every_single_bit_flip_is_rejected_by_both_implementations():
    f = P.Frame(P.LK_HELLO, bytes(range(60)), seq=77, ack=5)
    raw = P.pack_frame(f)
    assert P.unpack_frame(raw) == f
    b = _bridge_with_link()
    b.link_rx(1000, raw)
    assert b.status(1000).link_rx_ok == 1
    for i in range(len(raw) * 8):
        bad = bytearray(raw)
        bad[i // 8] ^= 1 << (i % 8)
        assert P.unpack_frame(bytes(bad)) is None
        b.link_rx(2000, bytes(bad))
    s = b.status(2000)
    assert s.link_rx_ok == 1 and s.link_rx_bad == 1024
    rng = random.Random(10)
    for _ in range(3000):  # up to 3 flipped bits: CRC-32 detects all of them at this length
        bad = bytearray(raw)
        for i in rng.sample(range(1024), rng.randint(2, 3)):
            bad[i // 8] ^= 1 << (i % 8)
        assert P.unpack_frame(bytes(bad)) is None
    assert P.unpack_frame(raw[:-1]) is None and P.unpack_frame(raw + b"\0") is None


def test_status_telemetry_and_timesync_frames_parse_back_to_what_the_core_knows():
    r = with_scene(rig("logi"), 600.0, 0.0)
    r.module.telemetry.clear()
    for k in range(200):
        r.move(6, -2, buttons=(k // 50) % 2)
        r.step()
    r.run(30)
    r.module.send(r.t, [P.tsync_frame(123456789)])
    r.module.send(r.t, [P.Frame(P.LK_NOP)] * 3)
    tel = r.module.telemetry
    assert len(tel) >= 190
    sample = tel[len(tel) // 2]
    assert (sample.raw_dx, sample.raw_dy) == (6, -2) and abs(sample.out_dx) <= 6 and abs(sample.out_dy) <= 2
    assert 0.1 <= sample.k <= 1.0 and sample.buttons in (0, 1)
    assert [x.t_us for x in tel] == sorted(x.t_us for x in tel)
    st = r.module.statuses[-1]
    s = r.status()
    assert st["state"] == s.state == S_ASSIST and st["gen_asc"] == s.gen_asc and st["vid"] == 0x046D and st["pid"] == 0xC077
    assert st["n_motion_if"] == 1 and st["params_asc_ok"] and st["link_ok"] and st["healthy"]
    assert r.module.tsync and r.module.tsync[-1]["m_t"] == 123456789
    assert r.module.bad_rx == 0


def test_telemetry_is_best_effort_it_drops_the_oldest_counts_them_and_marks_the_gap():
    r = rig("m16")
    r.module.silent = True  # nobody drains the ring
    for _ in range(2500):
        r.move(3, 1)
        r.step()
    s = r.status()
    assert s.telem_dropped > 0 and s.telem_pending == 2048
    r.module.silent = False
    r.module.telemetry.clear()
    r.module.send(r.t, [P.Frame(P.LK_NOP)] * 400)
    tel = r.module.telemetry
    assert tel and tel[0].gap_before  # the first sample after the loss is marked
    assert not any(x.gap_before for x in tel[1:])


# ------------------------------------------------------------------ the lock-in backstop
def test_a_stuck_chain_is_released_by_the_lock_in_backstop_and_the_person_keeps_moving():
    r = rig("m16", testing=True)
    r.b.inject(2)  # the chain outputs zero from now on (a bug / hostile parameters)
    r.pc_reports.clear()
    for _ in range(4000):
        r.move(3, 0)
        r.step()
    reps = bridge_reports(r)
    delivered_x = sum(xy("m16", x[3])[0] for x in reps)
    assert delivered_x >= 0.7 * 3 * 4000  # released most of the time (measured: ~86%)
    s = r.status()
    assert s.lockin_trips >= 5
    assert all(ok_axis(xy("m16", x[3])[0], xy("m16", x[4])[0]) for x in reps)


def test_oscillation_alone_can_not_trip_the_backstop():
    r = rig("m16", testing=True)
    r.b.inject(2)
    for k in range(4000):
        r.move(10 if (k // 5) % 2 else -10, 0)  # a pure 100 Hz shake: no net push
        r.step()
    assert r.status().lockin_trips == 0


def test_the_backstop_does_not_fire_for_people_the_chain_serves_well():
    for name in ("steady", "overshooter"):
        ap, tp = persona_params(name)
        trips = 0
        for i in range(40):
            ba = BridgeAssist(ap, tp)
            run_trial(PERSONAS[name], ba, np.random.default_rng(i))
            trips += ba.b.status(ba.rig.t).lockin_trips
        assert trips == 0, name


def test_with_the_big_tremor_persona_the_backstop_is_rare_and_never_corrupts_the_output():
    ap, tp = persona_params("tremor")
    trips = 0
    n = 40
    for i in range(n):
        ba = BridgeAssist(ap, tp)
        t = run_trial(PERSONAS["tremor"], ba, np.random.default_rng(i))
        trips += ba.b.status(ba.rig.t).lockin_trips > 0
        assert t.amp_violations == 0 and ba.b.status(ba.rig.t).invariant_viol == 0
    assert trips / n <= 0.3  # measured 7-17%: the hold well can swallow small corrective pushes


# ------------------------------------------------------------------ sizes, drift, hygiene
def test_wire_layouts_match_between_c_and_python(tmp_path):
    import subprocess
    from dataopen.bridge.cbridge import CHidMap, CCfg, CSetup, CStatus, CSRC, ASC_CSRC

    cc = shutil.which("gcc") or shutil.which("cc")
    src = tmp_path / "sz.c"
    src.write_text(
        '#include <stdio.h>\n#include <stddef.h>\n#include "bridge.h"\nint main(void){\n'
        'printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu\\n",\n'
        "  sizeof(asc_params_t), sizeof(tremor_params_t), sizeof(hid_motion_t), sizeof(hid_map_t), sizeof(usb_setup_t),\n"
        "  sizeof(bridge_status_t), sizeof(bridge_cfg_t), offsetof(hid_motion_t, btn_off), offsetof(hid_map_t, app));\n"
        "return 0;}\n"
    )
    exe = tmp_path / "sz"
    subprocess.run([cc, "-std=c99", f"-I{CSRC}", f"-I{ASC_CSRC}", str(src), "-o", str(exe)], check=True)
    out = list(map(int, subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()))
    assert out[0] == len(P.asc_blob_bytes(FixedParams())) == 152 and out[1] == len(P.tremor_blob_bytes(FixedTremorParams())) == 60
    assert out[2] == ctypes.sizeof(CMotion) and out[3] == ctypes.sizeof(CHidMap) and out[4] == ctypes.sizeof(CSetup)
    assert out[5] == ctypes.sizeof(CStatus) and out[6] == ctypes.sizeof(CCfg)
    assert out[7] == CMotion.btn_off.offset and out[8] == CHidMap.app.offset
    assert P.ASC_BLOB == 172 and P.TRM_BLOB == 80


BRIDGE_C = ["hid_desc.c", "usb_image.c", "usb_proxy.c", "link.c", "failsafe.c", "bridge.c"]


def _external_symbols(objs, nm="nm"):
    import subprocess

    und, dfn = set(), set()
    for o in objs:
        for ln in subprocess.run([nm, str(o)], capture_output=True, text=True).stdout.splitlines():
            parts = ln.split()
            if len(parts) >= 2 and parts[-2] == "U":
                und.add(parts[-1])
            elif len(parts) == 3 and parts[1] in "TDBRtdbr":
                dfn.add(parts[2])
    return und - dfn


def test_the_core_is_clean_c99_without_libc_dependencies(tmp_path):
    import subprocess
    from dataopen.bridge.cbridge import CSRC, ASC_CSRC

    cc = shutil.which("gcc") or shutil.which("cc")
    objs = []
    for src in [CSRC / n for n in BRIDGE_C] + [ASC_CSRC / "asc_core.c", ASC_CSRC / "tremor_core.c"]:
        o = tmp_path / (src.name + ".o")
        warn = ["-pedantic", "-Wall", "-Wextra", "-Wconversion", "-Wshadow", "-Werror"] if src.parent == CSRC else ["-Wall", "-Wextra"]
        r = subprocess.run(
            [
                cc,
                "-std=c99",
                *warn,
                "-O2",
                "-fno-stack-protector",
                "-U_FORTIFY_SOURCE",
                f"-I{CSRC}",
                f"-I{ASC_CSRC}",
                "-c",
                str(src),
                "-o",
                str(o),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr[:800]
        objs.append(o)
    assert _external_symbols(objs) <= {"memcpy", "memset", "memmove"}  # no malloc, printf, libm: the whole chain, as linked


@pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")
def test_the_whole_chain_cross_compiles_for_a_cortex_m7(tmp_path):
    import subprocess
    from dataopen.bridge.cbridge import CSRC, ASC_CSRC

    probe = subprocess.run(
        ["clang", "--target=thumbv7em-none-eabihf", "-mcpu=cortex-m7", "-ffreestanding", "-c", "-x", "c", "-", "-o", str(tmp_path / "p.o")],
        input="int f(int a){return a;}",
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("clang has no thumb backend")
    shim = tmp_path / "shim"  # a bare-metal toolchain ships its own libc headers; stand in for string.h
    shim.mkdir()
    (shim / "string.h").write_text(
        "#include <stddef.h>\nvoid *memcpy(void *, const void *, size_t);\nvoid *memset(void *, int, size_t);\n"
        "void *memmove(void *, const void *, size_t);\nint memcmp(const void *, const void *, size_t);\n"
    )
    objs = []
    for src in [CSRC / n for n in BRIDGE_C] + [ASC_CSRC / "asc_core.c", ASC_CSRC / "tremor_core.c"]:
        o = tmp_path / (src.name + ".arm.o")
        r = subprocess.run(
            [
                "clang",
                "--target=thumbv7em-none-eabihf",
                "-mcpu=cortex-m7",
                "-Os",
                "-ffreestanding",
                "-std=c99",
                f"-isystem{shim}",
                f"-I{CSRC}",
                f"-I{ASC_CSRC}",
                "-c",
                str(src),
                "-o",
                str(o),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr[:800]
        objs.append(o)
    nm = shutil.which("llvm-nm")
    if nm:
        ext = _external_symbols(objs, nm)
        runtime = {"memcpy", "memset", "memmove"} | {
            n for n in ext if n.startswith("__aeabi_")
        }  # compiler-rt helpers (64-bit division, memory)
        assert ext <= runtime, ext - runtime


def test_hostile_inputs_under_the_sanitizers(tmp_path):
    """Random and mutated descriptors, configurations, link frames, parameter blobs (in and out of range), scenes, reports and events, with
    ASan + UBSan trapping on the first finding. Parse results are checked for self-consistency, outputs for the per-axis invariants."""
    import subprocess
    from dataopen.bridge.cbridge import CSRC, ASC_CSRC, SOURCES

    cc = shutil.which("gcc") or shutil.which("cc")
    from dataopen.bridge.seeds import write_seeds_header

    write_seeds_header(tmp_path / "seeds.h", *fixed_params())
    exe = tmp_path / "fuzz"
    r = subprocess.run(
        [
            cc,
            "-O1",
            "-g",
            "-std=gnu99",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-DBRIDGE_TESTING",
            f"-I{CSRC}",
            f"-I{ASC_CSRC}",
            f"-I{tmp_path}",
            str(__import__("pathlib").Path(__file__).parent / "helpers" / "bridge_fuzz.c"),
            *[str(s) for s in SOURCES],
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 and ("sanitize" in r.stderr or "libasan" in r.stderr or "libubsan" in r.stderr):
        pytest.skip("no sanitizer runtime")
    assert r.returncode == 0, r.stderr[:1500]
    for seed in (1, 2, 3):
        p = subprocess.run([str(exe), "6000", str(seed)], capture_output=True, text=True, timeout=600)
        assert p.returncode == 0 and "fuzz ok" in p.stdout, p.stderr[-1500:]


def test_a_ballistic_tick_beyond_the_cores_exact_range_passes_unchanged_and_the_chain_carries_on():
    r = with_scene(rig("m16"), 600.0, 0.0)
    for _ in range(200):
        r.move(6, 0)
        r.step()
    n = len(bridge_reports(r))
    r.move(1200, -700)  # far beyond 511 counts in one tick
    r.step()
    last = bridge_reports(r)[n]
    assert last[3] == last[4]  # no help, no change
    for _ in range(100):
        r.move(6, 0)
        r.step()
    s = r.status()
    assert s.invariant_viol == 0 and s.k_q16 < 65536 and STATE[s.state] == "ASSIST"


def test_the_cli_runs_the_timeline_the_persona_table_the_bench_and_the_build(capsys, tmp_path):
    from dataopen.cli import build_parser

    def main(argv):
        a = build_parser().parse_args(argv)
        assert a.fn(a) == 0

    main(["bridge", "simulate", "failsafe"])
    out = capsys.readouterr().out
    assert "PANIC pressed" in out and "watchdog" in out and "(dead)" in out
    main(["bridge", "simulate", "persona", "--persona", "steady", "--trials", "3", "--json"])
    import json

    data = json.loads(capsys.readouterr().out)
    assert data["steady"]["through the bridge"]["amp_violations"] == 0
    main(["bridge", "bench", "-n", "20000"])
    out = capsys.readouterr().out
    assert "ASSIST, 1 kHz mouse" in out and "link frame" in out and "sizeof(bridge_t)" in out
    main(["bridge", "build", "--out", str(tmp_path)])
    assert list(tmp_path.glob("bridge_*.so"))
