"""Tremor suppression and the assistive chain: personalization, the one-sided invariants (float / fixed / C), spectral behaviour, what it
must NOT touch (intended motion), numeric agreement of the three implementations, the chain with ASC, and the closed-loop reach task."""
import json
import shutil
import subprocess
import time

import numpy as np
import pytest

from dataopen.assist.chain import AssistChain, overlap_hold
from dataopen.assist.params import AscParams
from dataopen.assist.sim_user import PERSONAS, build_profile, compare_all
from dataopen.assist.tremor import TremorConfig, TremorParams, TremorSuppressor
from dataopen.assist.tremor_fixed import FixedTremor, FixedTremorParams
from dataopen.assist.types import Guard, ObjectOfInterest
from dataopen.bioprofile import profile as P

HAVE_CC = shutil.which("gcc") is not None or shutil.which("cc") is not None


def view_from(jitter_deg=0.45, jitter_hz=8.0, dpc=0.02, n=40, overshoot=0.2, ov_rate=0.5, miss=0.5):
    st = P.ProfileState(profile_id=3, generation=1, deg_per_count=dpc)
    st.stats["jitter_amp"] = P.Stat(jitter_deg, 0.02, n)
    st.stats["jitter_hz"] = P.Stat(jitter_hz, 0.4, n)
    st.stats["t_motor"] = P.Stat(230.0, 25.0, n)
    st.stats["v_max"] = P.Stat(300.0, 30.0, n)
    st.stats["d_brake"] = P.Stat(0.4, 0.03, n)
    st.stats["overshoot"] = P.Stat(overshoot, 0.05, n)
    st.rates = {"overshoot": ov_rate, "undershoot": 0.0, "direction": 0.0, "miss": miss, "lapse": 0.0}
    return P.ProfileView(st.pack())


def strong_params(**kw):
    return TremorParams.from_view(view_from(**kw))


def make_cores(params):
    cores = {"float": TremorSuppressor(params), "fixed": FixedTremor(FixedTremorParams.from_params(params))}
    if HAVE_CC:
        from dataopen.assist.cimpl import CTremor
        cores["c"] = CTremor(FixedTremorParams.from_params(params))
    return cores


def run_stream(core, xs, ys=None, t0=0, dt=1000):
    outx, outy = [], []
    for i, x in enumerate(xs):
        y = 0 if ys is None else int(ys[i])
        o = core.tick(t0 + i * dt, int(x), y)
        outx.append(o[0])
        outy.append(o[1])
    return np.array(outx), np.array(outy)


def band_power(x, lo=3.0, hi=16.0, fs=1000.0):
    x = np.asarray(x, float)
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return float(p[(f >= lo) & (f <= hi)].sum())


def tremor_input(n, f_hz=8.0, amp_counts=30.0, drift=0.0, noise=0.0, seed=0):
    t = np.arange(n) / 1000.0
    v = drift + amp_counts * 2 * np.pi * f_hz / 1000.0 * np.cos(2 * np.pi * f_hz * t)
    if noise:
        v = v + np.random.default_rng(seed).normal(0, noise, n)
    return np.round(v).astype(int)


# ---------------------------------------------------------------- personalization
def test_off_without_confident_jitter_or_with_a_barely_visible_tremor():
    assert not TremorParams.from_view(P.ProfileView(P.ProfileState().pack())).enabled
    assert not TremorParams.from_view(view_from(n=5)).enabled
    assert not TremorParams.from_view(view_from(dpc=0.0)).enabled
    assert not TremorParams.from_view(view_from(jitter_deg=0.05)).enabled                 # below the point where it helps anyone
    assert TremorParams.from_view(view_from(dpc=0.0), deg_per_count=0.02).enabled


def test_strength_frequency_and_cap_follow_the_profile():
    weak, strong = strong_params(jitter_deg=0.15), strong_params(jitter_deg=0.5)
    assert weak.enabled and 0.0 < weak.s_max < strong.s_max <= 0.9
    p8, p5 = strong_params(jitter_hz=8.0), strong_params(jitter_hz=5.0)
    assert p5.a_lp < p8.a_lp                                                              # a slower tremor: a lower split frequency
    assert strong_params(jitter_hz=40.0).a_lp == p8.a_lp                                  # implausible frequency: default 8 Hz
    amp_counts = 0.5 / 0.02
    assert strong.trim_cap == int(np.ceil(1.5 * 2 * np.pi * 8.0 * amp_counts / 1000.0))   # never remove more than the tremor's own size
    assert strong.v_t == pytest.approx(2 * np.pi * 8.0 * amp_counts / 1000.0)


# ---------------------------------------------------------------- the one-sided invariants
def random_stream(rng, n=6000):
    t = 0
    for _ in range(n):
        t += int(rng.choice([1000, 1000, 1000, 1000, 2000, 300, 7000, 120_000]))
        mode = rng.integers(0, 5)
        dx = int(rng.integers(-4, 5)) if mode else 0
        dy = int(rng.integers(-3, 4)) if mode in (2, 3) else 0
        if rng.random() < 0.02:
            dx = int(rng.integers(-400, 400))
        yield t, dx, dy


@pytest.mark.parametrize("core", ["float", "fixed", "c"])
def test_never_amplifies_never_flips_never_creates_motion_and_removes_at_most_the_tremor(core):
    if core == "c" and not HAVE_CC:
        pytest.skip("no C compiler")
    for seed in range(4):
        prm = strong_params(jitter_deg=0.2 + 0.1 * seed, jitter_hz=5 + 2 * seed)
        c = make_cores(prm)[core]
        rng = np.random.default_rng(seed)
        for t, dx, dy in random_stream(rng):
            ox, oy = c.tick(t, dx, dy)
            assert abs(ox) <= abs(dx) and abs(oy) <= abs(dy)
            assert ox == 0 or (ox > 0) == (dx > 0)
            assert oy == 0 or (oy > 0) == (dy > 0)
            if dx == 0 and dy == 0:
                assert (ox, oy) == (0, 0)
            assert abs(dx - ox) <= prm.trim_cap + 1 and abs(dy - oy) <= prm.trim_cap + 1      # +1: the integer carry


def test_a_disabled_filter_is_the_identity():
    for core in make_cores(TremorParams.disabled()).values():
        for t, dx, dy in random_stream(np.random.default_rng(3), 500):
            assert core.tick(t, dx, dy) == (dx, dy)


def test_idle_gives_exactly_zero_and_nothing_stored_is_released_later():
    for name, core in make_cores(strong_params()).items():
        xs = tremor_input(800, amp_counts=30)
        run_stream(core, xs)
        t = 800_000
        for i in range(400):                                                              # the hand stops completely
            assert core.tick(t + i * 1000, 0, 0) == (0, 0), name
        o = core.tick(t + 400_000, 1, 0)
        assert abs(o[0]) <= 1


def test_a_long_pause_forgets_the_state_and_sparse_reports_are_not_idle():
    c = TremorSuppressor(strong_params())
    run_stream(c, tremor_input(500, amp_counts=30))
    assert c.e_band > 0
    c.tick(2_000_000, 0, 0)                                                              # a second later: stale state is gone
    assert c.e_band == 0.0 and c.e_lp == 0.0
    c2 = TremorSuppressor(strong_params())
    for i in range(20):                                                                  # one report every 3 ms
        c2.tick(i * 3000, 1, 0)
    assert c2.e_lp > 0


# ---------------------------------------------------------------- spectral behaviour and what must pass untouched
@pytest.mark.parametrize("f_hz", [6.0, 8.0, 12.0])
def test_a_hold_with_tremor_is_cut_by_at_least_15_db_in_the_tremor_band(f_hz):
    prm = strong_params(jitter_hz=f_hz)
    for name, core in make_cores(prm).items():
        xs = tremor_input(6000, f_hz=f_hz, amp_counts=30)
        ys, _ = run_stream(core, xs)
        pin, pout = band_power(np.cumsum(xs)[500:]), max(band_power(np.cumsum(ys)[500:]), 1e-9)
        assert 10 * np.log10(pin / pout) >= 15.0, (name, f_hz)


def test_intended_motion_passes_flick_glide_and_slow_sweeps():
    for name, core in make_cores(strong_params()).items():
        flick = np.zeros(4000, int)
        flick[1000:1150] = np.round(30 * np.sin(np.linspace(0, np.pi, 150)) ** 2)
        ys, _ = run_stream(core, flick)
        assert ys.sum() >= 0.98 * flick.sum() and (np.abs(ys) <= np.abs(flick)).all(), name
        core.reset()
        glide = np.zeros(4000, int)
        glide[500:3500] = 1
        ys, _ = run_stream(core, glide)
        assert ys.sum() >= 0.95 * glide.sum(), name
        core.reset()
        t = np.arange(8000) / 1000.0                                                     # a 0.5 Hz sweep of +-800 counts: a deliberate pass
        sweep = np.round(2 * np.pi * 0.5 * 800 / 1000 * np.cos(2 * np.pi * 0.5 * t)).astype(int)
        ys, _ = run_stream(core, sweep)
        assert abs(np.cumsum(ys)[-1] - np.cumsum(sweep)[-1]) <= 0.02 * np.abs(sweep).sum(), name
        assert np.abs(ys).sum() >= 0.97 * np.abs(sweep).sum(), name
        lag = np.argmax([np.dot(np.cumsum(ys)[200:7000], np.cumsum(sweep)[200 - L:7000 - L]) for L in range(0, 80)])
        assert lag <= 30, name                                                           # < 30 ms of delay on a slow deliberate movement
        core.reset()  # slow-sweep delay
        slow = np.round(2 * np.pi * 0.5 * 200 / 1000 * np.cos(2 * np.pi * 0.5 * t)).astype(int)
        ys, _ = run_stream(core, slow)
        assert np.abs(ys).sum() >= 0.5 * np.abs(slow).sum(), name                        # the honest cost: part is removed, never invented



def test_progress_is_kept_when_the_hand_clearly_moves_and_slowed_when_it_barely_does():
    prm = strong_params()
    kept = {}
    for drift in (0.5, 1.0, 2.0):                                                        # tremor velocity amplitude is ~1.5 counts/ms here
        c = TremorSuppressor(prm)
        xs = tremor_input(6000, amp_counts=30, drift=drift)
        ys, _ = run_stream(c, xs)
        kept[drift] = ys.sum() / xs.sum()
    assert kept[2.0] >= 0.97 and kept[1.0] >= 0.9 and kept[0.5] >= 0.55
    assert kept[0.5] < kept[1.0] <= kept[2.0] + 0.02


# ---------------------------------------------------------------- the implementations agree
def test_fixed_point_follows_the_float_reference_to_within_one_count_per_report():
    prm = strong_params()
    f, g = TremorSuppressor(prm), FixedTremor(FixedTremorParams.from_params(prm))
    rng = np.random.default_rng(11)
    xs = tremor_input(20_000, amp_counts=30, drift=0.4, noise=0.4, seed=1)
    xs[8000:8500] = 0
    xs[12_000:12_100] = np.round(25 * np.sin(np.linspace(0, np.pi, 100)) ** 2)
    ys = np.round(rng.normal(0, 0.6, len(xs))).astype(int)
    diff, sf, sg = 0, 0, 0
    for i in range(len(xs)):
        a, b = f.tick(i * 1000, int(xs[i]), int(ys[i])), g.tick(i * 1000, int(xs[i]), int(ys[i]))
        diff = max(diff, abs(a[0] - b[0]), abs(a[1] - b[1]))
        sf += a[0]
        sg += b[0]
    assert diff <= 1 and abs(sf - sg) <= 6


@pytest.mark.skipif(not HAVE_CC, reason="no C compiler")
def test_the_c_core_is_bit_identical_to_the_python_fixed_point_model():
    from dataopen.assist.cimpl import CTremor
    prm = strong_params()
    fp = FixedTremorParams.from_params(prm)
    total = 0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        g, c = FixedTremor(fp), CTremor(fp)
        for t, dx, dy in random_stream(rng, 8000):
            assert g.tick(t, dx, dy) == c.tick(t, dx, dy)
            total += 1
    g, c = FixedTremor(fp), CTremor(fp)
    xs = tremor_input(10_000, amp_counts=30, noise=0.5)
    for i, x in enumerate(xs):
        assert g.tick(i * 1000, int(x), 0) == c.tick(i * 1000, int(x), 0)
        total += 1
    assert total > 40_000


@pytest.mark.skipif(not HAVE_CC, reason="no C compiler")
def test_c_tremor_structs_match_their_ctypes_declarations(tmp_path):
    import ctypes

    from dataopen.assist import cimpl
    src = ('#include <stdio.h>\n#include "tremor_core.h"\nint main(void){printf("%zu %zu\\n", sizeof(tremor_params_t), '
           'sizeof(tremor_state_t));return 0;}\n')
    (tmp_path / "s.c").write_text(src)
    subprocess.run(["gcc", "-std=gnu99", "-I", str(cimpl.CSRC), "-o", str(tmp_path / "s"), str(tmp_path / "s.c")], check=True)
    a, b = (int(x) for x in subprocess.run([str(tmp_path / "s")], capture_output=True, text=True).stdout.split())
    assert (a, b) == (ctypes.sizeof(cimpl.CTremorParams), ctypes.sizeof(cimpl.CTremorState)) and b == 112


def test_a_tick_is_far_inside_the_1khz_budget():
    c = FixedTremor(FixedTremorParams.from_params(strong_params()))
    t0 = time.perf_counter()
    n = 4000
    for i in range(n):
        c.tick(i * 1000, 1 + (i % 3), i % 2)
    assert (time.perf_counter() - t0) / n * 1e6 < 200


# ---------------------------------------------------------------- the chain
def test_chain_only_shrinks_and_asc_sees_the_untouched_motion():
    view = build_profile(PERSONAS["tremor"], seed=1)
    ap, tp = AscParams.from_view(view), TremorParams.from_view(view)
    obj = ObjectOfInterest(1, 600.0, 0.0, 30.0)
    for impl in ("float", "fixed") + (("c",) if HAVE_CC else ()):
        chain = AssistChain.build(ap, tp, impl)
        solo = AssistChain.build(overlap_hold(ap, tp, 0.75), TremorParams.disabled(), impl, hold_overlap=0.0)  # ASC alone, same stream
        rng = np.random.default_rng(2)
        px = 0.0
        for i in range(3000):
            raw = int(rng.integers(-2, 9)) if i < 1500 else 0
            ry = int(rng.integers(-2, 3)) if i < 1500 else 0
            a = chain.tick(i * 1000, raw, ry, px, 0.0, obj)
            s = solo.tick(i * 1000, raw, ry, px, 0.0, obj)
            px += a.dx
            assert (a.k, a.guard, a.reason) == (s.k, s.guard, s.reason)                  # the guard and K never see the filter
            assert (a.asc_dx, a.asc_dy) == (s.dx, s.dy)
            assert abs(a.dx) <= abs(a.asc_dx) <= abs(raw) and abs(a.dy) <= abs(a.asc_dy) <= abs(ry)
            assert a.dx == 0 or (a.dx > 0) == (raw > 0)
            if raw == 0 and ry == 0:
                assert (a.dx, a.dy) == (0, 0)
            if a.guard != Guard.OPEN:
                assert a.k == 1.0


def test_the_hold_well_is_scaled_down_where_the_tremor_filter_already_does_the_job():
    view = build_profile(PERSONAS["tremor"], seed=1)
    ap, tp = AscParams.from_view(view), TremorParams.from_view(view)
    assert overlap_hold(ap, tp, 0.75).hold_scale == pytest.approx(ap.hold_scale * (1 - 0.75 * tp.s_max / 0.9))
    assert overlap_hold(ap, TremorParams.disabled(), 0.75) is ap and overlap_hold(AscParams.disabled(), tp, 0.75).hold_scale == 1.0


def test_swapping_parameters_at_run_time_keeps_working():
    view = build_profile(PERSONAS["tremor"], seed=1)
    ap, tp = AscParams.from_view(view), TremorParams.from_view(view)
    for impl in ("float", "fixed"):
        chain = AssistChain.build(AscParams.disabled(), TremorParams.disabled(), impl)
        chain.set_params_from(ap, tp)
        out = [chain.tick(i * 1000, 3, 0, float(i), 0.0, ObjectOfInterest(1, 600.0, 0.0, 30.0)) for i in range(50)]
        assert all(o.k <= 1.0 for o in out)


# ---------------------------------------------------------------- closed loop on simulated people
@pytest.fixture(scope="module")
def runs():
    return {n: compare_all(PERSONAS[n], n=40, seed=1, view=build_profile(PERSONAS[n], seed=1)) for n in PERSONAS}


def test_closed_loop_has_no_violations_anywhere(runs):
    for name, variants in runs.items():
        for v, s in variants.items():
            assert s["guard_violations"] == 0 and s["amp_violations"] == 0, (name, v)


def test_people_without_tremor_are_untouched_by_the_filter(runs):
    for name in ("steady", "overshooter"):
        v = runs[name]
        assert v["tremor"] == v["none"]                                                  # the filter is off for them, bit for bit
        assert v["chain"] == v["asc"]


def test_for_a_tremor_user_the_filter_alone_stabilizes_the_hold_at_almost_no_cost_in_time(runs):
    v = runs["tremor"]
    assert v["tremor"]["hold_rms_px"] <= 0.5 * v["none"]["hold_rms_px"]
    assert v["tremor"]["final_err_px"] <= 0.6 * v["none"]["final_err_px"]
    assert v["tremor"]["t_acquire_ms"] <= 1.15 * v["none"]["t_acquire_ms"] and v["tremor"]["acquired"] == 1.0


def test_the_chain_adds_overshoot_control_and_a_calmer_hold_than_asc_alone(runs):
    v = runs["tremor"]
    assert v["chain"]["overshoot_rate"] <= 0.5 * v["none"]["overshoot_rate"]
    assert v["chain"]["hold_rms_px"] <= v["asc"]["hold_rms_px"] and v["chain"]["acquired"] == 1.0
    assert v["chain"]["t_acquire_ms"] <= 1.3 * v["asc"]["t_acquire_ms"]                 # not much slower than ASC alone


def test_cli_chain_comparison(capsys):
    from dataopen.cli import main
    with pytest.raises(SystemExit) as e:
        main(["assist", "simulate", "--persona", "tremor", "--trials", "3", "--chain", "--json"])
    assert e.value.code == 0
    d = json.loads(capsys.readouterr().out)
    assert set(d["tremor"]) == {"none", "asc", "tremor", "chain"}
    assert TremorConfig().r_lo < TremorConfig().r_hi
