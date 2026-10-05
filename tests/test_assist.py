"""Adaptive Sensitivity Correction: the curve, the personalization, the safety invariants (float / fixed-point / C), the pre-reaction guard,
fatigue, release rules, numeric agreement between the three implementations, and a closed-loop reach task on simulated people."""
import json
import math
import shutil
import subprocess
import time

import numpy as np
import pytest

from dataopen.assist import fixed as fx
from dataopen.assist.feed import ProfileFeed, StaticObjects, blend_params
from dataopen.assist.fixed import FixedAsc, FixedParams
from dataopen.assist.model import AdaptiveSensitivity, smoothstep
from dataopen.assist.params import AscParams
from dataopen.assist.sim_user import PERSONAS, build_profile, compare, run_trial
from dataopen.assist.types import Guard, ObjectOfInterest, Reason
from dataopen.bioprofile import profile as P

HAVE_CC = shutil.which("gcc") is not None or shutil.which("cc") is not None
OBJ = ObjectOfInterest(1, 600.0, 0.0, 30.0, t_appear_us=None)


# ---------------------------------------------------------------- helpers
def view_from(overshoot=0.3, ov_rate=0.9, miss=0.7, jitter_deg=0.12, dpc=0.02, fatigue=None, n=40, d_brake=0.4, v_max=300.0,
              t_motor=230.0, t_sigma=25.0):
    st = P.ProfileState(profile_id=7, generation=1, deg_per_count=dpc)
    st.stats["t_motor"] = P.Stat(t_motor, t_sigma, n)
    st.stats["v_max"] = P.Stat(v_max, 30.0, n)
    st.stats["d_brake"] = P.Stat(d_brake, 0.03, n)
    st.stats["overshoot"] = P.Stat(overshoot, 0.05, n)
    st.stats["jitter_amp"] = P.Stat(jitter_deg, 0.01, n)
    st.rates = {"overshoot": ov_rate, "undershoot": 0.0, "direction": 0.0, "miss": miss, "lapse": 0.0}
    if fatigue:
        st.fatigue = P.Fatigue(3000.0, 40.0, 0.5, fatigue, fatigue, 220.0, True)
    return P.ProfileView(st.pack())


def feed_moving(asc, obj, n_ms=300, speed=8, start=0, dt_us=1000, px0=0.0, gain=1.0, t0=0):
    """Hand moves toward +x at `speed` counts/ms; the cursor follows the OUTPUT. Returns the list of TickOut."""
    outs, px = [], px0
    for i in range(n_ms):
        t = t0 + (start + i) * dt_us
        o = asc.tick(t, speed, 0, px, 0.0, obj)
        px += o.dx * gain
        outs.append(o)
    return outs


def make_cores(params):
    cores = {"float": AdaptiveSensitivity(params), "fixed": FixedAsc(FixedParams.from_params(params))}
    if HAVE_CC:
        from dataopen.assist.cimpl import CAsc
        cores["c"] = CAsc(FixedParams.from_params(params))
    return cores


@pytest.fixture(scope="module")
def views():
    return {n: build_profile(PERSONAS[n], seed=1) for n in PERSONAS}


# ---------------------------------------------------------------- the curve
def test_smoothstep_is_exactly_zero_and_one_outside_and_c2_at_the_ends():
    assert smoothstep(-1.0) == 0.0 and smoothstep(0.0) == 0.0 and smoothstep(1.0) == 1.0 and smoothstep(2.0) == 1.0
    xs = np.linspace(0, 1, 201)
    ys = np.array([smoothstep(x) for x in xs])
    assert (np.diff(ys) >= 0).all()
    h = 1e-4
    assert smoothstep(h) < 1e-10 and 1 - smoothstep(1 - h) < 1e-10                 # first AND second derivative vanish at both ends
    assert abs(smoothstep(0.5) - 0.5) < 1e-12


def test_fixed_point_smoothstep_matches_the_float_curve():
    for x in np.linspace(-0.2, 1.2, 141):
        assert abs(fx.smoothstep(fx.q(x)) / fx.ONE - smoothstep(x)) < 2e-4


def test_integer_helpers_follow_c_semantics():
    assert fx.sdiv(-7, 2) == -3 and fx.sdiv(7, -2) == -3 and fx.sdiv(-7, -2) == 3          # truncation toward zero, not floor
    assert fx.mulq(fx.q(0.5), fx.q(0.5)) == fx.q(0.25) and fx.mulq(-fx.q(0.5), fx.q(0.5)) == -fx.q(0.25)
    assert fx.isqrt(0) == 0 and fx.isqrt(15) == 3 and fx.isqrt(16) == 4 and fx.isqrt(10**18) == 10**9
    assert fx.scale_us(1000, 1000) == 1 and fx.scale_us(-1000, 1000) == -1 and fx.scale_us(400, 1000) == 0


# ---------------------------------------------------------------- personalization from the profile
def test_no_confident_profile_or_unknown_sensitivity_means_no_assistance():
    assert not AscParams.from_view(P.ProfileView(P.ProfileState().pack())).enabled
    assert not AscParams.from_view(view_from(n=5)).enabled                                # too few samples
    assert not AscParams.from_view(view_from(dpc=0.0)).enabled                            # degrees per count unknown
    assert AscParams.from_view(view_from(dpc=0.0), deg_per_count=0.02).enabled            # ... unless the caller knows it


def test_a_person_who_does_not_need_help_gets_none_and_a_struggling_one_gets_more():
    easy = AscParams.from_view(view_from(overshoot=0.03, ov_rate=0.0, miss=0.05, jitter_deg=0.05))
    hard = AscParams.from_view(view_from(overshoot=0.3, ov_rate=0.9, miss=0.7, jitter_deg=0.4))
    assert easy.enabled and easy.s_brake == 0.0 and hard.s_brake > 1.0
    assert hard.tremor_counts > easy.tremor_counts


def test_fatigue_weakens_the_assistance_never_strengthens_it():
    rested = AscParams.from_view(view_from())
    tired = AscParams.from_view(view_from(fatigue=4.0))
    half = AscParams.from_view(view_from(fatigue=2.0))
    assert tired.s_brake < half.s_brake < rested.s_brake
    assert tired.hold_scale == pytest.approx(0.5) and rested.hold_scale == 1.0
    # in the loop: the same input stream never gets MORE resistance when the person is tired
    ks = {}
    for name, prm in (("rested", rested), ("tired", tired)):
        asc = AdaptiveSensitivity(prm)
        ks[name] = min(o.k for o in feed_moving(asc, OBJ, 400, speed=10))
    assert ks["tired"] >= ks["rested"]


def test_zone_and_reaction_limits_come_from_the_profile():
    p = AscParams.from_view(view_from(t_motor=240.0, t_sigma=30.0))
    assert p.t_lo_us == 180_000                                                           # 240 - 2*30 ms
    assert AscParams.from_view(view_from(t_motor=120.0, t_sigma=40.0)).t_lo_us == 100_000    # never below a plausible human reaction
    assert p.v_on == pytest.approx(12.0 / 0.02 / 1000.0)
    assert 0.15 <= p.f_b <= 0.9


# ---------------------------------------------------------------- the pre-reaction guard
@pytest.mark.parametrize("core", ["float", "fixed", "c"])
def test_k_is_exactly_one_until_the_person_moves(core):
    if core == "c" and not HAVE_CC:
        pytest.skip("no C compiler")
    asc = make_cores(AscParams.from_view(view_from()))[core]
    for i in range(400):                                                # a still hand with an object right there
        o = asc.tick(i * 1000, 0, 0, 570.0, 0.0, OBJ)
        assert o.k == 1.0 and o.dx == 0 and o.dy == 0 and o.guard == Guard.LOCKED
    for i in range(3):                                                   # sensor noise: a count now and then is not a movement
        o = asc.tick((400 + i) * 1000, 1, 0, 570.0, 0.0, OBJ)
        assert o.k == 1.0 and o.guard == Guard.LOCKED


def test_correction_waits_for_the_movement_and_then_ramps_in_without_a_jump():
    asc = AdaptiveSensitivity(AscParams.from_view(view_from()))
    outs = feed_moving(asc, OBJ, 400, speed=8, px0=0.0)
    first = next(i for i, o in enumerate(outs) if o.guard == Guard.OPEN)
    assert first >= 5                                                    # not before the 6 ms movement test passed
    assert all(o.k == 1.0 for o in outs[:first + 1])
    ks = [o.k for o in outs]
    assert max(abs(a - b) for a, b in zip(ks, ks[1:])) < 0.05            # K never steps


def test_movement_that_began_sooner_than_the_person_can_react_does_not_open_the_guard():
    prm = AscParams.from_view(view_from(t_motor=230.0, t_sigma=25.0))     # T_lo = 180 ms
    early = ObjectOfInterest(2, 600.0, 0.0, 30.0, t_appear_us=100_000)    # appears at 100 ms, the hand starts at ~0
    asc = AdaptiveSensitivity(prm)
    outs = feed_moving(asc, early, 250, speed=8)
    assert all(o.k == 1.0 for o in outs) and any(o.reason == Reason.STIMULUS_LOCK for o in outs)
    late = ObjectOfInterest(3, 600.0, 0.0, 30.0, t_appear_us=0)           # appeared at 0, the hand starts at ~0 too: also too early
    asc2 = AdaptiveSensitivity(prm)
    assert all(o.k == 1.0 for o in feed_moving(asc2, late, 250, speed=8))
    # the same object, but the person starts moving 250 ms after it appeared (> T_lo): assistance may begin
    asc3 = AdaptiveSensitivity(prm)
    outs3 = feed_moving(asc3, late, 700, speed=8, start=250)
    assert any(o.guard == Guard.OPEN for o in outs3) and min(o.k for o in outs3) < 1.0


def test_stopping_relocks_the_guard_and_a_new_movement_is_a_new_reaction():
    asc = AdaptiveSensitivity(AscParams.from_view(view_from()))
    feed_moving(asc, OBJ, 300, speed=8)
    for i in range(300, 500):                                            # the hand stops for 200 ms (> 150 ms)
        o = asc.tick(i * 1000, 0, 0, 400.0, 0.0, OBJ)
    assert o.guard == Guard.LOCKED and o.k == 1.0


def test_no_object_means_no_correction():
    asc = AdaptiveSensitivity(AscParams.from_view(view_from()))
    outs = [asc.tick(i * 1000, 8, 0, float(i * 8), 0.0, None) for i in range(200)]
    assert all(o.k == 1.0 for o in outs) and outs[-1].reason == Reason.WAIT_OBJECT


# ---------------------------------------------------------------- shape: slope, well, direction, release
def run_to_object(prm, start_x=0.0, speed=10, n=600, obj=OBJ):
    asc = AdaptiveSensitivity(prm)
    xs, ks, px = [], [], start_x
    for i in range(n):
        o = asc.tick(i * 1000, speed, 0, px, 0.0, obj)
        px += o.dx
        xs.append(px)
        ks.append(o.k)
    return np.array(xs), np.array(ks)


def test_resistance_grows_smoothly_toward_the_object_and_is_absent_far_away():
    xs, ks = run_to_object(AscParams.from_view(view_from()), speed=2, n=500)       # a slow approach: the look-ahead is short
    far = ks[xs < 100]
    assert far.size and (far == 1.0).all()                               # exactly 1.0 far outside the zone (no sigmoid tail)
    inside = ks[(xs > 540) & (xs < 660)]
    assert inside.size and inside.min() < 0.5
    assert np.abs(np.diff(ks)).max() < 0.05
    xs2, ks2 = run_to_object(AscParams.from_view(view_from()), speed=10, n=300)    # a fast one is slowed earlier (it needs more room)
    assert ks2.min() < 1.0


def test_a_steady_person_is_left_alone_and_a_tremor_user_gets_a_hold_well_sized_to_the_object():
    steady = AscParams.from_view(view_from(overshoot=0.03, ov_rate=0.0, miss=0.05, jitter_deg=0.05))
    xs, ks = run_to_object(steady, n=700)
    assert (ks == 1.0).all()
    tremor = AscParams.from_view(view_from(jitter_deg=0.45))
    asc = AdaptiveSensitivity(tremor)
    assert asc._s_hold(30.0) > asc._s_hold(60.0) > 0.0                  # a smaller object needs a stronger well
    assert AdaptiveSensitivity(AscParams.from_view(view_from(jitter_deg=0.05)))._s_hold(30.0) == 0.0


def approach(prm, until=590.0, speed=10):
    """Drive the hand toward the object until the cursor (following the corrected output) is almost on it."""
    asc, px, i = AdaptiveSensitivity(prm), 0.0, 0
    while px < until and i < 3000:
        px += asc.tick(i * 1000, speed, 0, px, 0.0, OBJ).dx
        i += 1
    return asc, px, i


def test_moving_away_from_the_object_gets_extra_damping_but_a_fast_departure_releases_it():
    prm = AscParams.from_view(view_from())

    def leave(speed):
        asc, px, i = approach(prm)
        ks = []
        for j in range(i, i + 60):                                       # now going back at `speed` counts/ms
            o = asc.tick(j * 1000, -speed, 0, px, 0.0, OBJ)
            px += o.dx
            ks.append(o.k)
        return float(np.mean(ks[20:]))
    slow, fast = leave(3), leave(40)
    assert slow < 0.9 and fast > slow + 0.2                              # a slow retreat is damped, a quick deliberate leave is not


def test_the_person_can_never_be_trapped_the_damping_times_out():
    prm = AscParams.from_view(view_from())
    asc = AdaptiveSensitivity(prm)
    ks = []
    for i in range(3800):                                                # the cursor stays by the object for 3.8 s of continuous input
        ks.append(asc.tick(i * 1000, 1, 0, 590.0, 0.0, OBJ).k)
    assert min(ks) < 0.7 and ks[-1] > 0.999                              # helped at first, fully released by the end


# ---------------------------------------------------------------- safety invariants on random scenarios, all three cores
def random_scenario(rng, n=2500):
    t, px, py = 0, float(rng.uniform(0, 400)), float(rng.uniform(-50, 50))
    for _ in range(n):
        t += int(rng.choice([1000, 1000, 1000, 500, 2000, 8000]))
        mode = rng.integers(0, 4)
        dx = int(rng.integers(-12, 13)) if mode else 0
        dy = int(rng.integers(-5, 6)) if mode == 3 else 0
        obj = None if rng.random() < 0.05 else ObjectOfInterest(int(rng.integers(1, 4)), float(rng.uniform(300, 900)),
                                                                float(rng.uniform(-100, 100)), float(rng.uniform(0, 40)),
                                                                None if rng.random() < 0.5 else int(rng.integers(0, 50) * 10_000))
        yield t, dx, dy, px, py, obj
        px += dx * float(rng.uniform(0.2, 1.0))
        py += dy * 0.5


@pytest.mark.parametrize("core", ["float", "fixed", "c"])
def test_invariants_hold_on_random_input(core):
    if core == "c" and not HAVE_CC:
        pytest.skip("no C compiler")
    for seed in range(4):
        prm = AscParams.from_view(view_from(jitter_deg=0.1 + 0.1 * seed))
        asc = make_cores(prm)[core]
        rng = np.random.default_rng(seed)
        last_k = 1.0
        for t, dx, dy, px, py, obj in random_scenario(rng):
            o = asc.tick(t, dx, dy, px, py, obj)
            assert prm.cfg.k_floor - 1e-4 <= o.k <= 1.0                         # never amplifies, never below the floor
            if o.guard != Guard.OPEN:
                assert o.k == 1.0                                              # closed guard: exactly 1.0
            assert abs(o.dx) <= abs(dx) and abs(o.dy) <= abs(dy)               # never more than the hand did
            assert o.dx == 0 or (o.dx > 0) == (dx > 0)
            assert o.dy == 0 or (o.dy > 0) == (dy > 0)
            if dx == 0 and dy == 0:
                assert o.dx == 0 and o.dy == 0                                 # nothing from nothing
            last_k = o.k
        assert last_k <= 1.0


def test_hundreds_of_thousands_of_ticks_of_noise_never_break_the_contract():
    prm = AscParams.from_view(view_from())
    asc = FixedAsc(FixedParams.from_params(prm))
    rng = np.random.default_rng(99)
    t = 0
    for _ in range(40_000):
        t += int(rng.choice([1000, 1000, 999, 3000, 100_000]))
        o = asc.tick(t, int(rng.integers(-3000, 3000)), int(rng.integers(-3000, 3000)), float(rng.uniform(-1e4, 1e4)),
                     float(rng.uniform(-1e4, 1e4)), ObjectOfInterest(1, float(rng.uniform(-1e4, 1e4)), 0.0, 1e3, None))
        assert 0.1 - 1e-4 <= o.k <= 1.0 and abs(o.dx) <= 3000


def test_repeated_and_backward_timestamps_are_survivable():
    asc = AdaptiveSensitivity(AscParams.from_view(view_from()))
    for t in (1000, 1000, 500, 2000, 2000, 90_000, 91_000):
        o = asc.tick(t, 5, 0, 100.0, 0.0, OBJ)
        assert 0.1 <= o.k <= 1.0


# ---------------------------------------------------------------- the three implementations agree
def replay_inputs(persona, view, seeds=3):
    prm = AscParams.from_view(view)
    out = []
    for s in range(seeds):
        rec = []
        run_trial(persona, AdaptiveSensitivity(prm), np.random.default_rng(s), record=rec)
        out.append(rec)
    return prm, out


def test_fixed_point_tracks_the_float_reference_within_a_fraction_of_a_percent(views):
    for name in ("overshooter", "tremor"):
        prm, recs = replay_inputs(PERSONAS[name], views[name])
        obj = ObjectOfInterest(1, 600.0, 0.0, 30.0, t_appear_us=0)
        for rec in recs:
            f, g = AdaptiveSensitivity(prm), FixedAsc(FixedParams.from_params(prm))
            worst = max(abs(f.tick(t, dx, dy, px, py, obj).k - g.tick(t, dx, dy, px, py, obj).k) for t, dx, dy, px, py in rec)
            assert worst < 0.01, (name, worst)


@pytest.mark.skipif(not HAVE_CC, reason="no C compiler")
def test_the_c_core_is_bit_identical_to_the_python_fixed_point_model(views):
    from dataopen.assist.cimpl import CAsc
    total = 0
    for name in ("overshooter", "tremor"):
        prm, recs = replay_inputs(PERSONAS[name], views[name])
        fp = FixedParams.from_params(prm)
        obj = ObjectOfInterest(1, 600.0, 0.0, 30.0, t_appear_us=0)
        for rec in recs:
            g, c = FixedAsc(fp), CAsc(fp)
            for t, dx, dy, px, py in rec:
                a, b = g.tick(t, dx, dy, px, py, obj), c.tick(t, dx, dy, px, py, obj)
                assert (a.k, a.dx, a.dy, a.guard, a.reason, a.s) == (b.k, b.dx, b.dy, b.guard, b.reason, b.s)
                total += 1
    rng = np.random.default_rng(5)                                           # and on adversarial random input
    prm = AscParams.from_view(view_from())
    fp = FixedParams.from_params(prm)
    g, c = FixedAsc(fp), CAsc(fp)
    for t, dx, dy, px, py, obj in random_scenario(rng, 6000):
        a, b = g.tick(t, dx, dy, px, py, obj), c.tick(t, dx, dy, px, py, obj)
        assert (a.k, a.dx, a.dy, a.guard, a.reason) == (b.k, b.dx, b.dy, b.guard, b.reason)
        total += 1
    assert total > 10_000


@pytest.mark.skipif(not HAVE_CC, reason="no C compiler")
def test_c_structs_match_the_ctypes_declarations(tmp_path):
    from dataopen.assist import cimpl
    src = ('#include <stdio.h>\n#include "asc_core.h"\nint main(void){printf("%zu %zu %zu %zu\\n", sizeof(asc_params_t), '
           'sizeof(asc_state_t), sizeof(asc_obj_t), sizeof(asc_out_t));return 0;}\n')
    (tmp_path / "s.c").write_text(src)
    subprocess.run(["gcc", "-std=gnu99", "-I", str(cimpl.CSRC), "-o", str(tmp_path / "s"), str(tmp_path / "s.c")], check=True)
    sizes = [int(x) for x in subprocess.run([str(tmp_path / "s")], capture_output=True, text=True).stdout.split()]
    import ctypes
    assert sizes == [ctypes.sizeof(cimpl.CParams), ctypes.sizeof(cimpl.CState), ctypes.sizeof(cimpl.CObj), ctypes.sizeof(cimpl.COut)]


def test_a_tick_is_far_inside_the_1khz_budget():
    prm = AscParams.from_view(view_from())
    asc = FixedAsc(FixedParams.from_params(prm))
    t0 = time.perf_counter()
    n = 3000
    for i in range(n):
        asc.tick(i * 1000, 8, 1, float(i % 600), 0.0, OBJ)
    py_us = (time.perf_counter() - t0) / n * 1e6
    assert py_us < 200                                                      # the slowest implementation, 5x headroom on a busy CI box
    if HAVE_CC:
        from dataopen.assist.cimpl import CAsc
        c = CAsc(FixedParams.from_params(prm))
        t0 = time.perf_counter()
        for i in range(n):
            c.tick(i * 1000, 8, 1, float(i % 600), 0.0, OBJ)
        assert (time.perf_counter() - t0) / n * 1e6 < 100


# ---------------------------------------------------------------- closed loop on simulated people
@pytest.fixture(scope="module")
def results(views):
    return {n: compare(PERSONAS[n], n=40, seed=1, view=views[n]) for n in PERSONAS}


def test_closed_loop_never_violates_the_guard_or_the_amplitude_rule(results):
    for name, (_base, asc, _) in results.items():
        assert asc["guard_violations"] == 0 and asc["amp_violations"] == 0, name


def test_a_steady_person_is_not_touched(results):
    base, asc, _ = results["steady"]
    assert asc["k_mean"] == 1.0 and asc["t_acquire_ms"] == pytest.approx(base["t_acquire_ms"])
    assert asc["overshoot_rate"] == base["overshoot_rate"]


def test_an_overshoot_prone_person_overshoots_much_less_and_is_not_slower(results):
    base, asc, _ = results["overshooter"]
    assert asc["overshoot_rate"] <= 0.6 * base["overshoot_rate"]                   # >= 40% fewer overshoots
    assert asc["overshoot_px"] < 0.4 * base["overshoot_px"]
    assert asc["t_acquire_ms"] <= 1.15 * base["t_acquire_ms"] and asc["acquired"] == 1.0


def test_tremor_is_damped_at_the_target_and_the_cost_in_time_is_reported(results):
    base, asc, _ = results["tremor"]
    assert asc["hold_rms_px"] <= 0.65 * base["hold_rms_px"]                          # >= 35% less shake while holding
    assert asc["overshoot_rate"] < base["overshoot_rate"] and asc["acquired"] == 1.0
    assert asc["t_acquire_ms"] < 2.2 * base["t_acquire_ms"]                           # a real, bounded cost (it is ~1.8x in this model)


# ---------------------------------------------------------------- live profile feed, providers, CLI
def test_profile_feed_blends_new_parameters_in_and_never_breaks_on_a_failing_reader():
    now = [0.0]
    holder = {"v": None}

    def src():
        if holder["v"] == "boom":
            raise OSError("shm gone")
        return holder["v"]

    feed = ProfileFeed(src, blend_s=1.0, ttl_s=5.0, clock=lambda: now[0])
    assert not feed.update().enabled                                                  # nothing yet: off
    holder["v"] = view_from()
    p0 = feed.update()
    assert not p0.enabled or p0.s_brake == 0.0                                        # just arrived: starts from "off"
    now[0] = 2.0
    p1 = feed.update()
    assert p1.enabled and p1.s_brake > 1.0
    holder["v"] = view_from(overshoot=0.05, ov_rate=0.4, miss=0.4, jitter_deg=0.05)  # a new generation of the profile
    holder["v"] = P.ProfileView(_gen_bump(holder["v"]))
    now[0] = 2.5
    mid = feed.update()
    now[0] = 4.0
    end = feed.update()
    assert end.s_brake < p1.s_brake and (mid.s_brake == pytest.approx(end.s_brake, abs=1.0))
    holder["v"] = "boom"
    now[0] = 5.0
    assert feed.update().enabled                                                      # a failing reader: keep the last good profile ...
    now[0] = 20.0
    assert not feed.update().enabled                                                  # ... until the TTL, then assistance is off


def _gen_bump(view):
    st = P.ProfileState.unpack(view._state.pack())
    st.generation += 1
    return st.pack()


def test_blend_keeps_assistance_off_if_either_side_is_off():
    on = AscParams.from_view(view_from())
    off = AscParams.disabled()
    assert not blend_params(on, off, 0.5).enabled and not blend_params(off, on, 0.5).enabled
    mid = blend_params(AscParams.from_view(view_from(overshoot=0.05)), on, 0.5)
    assert mid.enabled and mid.ov_med == pytest.approx(0.5 * 0.05 + 0.5 * 0.3, abs=0.02)


def test_static_object_provider_returns_the_closest_one_and_the_core_never_needs_to_know_the_source():
    objs = StaticObjects([ObjectOfInterest(1, 100.0, 0.0, 10.0), ObjectOfInterest(2, 500.0, 0.0, 10.0)])
    assert objs.nearest(450.0, 0.0).id == 2 and objs.nearest(0.0, 0.0).id == 1 and StaticObjects([]).nearest(0, 0) is None


def test_profile_view_exposes_the_calibration_the_module_needs():
    v = view_from(dpc=0.0213)
    assert v.deg_per_count == pytest.approx(0.0213, rel=1e-4) and v.latency_comp_us == 0


def test_cli_simulate_and_bench(capsys):
    from dataopen.cli import main

    def run(*argv):
        with pytest.raises(SystemExit) as e:
            main(list(argv))
        return e.value.code

    assert run("assist", "simulate", "--persona", "steady", "--trials", "4", "--json") == 0
    d = json.loads(capsys.readouterr().out)
    assert set(d["steady"]) == {"with", "without"} and d["steady"]["with"]["amp_violations"] == 0
    assert run("assist", "bench", "--repeat", "1") == 0
    assert "us/tick" in capsys.readouterr().out
    assert math.isfinite(d["steady"]["with"]["k_mean"])
