"""BioProfile engine: rolling statistics, the packed profile and its storage, the engine against a simulator with known ground truth
(reaction time, flick metrics, tracking, error scenarios, fatigue), robustness to bad input, and the interface other modules read."""
import json
import math
import shutil
import subprocess
import time

import numpy as np
import pytest

from dataopen.bioprofile import profile as P
from dataopen.bioprofile.adapter import AdapterConfig, DetectorAdapter
from dataopen.bioprofile.engine import BioProfileEngine, EngineConfig
from dataopen.bioprofile.fatigue import DriftTracker, theil_sen
from dataopen.bioprofile.rolling import RollingMedianSigma
from dataopen.bioprofile.sim import SimPlayer, simulate
from dataopen.bioprofile.store import ProfilePublisher, ProfileReader, ProfileStore
from dataopen.bioprofile.types import TargetObs
from dataopen.detector import structs


def run(player, minutes=10.0, seed=0, mix=None, cfg=None, profile=None):
    s = simulate(player, minutes, seed, mix)
    eng = BioProfileEngine(cfg or EngineConfig(deg_per_count=player.dpc), profile)
    feed(eng, s)
    return eng, s


def feed(eng, s, t_off_us=0):
    for e in s.events:
        if e[0] == "mouse":
            eng.on_mouse(e[1] + t_off_us, e[2], e[3])
        else:
            eng.on_target(e[1] + t_off_us, e[2])
    eng.advance(int(s.duration_s * 1e6) + t_off_us + 3_000_000)


@pytest.fixture(scope="module")
def baseline():
    p = SimPlayer()
    eng, s = run(p, 12.0, 3)
    return p, eng, s


# ---------------------------------------------------------------- rolling statistics
def test_rolling_median_and_sigma_match_numpy_over_the_window():
    r = RollingMedianSigma(32)
    x = np.random.default_rng(0).normal(220, 25, 400)
    for v in x:
        r.add(v)
    w = x[-32:]
    assert r.median == pytest.approx(np.median(w)) and r.sigma == pytest.approx(1.4826 * np.median(np.abs(w - np.median(w))))
    assert len(r) == 32 and r.n_total == 400


def test_a_lapse_is_winsorized_but_a_real_shift_is_followed():
    r = RollingMedianSigma(32)
    for v in np.random.default_rng(1).normal(200, 10, 40):
        r.add(v)
    r.add(2000.0)
    assert r.n_clipped == 1 and r.median < 210                         # one 2000 ms lapse does not move the typical value
    for v in np.random.default_rng(2).normal(300, 10, 40):             # the player really became slower
        r.add(v)
    assert r.median == pytest.approx(300, abs=8)


def test_non_finite_values_are_rejected_and_seeding_continues_from_stored_values():
    r = RollingMedianSigma(32)
    assert not r.add(float("nan")) and not r.add(float("inf")) and len(r) == 0
    r.seed(200.0, 20.0, k=8)
    assert r.median == pytest.approx(200.0) and r.sigma == pytest.approx(20.0, rel=0.1)
    r.add(900.0)                                                       # the first sample after a restart cannot throw the statistic
    assert r.median < 215


# ---------------------------------------------------------------- the packed profile
def full_state():
    st = P.ProfileState(profile_id=0xABCDEF01, generation=7, deg_per_count=0.0213, latency_comp_us=2083)
    for i, s in enumerate(P.STATS):
        st.stats[s.name] = P.Stat((-12.5 if s.signed else 1.0) * (i + 1) * 3.3 * s.scale * 100, 0.7 * (i + 1) * s.scale * 100, 40 + i)
    st.fatigue = P.Fatigue(1800.0, 35.2, 0.4, 2.5, 3.1, 220.0, True)
    st.counts = {c: 10 * (i + 1) for i, c in enumerate(P.COUNTS)}
    st.rates = {r: 0.1 * (i + 1) for i, r in enumerate(P.RATES)}
    return st


def test_the_profile_is_under_100_bytes_and_survives_a_pack_unpack_roundtrip():
    assert P.SIZE < 100
    st = full_state()
    b = st.pack()
    assert len(b) == P.SIZE
    back = P.ProfileState.unpack(b)
    assert back.profile_id == st.profile_id and back.generation == 7 and back.deg_per_count == pytest.approx(0.0213, rel=1e-6)
    for s in P.STATS:
        a, c = st.stats[s.name], back.stats[s.name]
        assert c.median == pytest.approx(a.median, abs=s.scale) and c.sigma == pytest.approx(a.sigma, abs=s.scale) and c.n == a.n
    assert back.counts == st.counts
    assert all(back.rates[r] == pytest.approx(st.rates[r], abs=1 / 255) for r in P.RATES)
    assert back.fatigue.slope_t_ms_per_h == pytest.approx(35.2, abs=0.1) and back.fatigue.fatiguing
    assert back.pack() == b                                            # quantization is idempotent: a stored profile does not drift


def test_every_single_byte_flip_is_detected_by_the_crc_or_the_header_checks():
    b = bytearray(full_state().pack())
    for i in range(len(b)):
        c = bytearray(b)
        c[i] ^= 0x01
        with pytest.raises(P.ProfileError):
            P.ProfileState.unpack(bytes(c))


def test_wrong_size_magic_and_newer_versions_are_refused():
    b = full_state().pack()
    for bad in (b[:-1], b + b"\x00", b"", b"\x00" * P.SIZE):
        with pytest.raises(P.ProfileError):
            P.validate(bad)
    newer = bytearray(b)
    newer[4] = P.VERSION + 1
    with pytest.raises(P.ProfileVersionError):                          # newer data is "valid but not mine", distinct from corruption
        P.validate(bytes(newer))


def test_empty_state_packs_and_reads_back_as_empty():
    v = P.ProfileView(P.ProfileState().pack())
    assert not v.stat("t_motor").valid and not v.confident("t_motor") and v.percentile("t_motor", 0.5) is None
    assert not v.fatigue.valid and v.as_dict()["metrics"] == {}
    with pytest.raises(KeyError):
        v.stat("nope")


def test_percentile_uses_the_normal_approximation():
    s = P.Stat(230.0, 25.0, 40)
    assert s.percentile(0.5) == pytest.approx(230.0) and s.percentile(0.975) == pytest.approx(230 + 1.96 * 25, abs=0.1)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")
def test_the_generated_c_header_has_the_same_layout(tmp_path):
    h = tmp_path / "bioprofile.h"
    h.write_text(P.header_text())
    offs = " ".join(f"printf(\"%s %zu\\n\", \"{n}\", __builtin_offsetof(bio_profile_v1_t, {n}));" for n, _ in P.BioProfileV1._fields_)
    src = f'#include <stdio.h>\n#include <stddef.h>\n#include "bioprofile.h"\nint main(void){{ {offs} return 0; }}\n'
    (tmp_path / "t.c").write_text(src)
    subprocess.run(["gcc", "-o", str(tmp_path / "t"), str(tmp_path / "t.c"), "-I", str(tmp_path)], check=True)
    got = dict(line.split() for line in subprocess.run([str(tmp_path / "t")], capture_output=True, text=True).stdout.splitlines())
    for n, _ in P.BioProfileV1._fields_:
        assert int(got[n]) == getattr(P.BioProfileV1, n).offset, n


# ---------------------------------------------------------------- storage and sharing
def test_store_alternates_slots_and_loads_the_newest_generation(tmp_path):
    st = ProfileStore(tmp_path / "me.prof")
    assert st.load() is None
    a = full_state()
    a.generation = 0
    gens = [st.save(a) for _ in range(3)]
    assert gens == [1, 2, 3] and st.load().generation == 3
    assert all(p.exists() for p in st.slots)


def test_a_torn_or_corrupted_newest_slot_falls_back_to_the_previous_generation(tmp_path):
    st = ProfileStore(tmp_path / "me.prof")
    a = full_state()
    a.generation = 0
    st.save(a)
    a.counts["flick"] = 999
    st.save(a)
    newest = max(st.slots, key=lambda p: P.ProfileState.unpack(p.read_bytes()).generation)
    newest.write_bytes(newest.read_bytes()[:40])                       # a torn write
    back = st.load()
    assert back is not None and back.generation == 1 and back.counts["flick"] == 10
    for p in st.slots:
        p.write_bytes(b"garbage")
    assert st.load() is None                                           # both damaged: "no profile", never an exception or wrong data


def test_a_slot_written_by_a_newer_version_is_never_overwritten(tmp_path):
    st = ProfileStore(tmp_path / "me.prof")
    st.save(full_state())
    raw = bytearray(st.slots[0].read_bytes() if st.slots[0].exists() else st.slots[1].read_bytes())
    raw[4] = P.VERSION + 1
    newer = next(p for p in st.slots if p.exists())
    newer.write_bytes(bytes(raw))
    before = newer.read_bytes()
    with pytest.raises(P.ProfileVersionError):
        st.save(full_state())
    assert newer.read_bytes() == before


def test_publisher_and_reader_share_the_newest_profile_between_modules():
    name = f"bioprof_{time.monotonic_ns()}"
    pub = ProfilePublisher(name)
    rd = ProfileReader(name)
    assert rd.read() is None
    pub.publish(full_state())
    v = rd.read()
    assert v is not None and v.profile_id == 0xABCDEF01 and v.confident("t_motor") and v.age_ms is not None and v.age_ms < 1000
    assert v.error_rates["overshoot"] == pytest.approx(0.1, abs=0.01)
    rd.close()
    pub.close()


# ---------------------------------------------------------------- the engine against known ground truth
def matched(eng, s):
    fl = {e.t_appear_us // 1000: e for e in eng.episodes if e.kind == "flick"}
    out = []
    for t in s.truth:
        if t.kind == "flick" and fl:
            k = min(fl, key=lambda x: abs(x - t.t_appear_us // 1000))
            if abs(k - t.t_appear_us // 1000) < 100:                 # a low-visibility target is detected up to ~40 ms late
                out.append((fl[k], t))
    return out


def test_reaction_time_and_flick_metrics_are_recovered(baseline):
    p, eng, s = baseline
    st = eng.snapshot()
    pairs = matched(eng, s)
    assert len(pairs) >= 0.95 * sum(t.kind == "flick" for t in s.truth)             # every flick found
    d = np.array([e.t_motor_ms - t.t_motor_ms for e, t in pairs if e.t_motor_ms is not None and not t.lowvis])
    assert abs(d.mean()) < 5.0 and d.std() < 5.0
    tm = st.stats["t_motor"]
    assert tm.median == pytest.approx(p.t_motor_ms, abs=8) and tm.sigma == pytest.approx(p.t_motor_sigma_ms, rel=0.35)
    last = pairs[-32:]                                                              # the engine reports the last 32 episodes
    assert st.stats["v_max"].median == pytest.approx(np.median([t.v_max for _, t in last]), rel=0.04)
    assert st.stats["d_brake"].median == pytest.approx(np.median([t.d_brake_frac for _, t in last]), abs=0.03)
    assert st.stats["overshoot"].median == pytest.approx(np.median([t.overshoot for _, t in last]), abs=0.02)
    assert st.stats["err_final"].median == pytest.approx(np.median([t.err_final for _, t in last]), abs=0.15)


def test_tracking_metrics_phase_lag_and_micro_correction_jitter(baseline):
    p, eng, s = baseline
    st = eng.snapshot()
    assert st.stats["phase_lag"].median == pytest.approx(p.phase_lag_ms, abs=8)
    assert st.stats["jitter_hz"].median == pytest.approx(p.jitter_hz, abs=0.8)
    assert st.stats["jitter_amp"].median == pytest.approx(p.jitter_amp_deg, abs=0.03)
    assert st.counts["track"] >= 0.9 * sum(t.kind in ("track", "flick") for t in s.truth) * 0.9


def test_low_visibility_has_its_own_slower_reaction_statistic(baseline):
    p, eng, _ = baseline
    st = eng.snapshot()
    assert st.stats["t_motor_lowvis"].n >= 8
    assert st.stats["t_motor_lowvis"].median > st.stats["t_motor"].median + 15     # slower, but the late detection hides part of it
    assert st.counts["lowvis"] >= 8


def test_scenarios_are_classified_per_episode(baseline):
    _, eng, s = baseline
    kinds = [e.kind for e in eng.episodes]
    assert kinds.count("flick") >= 0.95 * sum(t.kind == "flick" for t in s.truth)
    flicks = [e for e in eng.episodes if e.kind == "flick" and e.main_known]
    assert flicks and all("wide_flick" in e.scenarios and "surprise" in e.scenarios for e in flicks)
    assert any("low_vis" in e.scenarios for e in eng.episodes)
    assert all("micro_track" in e.scenarios for e in eng.episodes if e.kind == "track")


def test_a_hand_that_is_already_moving_gives_a_censored_episode_and_teaches_nothing(baseline):
    p, _, _ = baseline
    only = {"surprise": 0.0, "lowvis": 0.0, "track": 0.0, "premove": 1.0}
    eng, s = run(p, 3.0, 5, mix=only)
    assert eng.episodes and all(e.kind == "censored" for e in eng.episodes)
    st = eng.snapshot()
    assert not st.stats["t_motor"].valid and not st.stats["v_max"].valid and st.counts["surprise"] == 0


def test_a_reaction_faster_than_humanly_possible_is_counted_as_anticipation_not_as_reaction_time():
    p = SimPlayer(anticipation_rate=0.5)
    eng, s = run(p, 6.0, 2)
    st = eng.snapshot()
    n_ant = sum(t.anticipation for t in s.truth if t.kind == "flick")
    assert n_ant >= 5 and st.counts["anticipation"] == pytest.approx(n_ant, abs=2)
    assert st.stats["t_motor"].median > 150                                        # the anticipations did not drag it down


def test_no_reaction_is_a_lapse_and_raises_the_lapse_rate_without_polluting_reaction_time():
    eng, s = run(SimPlayer(lapse_prob=0.5), 8.0, 4, mix={"surprise": 1.0, "lowvis": 0.0, "track": 0.0, "premove": 0.0})
    n_lapse = sum(t.kind == "lapse" for t in s.truth)
    assert n_lapse >= 4 and sum(e.kind == "miss" for e in eng.episodes) == pytest.approx(n_lapse, abs=1)
    st = eng.snapshot()
    assert st.rates["lapse"] > 0.1 and st.stats["t_motor"].median == pytest.approx(230, abs=15)


def test_overshoot_undershoot_wrong_direction_and_miss_scenarios_raise_their_own_rates():
    base = dict(anticipation_rate=0.0)
    mix = {"surprise": 1.0, "lowvis": 0.0, "track": 0.0, "premove": 0.0}
    clean, _ = run(SimPlayer(overshoot_mean=0.02, overshoot_sigma=0.02, final_err_deg=0.1, **base), 8.0, 1, mix=mix)
    over, _ = run(SimPlayer(overshoot_mean=0.35, overshoot_sigma=0.05, **base), 8.0, 1, mix=mix)
    under, _ = run(SimPlayer(overshoot_mean=-0.35, overshoot_sigma=0.05, **base), 8.0, 1, mix=mix)
    wrong, _ = run(SimPlayer(heading_err_prob=0.8, heading_err_deg=60.0, **base), 8.0, 1, mix=mix)
    sloppy, _ = run(SimPlayer(final_err_deg=1.5, **base), 8.0, 1, mix=mix)
    c = clean.snapshot().rates
    assert max(c.values()) < 0.1                                                    # a clean player triggers none of them
    assert over.snapshot().rates["overshoot"] > 0.5 > c["overshoot"]
    assert under.snapshot().rates["undershoot"] > 0.5 > c["undershoot"]
    assert wrong.snapshot().rates["direction"] > 0.4 > c["direction"]
    assert sloppy.snapshot().rates["miss"] > 0.5 > c["miss"]


# ---------------------------------------------------------------- fatigue
def test_fatigue_slopes_are_recovered_and_flagged_for_a_drifting_player():
    p = SimPlayer(t_motor_drift_ms_per_h=90.0, err_drift_deg_per_h=0.8)
    eng, _ = run(p, 40.0, 1)
    f = eng.snapshot().fatigue
    assert f.valid and f.slope_t_ms_per_h == pytest.approx(90.0, abs=25) and f.slope_err_deg_per_h == pytest.approx(0.8, abs=0.35)
    assert f.fatiguing and f.session_s == pytest.approx(2400, abs=200)


def test_a_steady_player_is_not_flagged_as_fatiguing():
    flagged = 0
    for seed in range(3):
        eng, _ = run(SimPlayer(), 30.0, seed)
        f = eng.snapshot().fatigue
        flagged += f.fatiguing
        assert not f.valid or abs(f.slope_t_ms_per_h) < 25
    assert flagged == 0


def test_a_long_rest_ends_the_session_and_fatigue_starts_over():
    p = SimPlayer(t_motor_drift_ms_per_h=80.0, err_drift_deg_per_h=0.6)
    s1 = simulate(p, 20.0, 1)
    s2 = simulate(SimPlayer(), 6.0, 2)
    eng = BioProfileEngine(EngineConfig(deg_per_count=p.dpc))
    feed(eng, s1)
    assert eng.snapshot().fatigue.valid
    feed(eng, s2, t_off_us=int(s1.duration_s * 1e6) + 20 * 60 * 1_000_000)          # twenty minutes later
    f = eng.snapshot().fatigue
    assert f.session_s < 8 * 60 and not f.valid                                     # a fresh, short session: no trend yet


def test_theil_sen_resists_a_single_bad_block_and_a_perfect_line_gives_a_finite_z():
    m, _ = theil_sen([0, 1, 2, 3, 4, 5, 6], [100, 102, 104, 400, 108, 110, 112])
    assert m == pytest.approx(2.0, abs=0.5)
    d = DriftTracker(block_s=60, min_blocks=4, min_span_s=200)
    for k in range(9):
        for j in range(4):
            d.add(k * 60 + 10 * j, 200 + 5 * k + (300 if k == 3 else 0), 0.3)       # one lapse block in reaction time, accuracy flat
    f = d.result()
    assert f.valid and f.slope_t_ms_per_h == pytest.approx(300.0, abs=60) and abs(f.slope_err_deg_per_h) < 1e-6
    assert math.isfinite(f.z_t) and abs(f.z_t) <= 50 and not f.fatiguing                # flat accuracy: not "fatiguing"


# ---------------------------------------------------------------- robustness
def test_garbage_input_never_raises_and_is_counted():
    eng = BioProfileEngine(EngineConfig())
    eng.on_mouse(1000, float("nan"), 3)
    eng.on_mouse(2000, 1, float("inf"))
    eng.on_target(3000, TargetObs(3000, float("nan"), 1.0))
    eng.on_mouse(10_000_000, 5, 5)
    eng.on_mouse(1_000_000, 5, 5)                                                   # time went backwards by seconds
    eng.on_mouse(10_000_500, 2**31, -(2**31))                                       # absurd counts
    eng.on_target(10_001_000, None)
    eng.advance(10_500_000)
    assert eng.dropped_samples >= 4
    eng.snapshot().pack()


def test_a_long_silence_does_not_cost_a_millisecond_per_millisecond():
    eng = BioProfileEngine(EngineConfig())
    eng.on_mouse(1_000, 1, 1)
    t0 = time.perf_counter()
    eng.on_target(1_000 + 3 * 3600 * 1_000_000, None)                               # three hours of nothing
    assert time.perf_counter() - t0 < 0.5


def test_the_engine_is_deterministic_and_cheap():
    p = SimPlayer()
    s = simulate(p, 4.0, 9)
    outs = []
    for _ in range(2):
        e = BioProfileEngine(EngineConfig(deg_per_count=p.dpc))
        t0 = time.perf_counter()
        feed(e, s)
        outs.append((e.snapshot().pack(), time.perf_counter() - t0))
    assert outs[0][0] == outs[1][0]
    assert outs[0][1] < 0.25 * s.duration_s                                         # far faster than real time (typically < 1%)


def test_a_restart_from_the_stored_profile_continues_instead_of_starting_over(baseline, tmp_path):
    p, eng, _ = baseline
    saved = eng.snapshot(clean=True)
    ProfileStore(tmp_path / "me").save(saved)
    loaded = ProfileStore(tmp_path / "me").load()
    short = SimPlayer()
    e2, _ = run(short, 0.6, 77, profile=loaded)
    a, b = saved.stats["t_motor"], e2.snapshot().stats["t_motor"]
    assert b.valid and b.median == pytest.approx(a.median, abs=15) and b.n >= a.n
    assert e2.snapshot().profile_id == saved.profile_id and e2.snapshot().counts["flick"] >= saved.counts["flick"]


def test_subscribers_get_a_snapshot_after_every_episode_and_other_modules_can_read_it():
    name = f"bioprof_e_{time.monotonic_ns()}"
    pub, rd = ProfilePublisher(name), None
    rd = ProfileReader(name)
    p = SimPlayer()
    eng = BioProfileEngine(EngineConfig(deg_per_count=p.dpc))
    seen = []
    eng.subscribe(pub.publish)
    eng.subscribe(seen.append)
    feed(eng, simulate(p, 3.0, 4))
    v = rd.read()
    assert seen and v is not None and v.stat("t_motor").valid and v.counts["flick"] >= 10
    assert v.percentile("t_motor", 0.05) < v.stat("t_motor").median < v.percentile("t_motor", 0.95)
    rd.close()
    pub.close()


# ---------------------------------------------------------------- detector adapter
def det(x, y, kc=0.9, score=0.9):
    k = np.zeros((12, 2))
    k[:] = (x, y)
    return {"cls": 0, "score": score, "box": np.array([x - 20, y - 20, x + 20, y + 60]), "kxy": k, "kscore": np.full(12, kc),
            "kvis": np.ones(12)}


def test_adapter_maps_pixels_to_angles_picks_the_target_nearest_the_crosshair_and_unwraps_the_clock():
    ad = DetectorAdapter(AdapterConfig(width=640, height=640, fov_h_deg=90.0))
    arr = structs.pack([det(640.0 - 1, 320.0), det(330.0, 330.0)], 1, 100, 100)
    t, obs = ad.observe(arr)
    assert obs is not None and obs.x_deg == pytest.approx(math.degrees(math.atan2(10, 320)), abs=0.01)    # the near one, not the edge one
    arr = structs.pack([det(639.0, 320.0)], 2, 200, 100)
    _, o2 = ad.observe(arr)
    assert o2.x_deg == pytest.approx(45.0, abs=0.2) and o2.y_deg == pytest.approx(0.0, abs=0.01)           # half the FOV
    a1 = structs.pack([], 3, 0xFFFFFF00, 100)
    a2 = structs.pack([], 4, 0x00000100, 100)                                                               # the 32-bit clock wrapped
    t1, _ = ad.observe(a1)
    t2, _ = ad.observe(a2)
    assert t2 > t1 and t2 - t1 == 0x200


def test_adapter_returns_none_for_errors_low_confidence_and_flags_dark_scenes():
    ad = DetectorAdapter()
    err = structs.pack([det(320, 320)], 1, 10, 100, flags=8)
    assert ad.observe(err)[1] is None
    assert ad.observe(structs.pack([det(320, 320, kc=0.1)], 2, 20, 100))[1] is None
    dark = ad.observe(structs.pack([det(320, 320)], 3, 30, 10))[1]
    assert dark is not None and dark.degraded


def test_detector_frames_through_the_adapter_drive_the_engine():
    """The whole chain: KeypointArray stream (what the runtime publishes) + mouse events -> a reaction time."""
    ad = DetectorAdapter(AdapterConfig(width=640, height=640, fov_h_deg=90.0))
    eng = BioProfileEngine(EngineConfig(deg_per_count=0.02))
    for rep in range(14):                                                       # appear at +10 deg, react after 220 ms, flick onto it
        base = rep * 3_000_000
        for k in range(150):                                                     # 0.6 s of empty frames: the hand is still
            _, o = ad.observe(structs.pack([], 0, base + k * 4167, 100))
            eng.on_target(base + k * 4167, o)
        t_app = base + 150 * 4167
        for k in range(200):
            now = t_app + k * 4167
            x = 320 + 320 * math.tan(math.radians(10.0)) * (1 - min(max((now - t_app - 220_000) / 120_000, 0), 1))
            _, o = ad.observe(structs.pack([det(320 + (x - 320) + 0.0, 320.0)], 0, now, 100))
            eng.on_target(now, o)
            if now > t_app + 220_000 and now < t_app + 340_000:
                eng.on_mouse(now, 10.0 * 0.5 / 0.02 / 24, 0)                     # about 10 degrees in 120 ms, in counts
    st = eng.snapshot()
    assert st.stats["t_motor"].valid or st.counts["surprise"] > 0


# ---------------------------------------------------------------- CLI
def test_cli_simulate_show_and_header(tmp_path, capsys):
    from dataopen.cli import main

    def run_cli(*argv):
        with pytest.raises(SystemExit) as e:
            main(list(argv))
        return e.value.code

    out = tmp_path / "me"
    assert run_cli("bioprofile", "simulate", "--minutes", "4", "--out", str(out), "--seed", "2") == 0
    shown = capsys.readouterr().out
    assert '"t_motor"' in shown
    assert run_cli("bioprofile", "show", str(out)) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["metrics"]["t_motor"]["n"] >= 8 and 180 < d["metrics"]["t_motor"]["median"] < 280
    assert run_cli("bioprofile", "show", str(tmp_path / "nothing")) == 1
    assert run_cli("bioprofile", "header", "--out", str(tmp_path / "b.h")) == 0 and "BIOPROFILE_MAGIC" in (tmp_path / "b.h").read_text()


