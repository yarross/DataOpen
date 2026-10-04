import json

import numpy as np

from dataopen.adapters.mock import MockGameAdapter
from dataopen.core.models import FrameKind
from dataopen.core.randomization import DomainRandomizationController
from dataopen.quality.balance import BalanceConfig
from dataopen.quality.feedback import AdaptiveRandomizer, FeedbackConfig, UnitWarp
from dataopen.quality.types import FrameFeatures, QualityMetrics, QualityVerdict, Tier


def verdict(tier=Tier.KEEP, utility=0.2, oks=0.9):
    m = QualityMetrics(evaluated=True, n_gt=1, mean_oks=oks)
    return QualityVerdict(tier, [], 0.3, 0.0, 0.2, 1.0, utility, m, FrameFeatures())


def make(seed=1, balance=None, **fb):
    """Balance control is off by default here: these tests exercise the per-bin learning mechanics in isolation
    (tests/test_quality_balance.py covers the dataset-level controller)."""
    cfg = FeedbackConfig(warmup_frames=60, update_every=10, balance=balance or BalanceConfig(enabled=False), **fb)
    return AdaptiveRandomizer(seed, MockGameAdapter().parameter_space(), feedback=cfg)


def test_unit_warp_identity_density_and_bin_consistency():
    w = UnitWarp(np.full(4, 0.25))
    assert all(abs(w.apply(u) - u) < 1e-9 for u in (0.0, 0.1, 0.49, 0.999))
    w.set_tilt(np.array([1.0, 1.0, 1.0, 5.0]))
    us = np.random.default_rng(0).random(20000)
    mapped = np.array([w.apply(u) for u in us])
    share_last = (mapped >= 0.75).mean()
    assert abs(share_last - 5 / 8) < 0.02                      # tilted mass 5/(1+1+1+5)
    assert mapped.min() >= 0 and mapped.max() < 1
    assert all(w.bin_of(m) == min(int(m * 4), 3) for m in mapped[:500])
    assert [w.apply(u) for u in np.linspace(0, 0.999, 50)] == sorted(w.apply(u) for u in np.linspace(0, 0.999, 50))  # monotone


def test_unit_warp_respects_unequal_base_masses():
    w = UnitWarp(np.array([0.7, 0.2, 0.1]))                    # a categorical with weights
    w.set_tilt(np.array([1.0, 1.0, 1.0]))
    assert abs(w.apply(0.8) - 0.8) < 1e-9
    w.set_tilt(np.array([0.1, 1.0, 1.0]))
    assert w.bin_of(w.apply(0.05)) == 0 and w.bin_of(w.apply(0.95)) == 2


def test_default_hook_is_identity_so_nothing_changes_without_feedback():
    plain = DomainRandomizationController(5, MockGameAdapter().parameter_space())
    adaptive = make(seed=5)                                     # still in warm-up: must behave identically
    for i in range(6):
        a, b = plain.sample_scene(i), adaptive.sample_scene(i)
        assert a.environment == b.environment and a.actors == b.actors
        fa = plain.sample_frame(a, 1, FrameKind.POSITIVE)
        fb = adaptive.sample_frame(b, 1, FrameKind.POSITIVE)
        assert fa.camera == fb.camera and fa.actor_frame == fb.actor_frame
    s = adaptive.sample_scene(0)
    assert len(s.units["env"]) == len(adaptive.env_space.params) and len(s.units["actors"]) == len(s.actors)


def drive(rz, n, utility_fn, start=0, per_scene=4):
    """Sample n frames and feed back verdicts computed from the values that were actually used."""
    used = []
    i = start
    while len(used) < n:
        scene = rz.sample_scene(i)
        for j in range(per_scene):
            frame = rz.sample_frame(scene, j, FrameKind.POSITIVE)
            tier, util = utility_fn(scene, frame)
            rz.observe(scene, frame, verdict(tier, util))
            used.append((scene, frame))
        i += 1
    return used


def fog_hard(scene, frame):
    """Fog above 0.02 is where the (pretend) model struggles; camera farther than 20 m is wasted (frames get dropped)."""
    if frame.camera.distance > 20:
        return Tier.DROP_INVISIBLE, 0.0
    if scene.environment["fog_density"] > 0.02:
        return Tier.KEEP_HARD, 1.0
    return Tier.KEEP, 0.1


def test_the_loop_concentrates_sampling_on_edge_cases_but_keeps_exploring():
    rz = make(seed=2)
    drive(rz, 400, fog_hard)                                    # warm-up + learning
    assert rz.adapting and rz.n_observed >= 400
    late = drive(rz, 600, fog_hard, start=1000)
    fog = np.array([s.environment["fog_density"] for s, _ in late])
    dist = np.array([f.camera.distance for _, f in late])
    base_fog = DomainRandomizationController(2, MockGameAdapter().parameter_space())
    base = np.array([base_fog.sample_scene(i).environment["fog_density"] for i in range(2000, 2600)])
    assert (fog > 0.02).mean() > (base > 0.02).mean() + 0.10    # more dense-fog scenes than uniform sampling
    assert (dist > 20).mean() < 0.10                            # the wasteful far-camera region shrank (natural share ~0.2)
    assert (fog <= 0.02).mean() > 0.05 and (dist > 20).mean() > 0.0   # ...but nothing was abandoned (floor + exploration)


def test_categorical_appearance_is_learned_without_enumerating_combinations():
    rz = make(seed=4)

    def armor_breaks_the_model(scene, frame):
        n_armor = sum(a["outfit"] == "armor" for a in scene.actors)
        return (Tier.KEEP_HARD, 1.0) if n_armor else (Tier.KEEP, 0.1)

    drive(rz, 500, armor_breaks_the_model)
    late = drive(rz, 500, armor_breaks_the_model, start=900)
    armor_share = np.mean([a["outfit"] == "armor" for s, _ in late for a in s.actors])
    assert armor_share > 0.45                                   # natural share 1/3
    rep = rz.report()
    rows = {r["bin"]: r for r in rep["parameters"]["actor.outfit"]["rows"]}
    assert rows["armor"]["hard_rate"] > rows["casual"]["hard_rate"] and rows["armor"]["tilt"] > rows["casual"]["tilt"]
    assert rep["pairwise_appearance_coverage"]["coverage"] > 0.5


def test_exploitation_reuses_remembered_hard_examples():
    rz = make(seed=6, exploit_p=1.0)
    drive(rz, 80, lambda s, f: (Tier.KEEP_HARD, 1.0))           # everything is "hard" -> memory fills
    assert len(rz._memory) > 0
    mem_cam = [m["frame"]["cam"] for m in rz._memory]
    scene = rz.sample_scene(500)
    frame = rz.sample_frame(scene, 0, FrameKind.POSITIVE)
    nearest = min(np.abs(np.array(frame.units["cam"]) - np.array(m)).max() for m in mem_cam)
    assert nearest < 0.35                                       # a jittered copy of something remembered (sigma 0.07)


def test_state_roundtrip_gives_identical_sampling_and_survives_json(tmp_path):
    a = make(seed=3)
    drive(a, 300, fog_hard)
    path = tmp_path / "feedback_state.json"
    a.save(path)
    b = make(seed=3)
    b.load_state(json.loads(path.read_text()))
    assert b.n_observed == a.n_observed and b.adapting
    for k in a._warps:
        assert np.allclose(a._warps[k].t, b._warps[k].t)
    for i in range(3000, 3004):
        sa, sb = a.sample_scene(i), b.sample_scene(i)
        assert sa.environment == sb.environment and sa.units == sb.units
    c = make(seed=3)
    c.load_state({"version": 99})                                # unknown versions are ignored, not fatal
    assert c.n_observed == 0


def test_report_is_json_serializable_and_lists_bins():
    rz = make(seed=7)
    drive(rz, 120, fog_hard)
    rep = rz.report()
    json.dumps(rep)
    fog = rep["parameters"]["env.fog_density"]
    assert fog["bins"] == 8 and fog["covered_bins"] >= 6 and all(set(r) >= {"bin", "n", "tilt", "drop_rate"} for r in fog["rows"])
