import numpy as np

from dataopen.adapters.mock import MockGameAdapter
from dataopen.core.models import FrameKind
from dataopen.quality.balance import BalanceConfig, BalanceController, drift_stats
from dataopen.quality.feedback import AdaptiveRandomizer, FeedbackConfig
from dataopen.quality.types import FrameFeatures, QualityMetrics, QualityVerdict, Tier, Verdict


def verdict(tier, utility=0.2):
    return QualityVerdict(tier, [], 0.3, 0.0, 0.2, 1.0, utility, QualityMetrics(evaluated=True, n_gt=1, mean_oks=0.8),
                          FrameFeatures())


def test_tier_to_public_verdict_mapping():
    expect = {Tier.KEEP: Verdict.KEEP_CLEAN, Tier.KEEP_HARD: Verdict.KEEP_HARD, Tier.HARD_NEGATIVE: Verdict.KEEP_HARD,
              Tier.SUSPECT: Verdict.QUARANTINE, Tier.DROP_PHANTOM: Verdict.QUARANTINE, Tier.DROP_INVISIBLE: Verdict.REJECT,
              Tier.DROP_GT_SANITY: Verdict.REJECT, Tier.DROP_RENDER: Verdict.REJECT, Tier.REJECTED: Verdict.REJECT}
    assert set(expect) == set(Tier)                              # a new tier must be given a verdict on purpose
    for tier, v in expect.items():
        assert tier.verdict is v and verdict(tier).to_dict()["verdict"] == v.value


def feed(bc, tiers):
    for t in tiers:
        bc.observe(verdict(t))
    bc.update()


def test_easy_overflow_sharpens_and_hard_overflow_widens_within_bounds():
    cfg = BalanceConfig(window=100)
    easy = BalanceController(cfg)
    for _ in range(30):
        feed(easy, [Tier.KEEP] * 9 + [Tier.KEEP_HARD])           # 10% hard against a 30% target
    assert easy.gamma_scale > 1.3 and easy.gamma_scale <= cfg.gamma_max
    assert easy.exploit_p(0.2) > 0.2 and easy.uniform_mix == cfg.uniform_min

    hard = BalanceController(cfg)
    for _ in range(30):
        feed(hard, [Tier.KEEP_HARD] * 8 + [Tier.KEEP] * 2)       # 80% hard: the generator is stuck in a corner
    assert hard.gamma_scale < 0.7 and hard.gamma_scale >= cfg.gamma_min
    assert hard.uniform_mix > cfg.uniform_min and hard.uniform_mix <= cfg.uniform_max


def test_inside_the_deadband_nothing_moves_and_reject_budget_sets_drop_pressure():
    bc = BalanceController(BalanceConfig(window=100))
    feed(bc, [Tier.KEEP] * 7 + [Tier.KEEP_HARD] * 3)             # exactly the target
    assert bc.gamma_scale == 1.0 and bc.drop_pressure == 0.0
    feed(bc, [Tier.DROP_INVISIBLE] * 30)                         # 30% of the window rejected
    assert bc.drop_pressure > 0
    s = bc.shares()
    assert s["reject_share"] > 0.15 and s["usable"] >= 10


def test_state_roundtrip():
    a = BalanceController(BalanceConfig())
    for _ in range(20):
        feed(a, [Tier.KEEP] * 9 + [Tier.KEEP_HARD])
    b = BalanceController(BalanceConfig())
    b.load_state(a.state())
    assert (b.gamma_scale, b.uniform_mix, b.shares()) == (a.gamma_scale, a.uniform_mix, a.shares())


def test_drift_stats_flag_a_collapsed_distribution_and_pass_a_uniform_one():
    m = np.full(8, 1 / 8)
    ok = drift_stats(np.full(8, 100.0), m)
    assert ok["kl"] < 0.01 and ok["entropy_ratio"] > 0.99 and ok["min_share_ratio"] > 0.95
    bad = drift_stats(np.array([700.0, 20, 20, 20, 20, 20, 20, 0]), m)
    assert bad["kl"] > 0.5 and bad["min_share_ratio"] == 0.0 and bad["entropy_ratio"] < 0.6


def make(seed, balance):
    return AdaptiveRandomizer(seed, MockGameAdapter().parameter_space(),
                              feedback=FeedbackConfig(warmup_frames=60, update_every=10, balance=balance))


def drive(rz, n, fn, start=0):
    out, i = [], start
    while len(out) < n:
        scene = rz.sample_scene(i)
        for j in range(4):
            frame = rz.sample_frame(scene, j, FrameKind.POSITIVE)
            tier, util = fn(scene, frame)
            rz.observe(scene, frame, verdict(tier, util))
            out.append((scene, frame))
        i += 1
    return out


def fog_hard(scene, frame):
    if scene.environment["fog_density"] > 0.02:
        return Tier.KEEP_HARD, 1.0
    return Tier.KEEP, 0.1


def test_balance_keeps_the_hard_share_closer_to_the_target_than_unchecked_mining():
    def hard_share(balance):
        rz = make(7, balance)
        drive(rz, 500, fog_hard)
        late = drive(rz, 800, fog_hard, start=1000)
        return float(np.mean([s.environment["fog_density"] > 0.02 for s, _ in late]))

    natural = 0.09                                               # share of dense-fog scenes under uniform sampling
    free = hard_share(BalanceConfig(enabled=False))
    ctrl = hard_share(BalanceConfig(target_hard_share=0.3, uniform_min=0.2))
    low = hard_share(BalanceConfig(target_hard_share=0.1, uniform_min=0.2))
    assert free > 0.35                                           # unchecked mining piles onto the hard region
    assert ctrl < free - 0.07 and abs(ctrl - 0.3) < 0.08         # the controller holds the requested mix...
    assert low < ctrl and abs(low - 0.1) < 0.08 and low > natural   # ...and follows a different target


def test_collapse_is_reported_and_coverage_floor_holds_with_the_controller():
    rz = make(8, BalanceConfig(target_hard_share=0.3))
    drive(rz, 600, fog_hard)
    drive(rz, 800, fog_hard, start=1000)
    rep = rz.report()
    assert rep["balance"]["enabled"] and "drift" in rep and isinstance(rep["drift_warnings"], list)
    for key, d in rep["drift"].items():
        assert d["min_share_ratio"] > 0.0, f"{key}: a bin was abandoned completely"
