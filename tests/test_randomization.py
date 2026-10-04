import numpy as np

from dataopen.core.randomization import (Categorical, DomainRandomizationController, LogUniform,
                                         ParameterSpace, derive_seed, latin_hypercube)
from dataopen.core.models import FrameKind
from dataopen.adapters.mock import MockGameAdapter


def make(seed=7):
    return DomainRandomizationController(seed, MockGameAdapter().parameter_space(), lhs_block=64)


def test_derive_seed_is_stable_and_order_sensitive():
    assert derive_seed(1, "a", 2) == derive_seed(1, "a", 2) != derive_seed(1, 2, "a")


def test_scene_and_frame_sampling_are_pure_functions_of_indices():
    a, b = make(), make()
    sa, sb = a.sample_scene(5), b.sample_scene(5)
    assert sa.environment == sb.environment and sa.actors == sb.actors
    b.sample_scene(99)                                  # unrelated draws must not disturb anything
    fa = a.sample_frame(sa, 3, FrameKind.POSITIVE)
    fb = b.sample_frame(b.sample_scene(5), 3, FrameKind.POSITIVE)
    assert fa == fb


def test_latin_hypercube_stratifies_every_dimension():
    u = latin_hypercube(50, 3, np.random.default_rng(0))
    for j in range(3):
        assert sorted((u[:, j] * 50).astype(int)) == list(range(50))


def test_time_of_day_covers_the_whole_day_uniformly_within_a_block():
    r = make()
    tod = np.array([r.sample_scene(i).environment["time_of_day"] for i in range(64)])
    hist, _ = np.histogram(tod, bins=8, range=(0, 24))
    assert hist.min() >= 7 and hist.max() <= 9          # LHS => near-perfectly flat (64/8 = 8)


def test_weather_rule_keeps_correlations_plausible():
    r = make()
    for i in range(200):
        e = r.sample_scene(i).environment
        if e["weather"] == "fog":
            assert e["fog_density"] >= 0.01
        if e["weather"] == "clear":
            assert e["cloud_cover"] <= 0.4 and e["fog_density"] <= 0.002


def test_categorical_inverse_cdf_respects_weights():
    c = Categorical(("a", "b"), (1, 3))
    assert c.from_unit(0.2) == "a" and c.from_unit(0.26) == "b" and c.from_unit(0.999999) == "b"
    assert 1.9 < LogUniform(2, 40).from_unit(0.0) < 2.1


def test_splits_are_per_scene_with_roughly_requested_fraction():
    r = make()
    n_val = sum(r.split_for_scene(i) == "val" for i in range(2000))
    assert 140 < n_val < 260                             # ~10% of 2000
