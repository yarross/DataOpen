import numpy as np
import pytest

from dataopen.core.models import EntityState
from dataopen.core.schema import HUMAN_13
from dataopen.core.viz import _dot, _line
from dataopen.quality.features import compute_features, static_difficulty
from dataopen.quality.metrics import DefaultMetricCalculator
from dataopen.quality.policy import DefaultQualityPolicy, PolicyConfig
from dataopen.quality.types import Prediction, Tier

from test_quality_metrics import gt_person, pred_from


def scene(bg=128, fg=(230, 40, 40), h=300, noise=0.0, seed=0):
    """Gray background with a limb-coloured skeleton drawn on it."""
    g = gt_person(h=h)
    img = np.full((480, 640, 3), bg, dtype=np.uint8)
    if noise:
        img = np.clip(img + np.random.default_rng(seed).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    kp = g.keypoints
    for a, b in HUMAN_13.edges:
        _line(img, kp[HUMAN_13.index(a), :2], kp[HUMAN_13.index(b), :2], fg, 5)
    for k in kp:
        _dot(img, k[0], k[1], fg, 5, False)
    return img, g


def feats(img, g, ents=()):
    return compute_features(img, [g], list(ents), HUMAN_13)


def test_perceptibility_high_for_contrasting_person_and_low_when_camouflaged_or_dark():
    img, g = scene()
    f = feats(img, g)
    assert f.persons[0].perceptibility > 0.9 and f.persons[0].contrast_rate > 0.2 and f.has_pixels
    camo, g2 = scene(bg=128, fg=(130, 126, 128), noise=20)                  # same colour as the wall
    assert feats(camo, g2).persons[0].perceptibility < 0.12
    dark, g3 = scene(bg=2, fg=(6, 4, 4))                           # pitch black
    assert feats(dark, g3).persons[0].perceptibility < 0.12
    tiny, g4 = scene(h=18)                                         # a few pixels tall
    assert feats(tiny, g4).persons[0].perceptibility < feats(img, g).persons[0].perceptibility


def test_occlusion_index_and_static_difficulty_ordering():
    img, g = scene()
    easy = feats(img, g)
    g.keypoints[[4, 5, 6, 7], 2] = 1                               # four joints hidden
    occluded = feats(img, g)
    assert easy.persons[0].occlusion_index == 0.0 and occluded.persons[0].occlusion_index == pytest.approx(4 / 13)
    camo, g2 = scene(fg=(130, 126, 128), noise=20)
    assert static_difficulty(easy) < static_difficulty(occluded) < static_difficulty(feats(camo, g2)) <= 1.0
    assert static_difficulty(compute_features(img, [], [], HUMAN_13)) == 0.0


def test_gt_sanity_issue_is_reported_without_pixels():
    img, g = scene()
    sk = np.zeros((13, 3))
    sk[:, 2] = np.arange(13) * 5.0                                  # absurdly long skeleton
    ent = EntityState(g.entity_id, "r", sk, np.ones(13, bool))
    f = compute_features(None, [g], [ent], HUMAN_13)
    assert f.gt_issues and not f.has_pixels and f.persons == []


def decide(img, g, preds, negative=False, cfg=None, ents=()):
    f = compute_features(img, [] if negative else [g], list(ents), HUMAN_13)
    m = DefaultMetricCalculator(HUMAN_13).compute([] if negative else [g], preds, "t")
    return DefaultQualityPolicy(cfg or PolicyConfig()).decide(m, f, negative)


def test_the_central_distinction_visible_but_model_blind_is_kept_as_hard_while_invisible_is_dropped():
    img, g = scene()
    v_hard = decide(img, g, [])                                    # perceptible person, the model finds nothing
    assert v_hard.tier is Tier.KEEP_HARD and "model_blind_but_visible" in v_hard.reasons[0]
    assert v_hard.utility == 1.0 and v_hard.weight > 1.5
    camo, g2 = scene(fg=(130, 126, 128), noise=20)                           # the same model failure, but the picture is unusable
    v_drop = decide(camo, g2, [])
    assert v_drop.tier is Tier.DROP_INVISIBLE and v_drop.utility == 0.0 and v_drop.weight == 0.0
    black = np.zeros((480, 640, 3), np.uint8)
    assert decide(black, g, [pred_from(g)]).tier is Tier.DROP_RENDER   # even a "good" prediction cannot save a black frame


def test_good_struggling_suspect_and_phantom_outcomes():
    img, g = scene()
    assert decide(img, g, [pred_from(g)]).tier is Tier.KEEP
    struggle = decide(img, g, [pred_from(g, offset=18.0)])
    assert struggle.tier is Tier.KEEP_HARD and "model_struggles" in struggle.reasons[0]
    bad = pred_from(g)
    bad.keypoints[:, :2] += 60
    sus = decide(img, g, [bad])
    assert sus.tier is Tier.SUSPECT and sus.tier.is_drop
    assert decide(img, g, [bad], cfg=PolicyConfig(suspect_action="keep_hard")).tier is Tier.KEEP_HARD
    other = gt_person(entity_id=9, x0=500, h=100)
    ph = decide(img, g, [pred_from(g), pred_from(other, score=0.95)])
    assert ph.tier is Tier.DROP_PHANTOM
    assert decide(img, g, [pred_from(g), pred_from(other, score=0.95)], cfg=PolicyConfig(phantom_action="hard")).tier is Tier.KEEP_HARD


def test_negative_frames_hard_negative_and_drop_rules():
    empty = np.full((480, 640, 3), 100, np.uint8)
    empty[::7] += 20                                               # not blank
    g = gt_person()
    assert decide(empty, g, [], negative=True).tier is Tier.KEEP
    fp = [Prediction((100, 100, 50, 150), 0.9, None)]
    assert decide(empty, g, fp, negative=True, cfg=PolicyConfig(phantom_action="hard")).tier is Tier.HARD_NEGATIVE
    assert decide(empty, g, fp, negative=True).tier is Tier.DROP_PHANTOM
    assert decide(np.full((480, 640, 3), 100, np.uint8), g, [], negative=True).tier is Tier.DROP_RENDER   # perfectly flat


def test_without_a_model_only_static_evidence_is_used():
    img, g = scene()
    f = compute_features(img, [g], [], HUMAN_13)
    m = DefaultMetricCalculator(HUMAN_13).compute([g], None)
    assert not m.evaluated
    assert DefaultQualityPolicy().decide(m, f, False).tier is Tier.KEEP
    camo, g2 = scene(fg=(130, 126, 128), noise=20)
    assert DefaultQualityPolicy().decide(m, compute_features(camo, [g2], [], HUMAN_13), False).tier is Tier.DROP_INVISIBLE
    g3 = scene()[1]
    g3.keypoints[3:11, 2] = 1                                      # heavily occluded
    img3 = scene()[0]
    v = DefaultQualityPolicy(PolicyConfig(hard_difficulty=0.2)).decide(m, compute_features(img3, [g3], [], HUMAN_13), False)
    assert v.tier is Tier.KEEP_HARD and "static_difficulty" in v.reasons[0]


def test_verdict_serializes():
    img, g = scene()
    v = decide(img, g, [pred_from(g, 10.0)])
    d = v.to_dict()
    assert d["tier"] in ("keep", "keep_hard") and set(d) >= {"difficulty", "occlusion_index", "contrast_rate", "weight",
                                                             "utility", "metrics", "features", "reasons"}
    import json
    json.dumps(d)
