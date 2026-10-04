"""The target-schema layer: rules, the 12-point aim schema, mapping from the mod's skeleton and from a COCO-17 model."""
import numpy as np
import pytest

from dataopen.core.derive import Rule, RuleError, SchemaMapping, identity_mapping
from dataopen.core.models import EntityState
from dataopen.core.schema import COCO_17, HUMAN_13, SkeletonSchema
from dataopen.core.schema_io import SchemaFileError, bundle_from_dict, load_schema

SHOOTER = ["head_top", "head_center", "l_shoulder", "neck", "r_shoulder", "l_elbow", "spine", "r_elbow", "l_wrist", "hip",
           "r_wrist", "center_of_mass"]


def standing(scale=1.0):
    """HUMAN_13 skeleton of a standing person (meters, z up), as a mod would report it."""
    pos = {"head": (0, 0, 1.70), "neck": (0, 0, 1.50), "l_shoulder": (0, 0.2, 1.45), "r_shoulder": (0, -0.2, 1.45),
           "l_elbow": (0, 0.26, 1.15), "r_elbow": (0, -0.26, 1.15), "l_wrist": (0, 0.28, 0.9), "r_wrist": (0, -0.28, 0.9),
           "pelvis": (0, 0, 0.95), "l_knee": (0, 0.1, 0.5), "r_knee": (0, -0.1, 0.5), "l_ankle": (0, 0.1, 0.08),
           "r_ankle": (0, -0.1, 0.08)}
    return np.array([pos[k] for k in HUMAN_13.keypoints], dtype=float) * scale


@pytest.fixture(scope="module")
def bundle():
    return load_schema("shooter12")


def test_the_12_points_are_exactly_the_requested_ones_in_order(bundle):
    s = bundle.schema
    assert list(s.keypoints) == SHOOTER and s.num_keypoints == 12
    assert s.primary == ("head_center",) and s.classes == ("player_ct", "player_t") and s.class_key == "team"
    assert s.flip_idx() == [0, 1, 4, 3, 2, 7, 6, 5, 10, 9, 8, 11]          # only the three limb pairs swap
    assert len(s.oks_sigmas()) == len(s.oks_weights()) == 12
    w = dict(zip(s.keypoints, s.oks_weights()))
    assert w["head_center"] == max(w.values()) and w["center_of_mass"] < w["l_shoulder"]      # aim point counts most
    assert {"neck", "hip", "spine", "center_of_mass"} <= set(s.derived)                        # computed points are flagged
    assert s.role("head") == "head_center" and s.role("pelvis") == "hip" and s.group_idx("head") == [0, 1]


def test_rules_parse_and_evaluate():
    p = {"a": np.array([0.0, 0, 0]), "b": np.array([0.0, 0, 1.0])}
    ev = SchemaMapping._eval
    pts = np.stack([p["a"], p["b"]])
    assert np.allclose(ev(Rule.parse("lerp(a, b, 0.25)"), pts), [0, 0, 0.25])
    assert np.allclose(ev(Rule.parse("extend(a, b, 0.5)"), pts), [0, 0, 1.5])            # relative to |a->b|
    assert np.allclose(ev(Rule.parse("extend_m(a, b, 0.12)"), pts), [0, 0, 1.12])          # absolute meters
    assert np.allclose(ev(Rule.parse("mean(a, b)"), pts), [0, 0, 0.5])
    assert np.allclose(ev(Rule.parse("weighted(a:3, b:1)"), pts), [0, 0, 0.25])
    assert Rule.parse("extend_m(a, b, {d})", {"d": 0.3}).num == 0.3                          # {param} substitution
    assert Rule.parse("mean(a, b); vis=majority").vis == "majority"
    for bad in ("frob(a)", "lerp(a, b)", "weighted(a, b)", "mean(a, b); vis=best", "extend(a, b, x)", "mean()", "copy(a, b)"):
        with pytest.raises(RuleError):
            Rule.parse(bad)
    with pytest.raises(RuleError, match="unknown parameter"):
        Rule.parse("extend_m(a, b, {missing})", {})


def test_mapping_from_the_mods_skeleton_geometry(bundle):
    m = bundle.mapping(HUMAN_13)
    sk, ok, _ = m.points(standing(), np.ones(13, bool))
    at = {n: sk[i] for i, n in enumerate(SHOOTER)}
    assert ok.all()
    assert np.allclose(at["head_center"], [0, 0, 1.70])                                  # head bone + 0.0 m
    assert np.allclose(at["head_top"], [0, 0, 1.82])                                     # + 0.12 m up the neck->head axis
    assert np.allclose(at["neck"], [0, 0, 1.45])                                         # mid-shoulders
    assert np.allclose(at["spine"], [0, 0, (0.95 + 1.45) / 2])                           # midpoint of the labeled hip and neck
    assert np.allclose(at["hip"], [0, 0, 0.95]) and np.allclose(at["l_wrist"], [0, 0.28, 0.9])
    com = at["center_of_mass"]
    assert 0.9 < com[2] < 1.3 and abs(com[0]) < 1e-9 and abs(com[1]) < 1e-9                 # inside the torso, on the axis
    m2 = bundle.mapping(HUMAN_13, {"head_top_offset_m": 0.2, "head_center_offset_m": 0.05})   # per-game calibration
    sk2, _, _ = m2.points(standing(), np.ones(13, bool))
    assert np.allclose(sk2[0], [0, 0, 1.90]) and np.allclose(sk2[1], [0, 0, 1.75])


def test_validity_and_visibility_propagate_through_derived_points(bundle):
    m = bundle.mapping(HUMAN_13)
    valid = np.ones(13, bool)
    valid[HUMAN_13.index("l_shoulder")] = False
    _, ok, _ = m.points(standing(), valid)
    bad = {SHOOTER[i] for i in np.where(~ok)[0]}
    assert bad == {"l_shoulder", "neck", "spine", "center_of_mass"}                       # everything built on it; nothing else
    vis = np.full(13, 2)
    vis[HUMAN_13.index("neck")] = 1
    vis[HUMAN_13.index("head")] = 1
    _, _, v = m.points(standing(), np.ones(13, bool), vis)
    got = dict(zip(SHOOTER, v))
    assert got["head_center"] == 1 and got["head_top"] == 1                               # head bone hidden -> aim point hidden
    assert got["l_shoulder"] == 2 and got["neck"] == 2                                    # neck = mean of the shoulders only
    assert got["spine"] == 2                                                              # built on the labeled neck + hip
    assert got["center_of_mass"] == 2                                                     # majority of the mass is visible


def test_entity_conversion_keeps_meta_and_adds_the_team(bundle):
    m = bundle.mapping(HUMAN_13)
    e = EntityState(3, "rig", standing(), np.ones(13, bool), meta={"outfit": "armor"})
    out = m.convert_entity(e, {"team": "t", "outfit": "ignored: the entity's own meta wins"})
    assert out.skeleton_world.shape == (12, 3) and out.meta == {"team": "t", "outfit": "armor"}
    assert bundle.schema.class_of(out.meta) == (1, None) and bundle.schema.class_of({"team": "CT"}) == (0, None)
    assert bundle.schema.class_of({})[1] and bundle.schema.class_of({"team": "red"})[1]    # warns, defaults to class 0


def coco_person(x=100.0, y=50.0, w=60.0, h=180.0, conf=0.9):
    rel = [(.5, .03), (.46, .02), (.54, .02), (.42, .04), (.58, .04), (.3, .2), (.7, .2), (.2, .38), (.8, .38), (.15, .5),
           (.85, .5), (.38, .52), (.62, .52), (.38, .74), (.62, .74), (.38, .97), (.62, .97)]
    return np.array([[x + rx * w, y + ry * h, conf] for rx, ry in rel])


def test_a_coco17_baseline_is_mapped_onto_the_same_12_points_in_2d(bundle):
    mm = bundle.model_mapping("coco17")
    kp = coco_person()
    out = mm.apply2d(kp)
    assert out.shape == (12, 3)
    at = {n: out[i] for i, n in enumerate(SHOOTER)}
    assert np.allclose(at["neck"][:2], (kp[5, :2] + kp[6, :2]) / 2)                       # same definition as the ground truth
    assert np.allclose(at["hip"][:2], (kp[11, :2] + kp[12, :2]) / 2)
    assert np.allclose(at["head_center"][:2], kp[[0, 3, 4], :2].mean(axis=0))
    assert at["head_top"][1] < at["head_center"][1]                                       # the crown is above (smaller y)
    assert np.allclose(at["spine"][:2], (at["neck"][:2] + at["hip"][:2]) / 2)
    low = kp.copy()
    low[3, 2] = 0.1                                                                       # one ear hidden (profile view)
    assert mm.apply2d(low)[1, 2] == pytest.approx((0.1 + 0.9 + 0.9) / 3)                  # mean: the head is still found
    low2 = kp.copy()
    low2[5, 2] = 0.1
    assert mm.apply2d(low2)[3, 2] == pytest.approx(0.1)                                   # neck = min over the shoulders


def test_misuse_is_caught_with_clear_errors(bundle):
    with pytest.raises(SchemaFileError, match="human13"):
        bundle.mapping(COCO_17)                                                           # rules are written against human13
    with pytest.raises(SchemaFileError, match="no mapping for model keypoints"):
        bundle.model_mapping("openpose25")
    t = SkeletonSchema("t", ("a", "b"), (("a", "b"),), ())
    with pytest.raises(RuleError, match="no rule"):
        SchemaMapping(HUMAN_13, t, {"a": Rule.parse("copy(head)")})
    with pytest.raises(RuleError, match="unknown keypoint"):
        SchemaMapping(HUMAN_13, t, {"a": Rule.parse("copy(head)"), "b": Rule.parse("copy(nose)")})
    with pytest.raises(RuleError, match="cycle"):
        SchemaMapping(HUMAN_13, t, {"a": Rule.parse("copy(@b)"), "b": Rule.parse("copy(@a)")})
    m = bundle.mapping(HUMAN_13)
    with pytest.raises(RuleError, match="3D"):
        m.points(np.zeros((13, 2)), None)                                                 # extend_m is meters: 3D only
    assert identity_mapping(HUMAN_13).apply2d(np.ones((13, 3))).shape == (13, 3)


def test_changing_the_point_set_is_a_data_change_not_a_code_change():
    """A different aim schema (8 points, three classes, another primary point) written only as data runs the same code."""
    d = {"schema": {
        "name": "tiny8", "source": "human13", "keypoints": ["crown", "chest", "l_hand", "r_hand", "belly", "l_foot", "r_foot",
                                                            "mass"],
        "edges": [["crown", "chest"], ["chest", "belly"]], "flip_pairs": [["l_hand", "r_hand"], ["l_foot", "r_foot"]],
        "primary": ["chest", "crown"], "classes": ["red", "blue", "neutral"], "class_key": "side",
        "roles": {"head": "crown", "neck": "chest", "pelvis": "belly", "l_shoulder": "l_hand", "r_shoulder": "r_hand"},
        "points": {
            "crown": {"sigma": 0.03, "weight": 2.0, "rule": "extend_m(neck, head, 0.1)"},
            "chest": {"sigma": 0.05, "derived": True, "rule": "lerp(neck, pelvis, 0.2)"},
            "l_hand": {"rule": "copy(l_wrist)"}, "r_hand": {"rule": "copy(r_wrist)"},
            "belly": {"rule": "copy(pelvis)"}, "l_foot": {"rule": "copy(l_ankle)"}, "r_foot": {"rule": "copy(r_ankle)"},
            "mass": {"derived": True, "weight": 0.2, "rule": "mean(head, pelvis)"}}}}
    b = bundle_from_dict(d)
    m = b.mapping(HUMAN_13)
    sk, ok, _ = m.points(standing(), np.ones(13, bool))
    assert b.schema.num_keypoints == 8 and ok.all() and np.allclose(sk[0], [0, 0, 1.8])
    assert b.schema.primary_idx() == [1, 0] and b.schema.flip_idx() == [0, 1, 3, 2, 4, 6, 5, 7]
    assert b.schema.class_of({"side": "blue"}) == (1, None)
    with pytest.raises(SchemaFileError, match="no \\[schema.points"):
        bundle_from_dict({"schema": {"name": "x", "keypoints": ["a"], "points": {}}})
    with pytest.raises(SchemaFileError, match="unknown keypoint"):
        bundle_from_dict({"schema": {"name": "x", "keypoints": ["a"], "primary": ["zzz"], "points": {"a": {}}}})
