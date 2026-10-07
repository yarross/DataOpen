"""The device manifest, the settings bundle and the two knobs' parameter mapping."""
import copy
import json
import random

import pytest

from dataopen.assist.params import AscParams
from dataopen.assist.sim_user import PERSONAS, build_profile
from dataopen.assist.tremor import TremorParams
from dataopen.ctl import bundle as B
from dataopen.ctl import manifest as M
from dataopen.ctl import protocol as P
from dataopen.ctl import tuning as T


# ---------------------------------------------------------------------------------------------------------------- manifest
def test_default_manifest_is_valid_small_and_fully_translated():
    m = M.default_manifest()
    assert M.validate_manifest(m) == []
    assert len(M.manifest_bytes(m)) < 8000                    # far under the 16 KB limit: it crosses BLE in a handful of seconds
    texts = []

    def walk(o):
        if isinstance(o, dict):
            if set(o) >= {"ru", "en"}:
                texts.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(m)
    assert len(texts) > 20 and all(t["ru"].strip() and t["en"].strip() and t["ru"] != t["en"] for t in texts)


def test_every_control_in_the_default_manifest_is_known_to_the_gateway_and_vice_versa():
    mm = M.Manifest()
    handled = {"assist.on", "assist.strength", "tremor.level", "calib.running", "profile.restore", "pairing.forget"}
    shown = {k for k, c in mm.by_key.items() if c["type"] in ("toggle", "stepper", "action")}
    assert shown == handled


def test_the_safety_shell_is_not_in_the_manifest():
    text = M.manifest_bytes(M.default_manifest()).decode()
    for word in ("stop", "bypass", "passthru", "panic"):
        assert word not in text.lower().replace("bypass_", "")      # the shell is fixed in the client and fed by `status`


BAD = [
    ("schema", lambda m: m.update(schema=2)),
    ("rev", lambda m: m.update(rev=70000)),
    ("rev type", lambda m: m.update(rev="1")),
    ("no pages", lambda m: m.update(pages=[])),
    ("too many pages", lambda m: m.update(pages=[copy.deepcopy(m["pages"][0]) | {"id": f"p{i}"} for i in range(9)])),
    ("title missing a language", lambda m: m["title"].pop("en")),
    ("empty text", lambda m: m["title"].update(ru="")),
    ("long label", lambda m: m["pages"][0]["controls"][0]["label"].update(en="x" * 81)),
    ("unknown type", lambda m: m["pages"][0]["controls"][0].update(type="slider3d")),
    ("duplicate id", lambda m: m["pages"][0]["controls"][1].update(id="assist")),
    ("bad id", lambda m: m["pages"][0]["controls"][0].update(id="Bad Id")),
    ("bad key", lambda m: m["pages"][0]["controls"][0].update(key="../etc")),
    ("bad confirm", lambda m: m["pages"][0]["controls"][0].update(confirm="maybe")),
    ("stepper range", lambda m: m["pages"][0]["controls"][1].update(min=0, max=500)),
    ("stepper mark outside", lambda m: m["pages"][0]["controls"][1]["marks"].update({"99": M.L("а", "b")})),
    ("meter fmt", lambda m: m["pages"][1]["controls"][0].update(fmt="no placeholder")),
    ("file op", lambda m: m["pages"][1]["controls"][3].update(op="rm")),
    ("file size", lambda m: m["pages"][1]["controls"][3].update(max_bytes=10**9)),
    ("controls not a list", lambda m: m["pages"][0].update(controls={})),
]


@pytest.mark.parametrize("name,mutate", BAD, ids=[b[0] for b in BAD])
def test_validator_rejects(name, mutate):
    m = M.default_manifest()
    mutate(m)
    assert M.validate_manifest(m), name
    with pytest.raises(ValueError):
        M.Manifest(m)


def test_validator_limits_depth_count_and_size():
    m = M.default_manifest()
    g = {"id": "g0", "type": "group", "label": M.L("а", "a"), "controls": []}
    cur = g
    for i in range(1, 6):
        nxt = {"id": f"g{i}", "type": "group", "label": M.L("а", "a"), "controls": []}
        cur["controls"].append(nxt)
        cur = nxt
    m["pages"][0]["controls"].append(g)
    assert any("deeper" in e for e in M.validate_manifest(m))
    m = M.default_manifest()
    m["pages"][0]["controls"] += [{"id": f"n{i}", "type": "note", "label": M.L("а", "a")} for i in range(70)]
    assert any("more than" in e for e in M.validate_manifest(m))
    m = M.default_manifest()
    m["pages"][0]["controls"][0]["help"] = M.L("я" * 240, "e" * 240)
    m["pages"][0]["controls"] += [{"id": f"h{i}", "type": "note", "label": M.L("я" * 80, "e" * 80), "help": M.L("я" * 240, "e" * 240)}
                                  for i in range(40)]
    assert M.validate_manifest(m)


def test_validator_survives_garbage():
    rng = random.Random(5)
    base = M.default_manifest()

    def mutate(o, depth=0):
        if isinstance(o, dict) and o and rng.random() < 0.3:
            o.pop(rng.choice(list(o)))
        for k in list(o) if isinstance(o, dict) else range(len(o)) if isinstance(o, list) else []:
            if rng.random() < 0.05:
                o[k] = rng.choice([None, 0, -1, 2**70, "", "x" * 500, [], {}, [1, 2], {"a": 1}, True, 1.5])
            elif isinstance(o[k], (dict, list)) and depth < 8:
                mutate(o[k], depth + 1)
    for _ in range(400):
        m = copy.deepcopy(base)
        mutate(m)
        M.validate_manifest(m)                                  # must return a list, never raise
    for junk in (None, 1, "x", [], [[]], {"schema": 1}, {"schema": 1, "rev": 1, "title": 5, "pages": [5]}):
        assert M.validate_manifest(junk)


def test_check_set_is_strict():
    mm = M.Manifest()
    ok, bad = mm.check_set("assist.strength", 7), mm.check_set("assist.strength", 11)
    assert ok.ok and not bad.ok and bad.code == P.E.BAD_VALUE
    for v in (True, 7.0, "7", None, -1, [7]):
        assert not mm.check_set("assist.strength", v).ok, v
    for v in (1, 0, "true", None):
        assert not mm.check_set("assist.on", v).ok, v
    assert mm.check_set("assist.on", True).ok
    for k in ("nope", None, 5, "profile.restore", "profile.fill"):
        assert mm.check_set(k, 1).code == P.E.BAD_KEY, k


def test_check_act_needs_confirmation_for_two_step():
    mm = M.Manifest()
    assert not mm.check_act("pairing.forget", False).ok
    assert mm.check_act("pairing.forget", True).ok
    assert mm.check_act("assist.on", True).code == P.E.BAD_KEY


def test_manifest_hash_changes_with_content():
    a, b = M.default_manifest(1), M.default_manifest(1)
    b["title"]["en"] = "Other"
    assert M.manifest_hash(a) != M.manifest_hash(b) and len(M.manifest_hash(a)) == 4


# ---------------------------------------------------------------------------------------------------------------- bundle
@pytest.fixture(scope="module")
def profile():
    return build_profile(PERSONAS["tremor"], minutes=3, seed=1)._state


def test_bundle_roundtrip(profile):
    raw = B.Bundle("Мой профиль", "2026-10-07", 7, 3, profile).pack()
    assert len(raw) < 400
    b = B.unpack(raw)
    assert (b.name, b.created, b.strength, b.tremor) == ("Мой профиль", "2026-10-07", 7, 3)
    assert b.profile.pack() == profile.pack()
    s = B.unpack(B.Bundle(strength=2, tremor=9).pack())
    assert s.profile is None and (s.strength, s.tremor) == (2, 9)


def test_bundle_rejects_every_single_bit_flip(profile):
    raw = B.Bundle("x", "2026-10-07", 5, 5, profile).pack()
    for i in range(len(raw) * 8):
        b = bytearray(raw)
        b[i // 8] ^= 1 << (i % 8)
        with pytest.raises(B.BundleError):
            B.unpack(bytes(b))


def test_bundle_rejects_truncation_growth_and_garbage(profile):
    raw = B.Bundle("x", "d", 5, 5, profile).pack()
    for n in range(0, len(raw), 7):
        with pytest.raises(B.BundleError):
            B.unpack(raw[:n])
    with pytest.raises(B.BundleError) as e:
        B.unpack(raw + bytes(B.MAX_BUNDLE))
    assert e.value.key == "too_big"
    rng = random.Random(2)
    for _ in range(300):
        with pytest.raises(B.BundleError):
            B.unpack(bytes(rng.randrange(256) for _ in range(rng.randrange(0, 300))))


def _resign(raw: bytes, edit) -> bytes:
    b = bytearray(raw[:-4])
    edit(b)
    return bytes(b) + P.crc32(bytes(b)).to_bytes(4, "little")


def test_bundle_version_and_level_checks(profile):
    raw = B.Bundle("x", "d", 5, 5, profile).pack()
    newer = _resign(raw, lambda b: b.__setitem__(4, 2))
    with pytest.raises(B.BundleError) as e:
        B.unpack(newer)
    assert e.value.key == "version"
    with pytest.raises(B.BundleError):
        B.unpack(_resign(raw, lambda b: b.__setitem__(4, 0)))
    bad_level = B.Bundle("x", "d", 5, 5, None).pack()
    head = json.dumps({"name": "x", "created": "d", "tuning": {"strength": 99, "tremor": 5}}, separators=(",", ":")).encode()
    import struct
    forged = struct.pack("<4sBBBBHH", B.MAGIC, 1, 1, 0, 0, len(head), 0) + head
    forged += P.crc32(forged).to_bytes(4, "little")
    with pytest.raises(B.BundleError):
        B.unpack(forged)
    assert bad_level


def test_bundle_profile_from_a_newer_version_or_damaged_is_refused(profile):
    raw = B.Bundle("x", "d", 5, 5, profile).pack()
    off = len(raw) - 4 - 98
    with pytest.raises(B.BundleError):                                  # a damaged profile inside a well-formed bundle
        B.unpack(_resign(raw, lambda b: b.__setitem__(off + 30, b[off + 30] ^ 1)))
    with pytest.raises(B.BundleError) as e:                             # newer profile version, even with a valid outer CRC
        B.unpack(_resign(raw, lambda b: b.__setitem__(off + 4, 9)))
    assert e.value.key == "version"


def test_bundle_names_are_sanitized():
    raw = B.Bundle("a\x00b\x1bc\n" + "z" * 100, "2026-10-07xxxxxxxx", 5, 5).pack()
    b = B.unpack(raw)
    assert "\x00" not in b.name and "\x1b" not in b.name and "\n" not in b.name and len(b.name) <= 40 and len(b.created) <= 10


# ---------------------------------------------------------------------------------------------------------------- tuning
def test_factor_shape():
    assert [T.factor(i) for i in (0, 5, 10)] == [0.0, 1.0, 1.5]
    assert all(T.factor(i) < T.factor(i + 1) for i in range(10))
    assert T.factor(-5) == 0.0 and T.factor(99) == 1.5


@pytest.fixture(scope="module")
def tview():
    return build_profile(PERSONAS["tremor"], minutes=8, seed=1)


@pytest.fixture(scope="module")
def sview():
    return build_profile(PERSONAS["steady"], minutes=8, seed=1)


def test_level_five_is_exactly_the_recommendation(tview):
    ap, tp = AscParams.from_view(tview), TremorParams.from_view(tview)
    assert T.scale_asc(ap, 5) == ap and T.scale_tremor(tp, 5) == tp


def test_level_zero_switches_a_layer_off(tview):
    d = T.derive(tview, 0, 5)
    assert not d.asc_on and d.tremor_on
    d = T.derive(tview, 5, 0)
    assert d.asc_on and not d.tremor_on
    assert not T.derive(tview, 0, 0).any_on


def test_no_profile_means_nothing_is_on():
    d = T.derive(None, 10, 10)
    assert not d.any_on and d.asc.enabled == 0 and d.tremor.enabled == 0


def test_levels_never_weaken_the_layers_as_they_rise(tview):
    prev = None
    for lv in range(1, 11):
        a, t = T.scale_asc(AscParams.from_view(tview), lv), T.scale_tremor(TremorParams.from_view(tview), lv)
        cur = (a.s_brake, -a.cfg.k_floor, a.hold_scale, t.s_max, t.trim_cap)
        if prev:
            assert all(c >= p - 1e-12 for c, p in zip(cur, prev)), (lv, cur, prev)
        prev = cur
    assert prev[3] <= T.S_MAX_CAP and prev[4] <= T.TRIM_CAP_MAX


def test_floors_and_caps_hold_at_the_extremes(tview, sview):
    for v in (tview, sview):
        for lv in (0, 1, 5, 10):
            d = T.derive(v, lv, lv)
            assert d.asc.k_floor >= int(0.1 * 65536) - 1 and d.asc.k_floor <= 65536
            assert d.tremor.s_max <= 65536 and d.tremor.trim_cap <= 64
            assert d.asc.s_cap >= d.asc.s_brake


def test_a_person_without_tremor_never_gets_a_filter(sview):
    d = T.derive(sview, 10, 10)
    assert d.asc_on and not d.tremor_on


def test_every_level_pair_produces_blobs_the_bridge_accepts(tview, sview, tmp_path):
    """The real C core's own range tables are the judge."""
    from dataopen.bridge import protocol as BP
    from dataopen.bridge.sim import Rig
    from dataopen.bridge.sim_usb import SimMouse
    rig = Rig(SimMouse("m16"), module=False)
    rig.run(50)
    gen = 1000
    for v in (tview, sview, None):
        for s in (0, 1, 5, 9, 10):
            for t in (0, 1, 5, 9, 10):
                d = T.derive(v, s, t)
                gen += 1
                for f in BP.asc_frames(d.asc, gen, d.profile_id, d.ppc, serial=gen & 15) + BP.tremor_frames(
                        d.tremor, gen, d.profile_id, serial=(gen + 1) & 15):
                    rig.b.link_rx(rig.t, BP.pack_frame(f))
    assert rig.status().params_rejected == 0


def test_knobs_on_the_closed_loop_simulator(tview):
    """The same simulated hand at three settings. It checks what can be checked without people: nothing is added to the motion, the
    guard holds, the hold gets steadier and the cost (time) is visible. It says nothing about how any of it feels."""
    import numpy as np

    from dataopen.assist.chain import AssistChain
    from dataopen.assist.sim_user import run_trial, summarize
    p = PERSONAS["tremor"]
    ap, tp = AscParams.from_view(tview), TremorParams.from_view(tview)

    def run(lv):
        def mk():                                                   # a fresh chain per reach, as the person-simulator comparisons do
            return None if lv is None else AssistChain.build(T.scale_asc(ap, lv), T.scale_tremor(tp, lv), "float")
        return summarize([run_trial(p, mk(), np.random.default_rng(1000 + i), 600.0, 30.0) for i in range(10)], 30.0)
    none, five, ten = run(None), run(5), run(10)
    for s in (five, ten):                                       # (the guard counter is only meaningful with a chain in the loop)
        assert s["amp_violations"] == 0 and s["guard_violations"] == 0
    assert none["acquired"] == five["acquired"] == ten["acquired"] == 1.0
    assert five["hold_rms_px"] < 0.5 * none["hold_rms_px"]
    assert ten["hold_rms_px"] <= 1.1 * five["hold_rms_px"]
    assert ten["overshoot_rate"] < none["overshoot_rate"]
