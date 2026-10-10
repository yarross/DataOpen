"""Safe defaults of the scene path and what happens when the pipeline is too slow (docs/LATENCY.md, "Умолчания v1 и деградация").

What is proved: the defaults are the ones the budget derives, a pipeline that falls behind switches the scene help OFF with hysteresis instead of
letting it blink (on the real C bridge core, in virtual time), a vanished target is withdrawn at once, and a model heavier than the budget is
kept but not run. What is not: any figure on a board, with an NPU, with a real screen or with a person."""
import random
import shutil

import pytest

from dataopen.assist.params import AscConfig
from dataopen.bridge import protocol as BP
from dataopen.latency import budget as B
from dataopen.latency import measure as M
from dataopen.runtime.frames import QueueSource
from dataopen.ui import policy as PL
from dataopen.ui.health import SceneHealth
from dataopen.ui.policy import V1
from dataopen.ui.scene import BridgeScenePublisher, Det, SceneConfig, UiSceneBuilder
from dataopen.ui.service import UiService, scene_service
from dataopen.ui.sim import GEOM_1080P

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")
Z = __import__("numpy").zeros((640, 640, 3), "uint8")


# ---------------------------------------------------------------------------------------------------------------- 1. the defaults
def test_the_defaults_are_the_safe_ones_and_the_old_behaviour_is_still_there_by_name():
    src = QueueSource(4)
    svc = UiService(src, None, UiSceneBuilder(GEOM_1080P), lambda s, f: None)
    assert svc.latest_only is True and svc.max_age_ms == 35.0 and (V1.latest_only, V1.max_age_ms) == (True, 35.0)
    assert SceneConfig().confirm_hits == V1.confirm_hits == 2
    old = UiService(src, None, UiSceneBuilder(GEOM_1080P), lambda s, f: None, latest_only=False, max_age_ms=None)
    assert old.latest_only is False and old.max_age_ms is None
    assert (V1.publish_max_age_ms, V1.recover_max_age_ms, V1.late_trip, V1.recover_n, V1.hold_ms) == (70.0, 50.0, 3, 15, 2000)
    assert V1.recover_max_age_ms < V1.publish_max_age_ms                                   # a real hysteresis, not one threshold


def test_the_defaults_are_derived_from_the_budget_not_typed_beside_it():
    sc = B.BY_KEY["v1-default"]
    s = B.summary(sc)
    assert s["healthy_age_hi_ms"] < V1.max_age_ms < s["safe_max_age_ms"]                   # healthy frames pass, frames too old to arrive in time do not
    for key in ("cpu-latest", "cpu-60", "v1-npu", "npu-4k60"):
        t = B.summary(B.BY_KEY[key])
        assert t["healthy_age_hi_ms"] < V1.max_age_ms < t["safe_max_age_ms"] + 0.01 or t["safe_max_age_ms"] > V1.max_age_ms, key
    assert PL.INFER_BUDGET_MS == V1.infer_budget_ms == int(B.max_infer_p95_ms(sc) * 2) / 2   # rounded down to half a ms
    assert V1.publish_max_age_ms + B.refresh_gap_ms(sc) <= B.bridge_cfg()["scene_ttl_ms"]  # a scene sent at the limit is still usable at its last use
    assert PL.INFER_MS["cpu"][1] <= PL.INFER_BUDGET_MS                                     # the reference network itself fits


def test_the_finding_is_kept_and_the_new_margins_count_the_refresh_gap():
    old, new = B.summary(B.BY_KEY["legacy-fifo"]), B.summary(B.BY_KEY["v1-default"])
    assert old["ttl_margin_ms"] < 0 and old["use_margin_ms"] < old["ttl_margin_ms"]          # the gap makes the old picture worse, not better
    assert new["use_margin_ms"] > 0 and new["use_margin_ms"] == pytest.approx(new["ttl_margin_ms"] - new["refresh_gap_ms"])
    assert new["B"][2] < old["B"][2] - 50                                                   # the worst case came down by more than 50 ms
    for key in ("v1-default", "cpu-latest", "v1-npu", "cpu-60", "npu-4k60", "npu-latest-c1"):
        t = B.summary(B.BY_KEY[key])
        assert t["use_margin_ms"] >= B.SAFETY_MARGIN_MS and t["guard_margin_ms"] > 0, key  # inside the TTL with the margin, and before the guard opens
    assert B.summary(B.BY_KEY["v1-default"])["guard_margin_ms"] > 0 and AscConfig().t_min_ms == 100
    assert B.summary(B.BY_KEY["v1-default"])["B"][2] < B._t_lo_ms()[0]                      # and far before the 'tremor' persona's guard (225 ms)


def test_confirm_hits_stays_two_because_one_costs_little_and_the_false_alarms_are_unmeasured():
    two, one = B.summary(B.BY_KEY["v1-default"]), B.summary(B.with_changes(B.BY_KEY["v1-default"], confirm_hits=1))
    assert 8.0 < two["B"][2] - one["B"][2] < 20.0                                           # the saving on the CPU: one detection cycle
    npu2, npu1 = B.summary(B.BY_KEY["v1-npu"]), B.summary(B.BY_KEY["npu-latest-c1"])
    assert npu2["B"][2] - npu1["B"][2] < 8.0                                                # and on the NPU only about one frame period
    assert two["use_margin_ms"] > B.SAFETY_MARGIN_MS                                         # the budget does not need the saving


# ---------------------------------------------------------------------------------------------------------------- 2. the state machine
def feed(h, ages, start=0.0, step=10.0):
    sent, t = [], start
    for a in ages:
        sent.append(h.scene(a, t))
        t += step
    return sent, t


def test_a_pipeline_that_warms_up_runs_a_late_scene_is_skipped_and_three_in_a_row_switch_off():
    h = SceneHealth()
    assert h.state == "warming"
    sent, t = feed(h, [30] * 14)
    assert not any(sent) and h.state == "warming"                                           # nothing is sent until the pipeline has proved itself
    sent, t = feed(h, [30], t)
    assert sent == [True] and h.state == "ok"
    sent, t = feed(h, [90, 30, 30], t)                                                       # one late scene: skipped, state unchanged
    assert sent == [False, True, True] and h.state == "ok" and h.skipped == 1
    sent, t = feed(h, [90, 95, 80], t)                                                       # three late in a row
    assert sent == [False, False, False] and h.state == "off" and h.reason == "late" and h.trips == 1
    assert h.take_announcement() is True and h.take_announcement() is False                  # one empty scene, once


def test_it_comes_back_only_when_held_long_enough_and_the_scenes_are_fresh_not_merely_usable():
    h = SceneHealth()
    feed(h, [30] * 15)
    feed(h, [90, 90, 90], 1000.0)
    assert h.state == "off"
    sent, t = feed(h, [30] * 30, 1100.0, step=10.0)                                          # fresh, but only 300 ms after the switch-off
    assert h.state == "off" and not any(sent)
    sent, t = feed(h, [60] * 30, 3100.0)                                                     # past the hold, but 60 ms: usable, not fresh enough
    assert h.state == "off" and not any(sent)
    sent, t = feed(h, [40] * 14, t)
    assert h.state == "off"
    sent, t = feed(h, [40], t)
    assert h.state == "ok" and sent == [True]


def test_a_scene_between_the_thresholds_neither_switches_off_nor_back_on():
    h = SceneHealth()
    feed(h, [30] * 15)
    sent, _ = feed(h, [55, 65, 69] * 20, 500.0)
    assert h.state == "ok" and all(sent) and h.trips == 0


def test_a_slow_detector_switches_off_at_once_with_its_own_reason_and_a_few_slow_frames_do_not():
    h = SceneHealth()
    feed(h, [30] * 15)
    for _ in range(5):
        h.detect(30.0)
    assert h.scene(30, 500.0) is True                                                        # fewer than 10 samples: no verdict
    for _ in range(20):
        h.detect(30.0)
    assert h.scene(30, 600.0) is False and h.state == "off" and h.reason == "slow_detector"
    for _ in range(60):
        h.detect(12.0)                                                                       # the detector recovers: the window flushes...
    sent, t = feed(h, [30] * 20, 3000.0)
    assert h.state == "ok"                                                                   # ...and after the hold the help returns


def test_dropped_frames_count_as_late_events():
    h = SceneHealth()
    feed(h, [30] * 15)
    for i in range(3):
        h.stale_frame(100.0 + i)
    assert h.state == "off" and h.reason == "late"


@pytest.mark.parametrize("seed", range(6))
def test_whatever_the_ages_do_the_help_does_not_blink(seed):
    """Random and adversarial age sequences around both thresholds: the number of switch-overs is bounded by the hold time, and no scene older
    than the publication limit is ever sent."""
    rng = random.Random(seed)
    h = SceneHealth()
    t, flips, last, total_ms = 0.0, 0, h.state, 0.0
    seqs = [lambda: rng.choice((30, 45, 52, 66, 72, 95, 110)), lambda: 72 if rng.random() < 0.5 else 48, lambda: 60 + 20 * rng.uniform(-1, 1)]
    draw = seqs[seed % 3]
    for _ in range(4000):
        age = draw()
        sent = h.scene(age, t)
        assert not (sent and age > V1.publish_max_age_ms)
        if h.state != last:
            flips, last = flips + 1, h.state
        t += 10.0
    total_ms = t
    assert flips <= 2 * (total_ms / V1.hold_ms) + 2, (flips, total_ms)                       # on->off->on at most once per hold period


# ---------------------------------------------------------------------------------------------------------------- 3. the publisher and the real bridge core
class Cap:
    def __init__(self):
        self.frames = []

    def __call__(self, f):
        self.frames.append(f)


def snap_with_target(builder, t_cap, with_target=True):
    box = (900.0, 508.0, 964.0, 572.0)
    dets = [Det("cursor", 0.9, (*GEOM_1080P.to_input(300, 540), *GEOM_1080P.to_input(311, 559)))]
    if with_target:
        dets.append(Det("button", 0.9, (*GEOM_1080P.to_input(box[0], box[1]), *GEOM_1080P.to_input(box[2], box[3]))))
    return builder.update(t_cap, dets)


def test_the_publisher_withdraws_a_target_that_is_gone_and_does_not_withdraw_on_one_missed_frame():
    b = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0))
    cap = Cap()
    pub = BridgeScenePublisher(b, cap, gate=False)
    for i in range(3):
        pub(snap_with_target(b, i * 7000), None)
    assert pub.sent == 2 and b.tracks                                                       # confirmed on the 2nd frame (frames 2 and 3 carry it)
    n0 = len(cap.frames)
    # the tracker keeps a target for one missed frame (misses <= 1), so the first empty snapshot still carries it
    for i in range(3, 9):
        pub(snap_with_target(b, i * 7000, with_target=False), None)
    kinds = [f.payload[4] for f in cap.frames[n0:]]                                         # the object count of each scene frame
    assert pub.retracted == 1 and kinds.count(0) == 1 and kinds[-1] == 0                       # exactly one empty scene, after the target was really gone


@needs_cc
def test_on_the_real_core_an_empty_scene_ends_the_help_now_not_at_the_ttl():
    r, _, _ = M._rig("overshooter")
    r.run(50)
    st0 = r.status()
    t_cap = r.t - 20_000
    r.module.send(r.t, [BP.scene_frame(t_cap, [BP.SceneObject(1, 600.0, 0.0, 24.0, None)])])
    r.step()
    assert r.status().scene_n == 1 and 0 <= r.status().scene_age_ms <= 100 and st0.scene_n == 0
    r.module.send(r.t, [BP.scene_frame(r.t - 15_000, [])])                                    # what the publisher sends when the targets are gone
    r.step()
    assert r.status().scene_n == 0 and r.status().scene_age_ms == -1                           # gone now, not at the TTL


@needs_cc
def test_without_the_guard_an_age_near_the_ttl_makes_the_help_blink_with_it_it_switches_off_once():
    """Scenes whose age alternates 90 / 110 ms (a pipeline sitting on the TTL). Count how often the bridge flips between 'has a usable scene'
    and 'has none'. Unguarded: it flips all the time. Guarded: it ends once and stays ended."""
    def run(guarded: bool):
        r, _, _ = M._rig("overshooter")
        cum0 = r.status().cum_x
        h = SceneHealth()
        flags, ttl, prev, carry = [], r.b.cfg.scene_ttl_ms, 0.0, 0.0
        for k in range(1500):
            pos = M._reach(1500.0, 1500, k)
            carry += pos - prev
            prev = pos
            i = int(round(carry))
            carry -= i
            r.move(i, 0)
            r.step()
            if k % 20 == 0:
                age = 30 if k < 600 else (90 if (k // 20) % 2 == 0 else 110)          # healthy for 600 ms, then sitting on the TTL
                send = True
                if guarded:
                    send = h.scene(age, r.t / 1000.0)
                    if h.take_announcement():
                        r.module.send(r.t, [BP.scene_frame(r.t - age * 1000, [])])
                if send:
                    r.module.send(r.t, [BP.scene_frame(r.t - age * 1000, [BP.SceneObject(1, 900.0 - (r.status().cum_x - cum0), 0.0, 24.0, None)])])
            s = r.status()
            flags.append(s.scene_n > 0 and s.scene_age_ms is not None and s.scene_age_ms <= ttl)
        return sum(1 for a, b in zip(flags, flags[1:]) if a != b), h, flags

    blink, _, _ = run(False)
    calm, h, flags = run(True)
    assert blink >= 20                                                                        # the old behaviour: on and off, over and over
    assert any(flags[:600]) and calm <= 3 and h.state == "off" and h.trips == 1               # the new one: was on while healthy, ends once, stays ended


# ---------------------------------------------------------------------------------------------------------------- 4. the whole path in virtual time
def test_the_v1_path_switches_the_scene_help_off_when_the_detector_is_too_slow():
    t = {"now": 0}
    clock = {"s": 0.0}

    class Slow:
        def detect(self, f):
            clock["s"] += 0.030                                                              # 30 ms of inference against a 19 ms budget
            return [Det("cursor", 0.9, (*GEOM_1080P.to_input(300, 540), *GEOM_1080P.to_input(311, 559))),
                    Det("button", 0.9, (*GEOM_1080P.to_input(900, 508), *GEOM_1080P.to_input(964, 572)))]

    src, sent = QueueSource(4), []
    builder = UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0))
    svc, pub = scene_service(src, Slow(), builder, sent.append, now=lambda: t["now"])
    svc.perf = lambda: clock["s"]
    for k in range(80):
        t["now"] = k * 7000
        src.push(Z, k, t["now"] - 3000, {"geometry": GEOM_1080P})
        svc.step(0.0)
    assert pub.health.state == "off" and pub.health.reason == "slow_detector"
    sent_before = len(sent)
    for k in range(80, 120):
        t["now"] = k * 7000
        src.push(Z, k, t["now"] - 3000, {"geometry": GEOM_1080P})
        svc.step(0.0)
    assert len(sent) == sent_before                                                           # off means silent


def test_the_v1_path_with_a_healthy_detector_warms_up_and_sends():
    t = {"now": 0}

    class Fast:
        def detect(self, f):
            return [Det("cursor", 0.9, (*GEOM_1080P.to_input(300, 540), *GEOM_1080P.to_input(311, 559))),
                    Det("button", 0.9, (*GEOM_1080P.to_input(900, 508), *GEOM_1080P.to_input(964, 572)))]

    src, sent = QueueSource(4), []
    svc, pub = scene_service(src, Fast(), UiSceneBuilder(GEOM_1080P, SceneConfig(warmup_frames=0)), sent.append, now=lambda: t["now"])
    for k in range(60):
        t["now"] = k * 7000
        src.push(Z, k, t["now"] - 3000, {"geometry": GEOM_1080P})
        svc.step(0.0)
    assert pub.health.state == "ok" and pub.sent > 30 and svc.errors == 0 and svc.trace.stale_dropped == 0


def test_a_frame_that_is_already_too_old_is_dropped_and_counted_as_late():
    t = {"now": 1_000_000}
    src = QueueSource(4)
    svc, pub = scene_service(src, type("D", (), {"detect": lambda self, f: []})(), UiSceneBuilder(GEOM_1080P), lambda f: None, now=lambda: t["now"])
    for k in range(4):
        src.push(Z, k, t["now"] - 80_000, {"geometry": GEOM_1080P})                           # every frame is 80 ms old when taken
        svc.step(0.0)
    assert svc.trace.stale_dropped == 4 and pub.health.state == "off"


# ---------------------------------------------------------------------------------------------------------------- 5. a heavy model of one's own
def test_the_estimate_is_calibrated_on_the_reference_network_and_grows_with_the_work():
    p50, p95 = PL.est_infer_ms(PL.REF_MACS)
    assert (p50, p95) == pytest.approx(PL.INFER_MS["cpu"]) and PL.fits_budget(PL.REF_MACS)
    assert not PL.fits_budget(int(PL.REF_MACS * 1.3)) and PL.fits_budget(int(PL.REF_MACS * 1.15))
    assert PL.est_infer_ms(PL.REF_MACS, "npu")[1] == pytest.approx(PL.INFER_MS["npu"][1])
    from dataopen.updates import models as MD
    assert PL.est_infer_ms(MD.MAX_MACS)[1] > PL.INFER_BUDGET_MS                              # the structural limit alone does not protect the budget


def test_check_model_says_what_the_budget_thinks_and_the_tool_warns(tmp_path, capsys):
    import json

    import pkg_helpers as H
    from dataopen.cli import build_parser
    from dataopen.updates import models as MD
    light, heavy = H.tiny_model(), H.tiny_model(conv_channels=190)
    a = MD.check_model(light, H.card_of(light))["est_infer_ms"]
    b = MD.check_model(heavy, H.card_of(heavy))
    assert a["fits"] and a["backend"] == "cpu" and not b["est_infer_ms"]["fits"] and 2.5e9 < b["macs"] < MD.MAX_MACS
    out = tmp_path / "heavy.onnx"
    out.write_bytes(heavy)
    ns = build_parser().parse_args(["update", "check-model", str(out)])
    assert ns.fn(ns) == 0                                                                    # accepted by the structure rules...
    said = capsys.readouterr().out
    assert "warning: over the latency budget" in said and json.loads(said.split("\nwarning")[0])["est_infer_ms"]["fits"] is False


@needs_cc
def test_a_heavy_model_is_kept_but_not_run_and_the_phone_is_told(tmp_path):
    import pkg_helpers as H
    from dataopen.ctl import residency as RS
    from dataopen.ctl.sim import World, seed_profile
    from test_ctl_gateway import ok
    from test_updates_gateway import give

    seed_profile(tmp_path / "seed", "tremor")
    shutil.copytree(tmp_path / "seed", tmp_path / "gw")
    w = World(tmp_path / "gw")
    w.phone.connect()
    light, heavy = H.tiny_model(), H.tiny_model(conv_channels=190)
    give(w, tmp_path, slot=0, model=(light, H.card_of(light, name="light", version=1)))
    assert w.phone.get_state()["model.state"] == "ok"
    give(w, tmp_path, slot=0, model=(heavy, H.card_of(heavy, name="heavy", version=2)))
    st = w.phone.get_state()
    assert st["model.state"] == "over_budget" and st["model.name"] == "heavy" and "over_budget" in RS.MODEL_STATES
    assert w.phone.get_packages()["models"][0]["state"] == "over_budget"
    ort = pytest.importorskip("onnxruntime")
    assert ort is not None
    from dataopen.ui.resident import detector_for_slot
    assert detector_for_slot(w.gw, 0) is None                                                 # kept, not run
    ok(w.phone.act("pkg.revert", True))                                                       # the lighter model, one step back
    assert w.phone.get_state()["model.state"] == "ok" and detector_for_slot(w.gw, 0) is not None
    ok(w.phone.act("model.clear", True))
    assert w.phone.get_state()["model.state"] == "none"


@needs_cc
def test_the_gateway_shows_the_scene_state_in_words_and_refuses_one_that_is_not_a_state(tmp_path):
    from dataopen.ctl import manifest as MF
    from dataopen.ctl import residency as RS
    from dataopen.ctl.sim import World

    w = World(tmp_path / "gw")
    w.phone.connect()
    assert w.phone.get_state()["scene.state"] == "none"
    for s in RS.SCENE_STATES:
        w.gw.set_scene_health(s)
        assert w.phone.get_state()["scene.state"] == s
    with pytest.raises(ValueError):
        w.gw.set_scene_health("blinking")
    c = {x["key"]: x for x in MF.Manifest().by_key.values() if x.get("key")}["scene.state"]
    assert set(c["map"]) == set(RS.SCENE_STATES) and all(v["ru"] and v["en"] for v in c["map"].values())
    assert "scene.state" in RS.STATE_KEYS
