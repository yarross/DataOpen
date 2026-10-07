"""The control gateway against the REAL bridge core: the phone's every action, and what the bridge actually does about it."""
import random
import shutil
import struct

import pytest

from dataopen.bioprofile.profile import ProfileState, ProfileView
from dataopen.bioprofile.store import ProfileStore
from dataopen.bridge.cbridge import REASONS, S_ASSIST, S_BYPASS, S_PASSTHRU
from dataopen.ctl import bundle as B
from dataopen.ctl import manifest as M
from dataopen.ctl import protocol as P
from dataopen.ctl.sim import SimLearner, World, seed_profile

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


@pytest.fixture(scope="module")
def tremor_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tremor-profile")
    seed_profile(d, "tremor")
    return d


def world(tmp_path, tremor_dir=None, **kw) -> World:
    d = tmp_path / "gw"
    if tremor_dir is not None:
        shutil.copytree(tremor_dir, d)
    return World(d, **kw)


def err(r) -> int:
    assert r.type == P.T_ERR, r
    return r.json()["code"]


def ok(r) -> dict:
    assert r.type == P.T_ACK, (r.type, r.body)
    return r.json()


def state(w):
    return w.bridge().state


def moves(w: World, n: int = 400, seed: int = 0, big: int = 12):
    """Feed random hand motion through the bridge; returns [(in_x, in_y, out_x, out_y)] of the reports the PC saw."""
    rng = random.Random(seed)
    start = len(w.rig.pc_reports)
    for _ in range(n):
        dx, dy = rng.randint(-big, big), rng.randint(-big, big)
        w.rig.move(dx, dy)
        w.rig.step()
        w.rig.step()
        w.rig.step()
    out = []
    for _, route, _, o, r in w.rig.pc_reports[start:]:
        if route != "bridge":
            continue
        _, ix, iy, _, _ = struct.unpack("<Bhhbb", r)
        _, ox, oy, _, _ = struct.unpack("<Bhhbb", o)
        out.append((ix, iy, ox, oy))
    return out


def never_adds(rows) -> bool:
    for ix, iy, ox, oy in rows:
        for i, o in ((ix, ox), (iy, oy)):
            if abs(o) > abs(i) or (o != 0 and (o > 0) != (i > 0)) or (i == 0 and o != 0):
                return False
    return True


# ---------------------------------------------------------------------------------------------------------------- basics
def test_a_fresh_device_does_not_assist_until_asked(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    assert state(w) == S_PASSTHRU and w.reason() in ("CMD_PASSTHRU", "WAIT_PC")
    w.run(2000)
    assert state(w) == S_PASSTHRU and w.reason() == "CMD_PASSTHRU"
    rows = moves(w, 200)
    assert rows and all((a, b) == (c, d) for a, b, c, d in rows)        # PASSTHRU: byte for byte what the mouse sent
    assert w.gw.settings.assist_wanted is False and w.bridge().params_rejected == 0


def test_session_handshake_and_manifest_over_small_chunks(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, chunk=20)
    r = w.phone.connect()
    j = r.json()
    assert r.type == P.T_HELLO_R and j["chunk"] == 20 and j["manifest_hash"] == M.Manifest().hash.hex() and j["fw"]
    assert w.phone.get_manifest() == M.default_manifest()
    assert w.phone.get_state()["assist.on"] is False
    assert w.phone.bad == 0


def test_hello_negotiates_the_chunk_size(tmp_path, tremor_dir):
    for want, got in ((10, 20), (100, 100), (9999, 244)):
        w = world(tmp_path / str(want), tremor_dir)
        w.phone.chunk = max(P.CHUNK_MIN, min(want, P.CHUNK_MAX))
        r = w.phone.call(P.T_HELLO, {"v": 1, "chunk": want})
        assert r.json()["chunk"] == got


def test_wrong_protocol_version_is_refused(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    assert err(w.phone.call(P.T_HELLO, {"v": 2})) == P.E.BAD_VERSION
    assert err(w.phone.call(P.T_GET, {"what": "state"})) == P.E.NO_SESSION


def test_nothing_but_the_safety_actions_works_before_hello(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    assert err(ph.set("assist.on", True)) == P.E.NO_SESSION
    assert err(ph.act("profile.restore", True)) == P.E.NO_SESSION
    assert err(ph.put_bundle(b"x")) == P.E.NO_SESSION
    assert ph.call(P.T_PING, {}).type == P.T_PONG
    assert ok(ph.stop())["ok"] and ok(ph.hard_bypass())["ok"]


# ---------------------------------------------------------------------------------------------------------------- enabling / stop
def test_enabling_assistance_with_a_profile_engages_the_bridge_and_never_adds_motion(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=20)
    ph = w.phone
    ph.connect()
    assert ok(ph.set("assist.on", True))["trial"] == 20
    w.run(500)
    assert state(w) == S_ASSIST
    st = w.gw.status()
    assert st.mode == S_ASSIST and st.flags & P.SF_ASSIST_WANTED and st.flags & P.SF_TRIAL and st.ready & P.RB_ASC
    rows = moves(w, 600, big=14)
    assert rows and never_adds(rows)
    assert any((a, b) != (c, d) for a, b, c, d in rows)                 # and it does something
    assert w.bridge().params_rejected == 0 and w.bridge().invariant_viol == 0


def test_stop_is_immediate_persisted_and_byte_exact(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    w.run(500)
    assert state(w) == S_ASSIST
    ok(ph.stop())
    assert state(w) == S_PASSTHRU and w.reason() == "CMD_PASSTHRU"      # no tick in between
    rows = moves(w, 200)
    assert all((a, b) == (c, d) for a, b, c, d in rows)
    assert w.gw.settings.assist_wanted is False
    w.run(3000)
    assert state(w) == S_PASSTHRU                                       # the gateway does not turn it back on
    w.make_gateway()                                                    # restart of the compute module
    w.run(3000)
    assert w.gw.settings.assist_wanted is False and state(w) == S_PASSTHRU


def test_stop_ends_a_trial_and_works_without_a_session_or_manifest(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.disconnect()
    assert w.gw.trial is not None
    ph.gw.on_connect()
    ok(ph.stop())                                                       # straight away, no HELLO
    assert w.gw.trial is None and state(w) == S_PASSTHRU


def test_losing_the_phone_changes_nothing_about_the_device(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    w.run(300)
    ph.disconnect()
    w.run(30_000)
    assert state(w) == S_ASSIST and w.gw.settings.assist_wanted


def test_enabling_without_a_profile_is_allowed_but_nothing_is_active(tmp_path):
    w = world(tmp_path)
    ph = w.phone
    ph.connect()
    ok(ph.set("assist.on", True))
    w.run(500)
    st = w.gw.status()
    assert st.ready == 0 and st.fill == 0
    rows = moves(w, 200)
    assert all((a, b) == (c, d) for a, b, c, d in rows)                 # a neutral chain is the identity
    assert w.bridge().params_rejected == 0


# ---------------------------------------------------------------------------------------------------------------- the hand's latches
def hold_panic(w, ms):
    w.rig.panic(True)
    w.run(ms)
    w.rig.panic(False)
    w.run(50)


def test_the_phone_can_not_lift_a_panic_latch_and_says_why(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    w.run(300)
    hold_panic(w, 100)                                                  # a short press: soft latch
    assert state(w) == S_PASSTHRU and w.reason() == "PANIC"
    ok(ph.set("assist.on", False))
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    w.run(3000)
    st = w.gw.status()
    assert state(w) == S_PASSTHRU and w.reason() == "PANIC"
    assert st.flags & P.SF_LATCH_SOFT and st.reason_name == "PANIC" and st.mode == S_PASSTHRU
    hold_panic(w, 2100)                                                 # the hand re-arms it
    w.run(500)
    assert state(w) == S_ASSIST                                         # and the wish the phone left is honoured again


def test_hardware_bypass_is_one_way_from_the_phone(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    w.run(300)
    ok(ph.hard_bypass())
    w.run(400)
    assert state(w) == S_BYPASS and w.reason() == "CMD_BYPASS"
    assert w.gw.settings.assist_wanted is False and w.gw.status().flags & P.SF_LATCH_HW
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    w.run(5000)
    assert state(w) == S_BYPASS                                         # only the hand's long press brings it back
    assert w.rig.route == "bypass"
    n = w.bridge()
    hold_panic(w, 2100)
    w.run(6000)
    assert state(w) >= S_PASSTHRU and w.reason() != "CMD_BYPASS" and n is not None
    w.run(3000)
    assert state(w) == S_ASSIST                  # the wish set while latched is honoured and the gateway never re-latched it


def test_hard_bypass_is_resent_until_the_bridge_confirms_it(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    w.spi_loss = 0.6                                                    # most SPI frames are lost
    ok(w.phone.hard_bypass())
    w.run(1200)
    assert state(w) == S_BYPASS
    w.spi_loss = 0.0
    w.run(5000)
    assert state(w) == S_BYPASS and w.reason() == "CMD_BYPASS"          # and once latched it is not hammered again


# ---------------------------------------------------------------------------------------------------------------- trial
def advance(w, s):
    w.run(s * 1000)


def test_raising_help_is_a_trial_that_undoes_itself(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=20)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    advance(w, 19)
    assert w.gw.settings.assist_wanted and state(w) == S_ASSIST
    advance(w, 2)
    assert not w.gw.settings.assist_wanted and state(w) == S_PASSTHRU and w.reason() == "CMD_PASSTHRU"
    assert w.gw.status().flags & P.SF_TRIAL == 0


def test_confirming_keeps_it(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=5)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ok(ph.confirm(True))
    advance(w, 30)
    assert w.gw.settings.assist_wanted and state(w) == S_ASSIST
    assert err(ph.confirm(True)) == P.E.NOT_ALLOWED                     # nothing to confirm any more


def test_undo_goes_back_at_once(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=60)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ok(ph.confirm(True))
    ok(ph.set("assist.strength", 9))
    assert w.gw.trial is not None
    ok(ph.confirm(False))
    assert w.gw.settings.strength == 5 and w.gw.settings.assist_wanted and w.gw.trial is None


def test_lowering_help_is_free(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=20)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    r = ok(ph.set("assist.strength", 2))
    assert r["trial"] == 0 and w.gw.trial is None
    advance(w, 40)
    assert w.gw.settings.strength == 2                                  # nothing came back
    r = ok(ph.set("assist.on", False))
    assert r["trial"] == 0


def test_raising_during_a_trial_restarts_the_timer_and_dropping_back_ends_it(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=20)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    ph.set("assist.strength", 7)
    advance(w, 15)
    ph.set("assist.strength", 9)
    advance(w, 15)
    assert w.gw.settings.strength == 9                                  # the timer restarted at the second raise
    advance(w, 6)
    assert w.gw.settings.strength == 5                                  # and it returns to the KEPT value, not to 7
    ph.set("assist.strength", 8)
    ph.set("assist.strength", 4)                                        # at or below the kept state: nothing left to confirm
    assert w.gw.trial is None
    advance(w, 40)
    assert w.gw.settings.strength == 4


def test_raising_a_knob_while_help_is_off_is_not_a_trial_but_turning_help_on_is(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=10)
    ph = w.phone
    ph.connect()
    assert ok(ph.set("assist.strength", 9))["trial"] == 0
    assert ok(ph.set("assist.on", True))["trial"] == 10
    advance(w, 12)
    assert not w.gw.settings.assist_wanted and w.gw.settings.strength == 9    # the preset stays, help is off again


def test_a_restart_in_the_middle_of_a_trial_is_a_no(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    ph.set("assist.strength", 9)
    assert w.gw.trial is not None
    w.make_gateway()
    assert w.gw.settings.strength == 5 and w.gw.settings.assist_wanted and w.gw.trial is None


def test_a_restart_in_the_middle_of_the_first_trial_leaves_help_off(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    w.phone.connect()
    w.phone.set("assist.on", True)
    w.make_gateway()
    assert w.gw.settings.assist_wanted is False
    w.run(3000)
    assert state(w) == S_PASSTHRU


# ---------------------------------------------------------------------------------------------------------------- parameters to the bridge
def test_parameter_generations_only_grow_and_survive_a_restart(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    w.run(1500)
    g0 = w.bridge()
    seen = [(w.gw.gen["asc"], w.gw.gen["tremor"])]
    for lv in (7, 3, 9, 1, 5):
        ph.set("assist.strength", lv)
        ph.confirm(True)
        ph.set("tremor.level", lv)
        ph.confirm(True)
        w.run(300)
        seen.append((w.gw.gen["asc"], w.gw.gen["tremor"]))
    assert all(b[0] > a[0] and b[1] > a[1] or b == a for a, b in zip(seen, seen[1:])) and len(set(seen)) > 3
    w.run(1500)
    assert w.bridge().params_rejected == 0
    g1 = w.gw.gen["asc"]
    w.make_gateway()                                                    # a fresh process starts from a higher epoch
    w.run(1500)
    assert w.gw.gen["asc"] > g1 and w.bridge().params_rejected == 0 and g0.params_rejected == 0


def test_unchanged_parameters_are_a_keepalive_not_a_new_generation(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    w.phone.connect()
    w.phone.set("assist.on", True)
    w.run(1500)
    g = dict(w.gw.gen)
    w.run(20_000)
    assert w.gw.gen == g and state(w) == S_ASSIST                       # 20 s of keepalives: well past the 5 s parameter TTL
    assert w.bridge().params_rejected == 0


def test_assistance_survives_a_lossy_link(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600, seed=3)
    w.phone.connect()
    w.phone.set("assist.on", True)
    w.phone.confirm(True)
    w.run(800)
    w.spi_loss = 0.5
    states = set()
    for _ in range(40):
        w.run(500)
        states.add(state(w))
    assert states == {S_ASSIST}


def test_a_silent_gateway_means_the_bridge_falls_back_by_itself(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    w.phone.connect()
    w.phone.set("assist.on", True)
    w.phone.confirm(True)
    w.run(500)
    assert state(w) == S_ASSIST
    w.gw_dead = True
    w.run(1500)
    assert state(w) == S_PASSTHRU and w.reason() == "STALE_LINK"


def test_the_whole_knob_range_never_adds_motion_through_the_bridge(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    seed = 0
    for s in (0, 3, 5, 10):
        for t in (0, 5, 10):
            ph.set("assist.strength", s)
            ph.set("tremor.level", t)
            ph.confirm(True)
            w.run(1200)
            rows = moves(w, 250, seed=seed, big=25)
            seed += 1
            assert rows and never_adds(rows), (s, t)
    b = w.bridge()
    assert b.params_rejected == 0 and b.invariant_viol == 0


# ---------------------------------------------------------------------------------------------------------------- manifest authority
def test_the_device_enforces_its_own_manifest(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    assert err(ph.set("assist.strength", 11)) == P.E.BAD_VALUE
    assert err(ph.set("assist.strength", True)) == P.E.BAD_VALUE
    assert err(ph.set("assist.on", 1)) == P.E.BAD_VALUE
    assert err(ph.set("no.such.key", 1)) == P.E.BAD_KEY
    assert err(ph.set("profile.fill", 100)) == P.E.BAD_KEY             # shown, not settable
    assert err(ph.act("assist.on")) == P.E.BAD_KEY
    assert err(ph.act("pairing.forget")) == P.E.NOT_ALLOWED            # two-step: needs "confirmed"
    assert w.gw.settings.strength == 5 and not w.gw.settings.assist_wanted


def test_forgetting_phones_needs_the_button_on_the_device(tmp_path, tremor_dir):
    forgot = []
    w = world(tmp_path, tremor_dir, on_forget=lambda: forgot.append(1))
    ph = w.phone
    ph.connect()
    assert err(ph.act("pairing.forget", True)) == P.E.PHYSICAL and not forgot
    w.gw.physical_press()
    ok(ph.act("pairing.forget", True))
    assert forgot == [1]
    assert err(ph.act("pairing.forget", True)) == P.E.PHYSICAL          # one press, one use
    w.gw.physical_press()
    w.run(31_000)
    assert err(ph.act("pairing.forget", True)) == P.E.PHYSICAL          # and it expires


def test_garbage_on_the_wire_never_breaks_the_gateway(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    rng = random.Random(1)
    good = P.pack_json(P.T_SET, 1, {"key": "assist.strength", "value": 7})
    for _ in range(3000):
        mode = rng.randrange(4)
        if mode == 0:
            c = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 30)))
        elif mode == 1:
            b = bytearray(good)
            b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
            chunks, _ = P.chunk_message(bytes(b), 20, rng.randrange(64))
            for c in chunks:
                w.gw.on_write(c)
            continue
        elif mode == 2:
            body = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 200)))
            chunks, _ = P.chunk_message(P.pack_message(rng.randrange(256), rng.randrange(65536), body), 20, 0)
            for c in chunks:
                w.gw.on_write(c)
            continue
        else:
            c = bytes([0x80 | 0x40 | rng.randrange(64)]) + bytes(rng.randrange(256) for _ in range(19))
        w.gw.on_write(c)
        w.run(1)
    assert not w.gw.settings.assist_wanted
    assert ph.connect().type == P.T_HELLO_R and ph.get_state()["assist.on"] is False
    w.run(100)
    assert w.bridge().params_rejected == 0


def test_json_that_is_valid_but_wrong_is_an_error_not_a_crash(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    for body in (b"[]", b"null", b"\xff\xfe", b"{", b'{"key":["x"]}', b'{"key":{},"value":{}}', b'{"what":5}'):
        for t in (P.T_SET, P.T_ACT, P.T_GET, P.T_CONFIRM, P.T_HELLO):
            r = ph.call(t, body=body)
            assert r.type in (P.T_ERR, P.T_ACK, P.T_DATA, P.T_HELLO_R)
    assert w.gw.settings == w.gw.settings.__class__(False, 5, 5)


# ---------------------------------------------------------------------------------------------------------------- status
def test_status_snapshot_tells_the_truth(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    w.run(500)
    s = w.gw.status()
    assert s.bridge == 1 and s.mode == S_PASSTHRU and s.flags & P.SF_LINK_OK and s.flags & P.SF_HEALTHY and s.flags & P.SF_PARAMS_OK
    assert s.fill > 90 and s.ready & P.RB_ASC and s.ready & P.RB_TREMOR and s.strength == 5 and s.tremor == 5
    w.phone.connect()
    w.phone.set("assist.on", True)
    w.run(500)
    s = w.gw.status()
    assert s.mode == S_ASSIST and s.flags & P.SF_ASSIST_WANTED and s.flags & P.SF_TRIAL and s.trial_left_s > 3000
    assert s.reason_name == REASONS[0]
    w.gw_dead = True
    w.run(2000)
    s = w.gw.status()
    assert s.bridge == 0 and s.mode == P.MODE_UNKNOWN                    # the bridge has not been heard from: say so, do not guess


def test_state_events_reach_the_phone(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    ph.set("assist.strength", 8)
    ph.set("tremor.level", 2)
    assert ph.state["assist.strength"] == 8 and ph.state["tremor.level"] == 2
    assert ph.get_state()["assist.strength"] == 8
    assert any(n.strength == 8 for n in ph.status_notes)


# ---------------------------------------------------------------------------------------------------------------- calibration
@pytest.fixture()
def learner():
    return SimLearner(minutes=5.0, seed=2, speed=240.0)


def test_calibration_learns_a_profile_while_assistance_is_held_off(tmp_path, learner):
    w = world(tmp_path, learner=learner, trial_s=3600)
    ph = w.phone
    ph.connect()
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    w.run(500)
    assert state(w) == S_ASSIST
    ok(ph.set("calib.running", True))
    w.run(300)
    assert state(w) == S_PASSTHRU and w.reason() == "CMD_PASSTHRU"
    assert w.gw.status().flags & P.SF_CALIBRATING
    assert err(ph.set("assist.on", True)) == P.E.BUSY
    fills = []
    for _ in range(12):
        w.run(250)
        fills.append(w.gw.status().fill)
    assert fills == sorted(fills) and fills[-1] > fills[0] >= 0
    w.run(3000)
    assert w.gw.status().fill >= 95
    assert ProfileStore(w.dir / "profile").load() is None               # nothing is stored until the calibration is finished
    ok(ph.set("calib.running", False))
    st = ProfileStore(w.dir / "profile").load()
    assert st is not None and st.generation == 1
    assert w.gw.status().flags & P.SF_CALIBRATING == 0
    w.run(500)
    assert state(w) == S_ASSIST and w.gw.status().ready & P.RB_ASC      # the wish was kept; the new profile is a trial
    assert w.gw.trial is not None and w.gw.trial.profile_changed


def test_a_profile_from_calibration_that_is_not_confirmed_turns_help_off_when_there_was_none(tmp_path, learner):
    w = world(tmp_path, learner=learner, trial_s=10)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    ph.set("calib.running", True)
    w.run(25_000)
    ph.set("calib.running", False)
    assert w.gw.trial is not None
    advance(w, 12)
    assert not w.gw.settings.assist_wanted and state(w) == S_PASSTHRU
    assert w.gw.view is not None                                        # the learned profile itself is kept


def test_calibration_without_a_learner_is_refused_politely(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    assert err(ph.set("calib.running", True)) == P.E.UNSUPPORTED
    assert not w.gw.calibrating


def test_stopping_calibration_with_no_data_keeps_the_old_profile(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, learner=SimLearner(speed=0.001))
    ph = w.phone
    ph.connect()
    before = ProfileStore(w.dir / "profile").load().generation
    ph.set("calib.running", True)
    w.run(500)
    ph.set("calib.running", False)
    assert ProfileStore(w.dir / "profile").load().generation == before


# ---------------------------------------------------------------------------------------------------------------- bundles
def profile_stats(view: ProfileView):
    return {k: (round(v["median"], 3), v["n"]) for k, v in view.as_dict()["metrics"].items()}


def test_a_bundle_moves_a_profile_and_the_knobs_between_devices(tmp_path, tremor_dir):
    a = world(tmp_path / "a", tremor_dir)
    a.phone.connect()
    a.phone.set("assist.strength", 8)
    a.phone.set("tremor.level", 3)
    raw = a.phone.get_bundle()
    assert len(raw) < 400 and B.unpack(raw).strength == 8
    b = world(tmp_path / "b")
    b.phone.connect()
    ok(b.phone.put_bundle(raw))
    assert (b.gw.settings.strength, b.gw.settings.tremor) == (8, 3)
    assert profile_stats(b.gw.view) == profile_stats(a.gw.view)
    assert b.gw.status().fill == a.gw.status().fill


def test_a_bad_bundle_changes_nothing(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    raw = ph.get_bundle()
    before = (w.gw.settings, ProfileStore(w.dir / "profile").load().generation)
    rng = random.Random(4)
    for i in range(0, len(raw) * 8, 13):
        bad = bytearray(raw)
        bad[i // 8] ^= 1 << (i % 8)
        assert err(ph.put_bundle(bytes(bad))) == P.E.BAD_BUNDLE
    for n in (0, 1, 50, len(raw) - 1):
        assert err(ph.put_bundle(raw[:n])) == P.E.BAD_BUNDLE
    assert err(ph.put_bundle(bytes(rng.randrange(256) for _ in range(200)))) == P.E.BAD_BUNDLE
    assert err(ph.put_bundle(bytes(5000))) == P.E.TOO_BIG
    assert (w.gw.settings, ProfileStore(w.dir / "profile").load().generation) == before


def test_importing_with_help_on_is_a_trial_and_undo_brings_the_old_profile_back(tmp_path, tremor_dir):
    other = tmp_path / "other"
    seed_profile(other, "overshooter")
    donor = World(other, start=False)
    donor.phone.connect()
    raw = donor.phone.get_bundle()
    w = world(tmp_path / "w", tremor_dir, trial_s=20)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    old = profile_stats(w.gw.view)
    ok(ph.put_bundle(raw))
    assert profile_stats(w.gw.view) != old and w.gw.trial is not None
    ok(ph.confirm(False))
    assert profile_stats(w.gw.view) == old and w.gw.settings.assist_wanted


def test_restore_the_previous_profile(tmp_path, tremor_dir):
    other = tmp_path / "other"
    seed_profile(other, "overshooter")
    donor = World(other, start=False)
    donor.phone.connect()
    raw = donor.phone.get_bundle()
    w = world(tmp_path / "w", tremor_dir, trial_s=20)
    ph = w.phone
    ph.connect()
    old = profile_stats(w.gw.view)
    assert err(ph.act("profile.restore", True)) == P.E.NO_PROFILE      # nothing before it yet
    ok(ph.put_bundle(raw))
    new = profile_stats(w.gw.view)
    assert new != old
    ok(ph.act("profile.restore", True))
    assert profile_stats(w.gw.view) == old
    ok(ph.act("profile.restore", True))
    assert profile_stats(w.gw.view) == new                              # it swaps: restoring twice is 'redo'


def test_bundle_import_is_refused_while_calibrating(tmp_path, learner):
    w = world(tmp_path, learner=learner)
    ph = w.phone
    ph.connect()
    ph.set("calib.running", True)
    assert err(ph.put_bundle(B.Bundle().pack())) == P.E.BUSY


def test_a_newer_profile_on_disk_is_never_overwritten(tmp_path):
    d = tmp_path / "gw"
    d.mkdir()
    raw = bytearray(ProfileState(profile_id=1).pack())
    raw[4] = 9
    import zlib
    raw[-4:] = (zlib.crc32(bytes(raw[:-4])) & 0xFFFFFFFF).to_bytes(4, "little")
    (d / "profile.a").write_bytes(bytes(raw))
    w = World(d, start=False)
    assert w.gw.view is None and w.gw.profile_error
    w.phone.connect()
    assert (d / "profile.a").read_bytes() == bytes(raw)


def test_settings_survive_damage_to_the_newest_slot(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    ph.set("assist.strength", 2)
    newest = max((w.dir / "settings.a", w.dir / "settings.b"), key=lambda p: p.stat().st_mtime_ns)
    newest.write_bytes(b"garbage")
    w.make_gateway()
    assert w.gw.settings.assist_wanted in (True, False) and 0 <= w.gw.settings.strength <= 10      # an older slot, never a crash
