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

from res_helpers import Clinic, profile_of

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
    assert pstore(w).load() is None               # nothing is stored until the calibration is finished
    ok(ph.set("calib.running", False))
    st = pstore(w).load()
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
    before = pstore(w).load().generation
    ph.set("calib.running", True)
    w.run(500)
    ph.set("calib.running", False)
    assert pstore(w).load().generation == before


# ---------------------------------------------------------------------------------------------------------------- sealed files
def profile_stats(view: ProfileView):
    return {k: (round(v["median"], 3), v["n"]) for k, v in view.as_dict()["metrics"].items()}


def pstore(w) -> ProfileStore:
    return w.gw.profile_store


def duo(tmp_path, tremor_dir, **kw):
    """Two devices: A with a learned profile, B empty."""
    return world(tmp_path / "a", tremor_dir, **kw), world(tmp_path / "b", **kw)


def clinic_for(tmp_path, name="clinic") -> Clinic:
    return Clinic(tmp_path, name)


def sent(clinic, b, profile, press_b=True, **kw):
    """The clinic seals `profile` (its own data) to B; B's button is pressed or not; the file goes in. Returns (file, reply)."""
    raw = clinic.file(b, profile=profile, tuning=kw.pop("tuning", (5, 5)), **kw)
    if press_b:
        b.gw.physical_press()
    return raw, b.phone.put_bundle(raw)


def hello(*ws):
    for w in ws:
        w.phone.connect()


def test_a_sealed_file_from_a_sender_sets_a_profile_and_the_knobs(tmp_path, tremor_dir):
    a, b = duo(tmp_path, tremor_dir)
    hello(a, b)
    clinic = clinic_for(tmp_path)
    prof = profile_of(a)
    raw, r = sent(clinic, b, prof, tuning=(8, 3))
    ok(r)
    assert raw[:4] == b"DOBS" and len(raw) < 1000
    assert (b.gw.settings.strength, b.gw.settings.tremor) == (8, 3)
    assert profile_stats(b.gw.view) == profile_stats(a.gw.view)
    assert b.gw.status().fill == a.gw.status().fill
    assert b.gw.state_tree()["trusted.count"] == 1


def test_the_card_is_served_and_matches_the_id_everywhere(tmp_path, tremor_dir):
    from dataopen.ctl.identity import Card
    w = world(tmp_path, tremor_dir)
    j = w.phone.connect().json()
    card = Card.from_json(w.phone.get_identity())
    assert card.verify() and card.id == j["device"] == w.gw.state_tree()["device.id"] == w.gw.identity.id
    assert w.gw.read_info()[2:6] == card.digest[:4]


def test_the_device_has_no_way_to_hand_a_file_out(tmp_path, tremor_dir):
    """There is no 'export': asking for the profile or a copy of it is refused as a matter of principle (docs/RESIDENCY.md)."""
    a = world(tmp_path, tremor_dir)
    a.phone.connect()
    for what in ("bundle", "profile", "backup", "export"):
        r = a.phone.call(P.T_GET, {"what": what, "for": "self", "scope": "all"})
        assert err(r) == P.E.RESIDENT
    assert not hasattr(a.gw, "_export_bundle") and not hasattr(a.phone, "get_bundle")


def test_a_file_from_an_unknown_sender_needs_the_button_and_is_then_remembered(tmp_path, tremor_dir):
    a, b = duo(tmp_path, tremor_dir)
    hello(a, b)
    clinic = clinic_for(tmp_path)
    prof = profile_of(a)
    raw, r = sent(clinic, b, prof, press_b=False)
    assert err(r) == P.E.PHYSICAL and r.json()["detail"] == f"trust:{clinic.id}"
    assert b.gw.view is None and b.gw.state_tree()["trusted.count"] == 0       # nothing was applied
    b.gw.physical_press()
    ok(b.phone.put_bundle(raw))
    assert b.gw.view is not None and b.gw.state_tree()["trusted.count"] == 1
    raw2 = clinic.file(b, profile=prof, tuning=(6, 6))
    ok(b.phone.put_bundle(raw2))                                                # a known sender: no button
    assert err(b.phone.put_bundle(raw)) == P.E.REPLAY                           # but an old file is not welcome twice
    assert err(b.phone.put_bundle(raw2)) == P.E.REPLAY


def test_a_refused_file_does_not_burn_the_press(tmp_path, tremor_dir):
    a, b = duo(tmp_path, tremor_dir)
    hello(a, b)
    raw = clinic_for(tmp_path).file(b, profile=profile_of(a), tuning=(5, 5))
    b.gw.physical_press()
    bad = bytearray(raw)
    bad[200] ^= 1
    assert err(b.phone.put_bundle(bytes(bad))) == P.E.BAD_SIGNATURE
    assert b.gw.physical_until > 0
    ok(b.phone.put_bundle(raw))


def test_a_file_made_for_one_device_is_refused_by_every_other(tmp_path, tremor_dir):
    a, b = duo(tmp_path, tremor_dir)
    c = world(tmp_path / "c")
    hello(a, b, c)
    raw, r = sent(clinic_for(tmp_path), b, profile_of(a))
    ok(r)
    c.gw.physical_press()
    assert err(c.phone.put_bundle(raw)) == P.E.WRONG_DEVICE
    a.gw.physical_press()
    assert err(a.phone.put_bundle(raw)) == P.E.WRONG_DEVICE
    assert c.gw.view is None and a.gw.state_tree()["trusted.count"] == 0


def test_a_file_that_says_this_very_device_made_it_is_refused(tmp_path, tremor_dir):
    """The device makes no files, so a file sealed with its own keys can only come from stolen keys (or a build that no longer exists)."""
    from dataopen.ctl import seal as SL
    from dataopen.ctl.identity import Card
    w = world(tmp_path, tremor_dir, trial_s=20)
    w.phone.connect()
    raw = SL.seal(w.gw.identity, Card.from_json(w.phone.get_identity()), w.gw.identity.next_seq(), tuning=(9, 9))
    before = (w.gw.settings, w.gw.view, dict(w.gw.trust["senders"]))
    r = w.phone.put_bundle(raw)
    assert err(r) == P.E.BAD_BUNDLE and r.json()["detail"] == "own_file"
    w.gw.physical_press()
    assert err(w.phone.put_bundle(raw)) == P.E.BAD_BUNDLE                       # the button does not help
    assert (w.gw.settings, w.gw.view, dict(w.gw.trust["senders"])) == before


def test_the_old_open_format_is_refused_unless_the_device_was_told_otherwise(tmp_path, tremor_dir):
    plain = B.Bundle("old", "2026-01-01", 7, 7).pack()
    w = world(tmp_path / "x", tremor_dir)
    w.phone.connect()
    assert err(w.phone.put_bundle(plain)) == P.E.PLAIN_REFUSED
    assert (w.gw.settings.strength, w.gw.settings.tremor) == (5, 5)
    legacy = world(tmp_path / "y", tremor_dir, allow_plain_import=True)
    legacy.phone.connect()
    ok(legacy.phone.put_bundle(plain))
    assert legacy.gw.settings.strength == 7


def test_damaged_or_foreign_files_change_nothing(tmp_path, tremor_dir):
    a, b = duo(tmp_path, tremor_dir)
    hello(a, b)
    raw = clinic_for(tmp_path).file(b, profile=profile_of(a), tuning=(5, 5))
    b.gw.physical_press()
    before = (b.gw.settings, b.gw.view, b.gw.trust["senders"].copy())
    codes = set()
    for i in range(0, len(raw) * 8, 37):
        x = bytearray(raw)
        x[i // 8] ^= 1 << (i % 8)
        codes.add(err(b.phone.put_bundle(bytes(x))))
    for n in (0, 4, 100, len(raw) - 1):
        codes.add(err(b.phone.put_bundle(raw[:n])))
    codes.add(err(b.phone.put_bundle(random.Random(1).randbytes(300))))
    codes.add(err(b.phone.put_bundle(b"DOBS" + bytes(300))))
    assert codes <= {P.E.BAD_BUNDLE, P.E.BAD_SIGNATURE}
    assert (b.gw.settings, b.gw.view, b.gw.trust["senders"]) == before
    assert err(b.phone.put_bundle(bytes(16200))) == P.E.TOO_BIG
    ok(b.phone.put_bundle(raw))                                                 # the press survived all of that


def test_a_stolen_paired_phone_can_not_take_the_profile_out_or_wipe_it(tmp_path, tremor_dir):
    """The attacker holds a bonded phone but not the device: there is nothing to ask for, and the dangerous things need the button."""
    a, b = duo(tmp_path, tremor_dir)
    hello(a, b)
    assert err(a.phone.call(P.T_GET, {"what": "bundle", "for": b.phone.get_identity()})) == P.E.RESIDENT    # profile out: not a thing
    assert err(a.phone.act("erase.profile", True)) == P.E.PHYSICAL              # wipe
    assert err(a.phone.act("factory.reset", True)) == P.E.PHYSICAL
    assert err(a.phone.act("pairing.forget", True)) == P.E.PHYSICAL
    stranger = clinic_for(tmp_path, "stranger")
    raw = stranger.file(a, profile=profile_of(b) or profile_of(a), tuning=(5, 5))   # a file from a stranger for this device ...
    assert err(a.phone.put_bundle(raw)) == P.E.PHYSICAL                         # ... is not applied without the button
    assert a.gw.view is not None and a.gw.identity.id == a.phone.get_identity()["id"]


def test_everything_the_device_stores_is_encrypted(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    clinic = clinic_for(tmp_path)
    w.gw.physical_press()
    ok(ph.put_bundle(clinic.file(w, profile=profile_of(w), tuning=(5, 5))))     # a new profile: the one before it is kept as 'previous' too
    probes = [w.gw.view._state.pack()[8:40], b"assist_wanted", b"strength", b"BIOP", b"senders"]
    files = [p for p in w.dir.rglob("*") if p.is_file() and p.name != "keys.json"]
    assert {p.name.split(".")[0] for p in files} >= {"profile", "settings"}
    for p in files:
        raw = p.read_bytes()
        assert raw[:4] == b"DOVT", p.name
        for pr in probes:
            assert pr not in raw, (p.name, pr)
    assert not [p for p in w.dir.rglob("*.tmp")]


def test_a_device_that_predates_the_vault_seals_its_plain_files_at_start(tmp_path, tremor_dir):
    d = tmp_path / "gw"
    shutil.copytree(tremor_dir, d)                                              # what an older version left: a plain profile
    assert any(p.read_bytes()[:4] == b"BIOP" for p in d.glob("profile.[ab]"))
    (d / "profile.prev").write_bytes((d / "profile.a").read_bytes() if (d / "profile.a").exists() else (d / "profile.b").read_bytes())
    w = World(d)
    assert w.gw.view is not None and w.gw.view.generation >= 1                    # still readable
    for p in d.glob("profile.*"):
        assert p.read_bytes()[:4] == b"DOVT", p.name
    assert all(p.read_bytes()[:4] == b"DOVT" for p in d.glob("settings.*"))


def test_erasing_personal_data_really_erases(tmp_path, tremor_dir):
    from dataopen.ctl.vault import Vault, VaultError
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    ph.set("assist.on", True)
    ph.confirm(True)
    ph.set("assist.strength", 9)
    ph.confirm(True)
    clinic = clinic_for(tmp_path)
    w.gw.physical_press()
    ok(ph.put_bundle(clinic.file(w, profile=profile_of(w), tuning=(9, 5))))
    ident = w.gw.identity.id
    old_key = w.gw.identity.storage_key
    old = {p.name: p.read_bytes() for p in w.dir.iterdir() if p.is_file()}
    assert err(ph.act("erase.profile", True)) == P.E.PHYSICAL
    w.gw.physical_press()
    ok(ph.act("erase.profile", True))
    w.run(500)
    assert state(w) == S_PASSTHRU and w.reason() == "CMD_PASSTHRU"
    assert w.gw.view is None and w.gw.settings.assist_wanted is False and w.gw.settings.strength == 5
    assert ph.get_state()["assist.on"] is False and ph.get_state()["profile.fill"] == 0 and ph.get_state()["trusted.count"] == 0
    assert w.gw.identity.id == ident and w.gw.identity.storage_key != old_key        # the same device, a new storage key
    assert pstore(w).load() is None
    for name, blob in old.items():                                              # the old ciphertext is gone, and unreadable even if kept
        if blob[:4] == b"DOVT":
            with pytest.raises(VaultError):
                Vault(w.gw.identity.storage_key).open(name.rsplit(".", 1)[0] if name.count(".") else name, blob)
    w.make_gateway()                                                            # and it stays erased across a restart
    assert w.gw.view is None and w.gw.settings.assist_wanted is False
    assert (w.gw.identity.id, len(w.gw.trust["senders"])) == (ident, 0)


def test_a_factory_reset_makes_a_new_device_and_kills_every_file_made_for_the_old_one(tmp_path, tremor_dir):
    a, b = duo(tmp_path, tremor_dir)
    hello(a, b)
    old_id = a.gw.identity.id
    clinic = clinic_for(tmp_path)
    raw_for_a = clinic.file(a, profile=profile_of(a), tuning=(5, 5))
    a.gw.physical_press()
    ok(a.phone.put_bundle(raw_for_a))
    a.gw.physical_press()
    ok(a.phone.act("factory.reset", True))
    assert a.gw.identity.id != old_id and a.phone.get_identity()["id"] == a.gw.identity.id
    assert a.gw.read_info()[2:6] != bytes.fromhex("00000000")
    a.gw.physical_press()
    assert err(a.phone.put_bundle(raw_for_a)) == P.E.WRONG_DEVICE
    assert a.gw.view is None


def test_a_manifest_from_a_trusted_sender_is_applied_validated_and_kept(tmp_path, tremor_dir):
    a, b = duo(tmp_path, tremor_dir)
    hello(a, b)
    custom = M.default_manifest(9)
    custom["title"]["en"] = "Clinic layout"
    old_hash = b.gw.manifest.hash
    clinic = clinic_for(tmp_path)
    b.gw.physical_press()
    raw = clinic.file(b, manifest=custom, tuning=(5, 5))
    mark = len(b.phone.inbox)
    ok(b.phone.put_bundle(raw))
    assert b.gw.manifest.rev == 9 and b.gw.manifest.hash != old_hash
    assert any(m.type == P.T_EVENT and m.json().get("manifest") == b.gw.manifest.hash.hex() for m in b.phone.inbox[mark:])
    assert b.phone.get_manifest()["title"]["en"] == "Clinic layout"
    b.make_gateway()
    b.phone.connect()
    assert b.gw.manifest.rev == 9 and b.gw.custom_manifest
    bad = dict(custom, pages=[])
    raw2 = clinic.file(b, manifest=bad, tuning=(5, 5))
    assert err(b.phone.put_bundle(raw2)) == P.E.BAD_BUNDLE
    assert b.gw.manifest.rev == 9
    b.gw.physical_press()
    ok(b.phone.act("erase.profile", True))
    assert b.gw.manifest.rev == 1 and not b.gw.custom_manifest                   # erasing brings the built-in layout back


def test_a_layout_with_a_control_that_would_hand_data_out_is_refused(tmp_path, tremor_dir):
    """A layout (from a sender, or inside a package) can not bring an 'export' button: no file operation of it returns device data."""
    b = world(tmp_path / "b")
    hello(b)
    clinic = clinic_for(tmp_path)
    for op in ("bundle_get", "bundle_for_card", "profile_get", "model_get"):
        evil = M.default_manifest(9)
        evil["pages"][0]["controls"].append({"id": "grab", "type": "file", "op": op, "accept": ".x", "max_bytes": 100,
                                             "label": M.L("Скачать", "Download")})
        b.gw.physical_press()
        assert err(b.phone.put_bundle(clinic.file(b, manifest=evil, tuning=(5, 5)))) == P.E.BAD_BUNDLE
        assert b.gw.manifest.rev == 1
    scoped = M.default_manifest(9)
    scoped["pages"][0]["controls"].append({"id": "grab", "type": "file", "op": "card_get", "scope": "all", "accept": ".x", "max_bytes": 100,
                                           "label": M.L("Карточка", "Card")})
    assert M.validate_manifest(scoped)


def test_the_sealed_file_survives_restarts_of_both_devices(tmp_path, tremor_dir):
    a, b = duo(tmp_path, tremor_dir)
    hello(a, b)
    clinic = clinic_for(tmp_path)
    raw, r = sent(clinic, b, profile_of(a))
    ok(r)
    a.make_gateway()
    b.make_gateway()
    hello(a, b)
    assert b.gw.state_tree()["trusted.count"] == 1 and b.gw.identity.id == b.phone.get_identity()["id"]
    raw2 = clinic.file(b, profile=profile_of(a), tuning=(6, 6))
    ok(b.phone.put_bundle(raw2))                                                # a trusted sender stays trusted, its counter kept
    assert err(b.phone.put_bundle(raw)) == P.E.REPLAY


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
    sl = w.gw.slot                                                                  # a pre-slot device: its profile became slot 0
    assert sl.vault.read(sl.dir / "profile.a", "profile") == bytes(raw)             # sealed (like every file), but its content is untouched


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
