"""The front panel: what a press and a hold mean (the model, then the gateway behind it on the real bridge core)."""
import shutil

import pytest

from dataopen.bioprofile.profile import ProfileState
from dataopen.ctl import panel as PN
from dataopen.ctl import protocol as P
from dataopen.ctl.sim import SimLearner, World, seed_profile

from test_ctl_gateway import ok  # noqa: F401

S = 1_000_000
needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


# ---------------------------------------------------------------------------------------------------------------- the model
def test_a_contact_bounce_is_not_a_press():
    p = PN.Panel()
    p.press("slot", 0)
    assert p.release("slot", 10_000) is None
    p.press("confirm", 0)
    assert p.release("confirm", 29_999) is None


def test_the_slot_button_is_short_press_only():
    p = PN.Panel()
    p.press("slot", 0)
    assert p.release("slot", 200_000) == PN.SLOT_NEXT
    p.press("slot", 0)
    assert p.release("slot", 3 * S) is None                                   # a long hold on it is nothing


def test_the_confirm_button_by_how_long_it_was_held():
    cases = [(0.1, PN.CONFIRM_SHORT), (1.9, PN.CONFIRM_SHORT), (2.5, None), (9.9, None), (10.0, PN.ERASE), (19.9, PN.ERASE),
             (20.0, PN.FACTORY), (60.0, PN.FACTORY)]
    for secs, want in cases:
        p = PN.Panel()
        p.press("confirm", 5 * S)
        assert p.release("confirm", 5 * S + int(secs * S)) == want, secs


def test_a_hold_warns_before_it_acts_and_a_release_in_between_cancels():
    p = PN.Panel()
    p.press("confirm", 0)
    got = [p.hold_warning(t * S) for t in (1, 9.9, 10, 15, 19.9, 20, 30)]
    assert got == ["", "", PN.M_ERASE, PN.M_ERASE, PN.M_ERASE, PN.M_FACTORY, PN.M_FACTORY]
    assert p.release("confirm", 12 * S) == PN.ERASE
    assert p.hold_warning(13 * S) == "" and p.release("confirm", 14 * S) is None            # released twice: nothing happens twice


def test_press_twice_without_a_release_is_one_press():
    p = PN.Panel()
    p.press("confirm", 0)
    p.press("confirm", 5 * S)
    assert p.release("confirm", 11 * S) == PN.ERASE                                          # counted from the FIRST edge
    with pytest.raises(ValueError):
        p.press("reset", 0)


def test_every_led_mode_looks_different_and_says_which_slot():
    sample = {m: [PN.leds(m, 2, t * 20_000) for t in range(200)] for m in (PN.M_STEADY, PN.M_TRIAL, PN.M_WINDOW, PN.M_ERASE, PN.M_FACTORY,
                                                                             PN.M_ERROR, PN.M_CALIB)}
    assert set(sample[PN.M_STEADY]) == {(False, False, True, False)}                         # steady: exactly the active one, always
    for m in (PN.M_TRIAL, PN.M_CALIB):
        assert set(sample[m]) == {(False, False, True, False), (False, False, False, False)}  # blinks, and only the active one
    for m in (PN.M_WINDOW, PN.M_ERASE, PN.M_FACTORY, PN.M_ERROR):
        assert set(sample[m]) == {(True,) * 4, (False,) * 4}                                   # all of them together
    rates = {m: sum(a != b for a, b in zip(v, v[1:])) for m, v in sample.items()}
    # faster and faster towards the point of no return
    assert rates[PN.M_FACTORY] > rates[PN.M_ERASE] > rates[PN.M_WINDOW]
    assert rates[PN.M_TRIAL] > rates[PN.M_CALIB]


# ---------------------------------------------------------------------------------------------------------------- the device
@pytest.fixture(scope="module")
def tremor_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("panel-tremor")
    seed_profile(d, "tremor")
    return d


@pytest.fixture(scope="module")
def other_state(tmp_path_factory):
    st = seed_profile(tmp_path_factory.mktemp("panel-other"), "overshooter")._state
    st.profile_id = 0x0B0B
    return st


def world(tmp_path, tremor_dir=None, **kw) -> World:
    d = tmp_path / "gw"
    if tremor_dir is not None:
        shutil.copytree(tremor_dir, d)
    return World(d, **kw)


def put(w, st, k):
    w.gw._install_profile(ProfileState.unpack(st.pack()), w.t, k)


def lit(w):
    return w.gw.leds(w.t)


@needs_cc
def test_the_slot_button_goes_round_the_slots_that_are_in_use(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    put(w, other_state, 2)
    seen = []
    for _ in range(5):
        w.press("slot")
        seen.append(w.gw.active)
    # two contexts: it toggles between exactly those
    assert seen == [2, 0, 2, 0, 2]
    put(w, other_state, 3)
    seen = []
    for _ in range(4):
        w.press("slot")
        seen.append(w.gw.active)
    assert seen == [3, 0, 2, 3]                                                              # three contexts: round those three


@needs_cc
def test_with_one_slot_in_use_the_button_walks_all_four_so_a_new_one_can_be_reached(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    seen = []
    for _ in range(5):
        w.press("slot")
        seen.append(w.gw.active)
    assert seen == [1, 2, 3, 0, 1]
    assert w.gw.settings.assist_wanted is False                                              # and none of it turned assistance on


@needs_cc
def test_the_led_follows_the_active_slot_and_the_phone_and_the_button_agree(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    put(w, other_state, 1)
    w.phone.connect()
    assert lit(w) == (True, False, False, False)
    w.press("slot")
    assert lit(w) == (False, True, False, False) and w.gw.status().slot == 1
    assert w.gw.state_tree()["slot.active"] == 1
    ok(w.phone.select_slot(0))
    assert lit(w) == (True, False, False, False)
    assert w.phone.state["slot.active"] == 0                                                 # the phone heard about the press too


@needs_cc
def test_confirm_short_press_opens_the_window_and_the_leds_say_so(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    assert w.gw.led_mode(w.t) == PN.M_STEADY
    w.press("confirm")
    assert w.gw.led_mode(w.t) == PN.M_WINDOW and w.gw.physical_until > w.t
    w.phone.connect()
    w.gw.physical_press(w.t)
    ok(w.phone.act("erase.profile", True))
    assert w.gw.led_mode(w.t) == PN.M_STEADY                                                 # one press, one use: the window closed with it


@needs_cc
def test_confirm_short_press_keeps_a_running_trial_and_does_not_open_the_window(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    w.phone.connect()
    ok(w.phone.set("assist.on", True))
    assert w.gw.trial is not None and w.gw.led_mode(w.t) == PN.M_TRIAL
    w.press("confirm")
    assert w.gw.trial is None and w.gw.settings.assist_wanted and w.gw.physical_until < w.t
    assert w.gw.led_mode(w.t) == PN.M_STEADY and w.gw.slot.vetted


@needs_cc
def test_the_slot_button_says_no_with_the_leds_when_it_cannot(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir, learner=SimLearner(minutes=5.0, seed=2, speed=240.0))
    put(w, other_state, 1)
    w.phone.connect()
    ok(w.phone.set("calib.running", True))
    assert w.gw.led_mode(w.t) == PN.M_CALIB
    w.press("slot")
    assert w.gw.active == 0 and w.gw.led_mode(w.t) == PN.M_ERROR
    w.run(700)
    assert w.gw.led_mode(w.t) == PN.M_CALIB


@needs_cc
def test_holding_confirm_ten_seconds_erases_all_personal_data_but_not_before(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    put(w, other_state, 2)
    w.phone.connect()
    ident = w.gw.identity.id
    w.gw.button("confirm", True, w.t)
    w.run(9000)
    assert w.gw.led_mode(w.t) == PN.M_STEADY
    w.gw.button("confirm", False, w.t)                                                       # let go early: nothing happened
    assert w.gw.slotset.mask() == 0b101 and w.gw.led_mode(w.t) == PN.M_STEADY
    w.gw.button("confirm", True, w.t)
    w.run(10_500)
    assert w.gw.led_mode(w.t) == PN.M_ERASE                                                  # the warning: 'let go now to erase'
    assert w.gw.slotset.mask() == 0b101                                                      # still nothing erased while the button is down
    w.gw.button("confirm", False, w.t)
    assert w.gw.slotset.mask() == 0 and w.gw.identity.id == ident and not w.gw.settings.assist_wanted
    assert w.bridge().state == 2                                                             # and the bridge was told: pass-through


@needs_cc
def test_holding_confirm_twenty_seconds_is_a_factory_reset(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    ident, epoch0 = w.gw.identity.id, w.gw.identity.created
    w.gw.button("confirm", True, w.t)
    w.run(20_500)
    assert w.gw.led_mode(w.t) == PN.M_FACTORY
    w.gw.button("confirm", False, w.t)
    assert w.gw.identity.id != ident and w.gw.slotset.mask() == 0 and epoch0 is not None
    assert w.gw.led_mode(w.t) == PN.M_STEADY


@needs_cc
def test_a_hold_stopped_between_the_thresholds_does_nothing(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.gw.button("confirm", True, w.t)
    w.run(5000)
    w.gw.button("confirm", False, w.t)
    assert w.gw.slotset.mask() == 1 and w.gw.physical_until < w.t                            # not a press either: the window stayed shut
    assert P.StatusSnapshot.unpack(w.gw.read_status()).slot == 0
