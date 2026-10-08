"""The status LED, the quiet buzzer and the mapping from the bridge's reason to what the person sees."""
import pytest

from dataopen.bridge.cbridge import REASONS
from dataopen.ctl import protocol as P
from dataopen.hw import indication as I


def snap(reason="NONE", mode=3, flags=P.SF_BRIDGE_SEEN, ready=P.RB_ASC, bridge=1):
    return P.StatusSnapshot(bridge, mode, REASONS.index(reason), flags, 50, ready)


def test_every_bridge_reason_has_a_state_and_every_state_exists():
    assert I.reasons_covered()
    assert set(I.REASON_STATE.values()) <= set(I.BY_KEY)
    assert set(I.PRIORITY) == set(I.BY_KEY)


def test_states_differ_by_shape_not_by_colour_alone():
    lit = [s for s in I.STATES if s.color]
    sigs = {(s.pattern, s.breathe) for s in lit}
    assert len(sigs) == len(lit)                                                     # no two lit states share a blinking pattern
    # states of one colour are all distinguishable by pattern alone
    for color in {s.color for s in lit}:
        same = [s for s in lit if s.color == color]
        assert len({(s.pattern, s.breathe) for s in same}) == len(same)
    off = I.BY_KEY["OFF"]
    assert off.color == "" and all(on == 0 for on, _ in off.pattern)                 # no power: dark


def test_the_buzzer_is_quiet_short_and_never_continuous():
    for notes in I.BEEPS.values():
        assert sum(ms for _, ms in notes) <= I.MAX_BEEP_MS
        assert all(0 <= f <= 4000 for f, _ in notes)
    quiet = {"OFF", "STARTING", "READY", "ACTIVE", "NO_MOUSE", "DEGRADED", "CALIBRATING", "TRIAL"}
    assert all(I.BY_KEY[k].beep is None for k in quiet)                              # ordinary operation is silent
    assert {s.beep for s in I.STATES if s.beep} <= set(I.BEEPS)
    assert I.BY_KEY["ERROR"].beep == "error" and I.BY_KEY["BYPASS"].beep == "bypass"


def test_no_power_is_off_and_nothing_else_overrides_it():
    assert I.indicate(snap(), powered=False, pairing=True, upload=True) == "OFF"
    assert I.indicate(None, powered=True) == "STARTING"
    # a bridge nobody has heard from is 'starting', not guessed
    assert I.indicate(P.StatusSnapshot(), powered=True) == "STARTING"


def test_the_ordinary_states():
    assert I.indicate(snap("NONE", 3)) == "ACTIVE"
    assert I.indicate(snap("NONE", 3, ready=0)) == "READY"                           # help is on, nothing to correct yet
    assert I.indicate(snap("CMD_PASSTHRU", 2)) == "READY"
    assert I.indicate(snap("PANIC", 2)) == "READY"                                   # the soft latch: help off by the person's own hand
    assert I.indicate(snap("PROBING", 1)) == "STARTING"
    assert I.indicate(snap("NO_DEVICE", 0)) == "NO_MOUSE"
    assert I.indicate(snap("CMD_BYPASS", 0)) == "BYPASS" and I.indicate(snap("PANIC_LONG", 0)) == "BYPASS"
    assert I.indicate(snap("STALE_LINK", 2)) == "DEGRADED"


def test_priorities():
    cal = snap("NONE", 3, flags=P.SF_BRIDGE_SEEN | P.SF_CALIBRATING)
    assert I.indicate(cal) == "CALIBRATING"
    assert I.indicate(cal, pairing=True) == "PAIRING"                                # pairing outranks calibration
    assert I.indicate(cal, pairing=True, upload=True) == "OTA"                       # an update outranks both
    assert I.indicate(snap("NONE", 3, flags=P.SF_BRIDGE_SEEN | P.SF_TRIAL)) == "TRIAL"
    assert I.indicate(snap("NONE", 3), fw_state="trial") == "OTA"
    # being in bypass is what matters, not the phone's business
    assert I.indicate(snap("CMD_BYPASS", 0), pairing=True, upload=True) == "BYPASS"
    for bad in ("USB_ERRORS", "CRASHLOOP", "FATAL", "ENGAGE_FAILED", "INVARIANT", "OVERRUN"):
        assert I.indicate(snap(bad, 0), pairing=True, upload=True) == "ERROR"        # a fault outranks everything that is powered
    assert I.indicate(snap(), both_mice=True) == "ERROR"


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_every_reason_in_every_mode_gives_a_known_state(reason, mode):
    assert I.indicate(snap(reason, mode)) in I.BY_KEY


def test_hold_priority_list_is_a_permutation_with_off_first_and_error_next():
    assert I.PRIORITY[:3] == ("OFF", "ERROR", "BYPASS") and len(set(I.PRIORITY)) == len(I.PRIORITY)
