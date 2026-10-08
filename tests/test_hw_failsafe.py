"""The fail-safe net: exhaustive over its inputs, and cross-checked with the bridge simulator's hardware model."""
import pytest

from dataopen.bridge.sim import HwModel
from dataopen.hw import failsafe as F
from dataopen.hw import spec as S


def test_the_whole_truth_table_obeys_the_rules():
    seen = 0
    for i in F.all_inputs():
        o = F.evaluate(i)
        seen += 1
        # no power, hung watchdog, open Panic loop or MODE on BYPASS: nothing is engaged, whatever else is true
        if not (i.vext_ok and i.wdg_ok and i.panic_loop_ok and i.mode_assist):
            assert not o.mouse_engaged and not o.video_engaged
        # the processors can only request
        if not i.mcu_engage:
            assert not o.mouse_engaged
        if not (i.video_en and i.video_pwr_ok):
            assert not o.video_engaged
        # and when everything is fine the requests are granted
        if i.vext_ok and i.wdg_ok and i.panic_loop_ok and i.mode_assist:
            assert o.mouse_engaged == i.mcu_engage and o.video_engaged == (i.video_en and i.video_pwr_ok)
        # the amber LED needs no firmware: it is lit exactly when powered and the mouse path is direct
        assert o.bypass_led == (i.vext_ok and not o.mouse_engaged)
    assert seen == 2 ** 7


def test_without_external_power_everything_is_direct_for_every_other_input():
    for i in F.all_inputs():
        if not i.vext_ok:
            o = F.evaluate(i)
            # direct, and the LED is dark with the rest
            assert (o.mouse_engaged, o.video_engaged, o.bypass_led) == (False, False, False)


def test_a_dead_processor_cannot_hold_a_path_engaged():
    for i in F.all_inputs():
        # the watchdog is not fed: the MCU (or its firmware) is dead
        if not i.wdg_ok:
            assert not F.evaluate(i).mouse_engaged and not F.evaluate(i).video_engaged


def test_video_defaults_to_direct():
    o = F.evaluate(F.Inputs(mcu_engage=True))
    # the detector is not wanted: the monitor stays wired straight
    assert o.mouse_engaged and not o.video_engaged


def test_the_spec_gives_the_coils_no_path_around_the_gate():
    src = F.coil_sources()
    assert src["COIL_DRV"] == {"GATE"}                                               # only the gate drives the coil drivers
    for coil in F.COILS:
        assert src[coil] == {"COIL_DRV"}
    for blk in F.firmware_blocks():                                                  # firmware blocks have no link to a coil or its driver
        assert blk not in src["COIL_DRV"] and all(blk not in src[c] for c in F.COILS)
    gate_inputs = {a for a, b, _ in S.LINKS if b == "GATE"}
    # the MCU is the only firmware input, and only as a request
    assert gate_inputs & F.firmware_blocks() == {"MCU"}
    assert {a for a, b, _ in S.LINKS if b == "GATE" and a == "MCU"} == {"MCU"}
    for a, b, what in S.LINKS:
        if a == "MCU" and b == "GATE":
            assert "request" in what
    assert S.BLOCKS["GATE"] == "D0" and S.BLOCKS["COIL_DRV"] == "D0" and all(S.BLOCKS[c] == "D0" for c in F.COILS)


@pytest.mark.parametrize("what", ["power", "panic_hold", "wire_cut", "mode", "hang"])
def test_the_scenario_times_match_the_bridge_simulators_hardware_model(what):
    """Same events on `HwModel` (microseconds): the moment ENGAGE drops must be the scenario's time minus the relay release."""
    hw = HwModel(wdg_ms=F.WDG_MS, rc_ms=F.PANIC_RC_MS)
    hw.mcu_engage = True
    hw.kick(0)
    key = {"power": "no_power", "panic_hold": "panic_long", "wire_cut": "cable_cut", "mode": "mode_switch", "hang": "mcu_hang"}[what]
    t0 = 1_000_000
    hw.kick(t0)
    if what == "power":
        hw.powered = False
    elif what == "panic_hold":
        hw.panic_since = t0
    elif what == "wire_cut":
        hw.panic_wire_cut = True
    elif what == "mode":
        hw.switch_on = False
    t = t0
    while hw.engage(t) and t < t0 + 10_000_000:
        t += 1000                                                                    # 1 ms steps
        if what in ("power", "panic_hold", "wire_cut", "mode") and t % 10_000 == 0:
            # a healthy firmware keeps kicking; only a hang stops it
            hw.kick(t)
    dropped_ms = (t - t0) // 1000
    expected = F.scenario(key).to_bypass_ms - F.RELEASE_MS
    # HwModel opens a cut wire at once; the spec says the RC of the loop applies
    if what == "wire_cut":
        assert dropped_ms <= expected
    else:
        assert abs(dropped_ms - expected) <= 2, (what, dropped_ms, expected)


def test_the_logic_agrees_with_the_bridge_model_on_every_combination():
    for powered in (False, True):
        for sw in (False, True):
            for eng in (False, True):
                for panic in (False, True):
                    for fed in (False, True):
                        hw = HwModel()
                        hw.powered, hw.switch_on, hw.mcu_engage, hw.panic_wire_cut = powered, sw, eng, panic
                        if fed:
                            hw.kick(0)
                        i = F.Inputs(vext_ok=powered, wdg_ok=fed, panic_loop_ok=not panic, mode_assist=sw, mcu_engage=eng)
                        mine = F.evaluate(i).mouse_engaged
                        assert mine == hw.engage(0), (powered, sw, eng, panic, fed)


def test_scenarios_are_honest_about_firmware_and_about_the_cost():
    by = {s.key: s for s in F.SCENARIOS}
    for k in ("no_power", "panic_long", "mcu_hang", "mode_switch", "cable_cut"):
        assert by[k].needs_firmware is False and by[k].to_bypass_ms is not None
    # the soft latch is firmware: that is why the long press exists
    assert by["panic_short"].needs_firmware and by["panic_short"].to_bypass_ms is None
    assert by["som_hang"].needs_firmware and by["som_hang"].to_bypass_ms is None
    assert by["no_power"].to_bypass_ms < by["mcu_hang"].to_bypass_ms < by["panic_long"].to_bypass_ms
    assert by["mcu_hang"].to_bypass_ms <= F.WDG_MS + F.RELEASE_MS
    assert F.REENUM_MS[0] < F.REENUM_MS[1] and F.MONITOR_RELOCK_MS[0] < F.MONITOR_RELOCK_MS[1]
