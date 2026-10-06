"""Fail-safe behavior of the assistive HID bridge, in simulated time: panic, long press, watchdog, power loss, mouse removal,
unsupported devices, stale module, hostile link, invariant/overrun trouble, USB errors, crash loops, and a random walk over
events checking that no latch can be lifted by anything but a deliberate action."""

import functools
import random
import shutil

import pytest

from dataopen.assist.fixed import FixedParams
from dataopen.assist.params import AscParams
from dataopen.assist.sim_user import PERSONAS, build_profile
from dataopen.assist.tremor import TremorParams
from dataopen.assist.tremor_fixed import FixedTremorParams
from dataopen.bridge import protocol as P
from dataopen.bridge.cbridge import HW_BYPASS, HW_ENGAGED, RESET_WATCHDOG, S_ASSIST, STATE, REASONS
from dataopen.bridge.sim import Rig
from dataopen.bridge.sim_usb import SimMouse

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


@functools.lru_cache(maxsize=None)
def _params():
    v = build_profile(PERSONAS["overshooter"], seed=1)
    return FixedParams.from_params(AscParams.from_view(v)), FixedTremorParams.from_params(TremorParams.from_view(v))


def rig(kind="m16", **kw):
    asc, trm = _params()
    r = Rig(SimMouse(kind), asc=asc, tremor=trm, **kw)
    assert r.engage_fast(), (STATE[r.status().state], REASONS[r.status().reason])
    return r


def state(r):
    s = r.status()
    return STATE[s.state], REASONS[s.reason]


def feed(r, n=50, dx=3, dy=1, buttons=0):
    for _ in range(n):
        r.move(dx, dy, buttons)
        r.step()


def delivered(r, since):
    return [x for x in r.pc_reports if x[0] >= since]


def test_it_starts_on_the_hardware_bypass_and_engages_only_after_checks():
    asc, trm = _params()
    r = Rig(SimMouse("m16"), asc=asc, tremor=trm)
    assert state(r)[0] == "HW_BYPASS" and r.route == "bypass" and r.hw.engage(0) is False
    r.run(1500)
    assert r.route == "bypass"  # the mouse must settle first (engage_hold)
    assert r.engage_fast()
    assert state(r) == ("ASSIST", "NONE") and r.route == "bridge"
    kinds = [e[1] for e in r.events]
    assert "route bypass -> none" in kinds and "route none -> bridge" in kinds  # the PC saw the bridge only after the image was built


def test_without_a_module_it_passes_through_and_waits_for_parameters():
    r = Rig(SimMouse("m16"), module=False)
    r.run(4000)
    assert state(r) == ("PASSTHRU", "STALE_LINK") and r.route == "bridge"
    feed(r, 20)
    assert all(rep[3] == rep[4] for rep in r.pc_reports if rep[1] == "bridge")


def test_panic_press_is_instant_and_leaves_no_gap():
    r = rig()
    feed(r, 100)
    n_before = len(r.pc_reports)
    r.panic(True)
    assert state(r) == ("PASSTHRU", "PANIC")  # in the very call that saw the press
    assert r.route == "bridge" and r.b.hw_select == HW_ENGAGED  # still on the bridge: nothing re-enumerated
    feed(r, 50)
    new = r.pc_reports[n_before:]
    assert len(new) == 50 and all(x[1] == "bridge" and x[3] == x[4] for x in new)  # output == input bit for bit, no report lost
    assert r.dropped == 0


def test_panic_latch_is_not_lifted_by_release_or_by_the_module():
    r = rig()
    r.panic(True)
    r.run(100)
    r.panic(False)
    r.run(100)
    assert state(r) == ("PASSTHRU", "PANIC")
    r.module.send(r.t, [P.cmd_frame(P.CMD_ASSIST)])  # the module asks for help again: it may not override
    r.run(6000)
    assert state(r) == ("PASSTHRU", "PANIC")
    r.panic(True)
    r.run(1500)  # a short deliberate press is not enough ...
    r.panic(False)
    assert state(r)[0] == "PASSTHRU"
    r.panic(True)
    r.run(2100)  # ... a hold of 2 s is
    r.panic(False)
    r.run(50)
    assert state(r) == ("ASSIST", "NONE")


def test_long_press_goes_to_the_hardware_bypass_and_the_pc_gets_the_real_mouse():
    r = rig()
    feed(r, 20)
    r.panic(True)
    r.run(3100)
    assert r.b.hw_select == HW_BYPASS and r.route == "bypass"
    assert state(r)[0] == "HW_BYPASS" and REASONS[r.status().reason] == "PANIC_LONG"
    r.panic(False)
    r.run(1000)
    assert state(r)[0] == "HW_BYPASS"  # latched: the release does not re-engage
    t0 = r.t
    feed(r, 40)
    assert r.pc_conn == "direct" and all(x[1] == "direct" and x[3] == x[4] for x in delivered(r, t0))
    r.panic(True)
    r.run(2100)
    r.panic(False)
    assert r.engage_fast() and state(r) == ("ASSIST", "NONE")  # a deliberate hold brings the help back


def test_chord_on_the_mouse_buttons_toggles_the_soft_latch():
    r = rig()
    t0 = r.t
    while r.t < t0 + 3200_000:
        r.move(0, 0, buttons=0x18)
        r.step()
    assert state(r) == ("PASSTHRU", "CHORD")
    r.move(0, 0, 0)
    r.step()
    t1 = r.t
    while r.t < t1 + 3200_000:
        r.move(0, 0, buttons=0x18)
        r.step()
    assert state(r) == ("ASSIST", "NONE")
    assert r.b.hw_select == HW_ENGAGED  # the chord never touched the switch


def test_a_hung_firmware_is_caught_by_the_watchdog_not_by_itself():
    r = rig()
    feed(r, 20)
    t_hang = r.t
    r.kill_firmware()
    for _ in range(250):  # the person keeps moving while the firmware is dead
        r.move(2, 1)
        r.step()
    assert r.route == "bypass"
    gap = [e for e in r.events if e[1] == "route bridge -> bypass"][0][0] - t_hang
    assert gap <= 101_000  # the supervisor's 100 ms (+ one step)
    assert 0 < r.dropped <= 110 + r.reenum_us // 1000  # reports are lost only for the watchdog window plus the PC's re-enumeration
    r.run(r.reenum_us // 1000 + 50)
    t0 = r.t
    feed(r, 30)
    assert r.pc_conn == "direct" and len(delivered(r, t0)) == 30


def test_power_loss_is_the_bypass():
    r = rig()
    r.power_off()
    r.run(1)
    assert r.route == "bypass"


def test_mechanical_switch_and_a_cut_panic_wire_both_mean_bypass():
    r = rig()
    r.hw.switch_on = False
    r.run(5)
    assert r.route == "bypass"
    r.hw.switch_on = True
    assert r.engage_fast(30000)
    r.hw.panic_wire_cut = True
    r.run(5)
    assert r.route == "bypass"


def test_the_panic_loop_rc_trips_the_hardware_even_if_the_firmware_ignores_the_button():
    r = rig()
    r.alive = True
    r.hw.panic_since = r.t  # the button is held but the firmware never sees it
    r.run(3050)
    assert r.route == "bypass"


def test_unplugging_and_replugging_a_different_mouse_clones_the_new_one():
    r = rig("logi")
    first = r.transcripts["bridge"][1]  # the device descriptor the PC got
    r.unplug()
    r.run(100)
    assert r.route in ("bypass", "none") and state(r)[0] == "HW_BYPASS"
    r.plug(SimMouse("boot", vid=0x1234, pid=0xABCD))
    assert r.engage_fast(30000)
    second = r.transcripts["bridge"][1]
    assert first[-1] != second[-1]
    assert second[-1][8:12] == bytes([0x34, 0x12, 0xCD, 0xAB])  # idVendor / idProduct of the new mouse, byte for byte


@pytest.mark.parametrize(
    "kw,code", [(dict(bulk=True), "UNSUPPORTED_EP"), (dict(n_conf=2), "MULTI_CONFIG"), (dict(alt=True), "ALT_SETTING")]
)
def test_devices_the_proxy_can_not_mirror_exactly_are_never_engaged(kw, code):
    from dataopen.bridge.cbridge import IMG

    r = Rig(SimMouse("m16", **kw), asc=_params()[0], tremor=_params()[1])
    r.run(8000)
    s = r.status()
    assert REASONS[s.reason] == "IMAGE" and IMG[s.img_code] == code and r.route == "bypass"
    r.unplug()
    r.plug(SimMouse("m16"))
    assert r.engage_fast(30000)  # a different device clears the device-specific latch


@pytest.mark.parametrize("kind", ["abs", "vendor"])
def test_pointing_devices_without_relative_xy_are_left_alone(kind):
    r = Rig(SimMouse(kind), asc=_params()[0], tremor=_params()[1])
    r.run(8000)
    assert REASONS[r.status().reason] == "IMAGE" and r.route == "bypass"


def test_mouse_only_scope_refuses_composite_inputs_transparent_scope_passes_them():
    from dataopen.bridge.cbridge import IMG, SCOPE_MOUSE_ONLY

    strict = Rig(SimMouse("logi", kbd=True), cfg={"scope": SCOPE_MOUSE_ONLY}, asc=_params()[0], tremor=_params()[1])
    strict.run(8000)
    assert IMG[strict.status().img_code] == "NOT_MOUSE_ONLY" and strict.route == "bypass"
    loose = Rig(SimMouse("logi", kbd=True, hidpp=True), asc=_params()[0], tremor=_params()[1])
    assert loose.engage_fast()
    loose.mouse.push_in(0x83, b"\x00\x00\x04\x00\x00\x00\x00\x00")  # a key press on the keyboard interface of the composite device
    loose.step()
    kb = [x for x in loose.pc_reports if x[2] == 0x83]
    assert kb and kb[0][3] == kb[0][4] == b"\x00\x00\x04\x00\x00\x00\x00\x00"


def test_stale_parameters_or_a_silent_module_drop_to_passthrough_and_recover_by_themselves():
    r = rig()
    r.module.silent = True
    r.run(700)
    assert state(r) == ("PASSTHRU", "STALE_LINK")  # the module went quiet for more than 500 ms
    r.module.silent = False
    r.module.next_params = 0
    r.run(2000)
    assert state(r) == ("ASSIST", "NONE")
    # the link is alive (keepalives) but the parameters stop being refreshed
    r.module.asc = r.module.tremor = None
    r.run(5500)
    assert state(r) == ("PASSTHRU", "STALE_PARAMS")


def test_a_corrupting_lossy_link_never_breaks_the_output_invariants():
    r = rig()
    r.module.corrupt = 0.3
    r.module.loss = 0.2
    rng = random.Random(5)
    bad = 0
    for _ in range(4000):
        r.move(rng.randint(-40, 40), rng.randint(-40, 40))
        r.step()
    for _, via, _, out, raw in r.pc_reports:
        if via != "bridge":
            continue
        ox, oy = int.from_bytes(out[1:3], "little", signed=True), int.from_bytes(out[3:5], "little", signed=True)
        rx, ry = int.from_bytes(raw[1:3], "little", signed=True), int.from_bytes(raw[3:5], "little", signed=True)
        for o, i in ((ox, rx), (oy, ry)):
            if abs(o) > abs(i) or (o != 0 and (o > 0) != (i > 0)) or (i == 0 and o != 0):
                bad += 1
    assert bad == 0
    s = r.status()
    assert s.link_rx_bad > 0 and s.invariant_viol == 0


def test_an_invariant_violation_is_clamped_counted_and_drops_to_passthrough_for_a_while():
    r = rig(testing=True)
    for _ in range(3):
        r.b.inject(1)
        r.move(10, 10)
        r.step()
        r.move(0, 0)
        r.step()
    s = r.status()
    assert s.invariant_viol == 3
    assert state(r) == ("PASSTHRU", "INVARIANT")
    assert all(
        abs(int.from_bytes(x[3][1:3], "little", signed=True)) <= abs(int.from_bytes(x[4][1:3], "little", signed=True))
        for x in r.pc_reports
        if x[1] == "bridge"
    )  # the doubled output never reached the PC
    r.run(5200)
    assert state(r) == ("ASSIST", "NONE")


def test_overrunning_the_time_budget_drops_to_passthrough():
    r = rig()
    for _ in range(3):
        r.b.note_cost(r.t, 400)
    assert state(r) == ("PASSTHRU", "OVERRUN")
    r.run(5200)
    assert state(r) == ("ASSIST", "NONE")
    r.b.note_cost(r.t, 400)
    r.b.note_cost(r.t, 10)  # a streak is needed, a single slow call is not enough
    r.b.note_cost(r.t, 400)
    assert state(r) == ("ASSIST", "NONE")


def test_usb_errors_leave_the_bridge_but_do_not_latch_a_working_device_forever():
    r = rig()
    for _ in range(8):
        r.b.usb_error(r.t)
    r.run(2)
    assert state(r)[0] == "HW_BYPASS" and REASONS[r.status().reason] == "USB_ERRORS" and r.route == "bypass"
    assert r.engage_fast(30000)  # one burst is retried automatically
    for _ in range(3):
        for _ in range(8):
            r.b.usb_error(r.t)
        r.run(2500)
    assert REASONS[r.status().reason] == "ENGAGE_FAILED"  # a device that keeps failing is given up on
    assert r.route == "bypass"


def test_a_host_that_never_configures_the_device_is_given_up_on():
    r = Rig(SimMouse("m16"), asc=_params()[0], tremor=_params()[1])
    r.pc_never_configures = True
    r.run(30000)
    assert state(r)[0] == "HW_BYPASS" and REASONS[r.status().reason] == "ENGAGE_FAILED" and r.route == "bypass"
    assert r.status().attach == 0


def test_a_crash_loop_ends_in_the_bypass_until_a_service_action():
    r = rig()
    for _ in range(3):
        r.reboot_firmware(RESET_WATCHDOG)
        r.run(300)
    assert state(r)[0] == "HW_BYPASS" and REASONS[r.status().reason] == "CRASHLOOP" and r.nv_crashes >= 3
    r.run(10000)
    assert state(r)[0] == "HW_BYPASS"  # it does not try again by itself
    r.panic(True)
    r.run(2100)
    assert state(r)[0] == "HW_BYPASS"  # the ordinary 2 s re-arm is not enough for a crash loop
    r.run(3000)
    r.panic(False)
    assert r.engage_fast(30000)


def test_a_clean_minute_clears_the_crash_counter():
    r = rig()
    r.reboot_firmware(RESET_WATCHDOG)
    assert r.nv_crashes == 1
    assert r.engage_fast(30000)
    r.run(61000)
    assert r.b.status(r.t).crashes == 0 and r.nv_crashes == 0


def test_the_hardware_never_engages_without_a_healthy_firmware():
    asc, trm = _params()
    r = Rig(SimMouse("m16"), asc=asc, tremor=trm)
    r.alive = False  # firmware never starts
    r.run(5000)
    assert r.route == "bypass" and r.hw.last_kick is None


def test_the_probe_phase_is_aborted_if_the_mouse_does_not_answer():
    asc, trm = _params()
    m = SimMouse("m16")
    m.fail_control = True
    r = Rig(m, asc=asc, tremor=trm)
    r.run(12000)
    assert r.route == "bypass"
    assert REASONS[r.status().reason] in ("IMAGE_TIMEOUT", "ENGAGE_FAILED")


def test_random_event_walks_never_let_help_run_under_a_latch():
    """Panic presses and releases, chords, module commands (including the forbidden raise), faults and plain time, in random order:
    whenever a latch is set the state is never ASSIST, a hard latch means HW_BYPASS, and the output invariants hold throughout."""
    for seed in range(12):
        rng = random.Random(seed)
        r = rig(testing=True)
        down = False
        for step in range(400):
            a = rng.random()
            if a < 0.10:
                down = not down
                r.panic(down)
            elif a < 0.15:
                r.module.send(r.t, [P.cmd_frame(rng.choice([0, 1, 1, 1]))])
            elif a < 0.17:
                r.b.inject(1)
            elif a < 0.19:
                r.b.note_cost(r.t, rng.choice([10, 500]))
            elif a < 0.21:
                r.b.usb_error(r.t)
            r.move(rng.randint(-5, 5), rng.randint(-5, 5), rng.choice([0, 0, 0x18]))
            r.run(rng.choice([1, 1, 5, 40, 700]))
            s = r.status()
            if s.latch_hw:
                assert STATE[s.state] == "HW_BYPASS" or r.b.hw_select == HW_BYPASS, (seed, step)
            if s.latch_soft:
                assert s.state != S_ASSIST, (seed, step, REASONS[s.reason])
            if s.state == S_ASSIST:
                assert r.b.hw_select == HW_ENGAGED and s.attach
        for _, via, _, out, raw in r.pc_reports:
            if via == "bridge":
                for a_, b_ in ((1, 3), (3, 5)):
                    o, i = int.from_bytes(out[a_:b_], "little", signed=True), int.from_bytes(raw[a_:b_], "little", signed=True)
                    assert abs(o) <= abs(i) and (o == 0 or (o > 0) == (i > 0)) and (i != 0 or o == 0)
