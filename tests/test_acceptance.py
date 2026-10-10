"""The acceptance scenarios of v1 (docs/V1.md), end to end on ONE device: made by the factory station, run by the gateway against the real C bridge
core, a simulated mouse, PC and person, and a phone that only carries bytes.

Every scenario is a function over a `Pilot`; each has a test of its own on a fresh device, and `test_a_pilot_day_in_one_directory` runs them
all in order on the same directory, with a power cut in between, which is how a device is actually used. What none of this can show is written
in docs/V1.md next to each scenario: it is a simulation of hardware, radio and people."""
import shutil

import pytest

from dataopen.bridge.cbridge import REASONS as BRIDGE_REASONS
from dataopen.bridge.cbridge import S_ASSIST, S_BYPASS, S_PASSTHRU
from dataopen.ctl import protocol as P
from dataopen.ctl.identity import Card
from dataopen.ctl.sim import dev_vendor
from dataopen.provisioning import records as R

from acc_helpers import Pilot

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


def err_code(r) -> tuple:
    assert r.type == P.T_ERR, r.body
    j = r.json()
    return j["code"], j["detail"]


def reason(p) -> str:
    return BRIDGE_REASONS[p.rig.status().reason]


def mouse_is_untouched(rows) -> bool:
    """Every report the PC got equals what the hand sent: the property of PASSTHRU and of the bypass."""
    return bool(rows) and all(i == o for _, i, o in rows)


def mouse_is_only_ever_reduced(rows) -> bool:
    return bool(rows) and all(abs(o) <= abs(i) and (o == 0 or (o > 0) == (i > 0)) for _, i, o in rows)


# ---------------------------------------------------------------------------------------------------------------- ACC-01 first run
def s01_first_run(p: Pilot) -> None:
    p.connect()
    rec = p.gw.device.record
    # the factory made it: a label, a chain from the manufacturer to this owner card, the first boot moved it into the field
    assert R.parse_serial(rec.serial)["sku"] == 1 and p.gw.device.lifecycle == R.IN_FIELD and p.report.serial == rec.serial
    card = p.ph.get_identity()
    assert R.verify_chain(Card.from_json(card), p.pub, p.hsm.hw_id) == rec.serial
    st = p.state()
    assert st["device.serial"] == rec.serial and st["assist.on"] is False and st["trusted.count"] == 0 and st["fw.state"] == "current"
    assert p.gw.slotset.mask() == 0 and st["pkg.state"] == "none" and st["model.state"] == "none"
    # with no profile and no module help it is a plain USB mouse: nothing changed, nothing lost
    assert (p.rig.status().state, reason(p)) == (S_PASSTHRU, "CMD_PASSTHRU") and p.rig.route == "bridge"
    rows = p.traffic()
    assert len(rows) == 200 and mouse_is_untouched(rows) and p.rig.dropped == 0
    # a stranger with the phone can not take anything out (there is nothing to ask for) or wipe anything without the hand on the device
    assert err_code(p.ph.call(P.T_GET, {"what": "bundle", "for": "self", "scope": "all"}))[0] == P.E.RESIDENT
    assert err_code(p.ph.act("erase.profile", True)) == (P.E.PHYSICAL, "erase")
    assert err_code(p.ph.act("pairing.forget", True))[0] == P.E.PHYSICAL
    # a phone is paired by a physical hold of CONFIRM (3-10 s); the window closes by itself
    assert p.state()["pairing.open"] is False
    p.gw.button("confirm", True, p.w.t)
    p.w.run(3500)
    p.gw.button("confirm", False, p.w.t)
    assert p.state()["pairing.open"] is True
    p.w.run(121_000)
    assert p.state()["pairing.open"] is False
    assert p.gw.slotset.mask() == 0 and not p.gw.settings.assist_wanted               # none of it started anything


def test_acc01_first_run(tmp_path):
    s01_first_run(Pilot(tmp_path))


# ---------------------------------------------------------------------------------------------------------------- ACC-02 calibration to help
def s02_calibration(p: Pilot) -> None:
    assert p.ph.set("calib.running", True).type == P.T_ACK
    # while it learns, help stays off and can not be turned on; the LEDs say so
    assert err_code(p.ph.set("assist.on", True))[0] == P.E.BUSY and p.gw.led_mode(p.w.t) == "calibrating"
    rows = p.traffic(50)
    assert mouse_is_untouched(rows)                                                      # learning does not touch the pointer
    p.w.run(2000)
    st = p.state()
    assert st["calib.running"] is True and st["profile.fill"] == 100 and st["profile.layers"] == "both"
    assert p.ph.set("calib.running", False).type == P.T_ACK
    assert p.gw.slot.has and p.state()["assist.on"] is False                             # a profile exists; help is still the person's decision
    # turning help on is a TRIAL that undoes itself; keeping it is a decision
    assert p.ph.set("assist.on", True).type == P.T_ACK
    assert p.state()["trial.left_s"] > 0
    p.w.run(1500)
    assert p.rig.status().state == S_ASSIST and p.rig.status().params_rejected == 0
    assert p.ph.confirm(True).type == P.T_ACK
    p.w.run(25_000)                                                                       # past the trial time: it stays
    st = p.state()
    assert st["assist.on"] is True and st["trial.left_s"] == 0 and p.rig.status().state == S_ASSIST
    # not keeping it: the trial runs out and help goes back off by itself
    assert p.ph.set("assist.strength", 9).type == P.T_ACK
    p.w.run(21_000)
    assert p.state()["assist.strength"] != 9 and p.rig.status().state == S_ASSIST


def test_acc02_calibration_to_help(tmp_path):
    p = Pilot(tmp_path)
    s01_first_run(p)
    s02_calibration(p)


# ---------------------------------------------------------------------------------------------------------------- ACC-03 everyday work
def s03_everyday_work(p: Pilot) -> None:
    assert p.rig.status().state == S_ASSIST
    # no target in sight: not a single count is changed (help needs an object)
    rows = p.traffic(150)
    assert mouse_is_only_ever_reduced(rows) and sum(abs(o) for _, _, o in rows) > 0
    # a reach towards a target with the detector's scene arriving: the approach is braked, never pushed
    r = p.reach()
    assert r["min_k"] < 0.8 and r["end"] < 700
    assert all(abs(o) <= abs(i) and (o == 0 or (o > 0) == (i > 0)) for i, o in r["reports"])
    assert all(o == 0 for i, o in r["reports"] if i == 0)                                  # nothing from nothing
    st = p.rig.status()
    assert st.invariant_viol == 0 and st.params_rejected == 0 and p.rig.route == "bridge"
    # the phone going away changes nothing about how the device works
    p.ph.disconnect()
    p.w.run(3000)
    assert p.rig.status().state == S_ASSIST and p.gw.settings.assist_wanted
    p.ph.connect()
    # STOP: at once, byte for byte, and it survives a power cut of the module (a restart never turns help back on)
    assert p.ph.stop().type == P.T_ACK
    rows = p.traffic(100)
    assert mouse_is_untouched(rows)
    p.w.power_cycle()
    p.ph.connect()
    p.w.run(3000)
    assert p.state()["assist.on"] is False and mouse_is_untouched(p.traffic(50))
    # and the person turns it on again, as a trial, and keeps it
    assert p.ph.set("assist.on", True).type == P.T_ACK
    p.w.run(1500)
    assert p.ph.confirm(True).type == P.T_ACK
    p.w.run(25_000)
    assert p.state()["assist.on"] is True and p.rig.status().state == S_ASSIST


def test_acc03_everyday_work(tmp_path):
    p = Pilot(tmp_path)
    s01_first_run(p)
    s02_calibration(p)
    s03_everyday_work(p)


# ---------------------------------------------------------------------------------------------------------------- helpers for B and A
def second_profile(tmp_path):
    from dataopen.ctl.sim import seed_profile
    return seed_profile(tmp_path / "second-profile", "overshooter")._state


def load_package(p: Pilot, raw: bytes, piece: int = 3000) -> dict:
    r = p.ph.pkg_send(raw, piece=piece)
    assert r.type == P.T_ACK, r.body
    return r.json()["pending"]


def apply_package(p: Pilot, button: bool) -> None:
    if button:
        p.press()
    r = p.ph.act("pkg.apply", True)
    assert r.type == P.T_ACK, r.body


def ensure_help(p: Pilot) -> None:
    """Help is the person's decision and several things switch it off on purpose (a slot clear, Panic, STOP): turn it on again the way they would."""
    if not p.gw.session:
        p.ph.connect()
    if not p.gw.settings.assist_wanted:
        assert p.ph.set("assist.on", True).type == P.T_ACK
        p.w.run(1500)
        assert p.ph.confirm(True).type == P.T_ACK
        p.w.run(3000)
    assert p.rig.status().state == S_ASSIST


def refusal(r, code=P.E.PKG_REJECTED) -> str:
    c, d = err_code(r)
    assert c == code, (c, d)
    return d


# ---------------------------------------------------------------------------------------------------------------- ACC-04 slots
def s04_slots(p: Pilot, tmp_path) -> None:
    assert p.gw.active == 0 and p.gw.slot.has
    # a second context arrives as a package from somebody the person trusts (the button makes them known)
    prof = second_profile(tmp_path)
    load_package(p, p.package(slot=2, profile=prof, tuning=(8, 2), name="Браузер"))
    apply_package(p, button=True)
    p.w.run(25_000)
    sl = {s["n"]: s for s in p.ph.get_slots()["slots"]}
    assert sl[0]["has"] and sl[2]["has"] and sl[2]["name"] == "Браузер" and (sl[2]["strength"], sl[2]["tremor"]) == (8, 2)
    assert p.gw.active == 0 and p.state()["assist.on"] is True                             # a package for another slot moves nothing
    # the SLOT button walks only the slots in use, the LED and the phone agree, help is not toggled by it
    p.w.press("slot")
    assert p.gw.active == 2 and p.state()["slot.active"] == 2 and p.gw.status().slot == 2
    leds = p.gw.leds(p.w.t)
    assert not (leds[0] or leds[1] or leds[3]) and p.gw.led_mode(p.w.t) in ("trial", "steady") and p.gw.settings.assist_wanted
    # new content in a slot the person has not kept in work yet is a trial: Undo goes back to the old slot at once
    assert p.state()["trial.left_s"] > 0
    assert p.ph.confirm(False).type == P.T_ACK
    assert p.gw.active == 0 and p.state()["trial.left_s"] == 0
    # choosing it from the phone is the same thing, and Keep makes it a slot "in work"
    assert p.ph.select_slot(2).type == P.T_ACK
    assert p.ph.confirm(True).type == P.T_ACK
    p.w.run(1500)
    assert p.gw.active == 2 and p.gw.slot.vetted
    # each slot has its own levels; the safety clamp holds in every one, and the bridge got the slot's own parameters
    assert p.state()["assist.strength"] == 8 and p.state()["tremor.level"] == 2
    r = p.reach()
    assert all(abs(o) <= abs(i) for i, o in r["reports"]) and p.rig.status().invariant_viol == 0 and p.rig.status().params_rejected == 0
    p.w.press("slot")
    assert p.gw.active == 0 and p.state()["assist.strength"] != 8                          # the other context has its own
    assert p.gw.settings.assist_wanted and p.rig.status().state == S_ASSIST                # and help stayed the person's choice throughout
    # clearing a slot is two steps and the hand: the other slot is untouched
    assert err_code(p.ph.act("slot.clear", True))[0] == P.E.PHYSICAL
    p.ph.select_slot(2)
    p.ph.confirm(True)
    p.press()
    assert p.ph.act("slot.clear", True).type == P.T_ACK
    sl = {s["n"]: s for s in p.ph.get_slots()["slots"]}
    assert not sl[2]["has"] and sl[0]["has"]
    assert p.ph.select_slot(0).type == P.T_ACK


def test_acc04_slots(tmp_path):
    p = Pilot(tmp_path)
    p.connect()
    p.calibrate()
    s04_slots(p, tmp_path)


# ---------------------------------------------------------------------------------------------------------------- ACC-05 packages (channel B)
def s05_packages(p: Pilot, tmp_path) -> None:
    from dataopen.updates import dev as UD
    from dataopen.updates import package as K
    ensure_help(p)
    assert p.gw.active == 0
    big = UD.tiny_model(pad=20000)
    model_big = (big, UD.card_of(big, name="big", version=1))
    # a forged package and one for another device are refused with the FIRST piece: not a megabyte is spent on them
    forged = bytearray(p.package(model=model_big))
    forged[30] ^= 1
    r = p.ph.call(P.T_PKG_BEGIN, {"size": len(forged)})
    assert r.type == P.T_ACK
    assert refusal(p.ph.call(P.T_PKG_CHUNK, body=(0).to_bytes(4, "little") + bytes(forged[:3000]))) == "bad_signature"
    stranger = UD.sender(tmp_path / "other-people", "stranger")
    other = K.build_package(p.sender, stranger.card(), p.seq + 1, model=model_big)
    p.seq += 1
    assert refusal(p.ph.pkg_send(other)) == "wrong_device"
    assert p.ph.get_packages()["upload"] is None and p.ph.get_packages()["pending"] is None
    # an honest one from somebody known (the button made them known in ACC-04, or does so here), received WHILE the person keeps working
    raw = p.package(slot=0, tuning=(6, 4), model=model_big)
    r = p.ph.call(P.T_PKG_BEGIN, {"size": len(raw)})
    assert r.type == P.T_ACK
    half = len(raw) // 2
    for off in range(0, half, 3000):
        assert p.ph.call(P.T_PKG_CHUNK, body=off.to_bytes(4, "little") + raw[off : min(off + 3000, half)]).type == P.T_ACK
    up = p.ph.get_packages()["upload"]
    assert up and up["next"] == half and p.state()["pkg.state"] == "receiving"
    rows = p.traffic(150)                                                                    # the pointer does not notice a transfer
    assert len(rows) == 150 and mouse_is_only_ever_reduced(rows) and p.rig.route == "bridge" and p.rig.dropped == 0
    for off in range(half, len(raw), 3000):                                                  # the phone resumes from where the device is
        assert p.ph.call(P.T_PKG_CHUNK, body=off.to_bytes(4, "little") + raw[off : off + 3000]).type == P.T_ACK
    known = p.sender.digest.hex() in p.gw.trust["senders"]
    pend = p.ph.call(P.T_PKG_END, body=b"").json()["pending"]
    want = "model" if known else "trust"                                                     # a stranger first; weights from anybody but oneself
    assert pend["kinds"] == ["tuning", "model"] and pend["from"] == p.sender.id and pend["button"] == want
    assert p.gw.slotset[0].strength != 6 and not (p.gw.slotset[0].dir / "model.bin").exists()   # received is not applied
    # a refusal spends nothing and applies nothing
    c, d = err_code(p.ph.act("pkg.apply", True))
    assert c == P.E.PHYSICAL and d == f"{want}:{p.sender.id}"
    assert p.state()["pkg.state"] == "pending" and not (p.gw.slotset[0].dir / "model.bin").exists()
    apply_package(p, button=True)
    p.w.run(1500)
    st = p.state()
    assert st["model.state"] == "ok" and st["model.name"] == "big" and st["pkg.state"] == "none"
    assert p.ph.confirm(True).type == P.T_ACK                                                 # levels from a package are a trial like any other
    # the same package again is a replay; the previous generation of a model can be brought back
    assert refusal(p.ph.pkg_send(raw)) == "replay"
    small = UD.tiny_model()
    load_package(p, p.package(slot=0, model=(small, UD.card_of(small, name="small", version=2))))
    apply_package(p, button=True)
    assert p.state()["model.name"] == "small"
    assert p.ph.act("pkg.revert", True).type == P.T_ACK and p.state()["model.name"] == "big"
    # the two channels do not stand in for each other, whichever way it is tried
    assert refusal(p.ph.pkg_send(p.image(9))) == "wrong_channel"
    r = p.ph.fw_send(p.package(tuning=(1, 1)), piece=2000)
    assert refusal(r, P.E.FW_REJECTED) == "wrong_channel"
    # personal and encrypted at rest: nothing readable in the slot's files
    assert big[:40] not in (p.gw.slotset[0].dir / "model.bin").read_bytes()
    assert p.rig.status().invariant_viol == 0 and mouse_is_only_ever_reduced(p.traffic(50))


def test_acc05_packages(tmp_path):
    p = Pilot(tmp_path)
    p.connect()
    p.calibrate()
    s05_packages(p, tmp_path)


# ---------------------------------------------------------------------------------------------------------------- ACC-06 system update (channel A)
def s06_system_update(p: Pilot) -> None:
    from dataopen.ctl import firmware as F
    ensure_help(p)
    fw = p.gw.fw
    assert (fw.active, fw.version, fw.floor) == ("A", 1, 1) and p.state()["fw.state"] == "current"
    models_before = {f.name: f.read_bytes() for f in (p.gw.dir / "slots").rglob("model.*")}
    # a wrong file never reaches a bank: unsigned, for other hardware, a package, junk
    signer = dev_vendor()[0]
    for bad, want in ((F.build_image(b"x" * 900, p.hsm.hw_id, 2, 1, p.sender), "signature"),             # signed by somebody who is not the manufacturer
                      (F.build_image(b"y" * 900, b"DOHW0002", 2, 1, signer), "hardware")):                # the manufacturer's, for another board
        assert refusal(p.ph.fw_send(bad, piece=2000), P.E.FW_REJECTED) == want
    assert refusal(p.ph.fw_send(b"\x00" * 900, piece=2000), P.E.FW_REJECTED) == "damaged"
    assert p.gw.fw.slots["B"].image.version == 1 and p.gw.fw.trial is None
    # a good one is received WHILE help is working; it waits in the other bank and nothing restarts
    r = p.ph.fw_send(p.image(2), piece=2000)
    assert r.type == P.T_ACK and r.json()["staged"] == 2
    st = p.state()
    assert st["fw.state"] == "staged" and st["fw.version"] == 1 and p.rig.status().state == S_ASSIST
    rows = p.traffic(100)
    assert mouse_is_only_ever_reduced(rows) and p.rig.route == "bridge"
    # applying: two steps and the hand; the module restarts; the bridge (a separate processor) never stops carrying the mouse
    assert err_code(p.ph.act("fw.apply", True)) == (P.E.PHYSICAL, "fw.apply")
    p.press()
    assert p.ph.act("fw.apply", True).type == P.T_ACK
    lost_before = p.rig.dropped
    rows = p.traffic(400)                                                                    # the restart happens inside this stretch
    assert len(rows) == 400 and p.rig.dropped == lost_before and mouse_is_only_ever_reduced(rows)
    p.ph.connect()
    assert p.gw.fw.running == "B" and p.state()["fw.state"] == "trial" and p.state()["fw.version"] == 2
    # it has to prove itself: healthy bridge and a PC that sees the mouse, for a while; only then the floor rises
    p.w.run(1500)
    assert p.gw.fw.trial == "B"
    p.w.run(8000)
    assert (p.gw.fw.active, p.gw.fw.trial, p.gw.fw.floor) == ("B", None, 2) and p.state()["fw.state"] == "current"
    assert p.rig.status().state == S_ASSIST and p.gw.settings.assist_wanted                   # help came back by itself, as the person left it
    # the old version can not be put back by a file (floor), and the update left the person's data alone
    assert refusal(p.ph.fw_send(p.image(1, 1), piece=2000), P.E.FW_REJECTED) == "rollback"
    assert {f.name: f.read_bytes() for f in (p.gw.dir / "slots").rglob("model.*")} == models_before
    # an update that never proves itself is replaced by the old one, with the mouse working all the way
    assert p.ph.fw_send(p.image(3), piece=2000).type == P.T_ACK
    p.w.gw_args["fw_confirm_s"] = 10**6                                                       # this one will not prove itself
    p.press()
    assert p.ph.act("fw.apply", True).type == P.T_ACK
    runs = []
    for _ in range(F.MAX_BOOTS):
        p.w.run(60)
        runs.append(p.gw.fw.running)
        assert mouse_is_only_ever_reduced(p.traffic(30))
        p.w.power_cycle()
    assert runs == ["A"] * F.MAX_BOOTS                                                         # the failed one went into the bank that was not running
    p.w.power_cycle()
    p.w.gw_args["fw_confirm_s"] = 4
    p.w.power_cycle()
    p.ph.connect()
    fw = p.gw.fw
    assert fw.trial is None and fw.version == 2 and fw.floor == 2 and any(s.bad for s in fw.slots.values())
    p.w.run(8000)
    assert p.rig.status().state == S_ASSIST and mouse_is_only_ever_reduced(p.traffic(50))


def test_acc06_system_update(tmp_path):
    p = Pilot(tmp_path)
    p.connect()
    p.calibrate()
    s06_system_update(p)


# ---------------------------------------------------------------------------------------------------------------- ACC-07 Panic and the bypass
def hold_panic(p: Pilot, ms: int) -> None:
    p.rig.panic(True)
    p.w.run(ms)
    p.rig.panic(False)


def s07_panic_and_bypass(p: Pilot) -> None:
    ensure_help(p)
    # a short press: help off at once, the pointer is NOT interrupted (no re-enumeration), nothing is lost, the output is the input
    lost = p.rig.dropped
    hold_panic(p, 100)
    st = p.rig.status()
    assert (st.state, reason(p)) == (S_PASSTHRU, "PANIC") and p.rig.route == "bridge"
    rows = p.traffic(100)
    assert mouse_is_untouched(rows) and p.rig.dropped == lost
    # the phone can ask for help again as much as it likes: only the hand lifts a latch it did not set, and the phone is told so
    assert p.ph.set("assist.on", False).type == P.T_ACK
    assert p.ph.set("assist.on", True).type == P.T_ACK
    p.w.run(1500)
    assert p.ph.confirm(True).type == P.T_ACK
    p.w.run(3000)
    assert (p.rig.status().state, reason(p)) == (S_PASSTHRU, "PANIC") and p.gw.status().flags & P.SF_LATCH_SOFT
    assert mouse_is_untouched(p.traffic(50))
    hold_panic(p, 2100)                                                                       # a deliberate hold of the hand re-arms it
    p.w.run(3000)
    assert p.rig.status().state == S_ASSIST and p.gw.settings.assist_wanted                   # and the wish the phone left is honoured again
    # a long hold: the contacts go back to the real mouse, whatever the firmware is doing (the loop is an RC, not code)
    hold_panic(p, 3200)
    assert p.rig.status().state == S_BYPASS and p.rig.route == "bypass" and reason(p) == "PANIC_LONG"
    p.w.run(p.rig.reenum_us // 1000 + 100)
    rows = p.traffic(60)
    assert p.rig.pc_conn == "direct" and rows and all(r == "direct" and i == o for r, i, o in rows)
    # the phone's hard bypass is one-way too: it can only go DOWN, never back
    assert p.ph.set("assist.on", True).type == P.T_ACK
    p.w.run(2000)
    assert p.rig.status().state == S_BYPASS
    hold_panic(p, 2100)
    p.w.run(8000)
    assert p.rig.status().state in (S_PASSTHRU, S_ASSIST)
    p.w.run(4000)
    assert p.rig.status().state == S_ASSIST
    # a dead firmware on the bridge processor: the supervisor (analog, no code) releases the contacts within its window
    t_hang = p.w.t
    p.rig.kill_firmware()
    for _ in range(250):
        p.rig.move(2, 1)
        p.rig.step()
    gap = [e for e in p.rig.events if e[1] == "route bridge -> bypass"][-1][0] - t_hang
    assert p.rig.route == "bypass" and gap <= 101_000
    p.w.run(p.rig.reenum_us // 1000 + 100)
    rows = p.traffic(50)
    assert p.rig.pc_conn == "direct" and rows and all(r == "direct" and i == o for r, i, o in rows)
    # STOP needs no session and no layout: after the phone vanished it still works the moment help comes back
    p.rig.reboot_firmware()
    assert p.rig.engage_fast(20000)
    p.ph.disconnect()
    assert p.ph.stop().type == P.T_ACK                                                        # no HELLO was said on this connection
    p.w.run(500)
    assert not p.gw.settings.assist_wanted and p.rig.status().state == S_PASSTHRU
    assert mouse_is_untouched(p.traffic(50))


def test_acc07_panic_and_bypass(tmp_path):
    p = Pilot(tmp_path)
    p.connect()
    p.calibrate()
    s07_panic_and_bypass(p)


# ---------------------------------------------------------------------------------------------------------------- ACC-08 recovery after a failure
def s08_recovery(p: Pilot) -> None:
    from dataopen.provisioning import recovery as RC
    ensure_help(p)
    # the compute module dies completely (both banks bad, a dead SoM): the bridge is another processor and keeps carrying the mouse
    lost = p.rig.dropped
    p.w.gw_dead = True
    p.w.run(6000)
    st = p.rig.status()
    assert st.state == S_PASSTHRU and p.rig.route == "bridge" and reason(p) in ("STALE_PARAMS", "STALE_LINK")
    rows = p.traffic(100)
    assert mouse_is_untouched(rows) and p.rig.dropped == lost                                  # no gap, nothing changed, help off
    # the module comes back on the same storage: the person's slots, trust and wish are where they were, and help returns by itself
    p.w.gw_dead = False
    p.w.make_gateway()
    p.ph.connect()
    p.w.run(6000)
    assert p.gw.slot.has and p.gw.settings.assist_wanted and p.rig.status().state == S_ASSIST
    # a damaged copy of the OTHER bank changes nothing about what runs, and is not trusted either
    other = p.gw.dir / "fw" / f"bank-{p.gw.fw.other}.img"
    other.write_bytes(other.read_bytes()[:-7] + b"junk123")
    p.w.make_gateway()
    p.ph.connect()
    assert p.gw.fw.slots[p.gw.fw.other].image is None and p.gw.fw.version == p.gw.fw.slots[p.gw.fw.active].image.version
    assert p.state()["fw.state"] == "current"
    p.w.run(6000)
    assert p.rig.status().state == S_ASSIST and mouse_is_only_ever_reduced(p.traffic(50))
    # the rest of the ladder is a MODEL of the rules (there is no bootloader): in every scenario the mouse has a path, and every one that is
    # not a return to the manufacturer ends in a normal system
    seen = 0
    for s in RC.SCENARIOS:
        o = RC.run(s)
        seen += 1
        assert o.first.mouse in ("assisted", "passthrough", "direct"), s.key
        assert o.final.stage == "normal" or o.rma, s.key
    assert seen >= 12
    assert RC.always_direct(RC.Unit()) and not RC.always_direct(RC.Unit(relays_ok=False))   # the one thing layer 0 can not cover


def test_acc08_recovery(tmp_path):
    p = Pilot(tmp_path)
    p.connect()
    p.calibrate()
    s08_recovery(p)


# ---------------------------------------------------------------------------------------------------------------- ACC-09 the four levels of reset
def s09_resets(p: Pilot, tmp_path) -> None:
    from dataopen.provisioning import device as D
    from dataopen.provisioning import service as SV
    from dataopen.ctl import firmware as F
    from dataopen.updates import dev as UD
    p.ph.connect()
    me = p.gw.identity
    prof = second_profile(tmp_path)
    load_package(p, p.package(slot=2, profile=prof, tuning=(8, 2), name="Браузер"))
    apply_package(p, button=p.sender.digest.hex() not in p.gw.trust["senders"])
    small = UD.tiny_model()
    load_package(p, p.package(slot=0, model=(small, UD.card_of(small, name="own", version=1))))
    apply_package(p, button=True)                                                               # weights always need the press
    backup = p.settings_file(profile=prof, tuning=(5, 5))                                       # a file for THIS owner (made by somebody else)
    load_package(p, p.package(slot=1, tuning=(3, 3)))                                          # and a package still waiting
    owner, serial, dak, att = me.id, p.gw.device_serial, p.gw.device.se.dak_pub, p.gw.device.record.att.to_json()
    fw_files = {f.name: f.read_bytes() for f in (p.gw.dir / "fw").glob("*")}
    # L1: this slot only
    assert p.ph.select_slot(2).type == P.T_ACK
    p.ph.confirm(True)
    p.press()
    assert p.ph.act("slot.clear", True).type == P.T_ACK
    assert not p.gw.slotset[2].has and p.gw.slotset[0].has and (p.gw.slotset[0].dir / "model.bin").exists()
    assert p.ph.get_packages()["pending"] is not None and p.gw.identity.id == owner and not p.gw.settings.assist_wanted
    # L2: every profile, model, waiting package and trusted sender; the owner and the phones stay
    p.press()
    assert p.ph.act("erase.profile", True).type == P.T_ACK
    assert p.gw.slotset.mask() == 0 and not list((p.gw.dir / "slots").rglob("model.*")) and p.ph.get_packages()["pending"] is None
    assert p.gw.identity.id == owner and not p.gw.trust["senders"] and p.forgets == [] and p.state()["trusted.count"] == 0
    assert err_code(p.ph.put_bundle(backup))[0] == P.E.PHYSICAL                                 # the senders are forgotten: the button again
    p.press()
    r = p.ph.put_bundle(backup)                                                                 # the owner is the same: the file still opens
    assert r.type == P.T_ACK and p.gw.slotset[0].has
    p.press()
    assert p.ph.act("erase.profile", True).type == P.T_ACK
    # L3: a new owner; phones forgotten; every file made for the old owner is dead; the device is the same device
    p.press()
    assert p.ph.act("factory.reset", True).type == P.T_ACK
    assert p.gw.identity.id != owner and p.forgets == [1] and p.gw.device_serial == serial and p.gw.device.se.dak_pub == dak
    assert err_code(p.ph.put_bundle(backup))[0] == P.E.WRONG_DEVICE
    assert R.verify_chain(Card.from_json(p.ph.get_identity()), p.pub, p.hsm.hw_id) == serial
    assert {f.name: f.read_bytes() for f in (p.gw.dir / "fw").glob("*")} == fw_files           # none of the three touches the system
    # L4 is not on the phone or on a button: a token from the manufacturer for THIS serial, a nonce this device issued, and a hand on it
    assert p.ph.act("fw.rollback", True).type == P.T_ERR                                        # nothing to return to by that route
    d = p.dir
    ag = D.DeviceAgent(d, hw_id=p.hsm.hw_id, vendor_pub=p.hsm.pub)
    floor = F.SlotManager.load(d / "fw", p.hsm.pub, p.hsm.hw_id).floor

    def token():
        return p.hsm.issue_service_token(serial, "factory_return", ag.service_challenge()).to_json()
    snapshot = {f: f.read_bytes() for f in d.rglob("*") if f.is_file() and f.name != "service.json"}
    if floor > 1:                                                                               # an update that raised the floor made the golden image stale
        with pytest.raises(SV.ServiceError) as e:
            SV.full_return(d, ag, token(), presence=True)
        assert e.value.key == "floor"                                                           # refused BEFORE the token is spent, and nothing was changed
        assert {f: f.read_bytes() for f in d.rglob("*") if f.is_file() and f.name != "service.json"} == snapshot
        (d / "golden" / "golden.img").write_bytes(p.image(floor, 1, b"golden-refreshed-" * 100))   # the support tool brings a current one
    with pytest.raises(SV.ServiceError) as e:
        SV.full_return(d, ag, token(), presence=False)                                          # no hand on the device
    assert e.value.key == "presence"
    out = SV.full_return(d, ag, token(), presence=True)
    assert out["serial"] == serial and out["floor"] == floor and out["lifecycle"] == R.SHIPPED
    mgr = F.SlotManager.load(d / "fw", p.hsm.pub, p.hsm.hw_id)
    assert mgr.slots["A"].image.digest == mgr.slots["B"].image.digest and mgr.trial is None and mgr.floor == floor
    assert not (d / "slots").exists() and not list(d.glob("settings.*")) and not (d / "pkg").exists()
    assert ag.record.serial == serial and ag.se.dak_pub == dak and ag.record.att.to_json() == att
    # and it starts again as a device nobody has used: the first-run state of ACC-01, a new owner on the old device
    p.w = type(p.w)(d, **p.kw)
    p.connect()
    st = p.state()
    assert st["assist.on"] is False and p.gw.slotset.mask() == 0 and st["device.serial"] == serial and st["trusted.count"] == 0
    assert mouse_is_untouched(p.traffic(50))


def test_acc09_resets(tmp_path):
    p = Pilot(tmp_path)
    p.connect()
    p.calibrate()
    s09_resets(p, tmp_path)


# ---------------------------------------------------------------------------------------------------------------- the whole day, one device
def test_a_pilot_day_in_one_directory(tmp_path):
    """The scenarios in the order a person meets them, on ONE device directory, with the power cut between them. A scenario that passes on a
    fresh device but not here means two pieces that work alone disagree about something stored."""
    p = Pilot(tmp_path)
    s01_first_run(p)
    s02_calibration(p)
    s03_everyday_work(p)
    p.w.power_cycle()
    p.connect()
    p.w.run(5000)
    assert p.gw.settings.assist_wanted and p.rig.status().state == S_ASSIST                  # the day continues after a power cut of the module
    s04_slots(p, tmp_path)
    s05_packages(p, tmp_path)
    s06_system_update(p)
    s07_panic_and_bypass(p)
    s08_recovery(p)
    s09_resets(p, tmp_path)
    # and through the whole of it the profile the device learned, the profile and the weights it was given, were on the device and never left it
    heard = p.egress()
    assert len(p.leaks.needles) > 100 and len(heard) > 20000 and len(p.worlds) >= 2
    assert p.leaks.hits(heard) == [] and all(w.gw.egress_blocked == 0 for w in p.worlds)
