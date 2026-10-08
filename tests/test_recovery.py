"""Recovery: the boot chain as a model, every way back, and the full return to factory state with its token."""
import itertools
import json

import pytest

from dataopen.ctl import firmware as FW
from dataopen.provisioning import device as D
from dataopen.provisioning import recovery as RC
from dataopen.provisioning import records as R
from dataopen.provisioning import service as SV
from dataopen.provisioning import station as ST
from prov_helpers import provisioned, vendor


# ------------------------------------------------------------------------------------------------------------------------ the model
def test_a_healthy_unit_boots_normally_and_assists():
    b = RC.boot(RC.Unit())
    assert (b.stage, b.bank, b.mcu, b.mouse, b.video, b.pwa, b.profiles_kept) == ("normal", "A", "running", "assisted", "tap", "full", True)


def test_whatever_is_broken_the_mouse_and_the_monitor_still_have_a_path():
    """Layer 0 of the recovery design: the model is exercised over every combination of software damage and of the person's actions."""
    flags = ["supply_ok", "loader_ok", "golden_ok", "data_ok", "chord"]
    banks = list(itertools.product((True, False), repeat=2))
    seen = set()
    for vals in itertools.product((True, False), repeat=len(flags)):
        for ba, bb in banks:
            for ma, mb in banks:
                u = RC.Unit(**dict(zip(flags, vals)), bank_ok={"A": ba, "B": bb}, mcu_ok={"A": ma, "B": mb})
                b = RC.boot(u)
                seen.add(b.stage)
                assert b.mouse in ("assisted", "passthrough", "direct") and b.video in ("tap", "direct") and b.led
                if b.stage != "normal":
                    # help and the video tap only exist in the normal system
                    assert b.mouse != "assisted" and b.video == "direct"
                if not (ma or mb) or not u.supply_ok:
                    assert b.mouse in ("direct",) or b.mcu in ("running", "rolled_back")
                if b.mouse in ("assisted", "passthrough"):
                    assert b.mcu in ("running", "rolled_back")                            # the pointer is only handled by an MCU that is up
                if not u.data_ok:
                    assert not b.profiles_kept
                if b.stage == "recovery":
                    assert b.pwa == "recovery" and b.led == "RECOVERY" and u.golden_ok
                if b.stage == "normal":
                    assert u.bank_ok["A"] or u.bank_ok["B"]
    assert seen == {"normal", "recovery", "maskrom", "dead"}
    # the one thing layer 0 can not cover is a welded contact
    assert RC.always_direct(RC.Unit()) and not RC.always_direct(RC.Unit(relays_ok=False))


def test_the_boot_prefers_the_active_bank_and_falls_back_to_the_other():
    assert RC.boot(RC.Unit(active="B")).bank == "B"
    b = RC.boot(RC.Unit(active="B", bank_ok={"A": True, "B": False}))
    assert b.stage == "normal" and b.bank == "A" and any("bank B is bad" in n for n in b.notes)
    assert RC.boot(RC.Unit(bank_ok={"A": False, "B": False})).stage == "recovery"
    assert RC.boot(RC.Unit(bank_ok={"A": False, "B": False}, golden_ok=False)).stage == "maskrom"
    assert RC.boot(RC.Unit(loader_ok=False)).stage == "maskrom"
    assert RC.boot(RC.Unit(emmc_ok=False)).stage == "dead" and RC.boot(RC.Unit(rom_som_ok=False)).stage == "dead"
    assert RC.boot(RC.Unit(chord=True)).stage == "recovery" and RC.boot(RC.Unit(chord=True, golden_ok=False)).stage == "maskrom"
    assert RC.boot(RC.Unit(supply_ok=False)).stage == "dead" and RC.boot(RC.Unit(supply_ok=False)).led == "OFF"


def test_the_mcu_falls_back_by_itself_and_is_re_flashed_by_the_som():
    b = RC.boot(RC.Unit(mcu_ok={"A": False, "B": True}))
    assert b.mcu == "rolled_back" and b.mouse == "assisted"
    u = RC.Unit(mcu_ok={"A": False, "B": False})
    b = RC.boot(u)
    # no MCU image: the mouse goes straight through
    assert b.mcu == "downloader" and b.mouse == "direct"
    RC.reflash_mcu_from_som(u)
    assert RC.boot(u).mouse == "assisted"
    dead = RC.Unit(mcu_ok={"A": False, "B": False}, rom_mcu_ok=False)
    assert RC.boot(dead).mcu == "dead"
    with pytest.raises(RC.RecoveryError) as e:
        RC.reflash_mcu_from_som(dead)
    assert e.value.key == "hardware"
    with pytest.raises(RC.RecoveryError) as e:
        # the SoM itself is down: it has to be fixed first
        RC.reflash_mcu_from_som(RC.Unit(loader_ok=False, mcu_ok={"A": False, "B": False}))
    assert e.value.key == "stage"


def test_restoring_from_the_golden_image_needs_the_recovery_system_a_hand_and_a_new_enough_image():
    u = RC.Unit(bank_ok={"A": False, "B": False})
    with pytest.raises(RC.RecoveryError) as e:
        RC.restore_banks_from_golden(u, presence=False)
    assert e.value.key == "presence"
    with pytest.raises(RC.RecoveryError) as e:
        # a healthy unit is not in the recovery system
        RC.restore_banks_from_golden(RC.Unit(), presence=True)
    assert e.value.key == "stage"
    old = RC.Unit(bank_ok={"A": False, "B": False}, golden_version=1, floor=3)
    with pytest.raises(RC.RecoveryError) as e:
        RC.restore_banks_from_golden(old, presence=True)
    assert e.value.key == "floor" and old.bank_ok == {"A": False, "B": False}                    # nothing was written
    RC.download_golden(old, 3)
    RC.restore_banks_from_golden(old, presence=True)
    assert RC.boot(old).stage == "normal"
    with pytest.raises(RC.RecoveryError):
        RC.download_golden(old, 2)                                                              # never an image older than the floor


def test_the_boot_rom_route_needs_the_service_port_and_the_recovery_button():
    u = RC.Unit(loader_ok=False)
    for kw, key in (({"service_port": False, "recovery_button": True}, "port"),
                    ({"service_port": True, "recovery_button": False}, "button"),
                    ({"service_port": True, "recovery_button": True, "image_floor_ok": False}, "floor")):
        with pytest.raises(RC.RecoveryError) as e:
            RC.maskrom_flash(u, **{"service_port": True, "recovery_button": True, **kw})
        assert e.value.key == key
    with pytest.raises(RC.RecoveryError) as e:
        # a healthy unit is not waiting for a tool
        RC.maskrom_flash(RC.Unit(), service_port=True, recovery_button=True)
    assert e.value.key == "stage"
    with pytest.raises(RC.RecoveryError) as e:
        RC.maskrom_flash(RC.Unit(emmc_ok=False), service_port=True, recovery_button=True)
    assert e.value.key == "hardware"
    steps = RC.maskrom_flash(u, service_port=True, recovery_button=True)
    assert RC.boot(u).stage == "normal" and any("data partition" in s for s in steps)


def test_the_profiles_survive_every_repair_of_the_software_but_not_a_dead_data_partition():
    for s in RC.SCENARIOS:
        o = RC.run(s)
        if s.key in ("data_dead", "se_dead"):
            assert not o.profiles_kept
        elif s.key != "supply_dead":
            assert o.profiles_kept, s.key
    assert RC.run(next(s for s in RC.SCENARIOS if s.key == "data_dead")).backup_restores
    # a copy sealed to this device needs this device's keys
    assert not RC.run(next(s for s in RC.SCENARIOS if s.key == "se_dead")).backup_restores
    assert RC.run(next(s for s in RC.SCENARIOS if s.key == "se_dead")).rma


def test_the_scenarios_say_when_a_computer_is_needed():
    needs = {s.key for s in RC.SCENARIOS if RC.run(s).computer}
    assert needs == {"som_loader_dead", "som_golden_dead", "som_mcu_dead", "golden_old"}
    assert not needs & {"ota_bad_image", "ota_power_cut", "som_banks_bad", "som_chord", "mcu_bad_image", "mcu_dead", "lost_phone"}


def test_every_scenario_ends_with_a_working_normal_system_except_the_ones_that_go_back_to_the_manufacturer():
    for s in RC.SCENARIOS:
        o = RC.run(s)
        assert o.first.mouse in ("assisted", "passthrough", "direct")
        if not o.rma:
            assert o.final.stage == "normal" and o.final.mouse == "assisted", s.key
        assert o.first.led in {"READY", "RECOVERY", "DEGRADED", "BYPASS", "OFF"}


def test_the_led_a_recovering_unit_shows_exists_and_is_distinct():
    from dataopen.hw import indication as I
    assert RC.boot(RC.Unit(chord=True)).led == "RECOVERY" in I.BY_KEY
    assert I.indicate(None, recovery=True) == "RECOVERY" and I.indicate(None, powered=False, recovery=True) == "OFF"


# ------------------------------------------------------------------------------------------------------------ L4 on a device directory
def shipped(tmp_path):
    d, hsm, rep = provisioned(tmp_path)
    ag = D.DeviceAgent(d, hw_id=hsm.hw_id, vendor_pub=hsm.pub)
    ag.first_boot()
    return d, hsm, rep, ag


def token(hsm, ag, serial, action="factory_return"):
    return hsm.issue_service_token(serial, action, ag.service_challenge()).to_json()


def used_device(d, hsm, floor=None):
    """Put a (fake) update and some personal files on the device, the way a year in the field does."""
    mgr = FW.SlotManager.load(d / "fw", hsm.pub, hsm.hw_id)
    from dataopen.ctl.sim import dev_image
    mgr.stage(dev_image(2, 1, b"update-" * 100))
    mgr.approve()
    mgr.reboot()
    mgr.confirm()
    if floor is not None:
        mgr.floor = floor
    mgr.save(d / "fw")
    (d / "slots" / "0").mkdir(parents=True)
    (d / "slots" / "0" / "profile.a").write_bytes(b"secret")
    (d / "settings.a").write_bytes(b"x")
    (d / "trust.a").write_bytes(b"x")


def test_the_full_return_reinstalls_the_images_and_clears_the_person_but_not_the_device(tmp_path):
    d, hsm, rep, ag = shipped(tmp_path)
    # an update to version 2 was confirmed (floor stays 1)
    used_device(d, hsm)
    floor = FW.SlotManager.load(d / "fw", hsm.pub, hsm.hw_id).floor
    before = {"dak": ag.se.dak_pub, "att": ag.record.att.to_json(), "owner": ag.owner().id, "floor": floor}
    out = SV.full_return(d, ag, token(hsm, ag, rep.serial), presence=True)
    mgr = FW.SlotManager.load(d / "fw", hsm.pub, hsm.hw_id)
    # the history is kept
    assert out["lifecycle"] == R.SHIPPED and ag.otp.read("lifecycle_log")[-3:] == [R.RMA, R.PROVISIONED, R.SHIPPED]
    assert mgr.slots["A"].image.version == mgr.slots["B"].image.version == 1 and mgr.active == "A" and mgr.trial is None
    assert mgr.floor == before["floor"]                                                            # the floor is never lowered
    assert not (d / "slots").exists() and not list(d.glob("settings.*")) and not list(d.glob("trust.*"))
    assert ag.owner().id != before["owner"] and ag.se.dak_pub == before["dak"] and ag.record.att.to_json() == before["att"]
    assert ag.record.serial == rep.serial


def test_the_full_return_refuses_when_the_golden_image_is_older_than_the_floor_and_changes_nothing(tmp_path):
    d, hsm, rep, ag = shipped(tmp_path)
    used_device(d, hsm, floor=2)
    snap = {p: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    tok = token(hsm, ag, rep.serial)
    with pytest.raises(SV.ServiceError) as e:
        SV.full_return(d, ag, tok, presence=True)
    assert e.value.key == "floor"
    now = {p: p.read_bytes() for p in d.rglob("*") if p.is_file() and p.name != "service.json"}
    assert now == {p: v for p, v in snap.items() if p.name != "service.json"}


def test_the_token_the_hand_and_the_challenge_are_all_required(tmp_path):
    d, hsm, rep, ag = shipped(tmp_path)
    tok = token(hsm, ag, rep.serial)
    with pytest.raises(SV.ServiceError) as e:
        SV.full_return(d, ag, tok, presence=False)                                                 # no hand on the device
    assert e.value.key == "presence"
    for bad, key in ((dict(tok, serial=R.make_serial(1, 26, 41, 99)), "token"), (dict(tok, sig=tok["sig"][:-4] + "AAAA"), "token"),
                     (dict(tok, nonce="00" * 16), "token"), ({}, "token"), (dict(tok, action="reprovision"), "token")):
        with pytest.raises(SV.ServiceError) as e:
            SV.full_return(d, ag, bad, presence=True)
        assert e.value.key == key, bad
    # the genuine token still works: refusals did not spend it
    out = SV.full_return(d, ag, tok, presence=True)
    assert out["serial"] == rep.serial
    with pytest.raises(SV.ServiceError):
        SV.full_return(d, ag, tok, presence=True)                                                  # but only once


def test_another_devices_token_and_a_foreign_manufacturers_token_are_worthless(tmp_path):
    d, hsm, rep, ag = shipped(tmp_path)
    from cryptography.hazmat.primitives.asymmetric import ed25519
    evil = ST.VendorHsm(ed25519.Ed25519PrivateKey.generate())
    evil.shipped.add(rep.serial)
    tok = evil.issue_service_token(rep.serial, "factory_return", ag.service_challenge()).to_json()
    with pytest.raises(SV.ServiceError) as e:
        SV.full_return(d, ag, tok, presence=True)
    assert e.value.key == "token"


def test_a_missing_golden_image_is_reported_not_guessed(tmp_path):
    d, hsm, rep, ag = shipped(tmp_path)
    (d / "golden" / "golden.img").unlink()
    with pytest.raises(SV.ServiceError) as e:
        SV.full_return(d, ag, token(hsm, ag, rep.serial), presence=True)
    assert e.value.key == "golden"


def test_a_device_that_is_not_provisioned_has_nothing_to_return(tmp_path):
    hsm = vendor()[0]
    ag = D.DeviceAgent(tmp_path / "blank", hw_id=hsm.hw_id, vendor_pub=hsm.pub)
    with pytest.raises(SV.ServiceError) as e:
        SV.full_return(tmp_path / "blank", ag, {}, presence=True)
    assert e.value.key == "lifecycle"


def test_the_service_cant_mint_a_new_serial_or_a_new_device_key(tmp_path):
    d, hsm, rep, ag = shipped(tmp_path)
    with pytest.raises(D.DeviceError):
        ag.se.generate_dak()
    with pytest.raises(D.DeviceError):
        ag.otp.write_once("record", {})
    assert json.loads((d / "otp" / "otp.json").read_text())["_locked"] is True
