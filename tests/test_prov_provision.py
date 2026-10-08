"""The factory procedure end to end: the device, the manufacturer's HSM, the serial database, and the jig on the real bridge core."""
import json
import shutil

import pytest

from dataopen.ctl import firmware as FW
from dataopen.ctl.identity import Card
from dataopen.provisioning import device as D
from dataopen.provisioning import records as R
from dataopen.provisioning import station as ST
from prov_helpers import FastJig, provisioned, vendor

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


def agent(d, hsm):
    return D.DeviceAgent(d, hw_id=hsm.hw_id, vendor_pub=hsm.pub)


def test_one_board_from_blank_to_shipped(tmp_path):
    d, hsm, rep = provisioned(tmp_path)
    assert [s.name for s in rep.steps] == [k for k, _ in ST.STEPS] and all(s.ok for s in rep.steps)
    ag = agent(d, hsm)
    rec = ag.record
    assert rec.serial == rep.serial == rep.label["serial"] and rec.lifecycle == R.SHIPPED and rec.report_hash == rep.digest
    assert rec.att.verify(hsm.pub) and rec.att.dak == ag.se.dak_pub and rec.att.board == ag.board_id
    assert ag.otp.locked and rec.serial in hsm.shipped and ("attest", rec.serial, True) in hsm.log
    log = ag.otp.read("lifecycle_log")
    assert log == [R.DEV_TESTED, R.PROVISIONED, R.SHIPPED]
    cj = ag.owner().card().to_json()
    cj["device"] = ag.cert().to_json()
    assert R.verify_chain(Card.from_json(cj), hsm.pub, hsm.hw_id) == rec.serial
    assert json.loads(rep.label["qr"])["serial"] == rec.serial and rep.label["birth_id"] == ag.owner().id


def test_the_images_are_on_the_board_and_signed(tmp_path):
    d, hsm, rep = provisioned(tmp_path)
    mgr = FW.SlotManager.load(d / "fw", hsm.pub, hsm.hw_id)
    assert mgr.slots["A"].image.version == mgr.slots["B"].image.version == 1 and mgr.active == "A"
    for f in ("golden/golden.img", "golden/golden-mcu.img", "mcu/bank-A.img", "mcu/bank-B.img"):
        FW.verify_image((d / f).read_bytes(), hsm.pub, hsm.hw_id)


def test_no_secret_ever_leaves_the_device_in_the_request_or_the_report(tmp_path):
    d, hsm, rep = provisioned(tmp_path)
    ag = agent(d, hsm)
    seed = __import__("base64").b64decode(json.loads((d / "se" / "se.json").read_text())["dak"])
    blob = (json.dumps(rep.to_json()) + json.dumps(rep.label) + repr(hsm.log) + repr(hsm.serials) + repr(hsm.boards)).encode()
    assert seed not in blob and seed.hex().encode() not in blob
    assert ag.owner().storage_key.hex().encode() not in blob


def test_serials_are_unique_and_follow_the_sku(tmp_path):
    hsm, db = vendor()[0], ST.SerialDb()
    serials = []
    for i, sku in enumerate(("DO-1", "DO-1", "DO-3", "DO-4")):
        _, _, rep = provisioned(tmp_path, f"d{i}", sku, hsm=hsm, serial_db=db)
        serials.append(rep.serial)
    assert len(set(serials)) == 4 and [R.parse_serial(s)["sku"] for s in serials] == [1, 1, 3, 4]
    assert len({hsm.serials[s] for s in serials}) == 4                                 # four different device keys


# ---------------------------------------------------------------------------------------------------------------------------- the HSM
def request(tmp_path, name="x"):
    hsm = vendor()[0]
    ag = D.DeviceAgent(tmp_path / name, hw_id=hsm.hw_id, vendor_pub=hsm.pub)
    ag.mark_tested()
    return hsm, ag, ag.begin()


def test_the_hsm_wants_proof_of_possession(tmp_path):
    hsm, ag, req = request(tmp_path)
    serial = R.make_serial(1, 26, 41, 1)
    for field, mutate in (("pop", lambda v: v[:-4] + ("AAAA" if v[-4:] != "AAAA" else "BBBB")), ("nonce", lambda v: "00" * 16),
                          ("hw", lambda v: "00" * 8), ("board", lambda v: v[:-4] + ("AAAA" if v[-4:] != "AAAA" else "BBBB"))):
        bad = dict(req, **{field: mutate(req[field])})
        with pytest.raises(ST.HsmError) as e:
            hsm.attest(bad, serial, "2026-10-08")
        assert e.value.key == "bad_pop", field
    other = D.DeviceAgent(tmp_path / "y", hw_id=hsm.hw_id, vendor_pub=hsm.pub)
    other.mark_tested()
    r2 = other.begin()
    with pytest.raises(ST.HsmError):
        hsm.attest(dict(req, dak=r2["dak"]), serial, "2026-10-08")                      # somebody else's key under this device's proof
    assert hsm.serials == {}


def test_the_hsm_refuses_duplicates_and_bad_serials(tmp_path):
    hsm, ag, req = request(tmp_path)
    s1 = R.make_serial(1, 26, 41, 1)
    hsm.attest(req, s1, "2026-10-08")
    with pytest.raises(ST.HsmError) as e:
        # the same device key / board asking for a second serial
        hsm.attest(req, R.make_serial(1, 26, 41, 2), "2026-10-08")
    assert e.value.key in ("duplicate_dak", "duplicate_board")
    ag2 = D.DeviceAgent(tmp_path / "z", hw_id=hsm.hw_id, vendor_pub=hsm.pub)
    ag2.mark_tested()
    with pytest.raises(ST.HsmError) as e:
        hsm.attest(ag2.begin(), s1, "2026-10-08")                                         # a second device asking for an issued serial
    assert e.value.key == "duplicate_serial"
    with pytest.raises(ST.HsmError) as e:
        hsm.attest(ag2.begin(), "DO1-2641-00099-Z", "2026-10-08")
    assert e.value.key == "bad_serial"
    hsm.revoke(s1)
    assert ("revoke", s1, True) in hsm.log


def test_service_tokens_only_for_shipped_serials_and_once_per_challenge(tmp_path):
    d, hsm, rep = provisioned(tmp_path)
    n = bytes(16)
    with pytest.raises(ST.HsmError):
        hsm.issue_service_token(R.make_serial(1, 26, 41, 99), "factory_return", n)          # not a device this manufacturer shipped
    with pytest.raises(ST.HsmError):
        hsm.issue_service_token(rep.serial, "format_disk", n)
    t = hsm.issue_service_token(rep.serial, "factory_return", n)
    assert t.serial == rep.serial and t.sig
    with pytest.raises(ST.HsmError):
        hsm.issue_service_token(rep.serial, "factory_return", n)                            # the same challenge twice
    hsm.revoke(rep.serial)
    with pytest.raises(ST.HsmError):
        hsm.issue_service_token(rep.serial, "reprovision", b"\1" * 16)


# --------------------------------------------------------------------------------------------------------------- what goes wrong
def test_an_attestation_for_one_board_does_not_work_on_another(tmp_path):
    hsm = vendor()[0]
    a = D.DeviceAgent(tmp_path / "a", hw_id=hsm.hw_id, vendor_pub=hsm.pub)
    b = D.DeviceAgent(tmp_path / "b", hw_id=hsm.hw_id, vendor_pub=hsm.pub)
    for x in (a, b):
        x.mark_tested()
    att = hsm.attest(a.begin(), R.make_serial(1, 26, 41, 1), "2026-10-08")
    b.begin()
    with pytest.raises(D.DeviceError) as e:
        b.accept(att.to_json(), sku="DO-1", factory_fw=1)
    assert e.value.key == "binding" and b.record is None
    a.accept(att.to_json(), sku="DO-1", factory_fw=1)
    with pytest.raises(D.DeviceError) as e:
        a.accept(att.to_json(), sku="DO-1", factory_fw=1)                                 # and only once
    assert e.value.key == "lifecycle"


def test_an_attestation_from_the_wrong_manufacturer_is_refused(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from dataopen.ctl.sim import dev_vendor
    _, pub = dev_vendor()
    evil = ST.VendorHsm(ed25519.Ed25519PrivateKey.generate())
    ag = D.DeviceAgent(tmp_path / "a", hw_id=evil.hw_id, vendor_pub=pub)
    ag.mark_tested()
    att = evil.attest(ag.begin(), R.make_serial(1, 26, 41, 1), "2026-10-08")
    with pytest.raises(D.DeviceError) as e:
        ag.accept(att.to_json(), sku="DO-1", factory_fw=1)
    assert e.value.key == "signature" and ag.record is None


def test_a_board_that_fails_its_tests_gets_no_serial_and_no_lock(tmp_path):
    class Bad(FastJig):
        def run(self):
            out = super().run()
            out[3] = ST.Check("panic_soft", False, "x", "y")
            return out
    hsm = vendor()[0]
    rep = ST.provision(tmp_path / "d", hsm, jig=Bad())
    assert rep.quarantined and rep.failed_step == "selftest" and not rep.serial and not rep.label
    assert hsm.serials == {} and not (tmp_path / "d" / "otp" / "otp.json").exists() or agent(tmp_path / "d", hsm).record is None
    assert not agent(tmp_path / "d", hsm).otp.locked


def test_an_unsigned_image_never_reaches_a_board(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    hsm = vendor()[0]
    img = ST.Images.dev()
    evil = FW.build_image(b"x" * 400, hsm.hw_id, 1, 1, ed25519.Ed25519PrivateKey.generate())
    rep = ST.provision(tmp_path / "d", hsm, jig=FastJig(), images=ST.Images(evil, img.mcu, img.golden))
    assert rep.quarantined and rep.failed_step == "flash" and hsm.serials == {}


def test_failing_after_the_attestation_revokes_the_serial(tmp_path, monkeypatch):
    hsm = vendor()[0]

    def boom(self, *a, **k):
        raise D.DeviceError("binding", "injected")
    monkeypatch.setattr(D.DeviceAgent, "accept", boom)
    rep = ST.provision(tmp_path / "d", hsm, jig=FastJig())
    assert rep.quarantined and rep.failed_step == "record"
    assert len(hsm.revoked) == 1 and not hsm.shipped and ("revoke", next(iter(hsm.revoked)), True) in hsm.log


def test_provisioning_the_same_board_twice_is_refused(tmp_path):
    d, hsm, rep = provisioned(tmp_path)
    before = {p.name: p.read_bytes() for p in (d / "fw").iterdir()}
    again = ST.provision(d, hsm, jig=FastJig())
    assert again.quarantined and again.failed_step == "flash" and hsm.serials == {rep.serial: hsm.serials[rep.serial]}
    assert {p.name: p.read_bytes() for p in (d / "fw").iterdir()} == before                  # and nothing was written to the board


def test_after_the_last_step_nothing_can_be_written_to_the_otp(tmp_path):
    d, hsm, rep = provisioned(tmp_path)
    ag = agent(d, hsm)
    with pytest.raises(D.DeviceError) as e:
        ag.otp.write_once("serial2", "DO1-0000-00000-0")
    assert e.value.key == "locked"
    with pytest.raises(D.DeviceError):
        ag.se.generate_dak()                                                              # the device key can not be replaced
    with pytest.raises(D.DeviceError):
        ag.finalize("0" * 64)
    ag.first_boot()
    # the lifecycle log is the one thing that still grows
    assert ag.lifecycle == R.IN_FIELD and ag.otp.read("lifecycle_log")[-1] == R.IN_FIELD


# -------------------------------------------------------------------------------------------------------- the secure element and the OTP
def test_the_secure_element_signs_only_what_it_is_meant_to(tmp_path):
    se = D.SecureElementSim(tmp_path / "d")
    assert not se.has_dak and se.dak_pub is None
    pub = se.generate_dak()
    assert se.dak_pub == pub
    for domain in (R.ATT_DOMAIN, R.SVC_DOMAIN, b"anything"):
        with pytest.raises(D.DeviceError) as e:
            se.sign(domain + b"x")
        assert e.value.key == "domain"
    sig = se.sign(R.DEV_DOMAIN + b"payload")
    assert len(sig) == 64
    public = [n for n in dir(se) if not n.startswith("_")]
    # nothing that returns the secret is part of the interface
    assert public == sorted(["dak_pub", "generate_dak", "has_dak", "sign", "SIGNS"])
    seed = __import__("base64").b64decode(json.loads((tmp_path / "d" / "se" / "se.json").read_text())["dak"])
    assert oct((tmp_path / "d" / "se" / "se.json").stat().st_mode & 0o777) == oct(0o600) and seed not in repr(vars(se)).encode()


def test_the_otp_is_write_once_monotonic_and_lockable(tmp_path):
    o = D.OtpSim(tmp_path)
    o.write_once("a", 1)
    with pytest.raises(D.DeviceError):
        o.write_once("a", 2)
    o.counter("floor", 3)
    o.counter("floor", 5)
    with pytest.raises(D.DeviceError):
        o.counter("floor", 4)
    o.append("log", "x")
    o.lock()
    assert o.locked and o.read("a") == 1 and o.read("floor") == 5
    with pytest.raises(D.DeviceError):
        o.write_once("b", 1)
    # the anti-rollback floor still rises in the field
    o.counter("floor", 6)
    o.append("log", "y")
    assert o.read("log") == ["x", "y"]


# ------------------------------------------------------------------------------------------------------ the jig on the real bridge core
@needs_cc
def test_a_healthy_board_passes_every_check_on_the_real_core(tmp_path):
    checks = ST.Jig().run()
    assert {c.name for c in checks} == set(ST.CHECK_DOCS)
    assert all(c.ok for c in checks), [(c.name, c.measured) for c in checks if not c.ok]
    by = {c.name: c.measured for c in checks}
    assert by["power_cut_bypass"] <= ST.LIMITS["power_cut_ms"] and 2900 <= by["panic_long"] <= 3500 and by["watchdog_hang"] <= 130


FAULT_CHECKS = {"relay_stuck": {"panic_long", "panic_loop_hardware", "watchdog_hang", "mode_switch", "power_cut_bypass"},
                "watchdog_dead": {"watchdog_hang"}, "panic_stuck": {"panic_loop_hardware"},
                "mouse_dead": {"mouse_through_bridge", "panic_long", "watchdog_hang"},
                "ble_silent": {"ble_visible"}, "ble_weak": {"ble_visible"}, "video_dead": {"video_direct"},
                "led_dead": {"indication"}, "rail_low": {"rails"},
                "se_dead": {"secure_element"}}


@needs_cc
@pytest.mark.parametrize("fault", sorted(FAULT_CHECKS))
def test_every_injected_fault_is_caught_by_the_check_meant_for_it(fault):
    failed = {c.name for c in ST.Jig(ST.BoardFaults(**{fault: True})).run() if not c.ok}
    assert FAULT_CHECKS[fault] & failed, (fault, failed)


@needs_cc
def test_a_slow_relay_is_caught_by_the_timing_limits():
    failed = {c.name for c in ST.Jig(ST.BoardFaults(relay_slow_ms=60)).run() if not c.ok}
    assert {"mode_switch", "power_cut_bypass"} <= failed


@needs_cc
def test_provisioning_with_the_real_jig_and_with_a_faulty_board(tmp_path):
    hsm = vendor()[0]
    ok = ST.provision(tmp_path / "ok", hsm, jig=ST.Jig())
    assert not ok.quarantined and len(ok.checks) == len(ST.CHECK_DOCS)
    bad = ST.provision(tmp_path / "bad", hsm, jig=ST.Jig(ST.BoardFaults(relay_stuck=True)))
    assert bad.quarantined and bad.failed_step == "selftest" and "power_cut_bypass" in bad.steps[-1].detail and not bad.serial
    assert len(hsm.serials) == 1
