"""The two-bank update through the control gateway, on the real bridge core: stream, stage, apply with the button, prove, or come back."""
import hashlib
import shutil

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from dataopen.ctl import firmware as F
from dataopen.ctl import protocol as P
from dataopen.ctl.sim import World, seed_profile

from test_ctl_gateway import err, ok

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")

HW = b"DOHW0001"
SIGNER = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
PUB = SIGNER.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
EVIL = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))


def img(v, minv=None, payload=None, signer=SIGNER, hw=HW):
    return F.build_image(payload or (b"fw%d-" % v) * 700, hw, v, v if minv is None else minv, signer)


@pytest.fixture(scope="module")
def tremor_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("update-tremor")
    seed_profile(d, "tremor")
    return d


def world(tmp_path, tremor_dir=None, **kw) -> World:
    d = tmp_path / "gw"
    if tremor_dir is not None:
        shutil.copytree(tremor_dir, d)
    kw.setdefault("vendor_pub", PUB)
    kw.setdefault("factory_image", img(1, 1))
    kw.setdefault("fw_confirm_s", 3)
    return World(d, **kw)


def detail(r) -> str:
    assert r.type == P.T_ERR and r.json()["code"] == P.E.FW_REJECTED, r.body
    return r.json()["detail"]


def stage(w, v=2, **kw):
    r = w.phone.fw_send(img(v, **kw), piece=1500)
    assert r.type == P.T_ACK, r.body
    return r.json()


def apply_(w):
    w.gw.physical_press(w.t)
    return w.phone.act("fw.apply", True)


def reboot_into(w):
    """The restart the gateway asked for happens at the next tick; the phone has to say HELLO again afterwards (the link dropped)."""
    w.run(50)
    w.phone.connect()


# ---------------------------------------------------------------------------------------------------------------- the happy path
def test_an_update_is_streamed_into_the_other_bank_and_waits_for_the_person(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    assert ph.get_state()["fw.state"] == "current" and ph.state["fw.version"] == 1
    assert w.gw.read_info()[1] & 8                                                    # the capability bit
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    assert stage(w, 2)["staged"] == 2
    st = ph.get_state()
    assert st["fw.state"] == "staged" and st["fw.version"] == 1                       # still running the old one
    info = ph.get_firmware()
    assert info["supported"] and info["running"] == "A" and info["trial"] == "B" and info["upload"] is None
    assert info["versions"] == {"A": 1, "B": 2}
    w.run(1500)
    assert w.gw.settings.assist_wanted and w.bridge().state == 3 and w.bridge().params_rejected == 0     # nothing about running changed
    w.make_gateway()                                                                  # a power cut does NOT apply it
    assert w.gw.fw.running == "A" and w.gw.fw.trial == "B" and w.gw.fw.boots == 0
    w.phone.connect()
    assert w.phone.get_state()["fw.state"] == "staged"


def test_applying_needs_two_steps_and_the_button_then_the_new_version_runs_on_probation(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    stage(w, 2)
    assert err(ph.act("fw.apply", False)) == P.E.NOT_ALLOWED
    r = ph.act("fw.apply", True)
    assert err(r) == P.E.PHYSICAL and r.json()["detail"] == "fw.apply"
    assert w.reboots == []
    ok(apply_(w))
    assert w.reboots == ["update"]
    reboot_into(w)
    assert w.gw.fw.running == "B" and w.gw.fw.boots == 1
    st = w.phone.get_state()
    assert st["fw.state"] == "trial" and st["fw.version"] == 2


def test_the_new_version_confirms_itself_once_the_bridge_and_the_pc_are_fine(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    stage(w, 2)
    ok(apply_(w))
    reboot_into(w)
    assert w.gw.fw.trial == "B"
    w.run(1500)
    assert w.gw.fw.trial == "B"                                                       # not yet: it has to prove itself for a while
    w.run(5000)
    fw = w.gw.fw
    assert (fw.active, fw.trial, fw.floor, fw.version) == ("B", None, 2, 2)
    assert w.phone.get_state()["fw.state"] == "current"
    w.make_gateway()                                                                  # and it stays
    assert w.gw.fw.running == "B" and w.gw.fw.boots == 0


def test_a_version_that_never_proves_itself_is_replaced_by_the_old_one(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, fw_confirm_s=10**6)                                # it will never be confirmed
    w.phone.connect()
    stage(w, 2)
    ok(apply_(w))
    runs = []
    for _ in range(F.MAX_BOOTS):
        w.run(50)
        runs.append(w.gw.fw.running)
        w.power_cycle()                                                               # it keeps crashing
    assert runs == ["B"] * F.MAX_BOOTS
    w.power_cycle()
    fw = w.gw.fw
    assert fw.running == "A" and fw.trial is None and fw.slots["B"].bad and fw.version == 1
    w.phone.connect()
    assert w.phone.get_state()["fw.state"] == "current" and w.phone.state["fw.version"] == 1
    assert w.phone.fw_send(img(3, 1), piece=2000).type == P.T_ACK                     # a good update can follow a failed one


def test_an_update_does_not_get_confirmed_while_the_bridge_is_unhealthy(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, fw_confirm_s=2)
    w.phone.connect()
    stage(w, 2)
    ok(apply_(w))
    reboot_into(w)
    # the bridge goes to hardware bypass: not a state to judge a new image in
    ok(w.phone.hard_bypass())
    w.run(5000)
    assert w.bridge().state == 0 and w.gw.fw.trial == "B"


# ---------------------------------------------------------------------------------------------------------------- taking it back
def test_the_previous_version_can_be_taken_back_by_hand(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    stage(w, 2, minv=1)                                                               # an ordinary update: the old one is still allowed
    ok(apply_(w))
    reboot_into(w)
    w.run(6000)
    assert w.gw.fw.active == "B"
    r = w.phone.act("fw.rollback", True)
    assert err(r) == P.E.PHYSICAL and r.json()["detail"] == "fw.rollback"
    assert err(w.phone.act("fw.rollback", False)) == P.E.NOT_ALLOWED
    w.gw.physical_press(w.t)
    ok(w.phone.act("fw.rollback", True))
    assert w.reboots[-1] == "rollback"
    reboot_into(w)
    assert w.gw.fw.running == "A" and w.phone.get_state()["fw.version"] == 1


def test_the_floor_forbids_going_back_and_the_button_is_not_spent_on_it(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    # this one is a security fix: nothing older may come back
    stage(w, 3, minv=3)
    ok(apply_(w))
    reboot_into(w)
    w.run(6000)
    assert w.gw.fw.floor == 3
    w.gw.physical_press(w.t)
    assert detail(w.phone.act("fw.rollback", True)) == "rollback"
    # the press is still there for something that can happen
    assert w.gw.physical_until > w.t


def test_there_is_nothing_to_take_back_on_a_new_device(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    assert detail(w.phone.act("fw.rollback", True)) == "no_previous"
    assert detail(w.phone.act("fw.apply", True)) == "no_trial"


def test_taking_back_during_the_probation_drops_the_new_version(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir, fw_confirm_s=10**6)
    w.phone.connect()
    stage(w, 2)
    ok(apply_(w))
    reboot_into(w)
    w.gw.physical_press(w.t)
    ok(w.phone.act("fw.rollback", True))
    reboot_into(w)
    assert w.gw.fw.running == "A" and w.gw.fw.trial is None and w.gw.fw.slots["B"].bad


# ---------------------------------------------------------------------------------------------------------------- what is refused
def test_only_the_manufacturers_signature_for_this_hardware_and_a_newer_version(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    cases = [(img(2, signer=EVIL), "signature"), (img(2, hw=b"OTHER-HW"), "hardware"), (img(1), "same"),
             (b"not firmware at all", "damaged")]
    for raw, why in cases:
        assert detail(ph.fw_send(raw, piece=1500)) == why, why
    assert w.gw.fw.trial is None and w.gw.fw.slots["B"].image is None
    stage(w, 5, minv=5)
    ok(apply_(w))
    reboot_into(w)
    w.run(6000)
    assert detail(w.phone.fw_send(img(4, 4), piece=1500)) == "rollback"                # signed, genuine, and older than the floor
    flipped = bytearray(img(6, 6))
    flipped[len(flipped) // 2] ^= 1
    assert detail(w.phone.fw_send(bytes(flipped), piece=1500)) in ("damaged", "signature")


def test_a_second_update_waits_until_the_first_is_confirmed_or_gone(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    stage(w, 2)
    assert detail(w.phone.fw_send(img(3), piece=1500)) == "no_trial"


def test_the_stream_is_strict_about_order_size_and_hash(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    raw = img(2)
    chunk = lambda off, data: ph.call(P.T_FW_CHUNK, body=off.to_bytes(4, "little") + data)         # noqa: E731
    assert detail(chunk(0, raw[:10])) == "sequence"                                                  # no FW_BEGIN yet
    assert detail(ph.call(P.T_FW_END, body=b"")) == "sequence"
    assert detail(ph.call(P.T_FW_BEGIN, {"size": 10**9})) == "too_large"
    assert detail(ph.call(P.T_FW_BEGIN, {"size": "big"})) == "too_large"
    assert detail(ph.call(P.T_FW_BEGIN, {"size": len(raw), "sha256": "zz"})) == "damaged"
    r = ph.call(P.T_FW_BEGIN, {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    assert ok(r)["next"] == 0
    assert ok(chunk(0, raw[:1000]))["next"] == 1000
    assert ok(chunk(0, raw[:1000]))["next"] == 1000                                                  # a repeat is fine
    assert detail(chunk(2000, raw[2000:3000])) == "sequence"                                         # a gap is not
    assert detail(chunk(1000, b"x" * (len(raw) + 5))) == "too_large"
    assert detail(ph.call(P.T_FW_CHUNK, body=b"\x00\x00")) == "damaged"                              # not even an offset
    assert detail(ph.call(P.T_FW_END, body=b"")) == "sequence"                                       # incomplete
    assert ok(chunk(1000, raw[1000:]))["next"] == len(raw)
    assert ok(ph.call(P.T_FW_END, body=b""))["staged"] == 2


def test_a_wrong_announced_hash_drops_the_upload(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    raw = img(2)
    ok(ph.call(P.T_FW_BEGIN, {"size": len(raw), "sha256": "00" * 32}))
    ok(ph.call(P.T_FW_CHUNK, body=(0).to_bytes(4, "little") + raw))
    assert detail(ph.call(P.T_FW_END, body=b"")) == "damaged"
    assert ph.get_firmware()["upload"] is None and w.gw.fw.trial is None


def test_an_interrupted_upload_resumes_where_it_stopped_and_never_touches_the_running_bank(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    raw = img(2, payload=b"x" * 9000)
    before = {p.name: p.read_bytes() for p in w.gw.fw_dir.iterdir()}
    ok(ph.call(P.T_FW_BEGIN, {"size": len(raw)}))
    ok(ph.call(P.T_FW_CHUNK, body=(0).to_bytes(4, "little") + raw[:3000]))
    ph.disconnect()                                                                   # the phone walked away
    w.phone.connect()
    up = w.phone.get_firmware()["upload"]
    assert up == {"next": 3000, "size": len(raw)}                                     # the device knows where it was
    assert {p.name: p.read_bytes() for p in w.gw.fw_dir.iterdir()} == before          # nothing written to a bank for half an image
    ok(w.phone.call(P.T_FW_CHUNK, body=(3000).to_bytes(4, "little") + raw[3000:]))
    assert ok(w.phone.call(P.T_FW_END, body=b""))["staged"] == 2


def test_a_half_upload_does_not_survive_a_restart_or_a_long_silence(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    raw = img(2)
    ok(w.phone.call(P.T_FW_BEGIN, {"size": len(raw)}))
    ok(w.phone.call(P.T_FW_CHUNK, body=(0).to_bytes(4, "little") + raw[:100]))
    w.run(61_000)
    assert w.phone.get_firmware()["upload"] is None
    ok(w.phone.call(P.T_FW_BEGIN, {"size": len(raw)}))
    w.make_gateway()
    w.phone.connect()
    assert w.phone.get_firmware()["upload"] is None and w.gw.fw.trial is None


def test_a_device_without_the_manufacturers_key_takes_no_updates(tmp_path, tremor_dir):
    d = tmp_path / "gw"
    shutil.copytree(tremor_dir, d)
    w = World(d)
    ph = w.phone
    ph.connect()
    assert ph.get_state()["fw.state"] == "unsupported" and not w.gw.read_info()[1] & 8
    assert detail(ph.call(P.T_FW_BEGIN, {"size": 100})) == "unsupported"
    assert detail(ph.act("fw.apply", True)) == "unsupported"
    assert detail(ph.act("fw.rollback", True)) == "unsupported"
    assert ph.get_firmware() == {"supported": False}


def test_firmware_is_not_personal_so_erasing_personal_data_leaves_it_alone(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    stage(w, 2)
    w.gw.physical_press(w.t)
    ok(w.phone.act("erase.profile", True))
    assert w.gw.fw.trial == "B" and w.gw.fw.slots["B"].image.version == 2
    w.gw.physical_press(w.t)
    ok(w.phone.act("factory.reset", True))
    assert w.gw.fw.trial == "B"
