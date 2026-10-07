"""The A/B firmware update rules (a simulation of what a bootloader and updater must do)."""
import struct

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from dataopen.ctl import firmware as F

HW = b"DO-RK35"


def keys():
    k = ed25519.Ed25519PrivateKey.generate()
    return k, k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


@pytest.fixture()
def rig():
    signer, pub = keys()
    mgr = F.SlotManager(pub, HW)
    mgr.install_factory(F.build_image(b"factory" * 100, HW, 1, 1, signer))
    return signer, pub, mgr


def img(signer, v, minv=None, payload=None, hw=HW):
    return F.build_image(payload or (b"fw%d" % v) * 200, hw, v, v if minv is None else minv, signer)


def test_an_update_goes_to_the_other_bank_and_the_running_one_is_untouched(rig):
    signer, _, m = rig
    before = m.slots["A"].image
    m.stage(img(signer, 2))
    assert m.active == "A" and m.slots["A"].image is before and m.slots["B"].image.version == 2 and m.trial == "B"
    assert m.reboot() == "B"
    m.confirm()
    assert (m.active, m.version, m.trial) == ("B", 2, None) and m.slots["A"].image.version == 1       # the old one stays as the fallback


def test_an_unconfirmed_update_rolls_back_by_itself(rig):
    signer, _, m = rig
    m.stage(img(signer, 2))
    assert [m.reboot() for _ in range(F.MAX_BOOTS)] == ["B"] * F.MAX_BOOTS
    assert m.reboot() == "A" and m.trial is None and m.slots["B"].bad and m.version == 1
    assert m.reboot() == "A"
    m.stage(img(signer, 3))                                                    # a good update can follow a failed one
    assert m.reboot() == "B"
    m.confirm()
    assert m.version == 3


def test_a_confirmed_update_survives_reboots(rig):
    signer, _, m = rig
    m.stage(img(signer, 2))
    m.reboot()
    m.confirm()
    assert [m.reboot() for _ in range(10)] == ["B"] * 10


def test_only_the_manufacturers_signature_is_accepted(rig):
    signer, pub, m = rig
    other, _ = keys()
    with pytest.raises(F.FirmwareError) as e:
        m.stage(img(other, 2))
    assert e.value.key == "signature" and m.trial is None and m.slots["B"].image is None


def test_every_single_bit_of_an_image_is_protected(rig):
    signer, _, m = rig
    raw = img(signer, 2, payload=b"x" * 40)
    seen = set()
    for i in range(len(raw) * 8):
        b = bytearray(raw)
        b[i // 8] ^= 1 << (i % 8)
        with pytest.raises(F.FirmwareError) as e:
            F.verify_image(bytes(b), rig[1], HW)
        seen.add(e.value.key)
    assert seen <= {"damaged", "signature", "hardware"}
    F.verify_image(raw, rig[1], HW)


def test_truncated_and_junk_images_are_refused(rig):
    signer, pub, m = rig
    raw = img(signer, 2)
    for n in (0, 3, F.HEAD, F.HEAD + F.SIG - 1, len(raw) - 1):
        with pytest.raises(F.FirmwareError):
            F.verify_image(raw[:n], pub, HW)
    with pytest.raises(F.FirmwareError):
        F.verify_image(raw + b"\0", pub, HW)
    with pytest.raises(F.FirmwareError):
        F.verify_image(b"DOFW" + bytes(300), pub, HW)


def test_an_image_for_other_hardware_is_refused(rig):
    signer, _, m = rig
    with pytest.raises(F.FirmwareError) as e:
        m.stage(img(signer, 2, hw=b"OTHER"))
    assert e.value.key == "hardware"


def test_anti_rollback_a_validly_signed_old_image_cannot_come_back(rig):
    signer, _, m = rig
    m.stage(img(signer, 5, minv=4))
    m.reboot()
    m.confirm()
    assert m.floor == 4
    for v in (1, 2, 3):
        with pytest.raises(F.FirmwareError) as e:
            m.stage(img(signer, v))
        assert e.value.key == "rollback"
    m.stage(img(signer, 4, minv=4))                                            # at the floor is fine
    m.reboot()
    m.confirm()
    assert m.version == 4


def test_the_floor_only_rises_when_an_update_is_confirmed_not_when_it_is_staged(rig):
    signer, _, m = rig
    m.stage(img(signer, 5, minv=5))
    assert m.floor == 1
    for _ in range(F.MAX_BOOTS + 1):
        m.reboot()                                                              # never confirmed: rolled back, floor unchanged
    assert m.floor == 1 and m.version == 1
    m.stage(img(signer, 2))
    m.reboot()
    m.confirm()
    assert m.floor == 2


def test_the_same_version_and_oversized_images_are_refused(rig):
    signer, _, m = rig
    with pytest.raises(F.FirmwareError) as e:
        m.stage(img(signer, 1))
    assert e.value.key == "same"
    m.slot_size = 100
    with pytest.raises(F.FirmwareError) as e:
        m.stage(img(signer, 2))
    assert e.value.key == "too_large"


def test_power_loss_during_the_write_leaves_the_running_firmware_booting(rig):
    signer, _, m = rig
    raw = img(signer, 2)
    for cut in (0, 10, F.HEAD, len(raw) // 2, len(raw) - 1):
        with pytest.raises(F.WriteInterrupted):
            m.stage(raw, crash_after=cut)
        assert m.reboot() == "A" and m.version == 1 and m.trial is None and m.slots["B"].image is None
    m.stage(raw)                                                               # and the update can simply be tried again
    assert m.reboot() == "B"


def test_a_second_update_waits_for_the_first_to_be_confirmed(rig):
    signer, _, m = rig
    m.stage(img(signer, 2))
    with pytest.raises(F.FirmwareError) as e:
        m.stage(img(signer, 3))
    assert e.value.key == "no_trial"
    with pytest.raises(F.FirmwareError):
        F.SlotManager(m.vendor_pub, HW).confirm()


def test_a_factory_image_must_be_genuine_too(rig):
    signer, pub, _ = rig
    other, _ = keys()
    with pytest.raises(F.FirmwareError):
        F.SlotManager(pub, HW).install_factory(img(other, 1))


def test_state_can_be_written_down_and_has_no_secrets(rig):
    signer, _, m = rig
    m.stage(img(signer, 2))
    d = m.to_dict()
    assert d["trial"] == "B" and d["slots"]["B"]["image"]["v"] == 2 and "payload" not in str(d)
    assert struct.calcsize(F.HEAD_FMT) == F.HEAD == 60
