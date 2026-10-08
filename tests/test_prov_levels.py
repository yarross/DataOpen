"""The four levels of reset against the REAL gateway: what each erases, what it never touches, and the device identity through all of it."""
import shutil

import pytest

from dataopen.bioprofile.profile import ProfileState
from dataopen.ctl import protocol as P
from dataopen.ctl.identity import Card
from dataopen.ctl.sim import World, seed_profile
from dataopen.provisioning import records as R
from dataopen.provisioning import station as ST
from dataopen.updates import package as K
from prov_helpers import provisioned, vendor
import pkg_helpers as H

from test_ctl_gateway import err, ok

MODEL = H.tiny_model()
CARD = H.card_of(MODEL, name="icons")

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


@pytest.fixture(scope="module")
def state(tmp_path_factory):
    return seed_profile(tmp_path_factory.mktemp("lv-profile"), "tremor")._state


def device_world(tmp_path, state, name="dev", hsm=None, **kw):
    hsm = hsm or vendor()[0]
    d, hsm, rep = provisioned(tmp_path, name, hsm=hsm)
    forgets = []
    w = World(d, vendor_pub=hsm.pub, hw_id=hsm.hw_id, on_forget=lambda: forgets.append(1), **kw)
    w.forgets = forgets
    w.hsm, w.rep = hsm, rep
    return w


def fill(w, state):
    """A device in use: two slots with profiles (slot 2 active), a trusted sender, assistance on, custom levels."""
    ph = w.phone
    ph.connect()
    now = w.t
    w.gw._install_profile(ProfileState.unpack(state.pack()), now, 0)
    w.gw._install_profile(ProfileState.unpack(state.pack()), now, 2)
    ok(ph.select_slot(2))
    ok(ph.set("slot.name", "Браузер"))
    ok(ph.set("assist.strength", 8))
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    w.gw.trust["senders"]["ab" * 8] = {"id": "ABCD-EFGH", "last_seq": 4}
    w.gw._save_trust()
    me = w.gw.identity
    for k in (0, 2):                                        # a model in the active slot and in another one
        w.phone.pkg_send(K.build_package(me, me.card(), me.next_seq(), slot=k, model=(MODEL, CARD)))
        ok(ph.act("pkg.apply", True))
    w.phone.pkg_send(K.build_package(me, me.card(), me.next_seq(), slot=1, tuning=(3, 3)))      # and one more, still waiting
    return ph


def snapshot(w):
    d = w.dir
    fw = {p.name: p.read_bytes() for p in (d / "fw").iterdir() if p.name.endswith(".img")}
    mcu = {p.name: p.read_bytes() for p in (d / "mcu").iterdir()}
    return {"id": w.gw.identity.id, "storage": w.gw.identity.storage_key, "serial": w.gw.device_serial, "dak": w.gw.device.se.dak_pub,
            "floor": w.gw.fw.floor, "golden": (d / "golden" / "golden.img").read_bytes(), "fw": fw, "mcu": mcu,
            "att": w.gw.device.record.att.to_json(), "lifecycle": w.gw.device.lifecycle}


def observe(w, before, active_before=2):
    gw, after = w.gw, snapshot(w)
    s = gw.settings
    settings = "erased" if (gw.active == 0 and s == s.__class__(False, 5, 5)) else "assist off" if not s.assist_wanted else "kept"
    return {"serial": "kept" if after["serial"] == before["serial"] and after["att"] == before["att"] else "changed",
            "dak": "kept" if after["dak"] == before["dak"] else "changed",
            "rollback_floor": "kept" if after["floor"] >= before["floor"] else "lowered",
            "lifecycle": "kept" if R.LIFECYCLE.index(after["lifecycle"]) >= R.LIFECYCLE.index(before["lifecycle"]) else "lowered",
            "golden": "kept" if after["golden"] == before["golden"] else "changed",
            "owner_keys": "kept" if after["id"] == before["id"] else "replaced",
            "storage_key": "kept" if after["storage"] == before["storage"] else "replaced",
            "slot_active": "kept" if gw.slotset[active_before].has and (gw.slotset[active_before].dir / "model.bin").exists() else "erased",
            "slots_other": "kept" if gw.slotset[0].has and (gw.slotset[0].dir / "model.bin").exists() else "erased",
            "pkg_pending": "kept" if gw.pkg.summary() is not None and (gw.pkg.dir / "pending.dopk").exists() else "erased",
            "settings": settings, "trust": "kept" if gw.trust["senders"] else "erased",
            "bonds": "erased" if w.forgets else "kept",
            "fw_som": "kept" if after["fw"] == before["fw"] else "changed",
            "fw_mcu": "kept" if after["mcu"] == before["mcu"] else "changed"}


def apply_level(w, level):
    ph = w.phone
    w.gw.physical_press(w.t)
    key = {1: "slot.clear", 2: "erase.profile", 3: "factory.reset"}[level]
    ok(ph.act(key, True))


@pytest.mark.parametrize("level", [1, 2, 3])
def test_each_level_does_exactly_what_the_matrix_says(tmp_path, state, level):
    w = device_world(tmp_path, state)
    fill(w, state)
    before = snapshot(w)
    apply_level(w, level)
    seen = observe(w, before)
    for it in R.ITEMS:
        if it.key in seen:
            expect = it.levels[level - 1]
            words = {"replaced": "replaced", "erased": "erased", "kept": "kept", "assist off": "assist off"}
            assert seen[it.key] == words[expect], (level, it.key, seen[it.key], expect)


def test_the_serial_the_device_key_and_the_floor_survive_every_level_and_a_restart(tmp_path, state):
    w = device_world(tmp_path, state)
    fill(w, state)
    before = snapshot(w)
    for level in (1, 2, 3):
        apply_level(w, level)
        w.make_gateway()
        w.phone.connect()
        after = snapshot(w)
        keys = ("serial", "dak", "att", "floor")
        assert tuple(after[k] for k in keys) == tuple(before[k] for k in keys)
        assert after["fw"] == before["fw"] and after["golden"] == before["golden"]
        assert w.phone.get_state()["device.serial"] == before["serial"]


def test_the_gateway_shows_the_serial_and_a_card_with_the_manufacturers_chain(tmp_path, state):
    w = device_world(tmp_path, state)
    ph = w.phone
    ph.connect()
    assert ph.get_state()["device.serial"] == w.rep.serial and w.gw.device.lifecycle == R.IN_FIELD          # the first boot in the field
    card = ph.get_identity()
    assert R.verify_chain(Card.from_json(card), w.hsm.pub, w.hsm.hw_id) == w.rep.serial
    assert "device" in card and card["id"] == ph.get_state()["device.id"]


def test_a_device_that_was_not_provisioned_has_no_serial_and_a_plain_card(tmp_path):
    w = World(tmp_path / "gw")
    w.phone.connect()
    assert w.phone.get_state()["device.serial"] == "" and "device" not in w.phone.get_identity() and w.gw.device is None


def test_a_factory_reset_gives_a_new_owner_number_and_the_same_device(tmp_path, state):
    w = device_world(tmp_path, state)
    ph = fill(w, state)
    copy = ph.get_bundle("self")
    old_id = w.gw.identity.id
    apply_level(w, 3)
    assert w.gw.identity.id != old_id and w.forgets == [1]                                     # new owner, phones forgotten
    card = w.gw.identity.card()
    # the same device vouches for the new owner
    assert R.verify_chain(Card.from_json(card.to_json()), w.hsm.pub, w.hsm.hw_id) == w.rep.serial
    assert err(ph.put_bundle(copy)) == P.E.WRONG_DEVICE                                         # and an old backup copy is dead
    assert w.gw.slotset.mask() == 0 and not w.gw.settings.assist_wanted


def test_clearing_the_profiles_keeps_the_owner_and_the_phones(tmp_path, state):
    w = device_world(tmp_path, state)
    ph = fill(w, state)
    copy = ph.get_bundle("self")
    apply_level(w, 2)
    assert w.forgets == [] and w.gw.slotset.mask() == 0
    # the owner's own copy still opens: the profile comes back
    assert ok(ph.put_bundle(copy))["ok"]


# ------------------------------------------------------------------------------------------------------------- attested cards
def other_card(tmp_path, hsm, name="other", signer=None):
    d, h, rep = provisioned(tmp_path, name, hsm=hsm, **({"images": ST.Images.dev(signer=signer)} if signer else {}))
    ag = ST.DeviceAgent(d, hw_id=hsm.hw_id, vendor_pub=hsm.pub)
    cj = ag.owner().card().to_json()
    cj["device"] = ag.cert().to_json()
    return cj


def test_a_device_that_requires_attestation_makes_files_only_for_genuine_cards(tmp_path, state):
    hsm = vendor()[0]
    w = device_world(tmp_path, state, hsm=hsm, require_attested=True)
    fill(w, state)
    ph = w.phone
    genuine = other_card(tmp_path, hsm)
    w.gw.physical_press(w.t)
    # a genuine card of this manufacturer: allowed (with the button)
    assert ph.get_bundle(genuine)[:4] == b"DOBS"
    plain = ST.DeviceAgent(tmp_path / "plain", hw_id=hsm.hw_id, vendor_pub=hsm.pub).owner().card().to_json()
    w.gw.physical_press(w.t)
    r = ph.try_bundle(plain)
    # refused, and the press is not spent
    assert err(r) == P.E.BAD_BUNDLE and r.json()["detail"] == "card_unattested" and w.gw.physical_until > w.t
    from cryptography.hazmat.primitives.asymmetric import ed25519
    evil = ed25519.Ed25519PrivateKey.generate()
    forged = other_card(tmp_path, ST.VendorHsm(evil), "forged", signer=evil)                     # a real device of ANOTHER manufacturer
    r = ph.try_bundle(forged)
    assert err(r) == P.E.BAD_BUNDLE and r.json()["detail"] == "card_unattested"
    # a genuine certificate on somebody else's keys
    stolen = dict(plain, device=genuine["device"])
    r = ph.try_bundle(stolen)
    assert err(r) == P.E.BAD_BUNDLE and r.json()["detail"] == "card_unattested"


def test_by_default_cards_without_attestation_still_work(tmp_path, state):
    hsm = vendor()[0]
    w = device_world(tmp_path, state, hsm=hsm)
    ph = fill(w, state)
    plain = ST.DeviceAgent(tmp_path / "plain", hw_id=hsm.hw_id, vendor_pub=hsm.pub).owner().card().to_json()
    assert err(ph.try_bundle(plain)) == P.E.PHYSICAL                                              # as before: only the button is missing
