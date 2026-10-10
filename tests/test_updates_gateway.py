"""Channel B through the control gateway, on the real bridge core: receive, keep aside, apply, take back, and every way it fails."""
import shutil

import pytest

from dataopen.ctl import protocol as P
from dataopen.ctl.sim import World, seed_profile
from dataopen.updates import channels as C
from dataopen.updates import package as K

import pkg_helpers as H
from test_ctl_gateway import ok

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")

MODEL = H.tiny_model()
CARD = H.card_of(MODEL, name="icons", version=1)
MODEL2 = H.tiny_model(opset=11)
CARD2 = H.card_of(MODEL2, name="icons", version=2, opset=11)


@pytest.fixture(scope="module")
def tremor_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("upd-tremor")
    seed_profile(d, "tremor")
    return d


def world(tmp_path, tremor_dir=None, **kw) -> World:
    d = tmp_path / "gw"
    if tremor_dir is not None:
        shutil.copytree(tremor_dir, d)
    w = World(d, **kw)
    w.phone.connect()
    return w


def detail(r, code=P.E.PKG_REJECTED) -> str:
    assert r.type == P.T_ERR and r.json()["code"] == code, r.body
    return r.json()["detail"]


def pending(w):
    return w.phone.get_packages()["pending"]


def give(w, tmp_path, **kw):
    """A package goes in and is applied, from a sender who is known (the first time the button makes them known; weights always need it).
    The device makes no packages of its own (docs/RESIDENCY.md), so there is no 'a copy for oneself'."""
    giver = H.sender(tmp_path, "giver")
    seq = w.gw.trust["senders"].get(giver.digest.hex(), {}).get("last_seq", 0) + 1
    r = w.phone.pkg_send(K.build_package(giver, w.gw.identity.card(), seq, **kw))
    assert r.type == P.T_ACK, r.body
    return ok(apply_(w, button=True))


def apply_(w, button=False):
    if button:
        w.gw.physical_press(w.t)
    return w.phone.act("pkg.apply", True)


def pkg_dir(w):
    return w.gw.dir / "pkg"


def leftovers(w) -> list:
    return sorted(p.name for p in pkg_dir(w).glob("*")) if pkg_dir(w).exists() else []


# ---------------------------------------------------------------------------------------------------------------- the normal way
def test_a_package_is_checked_and_kept_aside_and_changes_nothing_until_applied(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    before = (w.gw.slotset[2].name, w.gw.slotset[2].strength, w.gw.settings)
    raw = K.build_package(alice, w.gw.identity.card(), 1, slot=2, tuning=(7, 3), name="Браузер", model=(MODEL, CARD))
    r = w.phone.pkg_send(raw, piece=1000)
    assert r.type == P.T_ACK
    sm = r.json()["pending"]
    assert sm["kinds"] == ["tuning", "meta", "model"] and sm["slot"] == 2 and sm["from"] == alice.id
    assert not sm["self"] and sm["button"] == "trust"
    st = w.phone.get_state()
    assert st["pkg.state"] == "pending" and st["pkg.from"] == alice.id and st["pkg.kinds"] == "tuning,meta,model"
    assert (w.gw.slotset[2].name, w.gw.slotset[2].strength, w.gw.settings) == before            # received is not applied
    assert pending(w)["id"] == sm["id"] and w.phone.get_packages()["upload"] is None
    assert not (w.gw.slotset[2].dir / "model.bin").exists()


def test_the_first_package_from_a_stranger_needs_the_button_and_makes_them_known(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 1, slot=2, tuning=(7, 3), name="Браузер"))
    r = apply_(w)
    assert r.type == P.T_ERR and r.json()["code"] == P.E.PHYSICAL and r.json()["detail"] == f"trust:{alice.id}"
    assert pending(w) is not None and w.gw.slotset[2].name == ""                              # a refusal spends nothing and applies nothing
    ok(apply_(w, button=True))
    sl = w.gw.slotset[2]
    assert sl.name == "Браузер" and (sl.strength, sl.tremor) == (7, 3) and not sl.vetted
    assert pending(w) is None and w.phone.get_state()["pkg.state"] == "none" and leftovers(w) == []
    assert w.gw.trust["senders"][alice.digest.hex()]["last_seq"] == 1 and w.phone.state["trusted.count"] == 1


def test_a_known_sender_needs_no_button_for_a_profile_but_does_for_a_model(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 1, slot=1, tuning=(4, 4)))
    ok(apply_(w, button=True))
    w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 2, slot=1, tuning=(5, 5), name="Работа"))
    assert pending(w)["button"] == ""
    ok(apply_(w))                                                                           # no press: a known sender, no weights
    assert w.gw.slotset[1].name == "Работа" and w.gw.slotset[1].strength == 5
    w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 3, slot=1, model=(MODEL, CARD)))
    assert pending(w)["button"] == "model"
    r = apply_(w)
    assert r.json()["code"] == P.E.PHYSICAL and r.json()["detail"] == f"model:{alice.id}"
    assert not (w.gw.slotset[1].dir / "model.bin").exists()
    ok(apply_(w, button=True))
    # slot 1 is not the active one
    assert w.gw.slotset[1].dir.joinpath("model.bin").exists() and w.phone.get_state()["model.state"] == "none"
    ok(w.phone.select_slot(1))
    assert w.phone.get_state()["model.state"] == "ok" and w.phone.state["model.name"] == "icons"


def test_a_model_needs_the_button_every_time_and_the_old_model_is_kept(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    give(w, tmp_path, slot=0, model=(MODEL, CARD))
    assert w.phone.get_state()["model.state"] == "ok" and w.phone.state["model.version"] == 1
    giver = H.sender(tmp_path, "giver")
    w.phone.pkg_send(K.build_package(giver, w.gw.identity.card(), 2, slot=0, model=(MODEL2, CARD2)))
    assert pending(w)["button"] == "model" and not pending(w)["self"]
    assert apply_(w).json()["code"] == P.E.PHYSICAL                                    # a known sender, but weights always need the press
    ok(apply_(w, button=True))
    assert w.phone.get_state()["model.version"] == 2
    info = w.phone.get_packages()["models"][0]
    assert info["model"]["version"] == 2 and info["previous"] and info["state"] == "ok"
    ok(w.phone.act("pkg.revert", True))                                                     # swaps current and previous
    assert w.phone.get_state()["model.version"] == 1
    ok(w.phone.act("pkg.revert", True))
    assert w.phone.get_state()["model.version"] == 2


def test_a_package_the_device_itself_seems_to_have_made_is_refused_with_the_first_pieces(tmp_path, tremor_dir):
    """The device makes no files; one sealed with its own keys can only come from stolen keys or a build that no longer exists."""
    w = world(tmp_path, tremor_dir)
    me = w.gw.identity
    raw = K.build_package(me, me.card(), me.next_seq(), slot=0, tuning=(8, 8))
    assert detail(w.phone.pkg_send(raw)) == "own_file" and leftovers(w) == []
    w.gw.physical_press(w.t)
    assert detail(w.phone.pkg_send(raw)) == "own_file" and w.gw.slotset[0].strength == 5


def test_nothing_to_revert_is_said_plainly(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    assert detail(w.phone.act("pkg.revert", True)) == "no_previous"
    assert detail(w.phone.act("pkg.apply", True)) == "no_pending"
    ok(w.phone.act("pkg.discard", False))                                                   # nothing to drop is not an error


def test_the_weights_are_encrypted_at_rest_under_the_slots_key_and_die_with_the_slot(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    give(w, tmp_path, slot=0, model=(MODEL, CARD))
    d = w.gw.slotset[0].dir
    assert (d / "model.bin").exists() and MODEL[:30] not in (d / "model.bin").read_bytes()
    assert b"icons" not in (d / "model.json").read_bytes()
    assert w.gw.slotset[0].vault.read(d / "model.bin", "model.bin", allow_plain=False) == MODEL
    assert MODEL[:30] not in w.gw.slotset[1].vault.seal("x", b"") and not w.gw.slotset[1].dir.joinpath("model.bin").exists()
    w.gw.physical_press(w.t)
    ok(w.phone.act("slot.clear", True))
    assert w.phone.get_state()["model.state"] == "none" and not (w.gw.slotset[0].dir / "model.bin").exists()


def test_erasing_personal_data_takes_a_waiting_package_and_every_model_with_it(tmp_path, tremor_dir):
    for key in ("erase.profile", "factory.reset"):
        w = world(tmp_path / key, tremor_dir)
        me, alice = w.gw.identity, H.sender(tmp_path / key)
        give(w, tmp_path / key, slot=0, model=(MODEL, CARD))
        w.phone.pkg_send(K.build_package(alice, me.card(), 1, slot=1, tuning=(2, 2)))
        assert pending(w) is not None and (pkg_dir(w) / "pending.dopk").exists()
        w.gw.physical_press(w.t)
        ok(w.phone.act(key, True))
        st = w.phone.get_state()
        assert st["pkg.state"] == "none" and st["model.state"] == "none" and not pkg_dir(w).exists()
        assert not list((w.gw.dir / "slots").rglob("model.*"))
        assert w.phone.get_packages()["pending"] is None


# ---------------------------------------------------------------------------------------------------------------- refusals, early
def test_a_file_for_another_device_is_refused_with_the_first_pieces_and_nothing_is_kept(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice, other = H.sender(tmp_path), H.sender(tmp_path, "other")
    raw = K.build_package(alice, other.card(), 1, model=(MODEL, CARD))
    assert w.phone.call(P.T_PKG_BEGIN, {"size": len(raw)}).type == P.T_ACK
    r = w.phone.call(P.T_PKG_CHUNK, body=(0).to_bytes(4, "little") + raw[: K.PREFIX + 10])
    assert detail(r) == "wrong_device"
    assert w.phone.get_packages()["upload"] is None and leftovers(w) == [] and w.phone.get_state()["pkg.state"] == "none"


def test_a_changed_header_a_replay_and_a_package_from_the_future_are_refused_at_the_header(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    good = K.build_package(alice, w.gw.identity.card(), 1, slot=1, tuning=(4, 4))
    bad = bytearray(good)
    bad[30] ^= 1
    assert detail(w.phone.pkg_send(bytes(bad))) == "bad_signature"
    ok(w.phone.pkg_send(good))
    ok(apply_(w, button=True))
    assert detail(w.phone.pkg_send(good)) == "replay"                                         # the same package again
    # an older number from the same sender
    assert detail(w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 1, tuning=(1, 1)))) == "replay"
    assert detail(w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 9, tuning=(1, 1), schema=2))) == "needs_update"
    assert detail(w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 9, tuning=(1, 1), min_fw=3))) == "needs_update"
    assert leftovers(w) == [] and pending(w) is None


def test_a_package_that_changes_on_the_way_is_dropped_with_everything_received(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    blob = bytes(range(256)) * 30
    raw = K.build_package(alice, w.gw.identity.card(), 1, model=(blob, H.card_of(blob)), chunk_size=1024)
    bad = bytearray(raw)
    bad[K.PREFIX + 3000] ^= 1
    assert detail(w.phone.pkg_send(bytes(bad), piece=1500)) == "tampered"
    assert leftovers(w) == [] and w.phone.get_packages()["upload"] is None
    cut = raw[:-50]
    w.phone.call(P.T_PKG_BEGIN, {"size": len(raw)})
    for off in range(0, len(cut), 1500):
        w.phone.call(P.T_PKG_CHUNK, body=off.to_bytes(4, "little") + cut[off : off + 1500])
    assert detail(w.phone.call(P.T_PKG_END, body=b"")) == "truncated" and leftovers(w) == []


def test_content_the_device_will_not_keep_is_refused_at_the_end_not_stored(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    me = w.gw.card = w.gw.identity.card()
    cases = {
        "not_ui_model": (MODEL, dict(CARD, classes=["person"])),
        "bad_model": (H.tiny_model(op="Gemm"), None),
        "bad_model ": (H.tiny_model(external=True), None),
    }
    seq = 0
    for want, (raw, card) in cases.items():
        seq += 1
        r = w.phone.pkg_send(K.build_package(alice, me, seq, model=(raw, card or H.card_of(raw))))
        assert detail(r) == want.strip(), want
        assert pending(w) is None and leftovers(w) == []
    bad_layout = {"schema": 1, "rev": 1, "title": {"ru": "x", "en": "x"}, "pages": [{"id": "p", "title": 5, "controls": []}]}
    assert detail(w.phone.pkg_send(K.build_package(alice, me, 9, ui_manifest=bad_layout))) == "bad_part"


def test_a_pose_model_cannot_be_smuggled_in_with_a_ui_card(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    people = H.tiny_model(layout=dict(H.LAYOUT_OK, classes=["person"], n_keypoints=17))
    r = w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 1, model=(people, H.card_of(people))))
    assert detail(r) in ("bad_model", "not_ui_model")
    assert pending(w) is None


# ---------------------------------------------------------------------------------------------------------------- interruptions
def test_a_dropped_connection_resumes_from_where_the_device_has_it(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    blob = H.tiny_model(pad=10000)
    raw = K.build_package(alice, w.gw.identity.card(), 1, slot=3, model=(blob, H.card_of(blob)), chunk_size=2048)
    r = w.phone.pkg_send(raw, piece=3000, stop_after=15000)
    assert r.type == P.T_ACK
    up = w.phone.get_packages()["upload"]
    assert up["size"] == len(raw) and up["next"] == 15000 and w.phone.get_state()["pkg.state"] == "receiving"
    off = up["next"]
    # the lost acknowledgement: a repeat
    r = w.phone.call(P.T_PKG_CHUNK, body=(off - 3000).to_bytes(4, "little") + raw[off - 3000 : off])
    assert r.type == P.T_ACK and r.json()["next"] == off
    r = w.phone.call(P.T_PKG_CHUNK, body=(off + 3000).to_bytes(4, "little") + raw[off + 3000 : off + 6000])   # a gap
    # not fatal: the reception stays
    assert detail(r) == "sequence" and w.phone.get_packages()["upload"]["next"] == off
    for o in range(off, len(raw), 3000):
        r = w.phone.call(P.T_PKG_CHUNK, body=o.to_bytes(4, "little") + raw[o : o + 3000])
        assert r.type == P.T_ACK
    r = w.phone.call(P.T_PKG_END, body=b"")
    assert r.type == P.T_ACK and r.json()["pending"]["slot"] == 3 and pending(w)["model"]["size"] == len(blob)


def test_a_reception_nobody_touches_is_dropped_and_a_restart_drops_it_too_but_keeps_what_was_complete(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    ok(w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 1, slot=1, tuning=(3, 3))))
    big = H.tiny_model(pad=10000)
    raw = K.build_package(alice, w.gw.identity.card(), 2, tuning=(1, 1), model=(big, H.card_of(big)))
    w.phone.pkg_send(raw, piece=1000, stop_after=3000)
    assert w.phone.get_packages()["upload"]["next"] == 3000
    w.run(61_000)
    assert w.phone.get_packages()["upload"] is None and "incoming.part" not in leftovers(w)
    w.phone.pkg_send(raw, piece=1000, stop_after=3000)
    w.power_cycle()
    w.phone.connect()
    # the half is gone, the finished one is still there
    assert w.phone.get_packages()["upload"] is None and pending(w)["slot"] == 1
    ok(apply_(w, button=True))
    assert w.gw.slotset[1].strength == 3


def test_a_cut_in_the_middle_of_applying_is_finished_after_the_restart(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    alice = H.sender(tmp_path)
    w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 1, slot=1, tuning=(6, 6), name="Игра", model=(MODEL, CARD)))

    def cut(point):
        if point == "after_model":
            raise RuntimeError("power lost")
    w.gw.pkg.fault = cut
    w.gw.physical_press(w.t)
    with pytest.raises(RuntimeError):
        w.phone.act("pkg.apply", True)
    assert (pkg_dir(w) / "applying").exists() and w.gw.slotset[1].name == ""                      # the model went in, the rest did not
    w.power_cycle()
    w.phone.connect()
    sl = w.gw.slotset[1]
    assert sl.name == "Игра" and sl.strength == 6 and (sl.dir / "model.bin").exists() and leftovers(w) == []
    assert w.phone.get_packages()["pending"] is None


# ---------------------------------------------------------------------------------------------------------------- the two channels together
def test_a_package_and_a_firmware_image_cannot_stand_in_for_each_other_on_the_wire(tmp_path, tremor_dir):
    from test_ctl_update import PUB, img
    w = world(tmp_path, tremor_dir, vendor_pub=PUB, factory_image=img(1, 1), fw_confirm_s=3)
    alice = H.sender(tmp_path)
    pk = K.build_package(alice, w.gw.identity.card(), 1, tuning=(1, 1))
    r = w.phone.call(P.T_FW_BEGIN, {"size": len(pk)})
    r = w.phone.call(P.T_FW_CHUNK, body=(0).to_bytes(4, "little") + pk)
    assert r.type == P.T_ERR and r.json()["code"] == P.E.FW_REJECTED and r.json()["detail"] == "wrong_channel"
    assert w.phone.get_firmware()["upload"] is None and w.gw.fw.trial is None
    fw = img(2, 1)
    r = w.phone.pkg_send(fw)
    assert detail(r) == "wrong_channel" and leftovers(w) == []
    assert w.phone.get_state()["fw.version"] == 1


def test_a_waiting_package_does_not_touch_the_banks_and_an_update_does_not_touch_the_slots(tmp_path, tremor_dir):
    from test_ctl_update import PUB, apply_ as fw_apply, img, reboot_into, stage
    w = world(tmp_path, tremor_dir, vendor_pub=PUB, factory_image=img(1, 1), fw_confirm_s=3)
    fw_before = {p.name: p.read_bytes() for p in (w.gw.dir / "fw").glob("*")}
    give(w, tmp_path, slot=0, tuning=(8, 2), model=(MODEL, CARD))
    assert {p.name: p.read_bytes() for p in (w.gw.dir / "fw").glob("*")} == fw_before          # a package never writes the banks
    slot_before = {p.relative_to(w.gw.dir): p.read_bytes() for p in (w.gw.dir / "slots").rglob("*") if p.is_file()}
    stage(w, 2)
    fw_apply(w)
    reboot_into(w)
    w.run(5000)
    assert w.gw.fw.active == "B" and w.gw.fw.version == 2
    # nor an update the slots
    assert {p.relative_to(w.gw.dir): p.read_bytes() for p in (w.gw.dir / "slots").rglob("*") if p.is_file()} == slot_before
    assert w.phone.get_state()["model.state"] == "ok"


def test_while_the_new_system_is_on_trial_packages_wait(tmp_path, tremor_dir):
    from test_ctl_update import PUB, apply_ as fw_apply, img, reboot_into, stage
    w = world(tmp_path, tremor_dir, vendor_pub=PUB, factory_image=img(1, 1), fw_confirm_s=30)
    alice = H.sender(tmp_path)
    ok(w.phone.pkg_send(K.build_package(alice, w.gw.identity.card(), 1, slot=1, tuning=(8, 8))))
    stage(w, 2)
    fw_apply(w)
    reboot_into(w)
    assert w.gw.fw.running == w.gw.fw.trial
    r = apply_(w, button=True)
    assert r.json()["code"] == P.E.BUSY and r.json()["detail"] == "fw_trial"
    assert pending(w) is not None and w.gw.slotset[1].strength == 5                             # it waits; the button was not spent
    w.run(40_000)
    assert w.gw.fw.trial is None
    ok(apply_(w, button=True))
    assert w.gw.slotset[1].strength == 8


def test_a_model_that_needs_a_newer_system_is_kept_but_marked(tmp_path, tremor_dir):
    from test_ctl_update import PUB, img
    w = world(tmp_path, tremor_dir, vendor_pub=PUB, factory_image=img(1, 1), fw_confirm_s=3)
    give(w, tmp_path, slot=0, model=(MODEL, CARD), min_fw=1)
    assert w.phone.get_state()["model.state"] == "ok"
    # the system goes back to something older than the package asked for: the file stays, the state says so, nothing is deleted
    mf = w.gw.slotset[0].dir / "model.json"
    import json
    meta = json.loads(w.gw.slotset[0].vault.read(mf, "model.json").decode())
    meta["min_fw"] = 5
    mf.write_bytes(w.gw.slotset[0].vault.seal("model.json", json.dumps(meta).encode()))
    w.gw.pkg.touch()
    assert w.phone.get_state()["model.state"] == "needs_system" and (w.gw.slotset[0].dir / "model.bin").exists()


def test_the_controls_and_the_reasons_are_there(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    m = w.gw.manifest
    for key in ("pkg.apply", "pkg.discard", "pkg.revert"):
        assert key in m.by_key and m.by_key[key]["type"] == "action"
    assert m.by_key["pkg.apply"]["confirm"] == "two-step" and m.by_key["pkg.revert"]["confirm"] == "two-step"
    assert "pkg_put" in m.files and m.max_file_bytes("pkg_put") >= 20 * 1024 * 1024
    assert w.gw.read_info()[1] & 16                                                             # the capability bit
    assert set(w.phone.get_state()) >= {"pkg.state", "pkg.from", "pkg.kinds", "model.state", "model.name", "model.version"}
    assert detail(w.phone.call(P.T_PKG_CHUNK, body=b"\0\0\0\0abcd")) == "sequence"             # a piece with no reception
    assert detail(w.phone.call(P.T_PKG_END, body=b"")) == "sequence"
    assert detail(w.phone.call(P.T_PKG_BEGIN, {"size": 5})) == "too_large"
    assert detail(w.phone.call(P.T_PKG_BEGIN, {})) == "damaged"
    for kind in C.REASON_KEYS["B"]:
        assert C.reason("B", kind).ru
