"""Hardware slots against the REAL bridge core: independent contexts, switching, the try-it timer, per-slot keys, clearing, files."""
import json
import shutil

import pytest

from dataopen.bioprofile.profile import ProfileState
from dataopen.ctl import manifest as M
from dataopen.ctl import protocol as P
from dataopen.ctl import seal as SL
from dataopen.ctl.sim import SimLearner, World, seed_profile
from dataopen.ctl.slots import SLOT_COUNT
from dataopen.ctl.vault import Vault, VaultError

from test_ctl_gateway import err, hello, moves, never_adds, ok, state  # noqa: F401  (the same helpers, one place)

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")


@pytest.fixture(scope="module")
def tremor_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("slots-tremor")
    seed_profile(d, "tremor")
    return d


@pytest.fixture(scope="module")
def other_state(tmp_path_factory):
    """A second, different person: another slot's content."""
    d = tmp_path_factory.mktemp("slots-other")
    st = seed_profile(d, "overshooter")._state
    st.profile_id = 0x0B0B                                      # something to tell the two people apart by
    return st


def world(tmp_path, tremor_dir=None, **kw) -> World:
    d = tmp_path / "gw"
    if tremor_dir is not None:
        shutil.copytree(tremor_dir, d)
    return World(d, **kw)


def put(w, st, k):
    """Put a profile into slot k the way a file would (so it is NOT yet kept in work)."""
    w.gw._install_profile(ProfileState.unpack(st.pack()), w.t, k)


def keep_everything(w):
    ph = w.phone
    if not w.gw.settings.assist_wanted:
        ph.set("assist.on", True)
    if w.gw.trial is not None:
        ph.confirm(True)


# ---------------------------------------------------------------------------------------------------------------- the model
def test_slots_keep_independent_profiles_levels_and_names(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    put(w, other_state, 2)
    a_id = w.gw.view.profile_id
    ok(ph.set("assist.strength", 8))
    ok(ph.set("slot.name", "Работа"))
    ok(ph.select_slot(2))
    assert w.gw.active == 2 and w.gw.view.profile_id == other_state.profile_id != a_id
    assert (w.gw.settings.strength, w.gw.settings.tremor) == (5, 5)             # a slot has its own levels
    ok(ph.set("assist.strength", 3))
    ok(ph.set("slot.name", "Браузер"))
    ok(ph.select_slot(0))
    assert w.gw.view.profile_id == a_id and w.gw.settings.strength == 8
    assert ph.get_state()["slot.0.name"] == "Работа" and ph.get_state()["slot.2.name"] == "Браузер" and ph.state["slot.active"] == 0
    w.make_gateway()                                                              # a restart: every bit of it comes back
    w.phone.connect()
    assert (w.gw.active, w.gw.settings.strength) == (0, 8)
    ok(w.phone.select_slot(2))
    assert (w.gw.settings.strength, w.gw.slot.name) == (3, "Браузер") and w.gw.view.profile_id == other_state.profile_id
    info = w.phone.get_slots()
    assert info["active"] == 2 and info["count"] == SLOT_COUNT == 4
    assert [s["has"] for s in info["slots"]] == [True, False, True, False] and info["slots"][2]["name"] == "Браузер"


def test_the_status_says_which_slot_and_which_slots_hold_a_profile(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    w.phone.connect()
    s = w.gw.status()
    assert (s.slot, s.slot_mask) == (0, 0b0001)
    put(w, other_state, 3)
    ok(w.phone.select_slot(3))
    s = P.StatusSnapshot.unpack(w.gw.read_status())
    assert (s.slot, s.slot_mask) == (3, 0b1001)
    assert w.phone.status_notes[-1].slot == 3                                      # and the notification said it, without being asked
    assert w.phone.state["slot.active"] == 3                                       # as did the event


def test_switching_a_slot_never_turns_assistance_on(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    put(w, other_state, 1)
    ok(ph.select_slot(1))
    w.run(1500)
    assert not w.gw.settings.assist_wanted and state(w).__eq__(2) and w.reason() == "CMD_PASSTHRU"
    assert w.gw.trial is None


def test_a_slot_kept_in_work_switches_at_once_but_a_new_one_is_a_trial_that_undoes_to_the_old_slot(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir, trial_s=10)
    ph = w.phone
    ph.connect()
    put(w, other_state, 1)
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))                                      # slot 0 is now kept in work
    assert w.gw.slotset[0].vetted and not w.gw.slotset[1].vetted
    ok(ph.select_slot(1))                                     # slot 1 came from a file: a trial, assistance stays on
    assert w.gw.active == 1 and w.gw.trial is not None and w.gw.settings.assist_wanted
    assert ph.get_state()["trial.left_s"] > 0 and w.gw.status().flags & P.SF_TRIAL
    w.run(11_000)                                             # nobody said 'keep': back to where it came from
    assert w.gw.active == 0 and w.gw.trial is None and w.gw.view.profile_id != other_state.profile_id
    assert w.gw.settings.assist_wanted and w.bridge().params_rejected == 0
    ok(ph.select_slot(1))
    ok(ph.confirm(True))                                      # kept: next time there is nothing to confirm
    assert w.gw.slotset[1].vetted
    ok(ph.select_slot(0))
    ok(ph.select_slot(1))
    assert w.gw.trial is None and w.gw.active == 1


def test_leaving_a_slot_that_was_in_work_counts_as_keeping_it(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    put(w, other_state, 1)
    ok(ph.select_slot(1))                                     # assistance is off: nothing to try
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    ok(ph.select_slot(0))                                     # slot 1 was in work with assistance on and was left without complaint
    assert w.gw.slotset[1].vetted


def test_a_restart_in_the_middle_of_a_slot_trial_goes_back_to_the_previous_slot(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    put(w, other_state, 1)
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    ok(ph.select_slot(1))
    assert w.gw.trial is not None
    w.make_gateway()
    assert w.gw.active == 0 and w.gw.trial is None and w.gw.settings.assist_wanted


def test_the_safety_clamp_holds_in_every_slot(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir, trial_s=3600)
    ph = w.phone
    ph.connect()
    put(w, other_state, 1)
    ok(ph.set("assist.on", True))
    ok(ph.set("assist.strength", 10))
    ok(ph.confirm(True))
    for k in (1, 0, 2, 3, 1):                                  # 2 and 3 are empty: a neutral chain, not a gap
        ok(ph.select_slot(k))
        if w.gw.trial is not None:
            ok(ph.confirm(True))
        assert never_adds(moves(w, 150, seed=k))
        w.run(1200)
    b = w.bridge()
    assert b.params_rejected == 0 and b.invariant_viol == 0 and state(w) == 3          # S_ASSIST


def test_an_empty_slot_has_nothing_ready_and_makes_no_trial(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    ok(ph.select_slot(1))
    assert w.gw.trial is None and w.gw.view is None and w.gw.status().ready == 0 and ph.get_state()["profile.fill"] == 0


def test_switching_refuses_what_it_must(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir, learner=SimLearner(minutes=5.0, seed=2, speed=240.0), trial_s=3600)
    ph = w.phone
    ph.connect()
    for bad in (-1, 4, 99, True, "1", 1.5, None):
        assert err(ph.select_slot(bad)) == P.E.BAD_VALUE, bad
    ok(ph.select_slot(0))                                      # the slot it is on: nothing to do, nothing wrong
    put(w, other_state, 1)
    ok(ph.set("calib.running", True))
    r = ph.select_slot(1)
    assert err(r) == P.E.BUSY and r.json()["detail"] == "calibrating" and w.gw.active == 0
    ok(ph.set("calib.running", False))
    assert w.gw.trial is None                                  # assistance was off: the calibrated profile is simply there
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    ok(ph.select_slot(1))                                      # a trial
    r = ph.select_slot(2)
    assert err(r) == P.E.BUSY and r.json()["detail"] == "trial" and w.gw.active == 1


def test_the_way_to_the_next_slot_does_not_depend_on_the_layout_a_slot_brings(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    custom = {"schema": 1, "rev": 5, "title": M.L("Своё", "Own"),
              "pages": [{"id": "p", "title": M.L("С", "P"),
                         "controls": [{"id": "n", "type": "note", "label": M.L("Только заметка", "Only a note")}]}]}
    w.gw._set_manifest(M.Manifest(custom), 1)
    put(w, other_state, 1)
    mark = len(ph.inbox)
    ok(ph.select_slot(1))
    assert w.gw.manifest.rev == 5 and w.gw.custom_manifest
    assert any(m.type == P.T_EVENT and m.json().get("manifest") == w.gw.manifest.hash.hex() for m in ph.inbox[mark:])   # the phone is told
    assert ph.get_manifest()["rev"] == 5
    assert err(ph.set("assist.on", True)) == P.E.BAD_KEY                       # that layout has no such control ...
    ok(ph.select_slot(0))                                                      # ... but the way back is never taken away
    assert w.gw.manifest.rev == 1 and not w.gw.custom_manifest


def test_slot_names_are_short_and_plain(tmp_path, tremor_dir):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    assert err(ph.set("slot.name", "x" * 25)) == P.E.BAD_VALUE
    assert err(ph.set("slot.name", 5)) == P.E.BAD_VALUE
    ok(ph.set("slot.name", "A\x00\x07B  "))
    assert w.gw.slot.name == "AB"
    ok(ph.set("slot.name", "я" * 24))
    assert len(w.gw.slot.name) == 24


# ---------------------------------------------------------------------------------------------------------------- keys and clearing
def slot_files(w, k):
    return sorted(p for p in (w.dir / "slots" / str(k)).iterdir() if p.is_file())


def test_every_slot_is_encrypted_under_its_own_key(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    put(w, other_state, 1)
    w.gw.slotset[1].name = "x"
    w.gw.slotset[1].save_meta()
    ident = w.gw.identity
    assert len({ident.slot_key(k) for k in range(SLOT_COUNT)}) == SLOT_COUNT and ident.slot_key(0) != ident.storage_key
    for k in (0, 1):
        files = slot_files(w, k)
        assert files
        for p in files:
            assert p.read_bytes()[:4] == b"DOVT" and b"BIOP" not in p.read_bytes()
    p0 = next(p for p in slot_files(w, 0) if p.name.startswith("profile."))
    own, neighbour = Vault(ident.slot_key(0)), Vault(ident.slot_key(1))
    assert own.open("profile", p0.read_bytes())[:4] == b"BIOP"
    with pytest.raises(VaultError):
        neighbour.open("profile", p0.read_bytes())                                  # slot 1's key opens nothing of slot 0
    with pytest.raises(VaultError):
        Vault(ident.storage_key).open("profile", p0.read_bytes())                   # nor does the device key itself


def test_clearing_a_slot_makes_its_old_ciphertext_dead_and_leaves_the_others_alone(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    put(w, other_state, 1)
    ok(ph.set("assist.strength", 9))
    ok(ph.select_slot(1))
    ok(ph.set("slot.name", "Браузер"))
    stolen = {p.name: p.read_bytes() for p in slot_files(w, 1)}                      # a copy of the flash as it was
    other = {p.name: p.read_bytes() for p in slot_files(w, 0)}
    ok(ph.set("assist.on", True))
    ok(ph.confirm(True))
    assert err(ph.act("slot.clear", False)) == P.E.NOT_ALLOWED                       # two steps
    r = ph.act("slot.clear", True)
    assert err(r) == P.E.PHYSICAL and r.json()["detail"] == "slot:2"                 # and the button on the device
    w.gw.physical_press()
    ok(ph.act("slot.clear", True))
    assert w.gw.slot.name == "" and w.gw.view is None and not w.gw.slot.has and (w.gw.settings.strength, w.gw.settings.tremor) == (5, 5)
    assert not w.gw.settings.assist_wanted and state(w) == 2 and w.gw.identity.slot_epoch(1) == 1 and w.gw.identity.slot_epoch(0) == 0
    for name, blob in stolen.items():                                                # the old bytes, put back, are noise to the new key
        (w.dir / "slots" / "1" / name).write_bytes(blob)
    w.make_gateway()
    assert w.gw.active == 1 and not w.gw.slot.has and w.gw.view is None
    assert {p.name: p.read_bytes() for p in slot_files(w, 0)} == other               # the neighbour's files are byte for byte the same
    w.phone.connect()
    ok(w.phone.select_slot(0))
    assert w.gw.view is not None and w.gw.settings.strength == 9 and w.gw.slot.has             # slot 0: as it was


def test_erasing_all_personal_data_clears_every_slot_and_factory_reset_makes_a_new_device(tmp_path, tremor_dir, other_state):
    w = world(tmp_path, tremor_dir)
    ph = w.phone
    ph.connect()
    put(w, other_state, 3)
    ok(ph.select_slot(3))
    ok(ph.set("slot.name", "x"))
    old_id = w.gw.identity.id
    w.gw.physical_press()
    ok(ph.act("erase.profile", True))
    assert w.gw.active == 0 and not any(s.has or s.name for s in w.gw.slotset) and w.gw.identity.id == old_id
    assert w.gw.slotset.mask() == 0 and w.gw.identity.slot_epochs == []
    assert not [p for p in w.dir.rglob("*") if p.is_file() and p.read_bytes()[:4] == b"BIOP"]                # no plain profile anywhere
    put(w, other_state, 2)
    w.gw.physical_press()
    ok(ph.act("factory.reset", True))
    assert w.gw.identity.id != old_id and w.gw.slotset.mask() == 0 and w.gw.active == 0


def test_a_device_from_before_slots_becomes_slot_zero_with_everything_it_had(tmp_path, tremor_dir):
    d = tmp_path / "gw"
    shutil.copytree(tremor_dir, d)                                               # profile.a / profile.b in the old place, in the clear
    (d / "profile.prev").write_bytes((next(d.glob("profile.[ab]"))).read_bytes())
    custom = M.default_manifest(7)
    (d / "manifest.json").write_text(json.dumps(custom))
    from dataopen.ctl.slots import SettingsStore
    SettingsStore(d / "settings").save({"assist_wanted": False, "strength": 8, "tremor": 2, "epoch": 3})
    w = World(d, start=False)
    gw = w.gw
    assert gw.active == 0 and gw.view is not None and (gw.settings.strength, gw.settings.tremor) == (8, 2)
    assert gw.slot.vetted and gw.manifest.rev == 7 and gw.slot.prev_path.exists()
    assert not list(d.glob("profile.*")) and not (d / "manifest.json").exists()
    assert all(p.read_bytes()[:4] == b"DOVT" for p in (d / "slots" / "0").iterdir() if p.is_file())
    w.make_gateway()                                                              # and the second start finds nothing left to move
    assert (w.gw.settings.strength, w.gw.manifest.rev) == (8, 7)


# ---------------------------------------------------------------------------------------------------------------- the file
def two_devices(tmp_path, tremor_dir, other_state):
    a = world(tmp_path / "a", tremor_dir)
    b = world(tmp_path / "b")
    hello(a, b)
    put(a, other_state, 2)
    ok(a.phone.set("assist.strength", 8))
    ok(a.phone.select_slot(2))
    ok(a.phone.set("slot.name", "Браузер"))
    ok(a.phone.set("assist.strength", 3))
    ok(a.phone.select_slot(0))
    return a, b


def send_all(a, b):
    card = b.phone.get_identity()
    a.gw.physical_press()
    raw = a.phone.call(P.T_GET, {"what": "bundle", "for": card, "scope": "all"})
    assert raw.type == P.T_DATA, raw.body
    b.gw.physical_press()
    return raw.body, b.phone.put_bundle(raw.body)


def test_one_file_carries_every_slot_to_another_device_by_number(tmp_path, tremor_dir, other_state):
    a, b = two_devices(tmp_path, tremor_dir, other_state)
    b.gw._install_profile(ProfileState.unpack(other_state.pack()), b.t, 1)           # B already has something in slot 1
    b1 = b.gw.slotset[1].profile.load().generation
    raw, r = send_all(a, b)
    ok(r)
    o = SL.open_sealed(raw, b.gw.identity)
    assert [s.n for s in o.slots] == [0, 2] and o.profile is None and o.tuning is None
    assert b.gw.slotset[0].has and b.gw.slotset[2].has and not b.gw.slotset[3].has
    assert (b.gw.slotset[0].strength, b.gw.slotset[2].strength, b.gw.slotset[2].name) == (8, 3, "Браузер")
    assert b.gw.slotset[1].profile.load().generation == b1 and b.gw.slotset[1].has                  # slot 1 was not in the file: untouched
    assert not b.gw.slotset[0].vetted and not b.gw.slotset[2].vetted                                # arrived, not yet kept in work
    assert b.gw.active == 0 and b.gw.settings.strength == 8
    ok(b.phone.select_slot(2))                                                                      # works without any more files
    assert b.gw.view.profile_id == other_state.profile_id and b.gw.slot.name == "Браузер"


def test_a_file_for_all_slots_still_needs_the_buttons_and_never_opens_elsewhere(tmp_path, tremor_dir, other_state):
    a, b = two_devices(tmp_path, tremor_dir, other_state)
    card = b.phone.get_identity()
    r = a.phone.call(P.T_GET, {"what": "bundle", "for": card, "scope": "all"})
    assert err(r) == P.E.PHYSICAL and r.json()["detail"].startswith("export:")
    a.gw.physical_press()
    raw = a.phone.call(P.T_GET, {"what": "bundle", "for": card, "scope": "all"}).body
    c = world(tmp_path / "c")
    c.phone.connect()
    c.gw.physical_press()
    assert err(c.phone.put_bundle(raw)) == P.E.WRONG_DEVICE and not c.gw.slotset.mask()
    assert err(b.phone.put_bundle(raw)) == P.E.PHYSICAL                                              # B does not know the sender yet
    assert not b.gw.slotset.mask()
    assert err(a.phone.call(P.T_GET, {"what": "bundle", "for": "self", "scope": "some"})) == P.E.BAD_VALUE


def test_the_default_file_is_about_the_active_slot_only(tmp_path, tremor_dir, other_state):
    a, b = two_devices(tmp_path, tremor_dir, other_state)
    ok(a.phone.select_slot(2))
    card = b.phone.get_identity()
    a.gw.physical_press()
    raw = a.phone.get_bundle(card)
    o = SL.open_sealed(raw, b.gw.identity)
    assert not o.slots and o.profile.profile_id == other_state.profile_id and o.tuning == (3, 5)
    # B is in slot 3: that is where it lands
    ok(b.phone.select_slot(3))
    b.gw.physical_press()
    ok(b.phone.put_bundle(raw))
    assert b.gw.slotset[3].has and b.gw.slotset[3].strength == 3 and not b.gw.slotset[0].has


def test_all_slots_with_layouts_of_their_own_either_fit_or_say_so(tmp_path, tremor_dir, other_state):
    a, b = two_devices(tmp_path, tremor_dir, other_state)
    big = M.default_manifest(2)
    for k in range(4):
        put(a, other_state, k) if not a.gw.slotset[k].has else None
        a.gw._set_manifest(M.Manifest(big), k)
    card = b.phone.get_identity()
    a.gw.physical_press()
    r = a.phone.call(P.T_GET, {"what": "bundle", "for": card, "scope": "all"})
    # four 10 KB layouts are more than one message
    assert err(r) == P.E.TOO_BIG
    a.gw.physical_press()
    raw = a.phone.get_bundle(card)                                                                   # the active slot alone always fits
    assert SL.open_sealed(raw, b.gw.identity).manifest["rev"] == 2


def test_a_file_into_the_active_slot_is_the_same_trial_as_ever_and_the_slot_does_not_move(tmp_path, tremor_dir, other_state):
    """A file lands in the ACTIVE slot with assistance on: the same try-it-and-keep-it as ever."""
    a, b = two_devices(tmp_path, tremor_dir, other_state)
    ok(b.phone.set("assist.on", True))
    _, r = send_all(a, b)
    ok(r)
    # slot 0 is active on B and got a profile
    assert b.gw.trial is not None and b.gw.active == 0
    ok(b.phone.confirm(False))
    # nothing was there before: undone to 'off'
    assert b.gw.trial is None and not b.gw.settings.assist_wanted
    # a slot in the drawer is not part of a trial
    assert b.gw.slotset[2].has
