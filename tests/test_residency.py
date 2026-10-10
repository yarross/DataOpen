"""Model & Profile Residency (docs/RESIDENCY.md): the profile of a hand and the weights of a model never leave the device.

What is proved here is not 'the interface does not show it'. It is that (1) the protocol has no message that returns it, (2) the list of
everything the device can be asked and everything it can say is closed and pinned, (3) a profile and a model with KNOWN bytes, put on a
device the only way there is, are never found in any byte the device says, whatever is asked, fuzzed or reset, (4) on disk they are only
ever ciphertext and a reset makes even a kept copy of that ciphertext useless, and (5) the support tools only delete, they never read.
"""
import builtins
import json
import logging
import random
import re
import shutil
from pathlib import Path

import pytest

from dataopen.ctl import manifest as M
from dataopen.ctl import protocol as P
from dataopen.ctl import residency as RS
from dataopen.ctl import residency_probe as RP
from dataopen.ctl.sim import World
from dataopen.ctl.vault import Vault, VaultError

from res_helpers import Clinic
from test_ctl_gateway import err, ok

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("cc") is None, reason="no C compiler")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "dataopen"


# ------------------------------------------------------------------------------------------------------------------ a device that holds secrets
@pytest.fixture(scope="module")
def secrets():
    return RP.make_profile(), RP.make_model()


def loaded_world(tmp_path, secrets, **kw):
    """A device with a profile and a model of KNOWN bytes in slot 0, put there the only way there is (a package from a sender)."""
    profile, model = secrets
    w = World(tmp_path / "dev", **kw)
    w.phone.connect()
    sender = RP.put_resident(w, tmp_path, profile, model)
    w.phone.confirm(True)
    can = RP.canary_for(profile, model, w.gw.identity, tmp_path / "dev")
    return w, can, sender


def test_the_canary_really_is_on_the_device_and_the_scan_really_finds_it(tmp_path, secrets):
    """A probe that can not find anything proves nothing: the same needles ARE found in the plaintext, so they are not too weak."""
    w, can, _ = loaded_world(tmp_path, secrets)
    profile, model = secrets
    assert w.gw.slotset[0].has and w.phone.get_state()["model.state"] == "ok"
    assert len(can.needles) > 50 and not can.hits(b"nothing to see here")
    assert can.hits(profile.pack()) and can.hits(model) and can.hits(b"x" + str(RP.PROFILE_ID).encode())
    assert w.gw.slotset[0].vault.read(w.gw.slotset[0].dir / "model.bin", "model.bin", allow_plain=False) == model       # it is really there


# ------------------------------------------------------------------------------------------------------------------ 1. the surface is closed
def test_the_protocol_has_no_message_name_or_constant_that_hands_the_secret_out():
    bad = re.compile(r"BUNDLE_GET|GET_BUNDLE|EXPORT|GET_PROFILE|GET_MODEL|GET_WEIGHTS|DOWNLOAD|BACKUP", re.I)
    assert [n for n in dir(P) if bad.search(n)] == []
    assert [k for k in P.TYPES if bad.search(k)] == []
    assert P.GET_KINDS == ("manifest", "state", "identity", "slots", "firmware", "packages")
    assert not (set(RS.ASK_FOR_SECRET) & set(P.GET_KINDS))


def test_the_surface_is_pinned_so_that_adding_to_it_is_a_decision(tmp_path):
    """The sets below are the whole interface. A new GET kind, SET key, ACT or file operation fails here until somebody classifies it."""
    assert P.GET_KINDS == RS_EXPECTED_GET
    assert RS.SET_KEYS == ("assist.on", "assist.strength", "tremor.level", "calib.running", "slot.name", "slot.active")
    assert RS.ACT_KEYS == ("profile.restore", "pairing.forget", "erase.profile", "factory.reset", "slot.clear", "fw.apply", "fw.rollback",
                           "pkg.apply", "pkg.discard", "pkg.revert")
    assert P.FILE_OPS_IN == ("bundle_put", "fw_put", "pkg_put") and P.FILE_OPS_OUT == ("card_get",)
    names = {v: k for k, v in P.TYPES.items()}
    assert sorted(names[t] for t in RS.OUTGOING_TYPES) == ["ACK", "DATA", "ERR", "EVENT", "HELLO_R", "PONG"]
    assert set(RS.OUTGOING_TYPES) | set(RS.INCOMING_TYPES) <= set(P.TYPES.values())
    # what the manifest the device ships actually offers is inside that surface, and fills it
    m = M.Manifest()
    keys = {k: c["type"] for k, c in m.by_key.items()}
    settable = {k for k, t in keys.items() if t in ("toggle", "stepper", "text")}
    acts = {k for k, t in keys.items() if t == "action"}
    assert settable <= set(RS.SET_KEYS) and acts == set(RS.ACT_KEYS)
    assert set(m.files) == set(P.FILE_OPS)
    out = [c for c in m.files.values() if c["op"] in P.FILE_OPS_OUT]
    assert [c["op"] for c in out] == ["card_get"]


RS_EXPECTED_GET = ("manifest", "state", "identity", "slots", "firmware", "packages")


def test_every_question_the_protocol_has_is_answered_inside_the_schemas_and_every_other_is_refused(tmp_path, secrets):
    w, can, _ = loaded_world(tmp_path, secrets)
    for kind in P.GET_KINDS:
        r = w.phone.call(P.T_GET, {"what": kind})
        assert r.type == P.T_DATA and not can.hits(r.body), kind
        if kind != "manifest":
            assert len(r.body) <= RS.MAX_ANSWER + 16
            RS.scrub(kind, r.json())
    for name in RS.ASK_FOR_SECRET:
        r = w.phone.call(P.T_GET, {"what": name, "for": "self", "scope": "all"})
        assert err(r) == P.E.RESIDENT and r.json()["key"] == "err.resident", name
    for name in ("", "nothing", "../keys", "model.bin", "Model", " model", None, 7, ["model"], {"a": 1}):
        r = w.phone.call(P.T_GET, {"what": name})
        assert err(r) in (P.E.BAD_KEY, P.E.RESIDENT), name
    assert w.gw.egress_blocked == 0


def test_a_layout_can_not_bring_a_control_that_hands_data_out():
    for op in ("bundle_get", "bundle_for_card", "profile_get", "model_get", "weights", "export", ""):
        m = M.default_manifest(2)
        m["pages"][0]["controls"].append({"id": "grab", "type": "file", "op": op, "accept": ".x", "max_bytes": 100, "label": M.L("x", "x")})
        assert M.validate_manifest(m), op
    for op in P.FILE_OPS:
        m = M.default_manifest(2)
        m["pages"][0]["controls"].append({"id": "grab", "type": "file", "op": op, "accept": ".x", "max_bytes": 100, "label": M.L("x", "x")})
        assert not M.validate_manifest(m), op


# ------------------------------------------------------------------------------------------------------------------ 2. the door out
def test_the_schemas_refuse_a_field_that_is_not_on_the_list():
    ok_state = {"rev": 3, "state": {"assist.on": True, "profile.fill": 40, "profile.layers": "both", "slot.2.name": "Работа"}}
    assert RS.scrub("state", ok_state) is ok_state
    for extra in ({"profile.t_motor": 231}, {"profile.hex": "00ff"}, {"model.sha256": "ab" * 32}, {"slot.9.name": "x"}, {"keys": 1}):
        with pytest.raises(RS.ResidencyViolation):
            RS.scrub("state", {"rev": 1, "state": extra})
    with pytest.raises(RS.ResidencyViolation):
        RS.scrub("state", {"rev": 1, "state": {"profile.layers": "t_motor=231"}})                 # a value outside the allowed words
    with pytest.raises(RS.ResidencyViolation):
        RS.scrub("state", {"rev": 1, "state": {"slot.name": "x" * 200}})                          # a long string is where a blob would hide
    with pytest.raises(RS.ResidencyViolation):
        RS.scrub("slots", {"active": 0, "count": 4, "slots": [{"n": 0, "name": "a", "has": True, "profile": "AAAA"}]})
    with pytest.raises(RS.ResidencyViolation):
        RS.scrub("packages", {"supported": True, "models": [{"n": 0, "state": "ok", "model": {"name": "x", "sha256": "ab" * 32}}]})
    with pytest.raises(RS.ResidencyViolation):
        RS.scrub("bundle", {})
    with pytest.raises(RS.ResidencyViolation):
        RS.scrub("firmware", {"supported": True, "weights": "x"})


def test_the_gateway_refuses_to_send_what_the_schemas_refuse_and_counts_it(tmp_path, secrets, monkeypatch):
    w, can, _ = loaded_world(tmp_path, secrets)
    real, n = w.gw.state_tree, iter(range(1000, 9000))
    monkeypatch.setattr(w.gw, "state_tree", lambda now=None: {**real(now), "profile.t_motor": next(n), "profile.dump": can_text(secrets[0])})
    r = w.phone.call(P.T_GET, {"what": "state"})
    assert err(r) == P.E.RESIDENT and r.json()["detail"] == "egress" and w.gw.egress_blocked >= 1
    blocked = w.gw.egress_blocked                                                             # (the event that followed the answer was refused too)
    mark = len(w.phone.inbox)
    w.phone.set("assist.strength", 7)                                                         # a state change would normally go out as an EVENT ...
    assert w.gw.egress_blocked > blocked
    assert not any(m.type == P.T_EVENT for m in w.phone.inbox[mark:])                        # ... and the poisoned one did not
    assert not can.hits(bytes(w.wire))


def can_text(profile) -> str:
    return profile.pack().hex()


def test_the_one_door_lets_only_the_listed_message_types_out(tmp_path, secrets):
    w, can, _ = loaded_world(tmp_path, secrets)
    before = len(w.wire)
    n = w.gw.egress_blocked
    for t in (P.T_GET, P.T_SET, P.T_ACT, P.T_BUNDLE_PUT, P.T_FW_CHUNK, P.T_PKG_CHUNK, 0x00, 0x55, 0xFF):
        w.gw._send(P.pack_message(t, 1, secrets[0].pack()))
    assert w.gw.egress_blocked == n + 9 and len(w.wire) == before
    w.gw._send(P.pack_message(P.T_DATA, 2, b"ok"))                                           # a listed type passes
    assert len(w.wire) > before


def test_a_model_card_shows_a_short_id_never_the_hash_of_the_weights(tmp_path, secrets):
    import hashlib
    w, can, _ = loaded_world(tmp_path, secrets)
    model = secrets[1]
    info = w.phone.get_packages()["models"][0]["model"]
    assert info["id"] == hashlib.sha256(model).hexdigest()[:16] and "sha256" not in info
    assert hashlib.sha256(model).hexdigest()[:20].encode() not in bytes(w.wire)
    assert set(info) <= set(RS.MODEL_CARD_PUBLIC)
    assert RS.public_card({"name": "a", "sha256": "ab" * 32, "weights": "zz"}) == {"name": "a", "id": "ab" * 8}
    assert RS.public_card(None) is None


# ------------------------------------------------------------------------------------------------------------------ 3. canaries
def test_nothing_the_device_says_carries_the_profile_or_the_model(tmp_path):
    r = RP.run_probe(tmp_path)
    assert r.hits == [] and r.egress_blocked == 0 and r.storage_hits == []
    assert set(r.refused_secret.values()) == {P.E.RESIDENT}
    assert len(r.wire) > 5000                                                                 # and it really said a good deal


def test_the_whole_life_of_a_device_says_nothing_of_either(tmp_path, secrets):
    """Everything a person does, one after the other, with the canaries on the device the whole time, and every byte out scanned at the end."""
    profile, model = secrets
    w, can, sender = loaded_world(tmp_path, secrets)
    ph = w.phone
    clinic = Clinic(tmp_path)
    RP.ask_everything(w)
    ph.set("assist.on", True)
    ph.confirm(True)
    for k in (1, 2, 0):
        ph.select_slot(k)
        ph.set("slot.name", f"slot {k}")
    # a second model, then back to the first (the previous generation exists and is swapped, never shown)
    RP.put_resident(w, tmp_path, profile, RP.make_model(seed=12), seq=2)
    ph.act("pkg.revert", True)
    ph.act("pkg.revert", True)
    # files come in: a settings file, a damaged one, one for nobody
    w.gw.physical_press()
    ph.put_bundle(clinic.file(w, profile=profile, tuning=(6, 6)))
    ph.put_bundle(b"DOBS" + bytes(400))
    ph.put_bundle(random.Random(5).randbytes(900))
    # a power cut and a restart, calibration refused (no learner), a hand on STOP, a hardware bypass
    w.make_gateway()
    w.phone.connect()
    w.phone.set("calib.running", True)
    w.phone.stop()
    w.phone.hard_bypass()
    RP.ask_everything(w)
    # every level of reset, with the person's data put back before each
    w.gw.physical_press()
    ok(w.phone.act("slot.clear", True))
    RP.put_resident(w, tmp_path, profile, model, seq=3)
    w.gw.physical_press()
    ok(w.phone.act("erase.profile", True))
    RP.put_resident(w, tmp_path, profile, model, seq=4)
    w.gw.physical_press()
    ok(w.phone.act("factory.reset", True))
    RP.ask_everything(w)
    w.run(500)
    wire = bytes(w.wire) + w.gw.read_status() + w.gw.read_info()
    assert len(wire) > 20000
    assert can.hits(wire) == [] and w.gw.egress_blocked == 0


def test_a_fuzzed_protocol_never_makes_the_device_say_the_secret(tmp_path, secrets):
    """Thousands of random, mutated and made-up requests (all types, odd bodies, every name for the secret): the answers stay on the list."""
    w, can, _ = loaded_world(tmp_path, secrets)
    rng = random.Random(2026)
    words = list(RS.ASK_FOR_SECRET) + list(P.GET_KINDS) + list(RS.SET_KEYS) + list(RS.ACT_KEYS) + ["for", "self", "scope", "all", "value", "key",
                                                                                                    "what", "confirmed", "keep", "size", "sha256", "name"]

    def val(d=0):
        c = rng.randrange(8)
        if c == 0:
            return rng.choice(words)
        if c == 1:
            return rng.randint(-5, 300)
        if c == 2:
            return rng.random() < 0.5
        if c == 3:
            return None
        if c == 4 and d < 3:
            return [val(d + 1) for _ in range(rng.randrange(4))]
        if c == 5 and d < 3:
            return {rng.choice(words): val(d + 1) for _ in range(rng.randrange(4))}
        if c == 6:
            return "x" * rng.randrange(0, 300)
        return rng.choice(words)

    types = list(P.TYPES.values()) + [0, 3, 0x20, 0x77]
    start = len(w.phone.inbox)
    sent = 0
    for i in range(2500):
        t = rng.choice(types)
        if t in (P.T_HELLO,) and rng.random() < 0.9:
            continue
        if rng.random() < 0.15:
            raw = P.pack_message(t, rng.randrange(1, 60000), rng.randbytes(rng.randrange(0, 80)))
        else:
            body = {rng.choice(words): val() for _ in range(rng.randrange(1, 5))}
            raw = P.pack_json(t, rng.randrange(1, 60000), body if rng.random() < 0.85 else val())
        if t in (P.T_FW_BEGIN, P.T_PKG_BEGIN, P.T_STOP, P.T_HARD_BYPASS) and rng.random() < 0.7:
            continue
        w.phone.write(raw)
        sent += 1
        if i % 500 == 499:
            w.phone.connect()
    assert sent > 1500
    out = w.phone.inbox[start:]
    assert out and all(m.type in RS.OUTGOING_TYPES for m in out)
    for m in out:
        if m.type == P.T_DATA:
            assert len(m.body) <= max(RS.MAX_ANSWER + 16, len(w.gw.manifest.raw)), len(m.body)
    assert can.hits(bytes(w.wire)) == [] and w.gw.egress_blocked == 0
    assert w.gw.slotset[0].has and w.phone.get_state()["model.state"] in ("ok", "none")   # and the device is alive and still holds what it held


def test_the_scan_has_teeth_a_device_that_did_leak_would_be_caught(tmp_path, secrets, monkeypatch):
    """The same checks against a device with the very bug the rule forbids (a 'get profile'): the canary, the closed list and the door catch it."""
    from dataopen.ctl.gateway import Gateway
    profile, model = secrets
    w, can, _ = loaded_world(tmp_path, secrets)
    real = Gateway._get

    def leaky(self, m):
        if m.json().get("what") == "profile":
            return self._send(P.pack_message(P.T_DATA, m.req, self.profile_store.load().pack()))      # a type the door allows, a body it can not judge
        return real(self, m)

    monkeypatch.setattr(Gateway, "_get", leaky)
    r = w.phone.call(P.T_GET, {"what": "profile"})
    assert r.type == P.T_DATA and can.hits(bytes(w.wire)) != []                                    # the canary scan finds it ...
    assert "profile" not in P.GET_KINDS                                                          # ... the pinned list does not list it ...
    monkeypatch.undo()
    assert err(w.phone.call(P.T_GET, {"what": "profile"})) == P.E.RESIDENT                       # ... and the real gateway refuses


# ------------------------------------------------------------------------------------------------------------------ 4. storage
def snapshot(d: Path) -> dict:
    return {p: p.read_bytes() for p in d.rglob("*") if p.is_file() and p.name != "keys.json"}


def test_on_disk_the_secret_is_only_ever_ciphertext_under_the_slots_own_key(tmp_path, secrets):
    w, can, _ = loaded_world(tmp_path, secrets)
    profile, model = secrets
    files = snapshot(w.gw.dir)
    assert any(p.name.startswith("model.bin") for p in files) and any(p.name.startswith("profile") for p in files)
    for p, raw in files.items():
        assert can.hits(raw) == [], p
        rel = p.relative_to(w.gw.dir).parts
        if rel[0] == "slots" or p.name.startswith(("settings", "trust")):
            assert raw[:4] == b"DOVT", p                                                          # sealed under a key of the device
        if p.name == "pending.dopk":
            assert raw[:4] == b"DOPK"                                                              # still sealed to the device's keys
    slot = w.gw.slotset[0]
    blob = (slot.dir / "model.bin").read_bytes()
    for key in (w.gw.identity.slot_key(1), w.gw.identity.storage_key):                           # another slot's key, the disk key
        with pytest.raises(VaultError):
            Vault(key).open("model.bin", blob)
    assert Vault(w.gw.identity.slot_key(0)).open("model.bin", blob) == model


@pytest.mark.parametrize("level", ["slot.clear", "erase.profile", "factory.reset"])
def test_a_kept_copy_of_the_ciphertext_is_useless_after_a_reset(tmp_path, secrets, level):
    """Flash keeps old blocks. The reset replaces the key, so a copy of every file taken BEFORE it can not be read AFTER it."""
    w, can, _ = loaded_world(tmp_path, secrets)
    before = snapshot(w.gw.dir)
    assert any(p.name.startswith("model.bin") for p in before)
    w.gw.physical_press()
    ok(w.phone.act(level, True))
    for p, raw in before.items():                                                             # put back everything that was there
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    w.make_gateway()
    w.phone.connect()
    st = w.phone.get_state()
    assert st["profile.fill"] == 0 and st["model.state"] == "none" and not w.gw.slotset[0].has
    assert w.gw.view is None
    ms = w.phone.get_packages()["models"][0]
    assert ms["model"] is None and ms["state"] == "none"
    assert can.hits(bytes(w.wire)) == []


def test_the_reset_levels_erase_exactly_what_the_rules_table_says_about_the_secrets(tmp_path, secrets):
    """The table in docs/RESIDENCY.md is not a promise on paper: each row for the profile, the model and the waiting package is checked."""
    profile, model = secrets

    def build(name):
        w, can, sender = loaded_world(tmp_path / name, secrets)
        w.phone.pkg_send(Clinic(tmp_path / name, "waiter").package(w, slot=1, tuning=(3, 3)))              # and a package still waiting
        return w

    def present(w):
        slot0 = w.gw.slotset[0].dir
        return {"profile": w.gw.slotset[0].has, "model": (slot0 / "model.bin").exists(), "pending": (w.gw.dir / "pkg" / "pending.dopk").exists()}

    for lvl, key in (("l1", "slot.clear"), ("l2", "erase.profile"), ("l3", "factory.reset")):
        w = build(lvl)
        assert present(w) == {"profile": True, "model": True, "pending": True}
        w.gw.physical_press()
        ok(w.phone.act(key, True))
        now = present(w)
        for rule_id, field in (("profile", "profile"), ("model", "model"), ("pending", "pending")):
            expected = getattr(RS.RULES_BY_ID[rule_id], lvl)
            assert now[field] == (expected == RS.KEPT), (lvl, rule_id, now)


def test_the_rules_table_is_consistent_with_itself():
    ids = [r.id for r in RS.RULES]
    assert len(ids) == len(set(ids))
    for r in RS.RULES:
        assert r.cls in RS.CLASSES and all(x in (RS.KEPT, RS.ERASED, RS.REPLACED, RS.NA) for x in (r.l1, r.l2, r.l3, r.l4)), r.id
        if r.cls in (RS.SECRET, RS.PERSONAL, RS.DERIVED):
            assert r.export is False, r.id                                                    # nothing personal is ever exportable
    assert [r.id for r in RS.RULES if r.export] == ["card"]
    for rid in ("profile", "profile_prev", "model", "derived_params"):
        assert RS.RULES_BY_ID[rid].cls == RS.SECRET and (RS.RULES_BY_ID[rid].l2, RS.RULES_BY_ID[rid].l4) == (RS.ERASED, RS.ERASED)


# ------------------------------------------------------------------------------------------------------------------ 5. support and recovery
def test_the_support_tools_only_ever_delete_the_personal_area_they_never_read_it(tmp_path, monkeypatch):
    from dataopen.provisioning import service as SV
    from test_recovery import shipped, token, used_device
    d, hsm, rep, ag = shipped(tmp_path)
    used_device(d, hsm)
    (d / "slots" / "0").mkdir(parents=True, exist_ok=True)
    (d / "slots" / "0" / "model.bin").write_bytes(b"WEIGHTS-OF-THE-MODEL" * 20)
    (d / "pkg").mkdir(exist_ok=True)
    (d / "pkg" / "pending.dopk").write_bytes(b"A-WAITING-PACKAGE" * 20)
    guarded = (d / "slots", d / "pkg")
    touched = []
    real_open, real_rb, real_rt, real_pop = builtins.open, Path.read_bytes, Path.read_text, Path.open

    def under(p) -> bool:
        try:
            q = Path(p).resolve()
        except (OSError, TypeError, ValueError):
            return False
        return any(g.resolve() in q.parents or q == g.resolve() for g in guarded)

    def no_open(f, *a, **kw):
        mode = a[0] if a else kw.get("mode", "r")
        if isinstance(f, (str, Path)) and under(f) and not any(c in mode for c in "wax"):
            touched.append(str(f))
        return real_open(f, *a, **kw)

    def no_rb(self):
        if under(self):
            touched.append(str(self))
        return real_rb(self)

    def no_rt(self, *a, **kw):
        if under(self):
            touched.append(str(self))
        return real_rt(self, *a, **kw)

    def no_pop(self, mode="r", *a, **kw):
        if under(self) and not any(c in mode for c in "wax"):
            touched.append(str(self))
        return real_pop(self, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", no_open)
    monkeypatch.setattr(Path, "read_bytes", no_rb)
    monkeypatch.setattr(Path, "read_text", no_rt)
    monkeypatch.setattr(Path, "open", no_pop)
    out = SV.full_return(d, ag, token(hsm, ag, rep.serial), presence=True)
    monkeypatch.undo()
    assert touched == []                                                                      # the tool never opened a slot or a package to read it
    assert set(out) == {"serial", "som", "mcu", "floor", "lifecycle"}                          # and what it reports is metadata and nothing else
    assert not (d / "slots").exists() and not (d / "pkg").exists()


def test_what_the_factory_and_the_recovery_model_report_is_metadata_only(tmp_path):
    from dataopen.provisioning import recovery as RC
    from prov_helpers import provisioned
    d, hsm, rep = provisioned(tmp_path, "dev")
    j = rep.to_json()
    assert set(j) == {"sku", "serial", "quarantined", "failed_step", "steps", "checks"}
    for s in RC.SCENARIOS:
        o = RC.run(s)
        assert set(vars(o)) == {"scenario", "first", "final", "computer", "profiles_kept", "rma", "steps"}
        assert set(vars(o.first)) == {"stage", "bank", "mcu", "mouse", "video", "pwa", "led", "profiles_kept", "notes"}
        assert not hasattr(o, "backup_restores")


# ------------------------------------------------------------------------------------------------------------------ 6. the code and the client
def test_no_code_in_the_device_hands_the_secret_to_a_message():
    # the gateway never seals anything: files come in, they do not go out
    gw = (SRC / "ctl" / "gateway.py").read_text(encoding="utf-8")
    for needle in ("SL.seal(", ".seal(self.identity", "_export_bundle", "_seal_or_refuse", "GET_BUNDLE", "bundle_get", "bundle_for_card"):
        assert needle not in gw, needle
    # nobody reads the weights for anything but running the model, and the model is not run yet: there is no call site at all
    calls = []
    for p in SRC.rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.weights\(\)", line) and not line.lstrip().startswith(("#", '"""')):
                calls.append(f"{p.relative_to(ROOT)}:{i}")
    assert calls == []
    # no source file of the device names an operation that returns the secret
    for p in SRC.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        if p.name in ("residency.py", "residency_probe.py"):
            continue
        for needle in ("bundle_get", "bundle_for_card", "GET_BUNDLE", "get_bundle("):
            assert needle not in text, (p.name, needle)


def test_a_profile_and_a_model_do_not_print_themselves(tmp_path, secrets, caplog):
    profile, model = secrets
    assert repr(profile) == "ProfileState(<resident>)" and str(profile) == "ProfileState(<resident>)"
    assert str(RP.PROFILE_ID) not in f"{profile!r} {[profile]} {{'p': {profile}}}"
    with caplog.at_level(logging.DEBUG):
        w, can, _ = loaded_world(tmp_path, secrets)
        RP.ask_everything(w)
        w.gw.physical_press()
        ok(w.phone.act("slot.clear", True))
    assert can.hits(caplog.text.encode()) == [] and "ProfileState(" not in caplog.text.replace("<resident>", "")
    assert can.hits((w.gw.last_error or "").encode()) == []


def test_the_phone_client_has_no_way_to_ask_for_the_secret_or_to_save_it():
    js = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "pwa" / "js").glob("*.js")}
    for name, text in js.items():
        for needle in ("getBundle", "bundle_get", "bundle_for_card", "what: 'bundle'", "what: 'profile'", "what: 'model'"):
            assert needle not in text, (name, needle)
    assert js["app.js"].count("download(") == 1 and "dataopen-${card.id}.docard" in js["app.js"]       # the one thing the page saves: the public card
    assert "GET_KINDS" in js["session.js"] and "GET_KINDS" in js["constants.js"] and "FILE_OPS" in js["view.js"]
    const = js["constants.js"]
    assert json.dumps(list(P.GET_KINDS)) in const and json.dumps(list(P.FILE_OPS)) in const


def test_the_dev_server_does_not_serve_the_device_directory(tmp_path, secrets):
    """The one place a device-side HTTP file server exists is the development server; it serves the page and nothing of the device."""
    src = (SRC / "ctl" / "ws.py").read_text(encoding="utf-8")
    assert "rglob" not in src and "self.world.dir" not in src.split("def static")[1].split("async def sim")[0]
    assert "self.root / rel" in src                                                           # static files come from the PWA directory only


def test_a_file_the_device_seems_to_have_made_itself_is_refused_in_both_formats(tmp_path, secrets):
    from dataopen.ctl import seal as SL
    from dataopen.ctl.identity import Card
    from dataopen.updates import package as K
    w, can, _ = loaded_world(tmp_path, secrets)
    me = w.gw.identity
    card = Card.from_json(w.phone.get_identity())
    r = w.phone.put_bundle(SL.seal(me, card, me.next_seq(), profile=secrets[0], tuning=(9, 9)))
    assert err(r) == P.E.BAD_BUNDLE and r.json()["detail"] == "own_file"
    r = w.phone.pkg_send(K.build_package(me, card, me.next_seq(), slot=1, tuning=(9, 9)))
    assert err(r) == P.E.PKG_REJECTED and r.json()["detail"] == "own_file"


# ------------------------------------------------------------------------------------------------------------------ 7. the document
DOC = ROOT / "docs" / "RESIDENCY.md"


def test_the_document_is_generated_from_the_rules_and_is_current():
    assert RS.doc_is_current(DOC), "run `dataopen ctl residency docs --write`"
    text = DOC.read_text(encoding="utf-8")
    for r in RS.RULES:
        assert r.what in text, r.id
    for name in RS.ASK_FOR_SECRET:
        assert f"`{name}`" in text


def test_the_command_probes_and_says_so(capsys, tmp_path):
    from dataopen.cli import build_parser

    def run(*argv):
        a = build_parser().parse_args(["ctl", "residency", *argv])
        return a.fn(a)
    assert run("check") == 0 and "residency: holds" in capsys.readouterr().out
    assert run("rules") == 0 and "моторный профиль" in capsys.readouterr().out
    assert run("surface") == 0 and "`GET`" in capsys.readouterr().out
    assert run("docs", "--check") == 0
