"""The device's keys, its public ID and its card."""
import json
import os
import stat

import pytest

from dataopen.ctl import identity as I


def ident(tmp_path, name="k"):
    return I.load_identity(I.FileKeyStore(tmp_path / name / "keys.json"))


def test_keys_are_created_once_and_survive_restarts(tmp_path):
    a = ident(tmp_path)
    b = ident(tmp_path)
    assert (a.id, a.ed_pub, a.x_pub, a.storage_key) == (b.id, b.ed_pub, b.x_pub, b.storage_key)
    assert len(a.storage_key) == 32 and len(a.ed_pub) == 32 and len(a.x_pub) == 32 and len(a.digest) == 32


def test_two_devices_never_share_anything(tmp_path):
    a, b = ident(tmp_path, "a"), ident(tmp_path, "b")
    assert len({a.id, b.id}) == 2 and a.storage_key != b.storage_key and a.ed_pub != b.ed_pub and a.x_pub != b.x_pub


def test_the_key_file_is_private(tmp_path):
    a = ident(tmp_path)
    p = tmp_path / "k" / "keys.json"
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    a.next_seq()
    assert stat.S_IMODE(p.stat().st_mode) == 0o600                       # still after a rewrite
    assert not list(p.parent.glob("*.tmp"))


def test_id_format_and_digest_binding(tmp_path):
    a = ident(tmp_path)
    assert len(a.id) == 19 and a.id.count("-") == 3 and all(c in I.B32 + "-" for c in a.id)
    assert a.id == I.format_id(I.digest_of(a.ed_pub, a.x_pub)) and a.fp8 == a.digest[:8]
    assert I.digest_of(a.ed_pub, a.x_pub) != I.digest_of(a.x_pub, a.ed_pub)              # the order of the keys matters


def test_ids_typed_by_people_are_normalised():
    assert I.normalize_id("7k2m 9qx4 abcd wxy3") == "7K2M-9QX4-ABCD-WXY3"
    assert I.normalize_id("OIl1-0000-1111-uuuu") == "0111-0000-1111-0000"
    for bad in ("", "7K2M-9QX4", "7K2M-9QX4-ABCD-WXY3-ZZZZ", "7K2M-9QX4-ABCD-WXY!"):
        with pytest.raises(ValueError):
            I.normalize_id(bad)


def test_id_is_well_spread(tmp_path):
    ids = {I.format_id(os.urandom(32)) for _ in range(2000)}
    assert len(ids) == 2000
    firsts = {i[0] for i in ids}
    assert len(firsts) >= 28                                                              # the 32 symbols are all in use


def test_a_card_is_self_signed_stable_and_round_trips(tmp_path):
    a = ident(tmp_path)
    c = a.card()
    assert c.verify() and c.id == a.id and c.digest == a.digest
    assert a.card().to_json() == c.to_json()                  # Ed25519 is deterministic: the same bytes every time
    back = I.Card.from_json(json.loads(json.dumps(c.to_json())))
    assert back == c


def test_a_card_that_lies_is_refused(tmp_path):
    a, b = ident(tmp_path, "a"), ident(tmp_path, "b")
    good = a.card().to_json()
    swapped = dict(good, x=b.card().to_json()["x"])           # another device's agreement key under this device's ID
    resigned_id = dict(good, id=b.id)
    other_sig = b.card().to_json()["sig"]
    for bad in (swapped, resigned_id, dict(good, label="someone else"), dict(good, created="1999-01-01"), dict(good, sig=other_sig),
                dict(good, v=2), dict(good, ed="AAAA"), {k: v for k, v in good.items() if k != "sig"}, [], None, "card"):
        with pytest.raises(I.CardError):
            I.Card.from_json(bad)


def test_a_card_can_be_forged_only_with_the_private_key(tmp_path):
    a, b = ident(tmp_path, "a"), ident(tmp_path, "b")
    forged = I.Card(a.ed_pub, a.x_pub, "DataOpen", a.created, b.sign(b"DOCARD1" + a.card()._body()))     # b signs a's card body
    assert not forged.verify()


def test_the_export_counter_only_grows_and_persists(tmp_path):
    a = ident(tmp_path)
    assert [a.next_seq() for _ in range(3)] == [1, 2, 3]
    assert ident(tmp_path).export_seq == 3


def test_rotating_the_storage_key_keeps_the_identity(tmp_path):
    a = ident(tmp_path)
    old = a.storage_key
    new = a.rotate_storage_key()
    b = ident(tmp_path)
    assert new != old and b.storage_key == new and b.id == a.id


def test_provisioning_again_is_a_new_device(tmp_path):
    a = ident(tmp_path)
    b = I.provision(I.FileKeyStore(tmp_path / "k" / "keys.json"))
    assert b.id != a.id and ident(tmp_path).id == b.id


def test_a_damaged_key_file_starts_the_device_over_instead_of_crashing(tmp_path):
    a = ident(tmp_path)
    (tmp_path / "k" / "keys.json").write_text("{not json")
    b = ident(tmp_path)
    assert b.id != a.id and len(b.storage_key) == 32


def test_signing_and_agreement_work_and_differ_per_device(tmp_path):
    a, b = ident(tmp_path, "a"), ident(tmp_path, "b")
    assert a.exchange(b.x_pub) == b.exchange(a.x_pub)
    assert a.exchange(b.x_pub) != a.exchange(a.x_pub)
    assert a.sign(b"x") != b.sign(b"x") and len(a.sign(b"x")) == 64
