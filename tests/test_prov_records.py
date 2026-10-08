"""Serial numbers, the signed records, the chain of trust and the map of what lives where."""
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from dataopen.ctl import identity as I
from dataopen.provisioning import records as R

DATA = Path(__file__).parent / "data"
HW = b"DOHW0001"


def key(tag: str):
    import hashlib
    k = ed25519.Ed25519PrivateKey.from_private_bytes(hashlib.sha256(tag.encode()).digest())
    return k, k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def golden_att():
    vend, vpub = key("golden-vendor")
    _, dak = key("golden-dak")
    att = R.Attestation("DO1-2641-00017-" + R.make_serial(1, 26, 41, 17)[-1], HW, dak, bytes(range(32)), "2026-10-08", R.vendor_id(vpub))
    return R.Attestation(att.serial, att.hw_id, att.dak, att.board, att.date, att.vendor, vend.sign(att.body())), vpub


# ---------------------------------------------------------------------------------------------------------------------------- serials
def test_a_serial_round_trips_and_its_check_character_catches_every_single_slip():
    s = R.make_serial(2, 26, 41, 17)
    assert R.parse_serial(s) == {"sku": 2, "year": 26, "week": 41, "seq": 17}
    body = s.replace("-", "")
    for i in range(2, len(body) - 1):                                   # change any one character of the body: the check no longer fits
        for c in "0123456789":
            if c == body[i]:
                continue
            t = body[:i] + c + body[i + 1:]
            t = f"{t[:3]}-{t[3:7]}-{t[7:12]}-{t[12]}"
            with pytest.raises(R.RecordError):
                R.parse_serial(t)


def test_serials_are_strict_and_typing_from_a_label_is_forgiving():
    for bad in ("", "DO1-2641-00017", "XX1-2641-00017-A", "DO0-2641-00017-A", "DO1-2699-00017-A", None, 5):
        with pytest.raises(R.RecordError):
            R.parse_serial(bad)
    s = R.make_serial(1, 26, 41, 17)
    assert R.normalize_serial(s.lower()) == s and R.normalize_serial(s.replace("-", " ")) == s
    assert R.normalize_serial(s.replace("-", "")) == s
    with pytest.raises(R.RecordError):
        R.normalize_serial("DO1-2641-00017-" + ("A" if s[-1] != "A" else "B"))
    with pytest.raises(R.RecordError):
        R.make_serial(1, 26, 54, 1)


# ------------------------------------------------------------------------------------------------------------------------- attestation
def test_the_attestation_format_is_pinned_by_a_fixed_vector():
    att, vpub = golden_att()
    f = DATA / "attestation_v1.json"
    assert f.exists(), f"write tests/data/attestation_v1.json with:\n{json.dumps(att.to_json(), sort_keys=True, indent=1)}"
    assert json.loads(f.read_text()) == att.to_json()
    back = R.Attestation.from_json(json.loads(f.read_text()))
    assert back.verify(vpub) and back == att


def test_every_signed_field_of_the_attestation_is_protected():
    att, vpub = golden_att()
    j = att.to_json()
    others = {"serial": R.make_serial(1, 26, 41, 18), "hw": "444f485730303032", "dak": j["board"], "board": j["dak"], "date": "2027-01-01",
              "vendor": "0" * 16}
    for field, value in others.items():
        t = dict(j, **{field: value})
        try:
            a = R.Attestation.from_json(t)
        except R.RecordError:
            continue                                                         # refused already by the strict parser
        assert not a.verify(vpub), field
    sig = bytearray(att.sig)
    sig[0] ^= 1
    assert not R.Attestation(att.serial, att.hw_id, att.dak, att.board, att.date, att.vendor, bytes(sig)).verify(vpub)
    assert not att.verify(key("another-vendor")[1])


def test_a_malformed_attestation_is_refused_not_guessed():
    att, _ = golden_att()
    j = att.to_json()
    for bad in (None, [], {}, dict(j, v=2), dict(j, serial="nope"), dict(j, dak="***"), dict(j, hw="zz"), dict(j, sig=5),
                {k: v for k, v in j.items() if k != "sig"}):
        with pytest.raises(R.RecordError):
            R.Attestation.from_json(bad)


# ------------------------------------------------------------------------------------------------------------------------------ chain
def owner_card(tmp_path, name="o"):
    ident = I.load_identity(I.FileKeyStore(tmp_path / name / "keys.json"))
    return ident, ident.card()


def chained(tmp_path, att, dak_priv, name="o"):
    ident, card = owner_card(tmp_path, name)
    cert = R.DeviceCert(att, dak_priv.sign(R.DeviceCert.body(att.serial, ident.digest)))
    return ident, I.Card(card.ed, card.x, card.label, card.created, card.sig, cert.to_json())


def test_the_chain_manufacturer_to_dak_to_owner_card(tmp_path):
    att, vpub = golden_att()
    dak_priv, _ = key("golden-dak")
    ident, card = chained(tmp_path, att, dak_priv)
    assert R.verify_chain(card, vpub, HW) == att.serial
    again = I.Card.from_json(json.loads(json.dumps(card.to_json())))                  # and it survives the JSON the phone carries
    assert R.verify_chain(again, vpub) == att.serial and again.verify()


def test_the_chain_refuses_every_kind_of_forgery(tmp_path):
    att, vpub = golden_att()
    dak_priv, _ = key("golden-dak")
    ident, card = chained(tmp_path, att, dak_priv)
    other_vendor = key("evil-vendor")[1]
    with pytest.raises(R.RecordError) as e:
        R.verify_chain(card, other_vendor)
    assert e.value.key == "signature"
    with pytest.raises(R.RecordError) as e:
        R.verify_chain(card, vpub, b"OTHER-HW")
    assert e.value.key == "hardware"
    ident2, card2 = owner_card(tmp_path, "p")                                          # somebody else's card with this device's certificate
    stolen = I.Card(card2.ed, card2.x, card2.label, card2.created, card2.sig, card.device)
    with pytest.raises(R.RecordError) as e:
        R.verify_chain(stolen, vpub)
    assert e.value.key == "binding"
    with pytest.raises(R.RecordError) as e:
        R.verify_chain(card2, vpub)                                                    # no certificate at all
    assert e.value.key == "binding"
    # a certificate signed by a key the attestation does not name
    evil_dak, _ = key("evil-dak")
    _, forged = chained(tmp_path, att, evil_dak, "q")
    with pytest.raises(R.RecordError) as e:
        R.verify_chain(forged, vpub)
    assert e.value.key == "binding"


def test_a_card_without_the_device_block_is_still_a_valid_card_and_a_bad_block_is_refused(tmp_path):
    ident, card = owner_card(tmp_path)
    j = card.to_json()
    assert "device" not in j and I.Card.from_json(j).device is None
    for bad in ("x", [], 5):
        with pytest.raises(I.CardError):
            I.Card.from_json(dict(j, device=bad))
    with pytest.raises(I.CardError):
        I.Card.from_json(dict(j, device={"junk": "x" * 3000}))


# ----------------------------------------------------------------------------------------------------------------------- service token
def test_a_service_token_is_bound_to_serial_action_and_nonce():
    vend, vpub = key("golden-vendor")
    t = R.ServiceToken("DO1-2641-00017-" + R.make_serial(1, 26, 41, 17)[-1], "factory_return", bytes(range(16)))
    t = R.ServiceToken(t.serial, t.action, t.nonce, vend.sign(t.body()))
    assert R.ServiceToken.from_json(t.to_json()) == t
    for bad in (None, {}, dict(t.to_json(), action="format_disk"), dict(t.to_json(), nonce="00"), dict(t.to_json(), serial="x")):
        with pytest.raises(R.RecordError):
            R.ServiceToken.from_json(bad)
    assert t.body().startswith(R.SVC_DOMAIN) and R.SVC_DOMAIN not in (R.ATT_DOMAIN, R.DEV_DOMAIN, R.POP_DOMAIN)


# --------------------------------------------------------------------------------------------------------------------- lifecycle, map
def test_the_lifecycle_only_moves_forward_except_through_service():
    assert R.LIFECYCLE == (R.BLANK, R.DEV_TESTED, R.PROVISIONED, R.SHIPPED, R.IN_FIELD, R.RMA)
    for a, nxt in R.TRANSITIONS.items():
        for b in nxt:
            assert R.LIFECYCLE.index(b) > R.LIFECYCLE.index(a) or (a, b) == (R.RMA, R.PROVISIONED)
    assert R.TRANSITIONS[R.RMA] == (R.PROVISIONED,) and R.BLANK not in sum(R.TRANSITIONS.values(), ())


def test_the_storage_map_and_the_reset_levels_agree():
    keys = [i.key for i in R.ITEMS]
    assert len(keys) == len(set(keys))
    assert [l[0] for l in R.LEVELS] == ["L1", "L2", "L3", "L4"]
    allowed = {"kept", "erased", "replaced", "reinstalled", "assist off"}
    for it in R.ITEMS:
        assert len(it.levels) == 4 and set(it.levels) <= allowed and it.mutability in (R.W_ONCE, R.W_MONOTONIC, R.W_USER, R.W_VENDOR)
        # no level touches what can only be written once or only grows
        if it.mutability in (R.W_ONCE, R.W_MONOTONIC):
            assert set(it.levels) == {"kept"}, it.key
        if it.levels[3] == "reinstalled":                                              # only a vendor-controlled thing is ever reinstalled
            assert it.mutability == R.W_VENDOR
    for k in ("serial", "dak", "vendor_key", "rollback_floor", "golden", "lifecycle"):
        assert R.item(k).levels == ("kept",) * 4
    assert R.item("fw_som").levels[:3] == ("kept",) * 3 and R.item("fw_mcu").levels[3] == "reinstalled"
    # each level erases at least what the one below it erases
    for lvl in range(4):
        for it in R.ITEMS:
            if it.mutability == R.W_USER and lvl:
                lower, this = it.levels[lvl - 1], it.levels[lvl]
                assert not (lower in ("erased", "replaced") and this == "kept"), (it.key, lvl)
