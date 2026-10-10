"""Channel B's file format `DOPK` and the model check: what a package is, what the header alone decides, what each damage looks like."""
import re
import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from dataopen.ctl import firmware as F
from dataopen.ctl.manifest import default_manifest
from dataopen.ctl.sim import seed_profile
from dataopen.ui.taxonomy import NAMES
from dataopen.updates import channels as C
from dataopen.updates import models as MD
from dataopen.updates import package as K

import pkg_helpers as H

SRC = Path(__file__).resolve().parents[1] / "src" / "dataopen"


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    d = tmp_path_factory.mktemp("pkg-keys")
    return H.sender(d, "alice"), H.sender(d, "device"), H.sender(d, "other")


@pytest.fixture(scope="module")
def profile(tmp_path_factory):
    d = tmp_path_factory.mktemp("pkg-profile")
    return seed_profile(d, "tremor")._state


def err(fn, *a, **kw):
    with pytest.raises(C.UpdateError) as e:
        fn(*a, **kw)
    return e.value


MODEL = H.tiny_model()
CARD = H.card_of(MODEL, name="icons", version=3)
BIG = bytes((i * 7 + i // 256) & 255 for i in range(6000))         # not a model: the FORMAT does not look inside, the device does (after)
BIG_CARD = H.card_of(BIG)


# ---------------------------------------------------------------------------------------------------------------- the format
def test_a_package_carries_every_kind_and_comes_out_the_same(keys, profile):
    alice, dev, _ = keys
    raw = K.build_package(alice, dev.card(), 7, slot=2, profile=profile, tuning=(4, 6), ui_manifest=default_manifest(),
                          name="Браузер", model=(MODEL, CARD), min_fw=1)
    o = K.open_package(raw, dev, fw_version=1)
    assert o.kind_names == ["profile", "tuning", "ui_manifest", "meta", "model"] and o.slot == 2
    assert o.tuning == (4, 6) and o.name == "Браузер"
    assert o.profile.pack() == profile.pack() and o.model == MODEL and o.model_card["name"] == "icons"
    assert o.seq == 7 and o.min_fw == 1 and not o.is_self and o.sender_digest == alice.digest and o.schema == C.PKG_SCHEMA
    assert raw[:4] == b"DOPK" and len(raw) > K.PREFIX + len(MODEL)


def test_nothing_readable_is_left_in_the_file(keys, profile):
    alice, dev, _ = keys
    raw = K.build_package(alice, dev.card(), 1, model=(MODEL, CARD), name="секретное имя")
    assert MODEL[:40] not in raw and "секретное имя".encode() not in raw and b"icons" not in raw
    assert K.inspect_header(raw)["kinds"] == ["meta", "model"]           # the header says what is inside, nothing more


def test_a_package_the_device_itself_seems_to_have_made_is_refused(keys):
    """The device makes no packages (docs/RESIDENCY.md): one that says it did can only come from stolen keys."""
    _, dev, _ = keys
    raw = K.build_package(dev, dev.card(), 1, tuning=(3, 3))
    assert K.inspect_header(raw)["self"]
    assert err(K.open_package, raw, dev).key == "own_file"


def test_chunk_size_changes_the_bytes_but_not_the_meaning(keys):
    alice, dev, _ = keys
    a = K.build_package(alice, dev.card(), 1, model=(MODEL, CARD), chunk_size=512)
    b = K.build_package(alice, dev.card(), 1, model=(MODEL, CARD), chunk_size=8192)
    assert a != b and K.open_package(a, dev).model == K.open_package(b, dev).model == MODEL
    assert err(K.build_package, alice, dev.card(), 1, tuning=(1, 1), chunk_size=100).key == "bad_part"
    assert err(K.build_package, alice, dev.card(), 1).key == "bad_part"           # nothing inside
    assert err(K.build_package, alice, dev.card(), 1, tuning=(1, 1), slot=4).key == "bad_part"


# --------------------------------------------------------------------------------------------------------- the header is judged first
def feed_until_refused(raw, me, piece=100, **kw):
    r = K.PackageReader(me, size=len(raw), **kw)
    off = 0
    while off < len(raw):
        try:
            off = r.chunk(off, raw[off : off + piece])
        except C.UpdateError as e:
            return e, off + piece
    return r.finish(), off


def test_a_forged_or_misaddressed_file_is_refused_before_the_data_arrives(keys):
    alice, dev, other = keys
    raw = K.build_package(alice, dev.card(), 1, model=(BIG, BIG_CARD))
    assert len(raw) > 6000
    forged = bytearray(raw)
    forged[40] ^= 1                                                      # one bit of the signed header
    e, got = feed_until_refused(bytes(forged), dev)
    assert e.key == "bad_signature" and got <= K.PREFIX + 100
    e, got = feed_until_refused(raw, other)                              # made for another device
    assert e.key == "wrong_device" and got <= K.PREFIX + 100
    e, got = feed_until_refused(raw, dev, last_seq=lambda d: 5)          # a sender whose sequence is already past this one
    assert e.key == "replay" and got <= K.PREFIX + 100
    e, got = feed_until_refused(K.build_package(alice, dev.card(), 1, tuning=(1, 1), schema=2), dev)
    assert e.key == "needs_update" and got <= K.PREFIX + 100
    e, got = feed_until_refused(K.build_package(alice, dev.card(), 1, tuning=(1, 1), min_fw=9), dev, fw_version=1)
    assert e.key == "needs_update" and got <= K.PREFIX + 100
    o, _ = feed_until_refused(K.build_package(alice, dev.card(), 1, tuning=(1, 1), min_fw=9), dev, fw_version=9)
    assert o.tuning == (1, 1)                                            # the same file on a system that is new enough


def test_the_signature_is_the_senders_and_nobody_elses(keys):
    alice, dev, other = keys
    raw = K.build_package(alice, dev.card(), 1, tuning=(1, 1))
    forged = bytearray(raw)
    forged[K.HEAD : K.PREFIX] = other.sign(b"DOPK-v1-sig" + raw[: K.HEAD])        # a valid signature, by somebody else
    assert err(K.open_package, bytes(forged), dev).key == "bad_signature"
    # a header re-signed by its own sender after an edit passes the signature; the data is bound to the header, so the edit shows there

    edited = bytearray(raw)
    struct.pack_into("<Q", edited, 4 + 1 + 1 + 2 + 16 + 8 + 8, 99)                # seq 99
    edited[K.HEAD : K.PREFIX] = alice.sign(b"DOPK-v1-sig" + bytes(edited[: K.HEAD]))
    assert err(K.open_package, bytes(edited), dev).key == "tampered"


def test_unknown_kinds_and_formats_are_not_guessed_at(keys):
    alice, dev, _ = keys
    raw = bytearray(K.build_package(alice, dev.card(), 1, tuning=(1, 1)))
    for off, fmt, val, key in ((4, "<B", 2, "version"), (5, "<B", 9, "unsupported")):
        x = bytearray(raw)
        struct.pack_into(fmt, x, off, val)
        x[K.HEAD : K.PREFIX] = alice.sign(b"DOPK-v1-sig" + bytes(x[: K.HEAD]))
        assert err(K.open_package, bytes(x), dev).key == key
    x = bytearray(raw)
    struct.pack_into("<H", x, 4 + 1 + 1 + 2 + 16 + 8 + 8 + 8 + 2, 0x80)             # kinds: a bit nobody defined
    x[K.HEAD : K.PREFIX] = alice.sign(b"DOPK-v1-sig" + bytes(x[: K.HEAD]))
    assert err(K.open_package, bytes(x), dev).key == "unsupported"


# ---------------------------------------------------------------------------------------------------------------- damage along the way
def test_every_kind_of_damage_to_the_body_is_named(keys):
    alice, dev, _ = keys
    raw = K.build_package(alice, dev.card(), 1, tuning=(2, 2), model=(BIG, BIG_CARD), chunk_size=512)
    n = (len(raw) - K.PREFIX)
    flipped = bytearray(raw)
    flipped[K.PREFIX + n // 2] ^= 1
    assert err(K.open_package, bytes(flipped), dev).key == "tampered"
    assert err(K.open_package, raw[:-300], dev).key in ("truncated", "damaged")      # the announced size no longer fits the header
    r = K.PackageReader(dev)                                                          # a stream that simply stops
    r.chunk(0, raw[:-300])
    assert err(r.finish).key == "truncated"
    r = K.PackageReader(dev)
    r.chunk(0, raw)
    with pytest.raises(C.UpdateError) as e:
        r.chunk(len(raw), b"more")
    assert e.value.key == "too_large"
    cs = 512 + K.TAG                                                                   # two ciphertext chunks swapped
    sw = bytearray(raw)
    a, b = K.PREFIX, K.PREFIX + cs
    sw[a : a + cs], sw[b : b + cs] = raw[b : b + cs], raw[a : a + cs]
    assert err(K.open_package, bytes(sw), dev).key == "tampered"
    dropped = raw[: K.PREFIX + cs] + raw[K.PREFIX + 2 * cs :]                          # a middle chunk dropped
    assert err(K.open_package, dropped, dev).key in ("truncated", "damaged", "tampered")


def test_the_last_chunk_flag_stops_truncation_that_keeps_whole_chunks(keys):
    alice, dev, _ = keys
    raw = K.build_package(alice, dev.card(), 1, model=(BIG, BIG_CARD), chunk_size=512)
    cs = 512 + K.TAG
    body = raw[K.PREFIX :]
    cut = raw[: K.PREFIX + (len(body) // cs) * cs]                                     # only whole chunks, the last one missing
    r = K.PackageReader(dev)
    r.chunk(0, cut)
    assert err(r.finish).key in ("truncated", "tampered", "hash")


def test_the_reader_resumes_repeats_and_refuses_gaps(keys):
    alice, dev, _ = keys
    raw = K.build_package(alice, dev.card(), 1, model=(MODEL, CARD))
    r = K.PackageReader(dev, size=len(raw))
    assert r.chunk(0, raw[:1000]) == 1000
    # an acknowledgement was lost: repeated, not applied twice
    assert r.chunk(0, raw[:1000]) == 1000 and r.chunk(500, raw[500:900]) == 1000
    assert err(r.chunk, 1200, raw[1200:1300]).key == "sequence"
    assert r.next == 1000 and r.chunk(1000, raw[1000:]) == len(raw)
    assert r.finish().model == MODEL
    assert err(K.PackageReader(dev, size=10).chunk, 0, raw[:50]).key == "too_large"


# --------------------------------------------------------------------------------------------------------- the channels share nothing
def test_each_channel_takes_only_its_own_file(keys):
    alice, dev, _ = keys
    pkg = K.build_package(alice, dev.card(), 1, tuning=(1, 1))
    vendor = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    pub = vendor.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    fw = F.build_image(b"fw" * 100, b"DOHW0001", 2, 1, vendor)
    assert C.channel_of(fw) == "A" and C.channel_of(pkg) == "B" and C.channel_of(b"DOBS....") == "B" and C.channel_of(b"junk") is None
    assert err(K.open_package, fw, dev).key == "wrong_channel"                            # a system image sent as a package
    assert err(K.open_package, b"DOBS" + bytes(300), dev).key == "unsupported"
    assert err(K.open_package, b"junk" + bytes(300), dev).key == "damaged"
    with pytest.raises(F.FirmwareError) as e:
        F.verify_image(pkg, pub, b"DOHW0001")                                             # and a package sent as a system image
    assert e.value.key == "damaged"


def test_a_senders_key_does_not_sign_firmware_and_the_manufacturers_does_not_make_a_sender(keys):
    alice, dev, _ = keys
    vendor = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    pub = vendor.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    by_alice = F.build_image(b"fw" * 100, b"DOHW0001", 2, 1, alice)                       # an Identity signs like any signer object
    with pytest.raises(F.FirmwareError) as e:
        F.verify_image(by_alice, pub, b"DOHW0001")
    assert e.value.key == "signature"
    # the manufacturer's key as a package sender is just another sender: nothing grants it the manufacturer's standing, so the first
    # package from it asks for the button like any stranger's (checked at the gateway, see test_updates_gateway)
    raw = K.build_package(alice, dev.card(), 1, tuning=(1, 1))
    assert not K.PackageReader(dev).known_sender and K.open_package(raw, dev).sender_digest != pub


# ---------------------------------------------------------------------------------------------------------------- the model check
def test_a_model_has_to_be_the_kind_of_thing_the_ui_path_may_run():
    ok = MD.check_model(MODEL, CARD)
    assert set(ok["ops"]) <= MD.ALLOWED_OPS and ok["card"]["version"] == 3 and ok["outputs"] == ["p3", "p4", "p5"]


@pytest.mark.parametrize("kw,key", [(dict(op="Gemm"), "model_ops"), (dict(layout=None), "bad_model"), (dict(external=True), "model_files"),
                                    (dict(domain="com.evil"), "model_files"), (dict(extra_domain="com.evil"), "model_files"),
                                    (dict(layout=dict(H.LAYOUT_OK, classes=["person"])), "bad_model"), (dict(opset=99), "model_ops")])
def test_a_graph_outside_the_rules_is_refused(kw, key):
    raw = H.tiny_model(**kw)
    assert err(MD.check_model, raw, H.card_of(raw, opset=min(kw.get("opset", 13), 20))).key == key


def test_a_pose_or_people_model_is_not_a_ui_model():
    for classes in (["person"], ["button", "keypoint"], list(NAMES) + ["person"]):
        assert err(MD.check_model, MODEL, dict(CARD, classes=classes)).key == "not_ui_model"


def test_the_card_has_to_describe_the_bytes():
    assert err(MD.check_model, MODEL + b"\0", CARD).key == "bad_model"                    # size and hash no longer match
    assert err(MD.check_model, MODEL, dict(CARD, sha256="0" * 64)).key == "bad_model"
    assert err(MD.check_model, MODEL, dict(CARD, input_size=320)).key == "bad_model"
    assert err(MD.check_model, MODEL, dict(CARD, format="tflite")).key == "bad_model"
    assert err(MD.check_model, b"not onnx", H.card_of(b"not onnx")).key == "bad_model"
    assert err(MD.check_model, MODEL, None).key == "bad_model"
    assert err(MD.check_model, MODEL, dict(CARD, name="x" * 41)).key == "bad_model"
    big = bytes(MD.MODEL_MAX + 1)
    assert err(MD.check_model, big, H.card_of(big)).key == "too_large"


def test_the_allowed_operators_are_the_ones_the_exporter_writes():
    pytest.importorskip("torch")
    from dataopen.ui import export as E
    assert MD.ALLOWED_OPS == frozenset(E.ALLOWED_OPS)


def test_the_real_exported_model_passes(tmp_path):
    pytest.importorskip("torch")
    from dataopen.ui.export import export_onnx
    from dataopen.ui.model import UiNet
    net = UiNet()
    export_onnx(net, tmp_path / "m.onnx")
    raw = (tmp_path / "m.onnx").read_bytes()
    found = MD.check_model(raw, H.card_of(raw, net.cfg.classes, name="synthetic", opset=13))
    assert "Conv" in found["ops"] and len(raw) < MD.MODEL_MAX


# ---------------------------------------------------------------------------------------------------------------- the table of reasons
def test_every_reason_the_code_can_give_has_a_row_and_every_row_has_text():
    keys_b, keys_a = set(), set()
    for p in (SRC / "updates").glob("*.py"):
        keys_b |= set(re.findall(r'UpdateError\(\s*"B",\s*"(\w+)"', p.read_text(encoding="utf-8")))
    for p in (SRC / "ctl" / "firmware.py", SRC / "ctl" / "gateway.py"):
        keys_a |= set(re.findall(r'FirmwareError\(\s*"(\w+)"', p.read_text(encoding="utf-8")))
    assert keys_b <= C.REASON_KEYS["B"], keys_b - C.REASON_KEYS["B"]
    assert keys_a <= C.REASON_KEYS["A"], keys_a - C.REASON_KEYS["A"]
    for r in C.REASONS:
        assert r.ru.strip() and r.en.strip() and r.ru != r.en and r.at and r.left and r.do, r
    assert len({(r.channel, r.key) for r in C.REASONS}) == len(C.REASONS)
    assert len(C.CHANNELS) == 2 and {c.key for c in C.CHANNELS} == {"A", "B"} and C.BY_KEY["A"].signer != C.BY_KEY["B"].signer
    with pytest.raises(C.UpdateError):
        C.check_compat(2, 0, 1, 5)
    C.check_compat(1, 5, 1, 5)
