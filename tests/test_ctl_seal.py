"""The sealed settings file: it opens on one device only, and nothing about it can change without being noticed."""
import hashlib
import random
import struct
from pathlib import Path

import pytest

from dataopen.assist.sim_user import PERSONAS, build_profile
from dataopen.bioprofile.profile import ProfileState, Stat
from dataopen.ctl import identity as I
from dataopen.ctl import manifest as M
from dataopen.ctl import seal as S

DATA = Path(__file__).parent / "data"


class DetRng:
    """A reproducible byte stream (SHA-256 in counter mode): ONLY for the fixed test vector, never for keys that matter."""

    def __init__(self, tag: str) -> None:
        self.tag, self.n = tag.encode(), 0

    def __call__(self, k: int) -> bytes:
        out = b""
        while len(out) < k:
            self.n += 1
            out += hashlib.sha256(self.tag + struct.pack("<I", self.n)).digest()
        return out[:k]


def ident(tmp_path, name):
    return I.load_identity(I.FileKeyStore(tmp_path / name / "keys.json"))


@pytest.fixture(scope="module")
def profile():
    return build_profile(PERSONAS["tremor"], minutes=3, seed=1)._state


@pytest.fixture()
def pair(tmp_path):
    return ident(tmp_path, "a"), ident(tmp_path, "b"), ident(tmp_path, "c")


def test_round_trip_of_every_section(pair, profile):
    a, b, _ = pair
    man = M.default_manifest(7)
    raw = S.seal(a, b.card(), 5, profile=profile, tuning=(8, 2), manifest=man, meta={"name": "Моя копия", "created": "2026-10-07"})
    o = S.open_sealed(raw, b)
    assert o.sender_digest == a.digest and o.sender_id == a.id and o.seq == 5 and not o.is_self
    assert o.profile.pack() == profile.pack() and o.tuning == (8, 2) and o.manifest == man
    assert o.meta == {"name": "Моя копия", "created": "2026-10-07"}
    assert (o.sender_ed, o.sender_x) == (a.ed_pub, a.x_pub) and len(o.bundle_id) == 16


def test_a_copy_for_oneself_is_marked_and_recognised(pair, profile):
    a, _, _ = pair
    o = S.open_sealed(S.seal(a, a.card(), 1, profile=profile), a)
    assert o.is_self and o.flags & S.F_SELF


def test_the_self_flag_is_not_taken_on_trust(pair, profile):
    a, b, _ = pair
    raw = bytearray(S.seal(a, b.card(), 1, profile=profile))
    assert not S.open_sealed(bytes(raw), b).is_self                       # sealed by a for b: never 'self' on b


def test_a_file_made_for_one_device_is_noise_to_every_other(pair, profile):
    a, b, c = pair
    raw = S.seal(a, b.card(), 1, profile=profile, tuning=(5, 5))
    for who in (a, c):
        with pytest.raises(S.SealError) as e:
            S.open_sealed(raw, who)
        assert e.value.key == "wrong_device"
    assert S.open_sealed(raw, b).profile is not None


def test_a_genuine_file_cannot_be_readdressed_to_a_third_device(pair, profile):
    a, b, c = pair
    raw = bytearray(S.seal(a, b.card(), 1, profile=profile))
    raw[8 + 16 : 8 + 16 + 8] = c.fp8                                      # rewrite the recipient's short ID in the header
    with pytest.raises(S.SealError) as e:
        S.open_sealed(bytes(raw), c)
    assert e.value.key == "bad_signature"                                  # the signature covers the header


def test_a_sender_signed_file_whose_key_is_for_someone_else_is_not_accepted(pair, profile, monkeypatch):
    a, b, c = pair
    raw = S.seal(a, c.card(), 1, profile=profile)                         # encrypted to c ...
    h = bytearray(raw[: S.HEAD])
    h[8 + 16 : 8 + 24] = b.fp8                                            # ... but the (re-signed by a) header says b
    # a malicious sender can do this with its own key: a signs a header that names b over a ciphertext that only c can read
    from dataopen.ctl.seal import HEAD
    ct = raw[HEAD : len(raw) - S.SIG]
    forged = bytes(h) + ct + a.sign(b"DOBS-v2-sig" + bytes(h) + ct)
    with pytest.raises(S.SealError) as e:
        S.open_sealed(forged, b)
    assert e.value.key == "wrong_device"


def test_every_single_bit_of_the_file_is_protected(pair, profile):
    a, b, _ = pair
    raw = S.seal(a, b.card(), 3, profile=profile, tuning=(6, 4), meta={"name": "x"})
    assert len(raw) == 496
    bad = 0
    for i in range(len(raw) * 8):
        x = bytearray(raw)
        x[i // 8] ^= 1 << (i % 8)
        with pytest.raises(S.SealError):
            S.open_sealed(bytes(x), b)
        bad += 1
    assert bad == len(raw) * 8


def test_truncation_growth_and_junk_are_refused(pair, profile):
    a, b, _ = pair
    raw = S.seal(a, b.card(), 1, profile=profile)
    for n in list(range(0, S.HEAD + 80, 9)) + [len(raw) - 1, len(raw) - 64]:
        with pytest.raises(S.SealError):
            S.open_sealed(raw[:n], b)
    for extra in (b"\0", b"x" * 100):
        with pytest.raises(S.SealError):
            S.open_sealed(raw + extra, b)
    with pytest.raises(S.SealError):
        S.open_sealed(bytes(S.MAX_FILE + 2000), b)


def test_versions_and_suites_are_checked_not_guessed(pair, profile):
    a, b, _ = pair
    raw = S.seal(a, b.card(), 1, profile=profile)

    def with_(off, val):
        x = bytearray(raw)
        x[off] = val
        h, ct = bytes(x[: S.HEAD]), bytes(x[S.HEAD : len(x) - S.SIG])
        return h + ct + a.sign(b"DOBS-v2-sig" + h + ct)                  # a validly re-signed file: only the field differs
    with pytest.raises(S.SealError) as e:
        S.open_sealed(with_(4, 3), b)
    assert e.value.key == "version"
    with pytest.raises(S.SealError) as e:
        S.open_sealed(with_(5, 2), b)
    assert e.value.key == "unsupported"
    with pytest.raises(S.SealError) as e:
        S.open_sealed(b"DOBN" + raw[4:], b)
    assert e.value.key == "damaged"


def test_a_sender_cannot_claim_to_be_somebody_else(pair, profile):
    a, b, c = pair
    raw = bytearray(S.seal(a, b.card(), 1, profile=profile))
    raw[8 + 16 + 8 : 8 + 16 + 16] = c.fp8                                  # claim to be c
    with pytest.raises(S.SealError) as e:
        S.open_sealed(bytes(raw), b)
    assert e.value.key == "bad_signature"
    forged = bytearray(S.seal(a, b.card(), 1, profile=profile))
    forged[8 + 16 + 16 + 8 : 8 + 16 + 16 + 8 + 32] = c.ed_pub              # c's public key, a's signature
    with pytest.raises(S.SealError):
        S.open_sealed(bytes(forged), b)


def test_the_sealed_file_does_not_leak_the_profile(pair, profile):
    a, b, _ = pair
    raw = S.seal(a, b.card(), 1, profile=profile, tuning=(6, 4), meta={"name": "secret name"})
    body = profile.pack()
    assert b"BIOP" not in raw and body[:16] not in raw and b"secret name" not in raw and b"strength" not in raw


def test_size_is_padded_so_it_says_little(pair, profile):
    a, b, _ = pair
    sizes = {len(S.seal(a, b.card(), 1, profile=profile, meta={"name": "x" * n})) for n in range(0, 30)}
    assert len(sizes) == 1                                                  # the name's length does not show
    big = len(S.seal(a, b.card(), 1, profile=profile, manifest=M.default_manifest()))
    assert big > min(sizes) and (big - S.HEAD - S.SIG - S.TAG) % S.PAD == 0


def test_two_seals_of_the_same_content_differ(pair, profile):
    a, b, _ = pair
    one, two = S.seal(a, b.card(), 1, profile=profile), S.seal(a, b.card(), 1, profile=profile)
    assert one != two and one[S.HEAD:-S.SIG] != two[S.HEAD:-S.SIG]


def test_what_is_inside_is_checked_before_anything_is_returned(pair, profile):
    a, b, _ = pair
    bad_levels = S.seal(a, b.card(), 1, tuning=(99, 1))
    with pytest.raises(S.SealError):
        S.open_sealed(bad_levels, b)
    for tun in ((True, 1), (1.5, 1)):
        with pytest.raises(S.SealError):
            S.open_sealed(S.seal(a, b.card(), 1, tuning=tun), b)
    broken = ProfileState.unpack(profile.pack())
    raw = S.seal(a, b.card(), 1, profile=broken)
    assert S.open_sealed(raw, b).profile is not None
    with pytest.raises(S.SealError):                                       # nothing inside at all
        S.open_sealed(S.seal(a, b.card(), 1), b)


def test_unknown_and_reserved_sections_are_refused(pair):
    a, b, _ = pair
    # build files that carry a model section (reserved), an unknown one, a duplicate and bad padding, each properly sealed and signed
    def forge(sections=None, inner=None):
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        ct_plain = inner if inner is not None else S._tlv(sections)
        rng = DetRng("forge")
        nonce, bid, seed = rng(12), rng(16), rng(32)
        from cryptography.hazmat.primitives.asymmetric import x25519
        eph = x25519.X25519PrivateKey.from_private_bytes(seed)
        from cryptography.hazmat.primitives import serialization as ser
        ep = eph.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)
        key = S._key(eph.exchange(x25519.X25519PublicKey.from_public_bytes(b.x_pub)), ep, b.x_pub, a.digest, b.digest, bid)
        h = struct.pack(S.HEAD_FMT, S.MAGIC, 2, 1, 0, bid, b.fp8, a.fp8, 1, a.ed_pub, a.x_pub, ep, nonce, len(ct_plain) + 16)
        ct = ChaCha20Poly1305(key).encrypt(nonce, ct_plain, h)
        return h + ct + a.sign(b"DOBS-v2-sig" + h + ct)
    ok = forge([(S.S_TUNING, b'{"strength":5,"tremor":5}')])
    assert S.open_sealed(ok, b).tuning == (5, 5)
    for secs, key in (([(S.S_MODEL, b"weights")], "unsupported"), ([(99, b"?")], "unsupported"),
                      ([(S.S_TUNING, b'{"strength":5,"tremor":5}')] * 2, "damaged")):
        with pytest.raises(S.SealError) as e:
            S.open_sealed(forge(secs), b)
        assert e.value.key == key
    padded = bytearray(S._tlv([(S.S_TUNING, b'{"strength":5,"tremor":5}')]))
    padded[-1] = 1                                                          # a non-zero byte in the padding
    with pytest.raises(S.SealError) as e:
        S.open_sealed(forge(inner=bytes(padded)), b)
    assert e.value.key == "damaged"


def test_the_recipients_card_must_be_genuine(pair, profile):
    a, b, _ = pair
    good = b.card()
    forged = I.Card(good.ed, good.x, "x", good.created, good.sig)           # the signature no longer matches the card
    with pytest.raises(S.SealError):
        S.seal(a, forged, 1, profile=profile)


def test_garbage_never_crashes_the_opener(pair, profile):
    a, b, _ = pair
    rng = random.Random(7)
    raw = S.seal(a, b.card(), 1, profile=profile, tuning=(5, 5))
    for _ in range(600):
        x = bytearray(raw)
        for _ in range(rng.randrange(1, 6)):
            i = rng.randrange(len(x))
            x[i] = (x[i] + rng.randrange(1, 256)) % 256                 # a byte that really changes
        if rng.random() < 0.3:
            x = x[: rng.randrange(len(x))]
        if bytes(x) == raw:                                             # two edits of one byte can cancel out: that is no mutation
            continue
        with pytest.raises(S.SealError):
            S.open_sealed(bytes(x), b)
    for _ in range(300):
        junk = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 700)))
        with pytest.raises(S.SealError):
            S.open_sealed(b"DOBS" + junk, b)


def _golden_parties():
    ra, rb = DetRng("golden-a"), DetRng("golden-b")

    def mk(rng, name):
        class Mem:
            d = None

            def load(self):
                return self.d

            def save(self, d):
                self.d = {"v": 1, **d}

            def wipe(self):
                self.d = None
        return I.provision(Mem(), rng=rng)
    return mk(ra, "a"), mk(rb, "b")


def _golden_state():
    s = ProfileState(profile_id=1234, generation=3, deg_per_count=0.02, latency_comp_us=2083)
    s.stats["t_motor"] = Stat(231.5, 20.0, 30)
    s.stats["jitter_amp"] = Stat(0.31, 0.04, 30)
    s.stats["jitter_hz"] = Stat(7.5, 0.4, 30)
    s.counts["flick"] = 40
    s.rates["overshoot"] = 0.25
    return s


def test_the_file_format_is_pinned_by_a_fixed_vector():
    """If this fails the FORMAT changed. That needs a new version number, not a new vector."""
    a, b = _golden_parties()
    meta = {"name": "golden", "created": "2026-10-07"}
    raw = S.seal(a, b.card(), 9, profile=_golden_state(), tuning=(7, 3), meta=meta, rng=DetRng("seal"))
    golden = (DATA / "dobs_v2.hex").read_text().strip() if (DATA / "dobs_v2.hex").exists() else None
    assert golden is not None, f"write tests/data/dobs_v2.hex with:\n{raw.hex()}"
    assert raw.hex() == golden
    o = S.open_sealed(bytes.fromhex(golden), b)
    assert (o.seq, o.tuning, o.meta["name"], o.profile.profile_id) == (9, (7, 3), "golden", 1234)
    assert (a.id, b.id) == ("TTD9-SX68-FH7N-88T2", "G3K2-QKCD-F94N-3A5S")           # the fixed keys give fixed IDs


# ---------------------------------------------------------------------------------------------------------------- slots in a file
def forge_file(a, b, inner):
    """A properly sealed and signed file whose INSIDE is whatever the test builds (what a hostile but genuine sender could send)."""
    from cryptography.hazmat.primitives import serialization as ser
    from cryptography.hazmat.primitives.asymmetric import x25519
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    rng = DetRng("forge-slots")
    nonce, bid, seed = rng(12), rng(16), rng(32)
    eph = x25519.X25519PrivateKey.from_private_bytes(seed)
    ep = eph.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)
    key = S._key(eph.exchange(x25519.X25519PublicKey.from_public_bytes(b.x_pub)), ep, b.x_pub, a.digest, b.digest, bid)
    h = struct.pack(S.HEAD_FMT, S.MAGIC, 2, 1, 0, bid, b.fp8, a.fp8, 1, a.ed_pub, a.x_pub, ep, nonce, len(inner) + 16)
    ct = ChaCha20Poly1305(key).encrypt(nonce, inner, h)
    return h + ct + a.sign(b"DOBS-v2-sig" + h + ct)


def slot_body(entries):
    """entries: [(number, [(section, bytes)])] -> the content of the slot section."""
    body = bytes([len(entries)])
    for n, secs in entries:
        inner = S._tlv(secs, pad=1)
        body += struct.pack("<BI", n, len(inner)) + inner
    return body


TUN = (S.S_TUNING, b'{"strength":5,"tremor":5}')


def test_slots_round_trip_by_number_with_everything_a_slot_holds(pair, profile):
    a, b, _ = pair
    man = M.default_manifest(3)
    slots = [S.SlotData(0, profile, (8, 2), None, "Работа"), S.SlotData(3, None, (1, 9), man, "")]
    o = S.open_sealed(S.seal(a, b.card(), 4, slots=slots, meta={"name": "x"}), b)
    assert [s.n for s in o.slots] == [0, 3] and o.profile is None and o.tuning is None and o.manifest is None
    s0, s3 = o.slots
    assert s0.profile.pack() == profile.pack() and s0.tuning == (8, 2) and s0.name == "Работа" and s0.manifest is None
    assert s3.profile is None and s3.tuning == (1, 9) and s3.manifest == man and s3.name == ""


def test_a_file_is_about_the_active_slot_or_about_named_slots_never_both(pair, profile):
    a, b, _ = pair
    with pytest.raises(S.SealError):
        S.seal(a, b.card(), 1, profile=profile, slots=[S.SlotData(1, None, (5, 5))])
    mixed = S._tlv([(S.S_TUNING, b'{"strength":5,"tremor":5}'), (S.S_SLOT, slot_body([(1, [TUN])]))])
    with pytest.raises(S.SealError) as e:
        S.open_sealed(forge_file(a, b, mixed), b)
    assert e.value.key == "damaged"


def test_slot_numbers_must_exist_and_be_unique(pair):
    a, b, _ = pair
    for n in (4, 9, 255):
        with pytest.raises(S.SealError):
            S.seal(a, b.card(), 1, slots=[S.SlotData(n, None, (5, 5))])
        with pytest.raises(S.SealError) as e:
            S.open_sealed(forge_file(a, b, S._tlv([(S.S_SLOT, slot_body([(n, [TUN])]))])), b)
        assert e.value.key == "damaged"
    with pytest.raises(S.SealError):
        S.seal(a, b.card(), 1, slots=[S.SlotData(1, None, (5, 5)), S.SlotData(1, None, (6, 6))])
    with pytest.raises(S.SealError) as e:
        S.open_sealed(forge_file(a, b, S._tlv([(S.S_SLOT, slot_body([(1, [TUN]), (1, [TUN])]))])), b)
    assert e.value.key == "damaged"


def test_a_slot_holds_only_what_belongs_in_a_slot(pair):
    a, b, _ = pair
    nested = slot_body([(0, [(S.S_SLOT, slot_body([(1, [TUN])]))])])
    for body, key in ((nested, "unsupported"), (slot_body([(0, [(S.S_MODEL, b"weights")])]), "unsupported"),
                      (slot_body([(0, [(99, b"?")])]), "unsupported"), (slot_body([(0, [])]), "damaged"), (b"", "damaged"),
                      (b"\x00", "damaged"), (slot_body([(0, [TUN])]) + b"x", "damaged"),
                      (slot_body([(0, [TUN])])[:-3], "damaged"), (bytes([2]) + slot_body([(0, [TUN])])[1:], "damaged")):
        with pytest.raises(S.SealError) as e:
            S.open_sealed(forge_file(a, b, S._tlv([(S.S_SLOT, body)])), b)
        assert e.value.key == key, body[:12]
    twice = S._tlv([(S.S_SLOT, slot_body([(0, [TUN])]))] * 2)
    with pytest.raises(S.SealError):
        S.open_sealed(forge_file(a, b, twice), b)                          # the section itself can come only once: a slot only once


def test_the_content_of_every_slot_is_checked_like_a_flat_file(pair, profile):
    a, b, _ = pair
    for bad in (b'{"strength":99,"tremor":1}', b'{"strength":true,"tremor":1}', b"not json", b'{"strength":1}'):
        with pytest.raises(S.SealError):
            S.open_sealed(forge_file(a, b, S._tlv([(S.S_SLOT, slot_body([(2, [(S.S_TUNING, bad)])]))])), b)
    broken = bytearray(profile.pack())
    broken[10] ^= 0xFF
    with pytest.raises(S.SealError):
        S.open_sealed(forge_file(a, b, S._tlv([(S.S_SLOT, slot_body([(2, [(S.S_PROFILE, bytes(broken))])]))])), b)


def test_slots_do_not_change_what_a_flat_file_looks_like(pair, profile):
    """A device that knows nothing of slots wrote exactly these bytes; the section is purely additive."""
    a, b, _ = pair
    assert S.encode_inner(profile, (5, 5)) == S._tlv([(S.S_PROFILE, profile.pack()), (S.S_TUNING, b'{"strength":5,"tremor":5}')])


def test_the_slot_file_format_is_pinned_by_a_fixed_vector():
    a, b = _golden_parties()
    slots = [S.SlotData(0, _golden_state(), (7, 3), None, "Work"), S.SlotData(2, None, (4, 6), None, "Browser")]
    raw = S.seal(a, b.card(), 10, slots=slots, meta={"name": "golden", "created": "2026-10-07"}, rng=DetRng("seal-slots"))
    f = DATA / "dobs_v2_slots.hex"
    assert f.exists(), f"write tests/data/dobs_v2_slots.hex with:\n{raw.hex()}"
    assert raw.hex() == f.read_text().strip()
    o = S.open_sealed(bytes.fromhex(f.read_text().strip()), b)
    assert [(s.n, s.name, s.tuning) for s in o.slots] == [(0, "Work", (7, 3)), (2, "Browser", (4, 6))]
    assert o.slots[0].profile.profile_id == 1234
