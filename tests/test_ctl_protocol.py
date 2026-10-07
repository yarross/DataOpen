"""CtlLink v1: messages, chunks, the fixed status/info layouts, and the generated artifacts the browser client is built from."""
import json
import random
import zlib
from pathlib import Path

import pytest

from dataopen.ctl import protocol as P

ROOT = Path(__file__).resolve().parents[1]


def rnd_message(rng: random.Random, n: int | None = None) -> bytes:
    n = rng.randrange(0, 3000) if n is None else n
    body = bytes(rng.randrange(256) for _ in range(n))
    return P.pack_message(rng.choice(list(P.TYPES.values())), rng.randrange(65536), body, rng.randrange(65536))


def test_crc32_is_zlib():
    assert P.crc32(b"123456789") == 0xCBF43926
    for b in (b"", b"a", bytes(range(256)), bytes(5000)):
        assert P.crc32(b) == zlib.crc32(b)


def test_message_roundtrip_for_every_type_and_size():
    rng = random.Random(1)
    for typ in P.TYPES.values():
        for n in (0, 1, 19, 20, 21, 255, 1000, P.MAX_BODY):
            body = bytes(rng.randrange(256) for _ in range(n))
            raw = P.pack_message(typ, 0xBEEF, body, 0x1234)
            m = P.unpack_message(raw)
            assert m == P.Message(typ, 0xBEEF, body, 0x1234)
            assert len(raw) == P.HDR + n + 4


def test_json_helpers_are_utf8_and_compact():
    raw = P.pack_json(P.T_EVENT, 3, {"msg": "Помощь выключена"})
    m = P.unpack_message(raw)
    assert m.json() == {"msg": "Помощь выключена"} and m.body == '{"msg":"Помощь выключена"}'.encode()
    with pytest.raises(ValueError):
        P.Message(P.T_SET, 0, b"[1,2]").json()


def test_body_over_limit_cannot_be_packed():
    with pytest.raises(ValueError):
        P.pack_message(P.T_DATA, 0, bytes(P.MAX_BODY + 1))


def test_every_single_bit_flip_is_rejected():
    raw = P.pack_json(P.T_SET, 7, {"key": "assist.strength", "value": 7})
    for i in range(len(raw) * 8):
        b = bytearray(raw)
        b[i // 8] ^= 1 << (i % 8)
        assert P.unpack_message(bytes(b)) is None, i


def test_truncated_extended_and_wrong_version_messages_are_rejected():
    raw = P.pack_message(P.T_DATA, 1, b"hello")
    for n in range(len(raw)):
        assert P.unpack_message(raw[:n]) is None
    assert P.unpack_message(raw + b"\0") is None
    bad = bytearray(raw)
    bad[0] = 2
    bad[-4:] = zlib.crc32(bytes(bad[:-4])).to_bytes(4, "little")
    assert P.unpack_message(bytes(bad)) is None                     # right CRC, wrong version


@pytest.mark.parametrize("size", [20, 21, 64, 100, 185, 244])
def test_chunks_roundtrip_at_every_size(size):
    rng = random.Random(size)
    seq = 0
    r = P.Reassembler()
    for _ in range(40):
        raw = rnd_message(rng)
        chunks, seq = P.chunk_message(raw, size, seq)
        assert all(len(c) <= size for c in chunks)
        assert chunks[0][0] & 0x80 and chunks[-1][0] & 0x40
        got = None
        for c in chunks:
            got = r.feed(c, 0)
        assert got == raw and P.unpack_message(got) is not None
    assert r.dropped == 0


def test_safety_messages_fit_in_one_default_chunk():
    for typ in (P.T_STOP, P.T_HARD_BYPASS):
        chunks, _ = P.chunk_message(P.pack_message(typ, 0xFFFF), P.CHUNK_DEFAULT)
        assert len(chunks) == 1 and len(chunks[0]) <= 20


def test_chunk_size_limits():
    with pytest.raises(ValueError):
        P.chunk_message(b"x", 19)
    with pytest.raises(ValueError):
        P.chunk_message(b"x", 245)


def test_reassembler_never_wedges():
    rng = random.Random(3)
    a, b = rnd_message(rng, 400), rnd_message(rng, 300)
    ca, sa = P.chunk_message(a, 40, 0)
    cb, _ = P.chunk_message(b, 40, sa)
    r = P.Reassembler()
    # a lost middle chunk drops the message, and the next one is still received
    for c in ca[:2] + ca[3:]:
        assert r.feed(c, 0) is None
    out = None
    for c in cb:
        out = r.feed(c, 0)
    assert out == b
    # a new FIRST in the middle of a message restarts cleanly
    r = P.Reassembler()
    r.feed(ca[0], 0)
    r.feed(ca[1], 0)
    out = None
    for c in cb:
        out = r.feed(c, 0)
    assert out == b and r.dropped == 1
    # duplicates and chunks with no FIRST are dropped
    r = P.Reassembler()
    assert r.feed(ca[1], 0) is None and r.feed(b"", 0) is None
    r.feed(ca[0], 0)
    before = r.dropped
    assert r.feed(ca[0], 0) is None and r.dropped == before + 1         # a repeated FIRST is a restart, counted
    # a stalled message times out and does not poison the next
    r = P.Reassembler(timeout_us=1000)
    r.feed(ca[0], 0)
    assert r.feed(ca[1], 5000) is None and not r.active
    # an endless message is cut off at the size limit
    r = P.Reassembler()
    r.feed(bytes([0x80]) + bytes(100), 0)
    seq = 1
    for _ in range(400):
        r.feed(bytes([seq & 0x3F]) + bytes(P.CHUNK_MAX - 1), 0)
        seq += 1
    assert len(r.buf) <= P.MAX_MSG and r.dropped >= 1


def test_reassembler_survives_random_chunks():
    rng = random.Random(9)
    r = P.Reassembler()
    for i in range(20000):
        c = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40)))
        out = r.feed(c, i)
        if out is not None:
            P.unpack_message(out)
        assert len(r.buf) <= P.MAX_MSG


def test_status_layout_is_fixed_and_fits_the_default_mtu():
    assert P.STATUS_SIZE == 20 <= 23 - 3
    s = P.StatusSnapshot(1, 3, 8, P.SF_ASSIST_WANTED | P.SF_TRIAL, 70, P.RB_ASC | P.RB_TREMOR, 513, 4, 86400, 12, 6, 4)
    b = s.pack()
    assert len(b) == 20 and P.StatusSnapshot.unpack(b) == s
    assert s.reason_name == "PANIC"
    assert P.StatusSnapshot.unpack(b[:-1]) is None and P.StatusSnapshot.unpack(b"\x02" + b[1:]) is None


def test_status_values_saturate_instead_of_wrapping():
    s = P.StatusSnapshot(uptime_s=2**40, trial_left_s=10**6, state_rev=70000)
    u = P.StatusSnapshot.unpack(s.pack())
    assert u.trial_left_s == 0xFFFF and u.uptime_s == 2**40 & 0xFFFFFFFF and u.state_rev == 70000 & 0xFFFF


def test_info_layout():
    i = P.Info(3, b"\x01\x02\x03\x04", b"\xde\xad\xbe\xef", 4, 100)
    assert len(i.pack()) == 16 and P.Info.unpack(i.pack()) == i


def test_uuids_are_distinct_valid_and_private():
    import uuid
    u = [P.UUID_SERVICE, P.UUID_INFO, P.UUID_STATUS, P.UUID_CTL_IN, P.UUID_CTL_OUT]
    assert len(set(u)) == 5
    for x in u:
        assert str(uuid.UUID(x)) == x
        assert not x.endswith("-0000-1000-8000-00805f9b34fb")        # not a 16-bit SIG assigned number


def test_reason_table_matches_the_bridge_enum():
    from dataopen.bridge.cbridge import REASONS
    assert P.StatusSnapshot(reason=REASONS.index("STALE_LINK")).reason_name == "STALE_LINK"
    assert P.StatusSnapshot(reason=250).reason_name == "R250"


def test_error_codes_are_unique_and_nonzero():
    v = list(P.ERRORS.values())
    assert len(set(v)) == len(v) and min(v) >= 1


# ---- generated artifacts: what is committed must be what the generator writes
def test_golden_vectors_are_internally_consistent():
    g = P.golden()
    for m in g["messages"]:
        raw = bytes.fromhex(m["hex"])
        msg = P.unpack_message(raw)
        assert (msg.type, msg.req, msg.body.hex()) == (m["type"], m["req"], m["body"])
        for size, chunks in m["chunks"].items():
            r = P.Reassembler()
            out = None
            for c in chunks:
                assert len(bytes.fromhex(c)) <= int(size)
                out = r.feed(bytes.fromhex(c), 0)
            assert out == raw
    for s in g["status"]:
        assert P.StatusSnapshot.unpack(bytes.fromhex(s["hex"])).pack().hex() == s["hex"]


def test_committed_golden_file_is_current():
    f = ROOT / "pwa" / "tests" / "golden.json"
    assert f.exists(), "generate it: dataopen ctl golden --out pwa/tests/golden.json"
    assert json.loads(f.read_text()) == P.golden(), "regenerate: dataopen ctl golden --out pwa/tests/golden.json"


def test_committed_constants_are_current():
    f = ROOT / "pwa" / "js" / "constants.js"
    assert f.exists(), "generate it: dataopen ctl constants --out pwa/js/constants.js"
    assert f.read_text() == P.constants_js(), "regenerate: dataopen ctl constants --out pwa/js/constants.js"
