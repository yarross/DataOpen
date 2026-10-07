"""Encrypted data at rest and the crypto-erase."""
import pytest

from dataopen.bioprofile.profile import ProfileState, Stat
from dataopen.bioprofile.store import ProfileStore
from dataopen.ctl.vault import Vault, VaultError


def state(n=20):
    s = ProfileState(profile_id=77, deg_per_count=0.02)
    s.stats["t_motor"] = Stat(231.5, 20.0, n)
    s.stats["jitter_hz"] = Stat(7.77, 0.5, n)
    return s


def test_seal_and_open_round_trip_and_hide_the_content():
    v = Vault(bytes(range(32)))
    data = b"the quick brown fox jumps over the lazy dog" * 3
    blob = v.seal("profile", data)
    assert v.open("profile", blob) == data and data[:20] not in blob and blob[:4] == b"DOVT"
    assert v.seal("profile", data) != blob                                   # a fresh nonce every time


def test_a_blob_belongs_to_its_file_name_and_its_key():
    v, w = Vault(bytes(32)), Vault(bytes([1] * 32))
    blob = v.seal("settings", b"x" * 50)
    with pytest.raises(VaultError):
        v.open("profile", blob)                                              # dropped into the wrong file
    with pytest.raises(VaultError):
        w.open("settings", blob)                                             # another key


def test_every_bit_flip_is_detected():
    v = Vault(bytes(32))
    blob = v.seal("p", b"hello world, this is a profile")
    for i in range(len(blob) * 8):
        b = bytearray(blob)
        b[i // 8] ^= 1 << (i % 8)
        with pytest.raises(VaultError):
            v.open("p", bytes(b))


def test_the_profile_store_encrypts_its_slots_and_reads_them_back(tmp_path):
    v = Vault(bytes(32))
    st = ProfileStore(tmp_path / "profile", codec=v)
    assert st.save(state()) == 1 and st.save(state(30)) == 2
    for slot in st.slots:
        raw = slot.read_bytes()
        assert raw[:4] == b"DOVT" and b"BIOP" not in raw
    got = st.load()
    assert got.generation == 2 and got.stats["t_motor"].n == 30
    assert ProfileStore(tmp_path / "profile", codec=v).load().generation == 2


def test_an_older_plain_store_is_read_and_becomes_sealed_on_the_next_save(tmp_path):
    plain = ProfileStore(tmp_path / "profile")
    plain.save(state())
    v = Vault(bytes(32))
    st = ProfileStore(tmp_path / "profile", codec=v)
    assert st.load().generation == 1
    st.save(state())
    st.save(state())
    assert all(s.read_bytes()[:4] == b"DOVT" for s in st.slots)


def test_without_a_codec_the_store_is_unchanged(tmp_path):
    st = ProfileStore(tmp_path / "profile")
    st.save(state())
    assert st.slots[0].read_bytes()[:4] == b"BIOP" or st.slots[1].read_bytes()[:4] == b"BIOP"


def test_after_the_key_is_replaced_the_old_files_are_unreadable(tmp_path):
    v = Vault(bytes(32))
    st = ProfileStore(tmp_path / "profile", codec=v)
    st.save(state())
    st.save(state())
    saved = [s.read_bytes() for s in st.slots]
    fresh = ProfileStore(tmp_path / "profile", codec=Vault(bytes([9] * 32)))
    assert fresh.load() is None                                              # the old ciphertext is just noise now
    for b in saved:
        with pytest.raises(VaultError):
            Vault(bytes([9] * 32)).open("profile", b)
    assert fresh.save(state()) == 1                                          # and the store starts afresh instead of failing
