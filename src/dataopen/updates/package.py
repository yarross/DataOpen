"""`DOPK` v1: a channel B package, sealed to ONE device, signed by its sender, made to be received in pieces (docs/UPDATES.md section 3).

    header (208 bytes, plain, covered by the signature):
        'DOPK' | ver=1 | suite=1 | flags u16 | pkg_id[16] | recipient_fp[8] | sender_fp[8] | seq u64 | schema u16 | kinds u16 | min_fw u32 |
        sender_ed_pub[32] | sender_x_pub[32] | eph_x_pub[32] | nonce_base[12] | chunk_size u32 | ct_len u64 | ct_sha256[32]
    signature (64 bytes)       Ed25519 by the sender over 'DOPK-v1-sig' || header          <- right after the header
    ciphertext (ct_len bytes)  STREAM: chunk i is ChaCha20-Poly1305(key, nonce_base xor i, aad = sha256(header) || i || last_flag)

    plaintext = u32 manifest_len || manifest (canonical JSON: slot, parts[type, size, sha256, info]) || part bytes in manifest order

Why the signature comes BEFORE the data. The header names the recipient, the sender, the sequence number, the schema, the system version
needed, the kinds of content and (through `ct_sha256`) every ciphertext byte that follows. A device checks all of that from the first 272
bytes: a forged file, a file for another device, a replay, a package that needs a newer system, one that is too large: all are refused before
a megabyte of it has been sent. What follows is then authenticated twice, chunk by chunk (the AEAD tag, which also catches reordering,
dropping and truncation through the index and the last-chunk flag) and as a whole (`ct_sha256`).

Suite 1 is X25519 + HKDF-SHA256 + ChaCha20-Poly1305 + Ed25519 from the `cryptography` package, the same as `DOBS` (ctl/seal.py); only
the labels differ, so a key made for one format can never open the other. The parts are the same section encodings as `DOBS` and are
decoded by the same function, so a profile, levels or layout mean exactly one thing in both formats."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from typing import Callable, Optional

from ..bioprofile.profile import ProfileState
from ..ctl import seal as SL
from ..ctl.identity import Card, Identity, _lib, digest_of, format_id
from ..ctl.slots import SLOT_COUNT, clean_name
from .channels import MAGIC_A, MAGIC_B_SMALL, PKG_SCHEMA, UpdateError, check_compat

MAGIC, VERSION, SUITE = b"DOPK", 1, 1
HEAD_FMT = "<4sBBH16s8s8sQHHI32s32s32s12sIQ32s"
HEAD = struct.calcsize(HEAD_FMT)
SIG, TAG = 64, 16
PREFIX = HEAD + SIG
CHUNK = 4096                           # plaintext bytes per AEAD chunk
MAX_PACKAGE = 20 * 1024 * 1024         # a 16 MiB model plus the rest
MAX_MANIFEST = 65536
F_SELF = 1
K_PROFILE, K_TUNING, K_UI, K_META, K_MODEL = 1, 2, 4, 8, 16
KINDS = {"profile": K_PROFILE, "tuning": K_TUNING, "ui_manifest": K_UI, "meta": K_META, "model": K_MODEL}
ALL_KINDS = sum(KINDS.values())
ORDER = ("profile", "tuning", "ui_manifest", "meta", "model")
assert HEAD == 208
# the header, field by field (the document's table is generated from this; a test keeps it equal to HEAD_FMT)
FIELDS = (
    ("magic", 4, "'DOPK'"), ("ver", 1, "формат, 1"), ("suite", 1, "набор примитивов, 1"), ("flags", 2, "бит 0: «для себя» (отправитель = получатель)"),
    ("pkg_id", 16, "случайный номер пакета (входит в ключ)"), ("recipient_fp", 8, "короткий отпечаток получателя: Device Public ID"),
    ("sender_fp", 8, "короткий отпечаток отправителя"), ("seq", 8, "порядковый номер отправителя, только растёт"),
    ("schema", 2, "схема пакета, которую нужно понимать"), ("kinds", 2, "что внутри (битовая маска: профиль, уровни, раскладка, имя, модель)"),
    ("min_fw", 4, "версия системы, не ниже которой пакет читается"), ("sender_ed", 32, "открытый ключ подписи отправителя"),
    ("sender_x", 32, "открытый ключ обмена отправителя"), ("eph_x", 32, "одноразовый открытый ключ обмена"), ("nonce_base", 12, "основа nonce частей"),
    ("chunk_size", 4, "байт открытого текста в одной части"), ("ct_len", 8, "длина шифртекста"), ("ct_sha256", 32, "SHA-256 всего шифртекста"),
)
assert sum(n for _, n, _ in FIELDS) == HEAD


@dataclass
class Opened:
    pkg_id: bytes
    sender_ed: bytes
    sender_x: bytes
    sender_digest: bytes
    seq: int
    is_self: bool
    schema: int
    min_fw: int
    kinds: int
    slot: Optional[int]
    profile: Optional[ProfileState]
    tuning: Optional[tuple[int, int]]
    ui_manifest: Optional[dict]
    name: str
    model: Optional[bytes]
    model_card: Optional[dict]
    size: int = 0

    @property
    def sender_id(self) -> str:
        return format_id(self.sender_digest)

    @property
    def kind_names(self) -> list[str]:
        return [n for n in ORDER if self.kinds & KINDS[n]]


def _key(shared: bytes, eph_pub: bytes, recipient_x: bytes, sender_digest: bytes, recipient_digest: bytes, pkg_id: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=eph_pub + recipient_x,
                info=b"DOPK v1 key" + sender_digest + recipient_digest + pkg_id).derive(shared)


def _nonce(base: bytes, i: int) -> bytes:
    return base[:4] + (int.from_bytes(base[4:], "big") ^ i).to_bytes(8, "big")


def _aad(head_hash: bytes, i: int, last: bool) -> bytes:
    return head_hash + i.to_bytes(8, "big") + (b"\x01" if last else b"\x00")


# ---------------------------------------------------------------------------------------------------------------- building
def build_package(sender: Identity, recipient: Card, seq: int, *, slot: Optional[int] = None, profile: Optional[ProfileState] = None,
                  tuning: Optional[tuple[int, int]] = None, ui_manifest: Optional[dict] = None, name: Optional[str] = None,
                  model: Optional[tuple[bytes, dict]] = None, min_fw: int = 0, schema: int = PKG_SCHEMA, chunk_size: int = CHUNK,
                  rng: Callable[[int], bytes] = os.urandom) -> bytes:
    """Seal what the arguments carry to `recipient` (a verified Card). `model` is (ONNX bytes, card from `models.card_for`)."""
    ser, _, x25519 = _lib()
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    if not recipient.verify():
        raise UpdateError("B", "bad_signature", "the recipient's card is not valid")
    if slot is not None and not (isinstance(slot, int) and 0 <= slot < SLOT_COUNT):
        raise UpdateError("B", "bad_part", "slot")
    if not 512 <= chunk_size <= 65536:
        raise UpdateError("B", "bad_part", "chunk size")
    parts: list[tuple[str, bytes, Optional[dict]]] = []
    if profile is not None:
        parts.append(("profile", profile.pack(), None))
    if tuning is not None:
        parts.append(("tuning", json.dumps({"strength": tuning[0], "tremor": tuning[1]}, separators=(",", ":")).encode(), None))
    if ui_manifest is not None:
        parts.append(("ui_manifest", json.dumps(ui_manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), None))
    if name is not None:
        parts.append(("meta", json.dumps({"name": clean_name(name)}, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), None))
    if model is not None:
        parts.append(("model", model[0], model[1]))
    if not parts:
        raise UpdateError("B", "bad_part", "nothing inside")
    entries = [{"t": t, "n": len(b), "h": hashlib.sha256(b).hexdigest(), **({"info": info} if info is not None else {})} for t, b, info in parts]
    manifest = json.dumps({"v": 1, "slot": slot, "parts": entries}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    plain = struct.pack("<I", len(manifest)) + manifest + b"".join(b for _, b, _ in parts)
    pkg_id, nonce, eph_seed = rng(16), rng(12), rng(32)
    eph = x25519.X25519PrivateKey.from_private_bytes(eph_seed)
    eph_pub = eph.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)
    key = _key(eph.exchange(x25519.X25519PublicKey.from_public_bytes(recipient.x)), eph_pub, recipient.x, sender.digest, recipient.digest, pkg_id)
    n = max(1, -(-len(plain) // chunk_size))
    ct_len = len(plain) + n * TAG
    kinds = sum(KINDS[t] for t, _, _ in parts)
    flags = F_SELF if recipient.digest == sender.digest else 0
    # the header carries the ciphertext hash, and the ciphertext is bound to the header: the associated data is the header with that one field
    # zeroed (the real hash is covered by the signature)
    head_core = struct.pack(HEAD_FMT, MAGIC, VERSION, SUITE, flags, pkg_id, recipient.digest[:8], sender.fp8, seq, schema, kinds, min_fw,
                            sender.ed_pub, sender.x_pub, eph_pub, nonce, chunk_size, ct_len, b"\0" * 32)
    aead, ct = ChaCha20Poly1305(key), bytearray()
    head_hash = hashlib.sha256(head_core).digest()
    for i in range(n):
        piece = plain[i * chunk_size : (i + 1) * chunk_size]
        ct += aead.encrypt(_nonce(nonce, i), piece, _aad(head_hash, i, i == n - 1))
    header = head_core[:-32] + hashlib.sha256(bytes(ct)).digest()
    if len(header) + SIG + len(ct) > MAX_PACKAGE + 4096:
        raise UpdateError("B", "too_large", "package")
    return header + sender.sign(b"DOPK-v1-sig" + header) + bytes(ct)


# ---------------------------------------------------------------------------------------------------------------- receiving
class PackageReader:
    """A package arriving in order, in pieces. The header and the signature are judged as soon as they are complete; the ciphertext is
    decrypted chunk by chunk as it arrives; `finish()` checks the whole and returns `Opened`. Any `UpdateError` ends the reception: the
    caller drops this reader and with it everything that was received."""

    def __init__(self, me: Identity, *, max_size: int = MAX_PACKAGE, supported_schema: int = PKG_SCHEMA, fw_version: int = 0,
                 last_seq: Callable[[bytes], Optional[int]] = lambda digest: None, size: Optional[int] = None, now_us: int = 0) -> None:
        self.me, self.max_size, self.supported_schema, self.fw_version, self.last_seq = me, max_size, supported_schema, fw_version, last_seq
        self.size = size
        self.last_us = now_us
        self._pre = bytearray()
        self._raw_len = 0
        self.header: Optional[tuple] = None
        self.sender_digest = b""
        self.is_self = False
        self.known_sender = False
        self._aead = None
        self._head_hash = b""
        self._pending_ct = bytearray()
        self._i = 0
        self._n = 0
        self._ct_seen = 0
        self._hash = hashlib.sha256()
        self._plain = bytearray()
        self._done = False

    @property
    def next(self) -> int:
        return self._raw_len

    # -- order and repeats, like the firmware upload
    def chunk(self, offset: int, data: bytes, now_us: int = 0) -> int:
        self.last_us = now_us
        if offset + len(data) <= self._raw_len:                      # a repeat (the acknowledgement was lost)
            return self._raw_len
        if offset != self._raw_len:
            raise UpdateError("B", "sequence", f"expected offset {self._raw_len}")
        if self.size is not None and self._raw_len + len(data) > self.size:
            raise UpdateError("B", "too_large", "more than announced")
        self.feed(data)
        return self._raw_len

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self._raw_len += len(data)
        if self.header is None:
            self._pre += data
            self._early_magic()
            if len(self._pre) < PREFIX:
                return
            buf, self._pre = bytes(self._pre), bytearray()
            self._open_header(buf[:HEAD], buf[HEAD:PREFIX])
            data = buf[PREFIX:]
            if not data:
                return
        self._ciphertext(data)

    def _early_magic(self) -> None:
        if len(self._pre) >= 4:
            m = bytes(self._pre[:4])
            if m == MAGIC:
                return
            if m == MAGIC_A:
                raise UpdateError("B", "wrong_channel", "a system update was sent as a package")
            if m == MAGIC_B_SMALL:
                raise UpdateError("B", "unsupported", "a small sealed file goes through the settings path")
            raise UpdateError("B", "damaged", "not a package")

    def _open_header(self, header: bytes, sig: bytes) -> None:
        _, ed25519, _ = _lib()
        (magic, ver, suite, flags, pkg_id, rfp, sfp, seq, schema, kinds, min_fw, s_ed, s_x, eph_pub, nonce, chunk_size, ct_len,
         ct_sha) = struct.unpack(HEAD_FMT, header)
        if magic != MAGIC:
            raise UpdateError("B", "damaged", "not a package")
        if ver != VERSION:
            raise UpdateError("B", "version", f"format {ver}")
        if suite != SUITE:
            raise UpdateError("B", "unsupported", f"suite {suite}")
        sdig = digest_of(s_ed, s_x)
        if sdig[:8] != sfp:
            raise UpdateError("B", "bad_signature", "the sender's short ID does not match its keys")
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(s_ed).verify(sig, b"DOPK-v1-sig" + header)
        except Exception:
            raise UpdateError("B", "bad_signature", "changed or damaged") from None
        if rfp != self.me.fp8:
            raise UpdateError("B", "wrong_device", "made for another device")
        if not 512 <= chunk_size <= 65536 or ct_len < TAG:
            raise UpdateError("B", "damaged", "length")
        n = -(-ct_len // (chunk_size + TAG))
        if ct_len - (n - 1) * (chunk_size + TAG) < TAG:
            raise UpdateError("B", "damaged", "chunk layout")
        if ct_len - n * TAG > self.max_size:
            raise UpdateError("B", "too_large", "package")
        if self.size is not None and self.size != PREFIX + ct_len:
            raise UpdateError("B", "damaged", "the announced size does not match the header")
        if kinds == 0 or kinds & ~ALL_KINDS:
            raise UpdateError("B", "unsupported", f"kinds {kinds:#x}")
        check_compat(schema, min_fw, self.supported_schema, self.fw_version)
        self.is_self = bool(flags & F_SELF) and sdig == self.me.digest
        if bool(flags & F_SELF) != (sdig == self.me.digest):
            raise UpdateError("B", "damaged", "the self flag does not match the sender")
        self.sender_digest = sdig
        last = self.last_seq(sdig)
        self.known_sender = self.is_self or last is not None
        if not self.is_self and last is not None and seq <= last:
            raise UpdateError("B", "replay", "replay")
        self._n = n
        self._chunk = chunk_size
        self._ct_len = ct_len
        self._ct_sha = ct_sha
        self._nonce = nonce
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        key = _key(self.me.exchange(eph_pub), eph_pub, self.me.x_pub, sdig, self.me.digest, pkg_id)
        self._aead = ChaCha20Poly1305(key)
        self._head_hash = hashlib.sha256(header[:-32] + b"\0" * 32).digest()
        self.header = (pkg_id, seq, schema, kinds, min_fw, s_ed, s_x, flags)

    def _ciphertext(self, data: bytes) -> None:
        from cryptography.exceptions import InvalidTag
        if self._ct_seen + len(data) > self._ct_len:
            raise UpdateError("B", "too_large", "more than the header says")
        self._ct_seen += len(data)
        self._hash.update(data)
        self._pending_ct += data
        while self._i < self._n:
            last = self._i == self._n - 1
            need = (self._ct_len - (self._n - 1) * (self._chunk + TAG)) if last else self._chunk + TAG
            if len(self._pending_ct) < need:
                break
            piece, self._pending_ct = bytes(self._pending_ct[:need]), self._pending_ct[need:]
            try:
                self._plain += self._aead.decrypt(_nonce(self._nonce, self._i), piece, _aad(self._head_hash, self._i, last))
            except InvalidTag:
                # the header (signed, and naming this device) is fine, so a chunk that does not authenticate was changed on the way
                raise UpdateError("B", "tampered", f"chunk {self._i}") from None
            self._i += 1
        self._done = self._i == self._n

    def finish(self) -> Opened:
        if self.header is None:
            raise UpdateError("B", "truncated", "the header never completed")
        if not self._done or self._ct_seen != self._ct_len:
            raise UpdateError("B", "truncated", f"{self._ct_seen} of {self._ct_len} bytes")
        if self._hash.digest() != self._ct_sha:
            raise UpdateError("B", "hash", "the ciphertext does not match the signed hash")
        pkg_id, seq, schema, kinds, min_fw, s_ed, s_x, flags = self.header
        slot, sec, info = _parse_plain(bytes(self._plain), kinds)
        try:
            profile, tuning, ui, meta = SL._decode({t: v for t, v in sec.items() if t != SL.S_MODEL})
        except SL.SealError as e:
            raise UpdateError("B", {"version": "version", "unsupported": "unsupported"}.get(e.key, "bad_part"), str(e)) from None
        model = sec.get(SL.S_MODEL)
        return Opened(pkg_id, s_ed, s_x, self.sender_digest, seq, self.is_self, schema, min_fw, kinds, slot, profile, tuning, ui,
                      clean_name(meta.get("name", "")), model, info.get("model"), len(self._plain))


def _parse_plain(plain: bytes, kinds: int) -> tuple[Optional[int], dict, dict]:
    """(slot, {section type: bytes} for `SL._decode` plus S_MODEL, {part type: info}); everything inside is checked against the manifest."""
    if len(plain) < 4:
        raise UpdateError("B", "damaged", "no manifest")
    (n,) = struct.unpack_from("<I", plain)
    if n > MAX_MANIFEST or 4 + n > len(plain):
        raise UpdateError("B", "damaged", "manifest length")
    try:
        man = json.loads(plain[4 : 4 + n].decode("utf-8"))
        if not isinstance(man, dict) or man.get("v") != 1 or not isinstance(man.get("parts"), list) or not 1 <= len(man["parts"]) <= len(KINDS):
            raise ValueError("shape")
        slot = man.get("slot")
        if slot is not None and (isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < SLOT_COUNT):
            raise ValueError("slot")
    except (ValueError, UnicodeDecodeError):
        raise UpdateError("B", "damaged", "manifest") from None
    pos, seen, sec, info = 4 + n, set(), {}, {}
    secmap = {"profile": SL.S_PROFILE, "tuning": SL.S_TUNING, "ui_manifest": SL.S_MANIFEST, "meta": SL.S_META, "model": SL.S_MODEL}
    for e in man["parts"]:
        try:
            t, size, h = e["t"], e["n"], e["h"]
        except (KeyError, TypeError):
            raise UpdateError("B", "damaged", "part entry") from None
        if t not in KINDS or t in seen or isinstance(size, bool) or not isinstance(size, int) or size < 0 or not isinstance(h, str):
            raise UpdateError("B", "unsupported" if t not in KINDS else "damaged", f"part {t!r}")
        if pos + size > len(plain):
            raise UpdateError("B", "damaged", "part runs past the end")
        body = plain[pos : pos + size]
        pos += size
        if hashlib.sha256(body).hexdigest() != h:
            raise UpdateError("B", "hash", f"part {t}")
        seen.add(t)
        sec[secmap[t]] = body
        if "info" in e:
            info[t] = e["info"]
    if pos != len(plain):
        raise UpdateError("B", "damaged", "bytes after the last part")
    if sum(KINDS[t] for t in seen) != kinds:
        raise UpdateError("B", "damaged", "the header's kinds do not match the parts")
    return slot, sec, info


def open_package(raw: bytes, me: Identity, **kw) -> Opened:
    """A whole package in one go (the stored one, a file for the CLI). The same checks as the streaming path, because it IS the same code."""
    r = PackageReader(me, size=len(raw), **kw)
    r.chunk(0, raw)
    return r.finish()


def inspect_header(raw: bytes) -> dict:
    """What anyone can read without any key: the plain header (the CLI's `inspect`). Nothing here is checked."""
    if len(raw) < HEAD or raw[:4] != MAGIC:
        raise UpdateError("B", "damaged", "not a package")
    (magic, ver, suite, flags, pkg_id, rfp, sfp, seq, schema, kinds, min_fw, s_ed, s_x, eph, nonce, chunk, ct_len, ct_sha) = struct.unpack_from(HEAD_FMT, raw)
    return {"version": ver, "suite": suite, "self": bool(flags & F_SELF), "package_id": pkg_id.hex(), "recipient_fp": rfp.hex(), "sender_fp": sfp.hex(),
            "seq": seq, "schema": schema, "min_fw": min_fw, "kinds": [n for n in ORDER if kinds & KINDS[n]], "chunk_size": chunk, "ciphertext_bytes": ct_len,
            "total_bytes": len(raw), "sender_id": format_id(digest_of(s_ed, s_x))}
