"""`DOBS` v2: the settings/profile file, sealed to ONE device and signed by its sender (docs/SECURITY.md section 3).

    header (160 bytes, plain, authenticated):
        'DOBS' | ver=2 | suite=1 | flags u16 | bundle_id[16] | recipient_fp[8] | sender_fp[8] | seq u64 |
        sender_ed_pub[32] | sender_x_pub[32] | eph_x_pub[32] | nonce[12] | ct_len u32
    ciphertext (ct_len bytes)                  ChaCha20-Poly1305 over the inner sections, header as associated data
    signature (64 bytes)                       Ed25519 by the sender over 'DOBS-v2-sig' || header || ciphertext

Suite 1 = X25519 + HKDF-SHA256 + ChaCha20-Poly1305 + Ed25519, all from the `cryptography` package; nothing here is a primitive of our own.
The key is HKDF(X25519(ephemeral, recipient_x), salt = eph || recipient_x, info = label || SENDER digest || RECIPIENT digest || bundle_id):
the recipient's FULL digest is inside the key, so a file made for one device is noise to every other, and the signed header names the
recipient, so a genuine file cannot be re-addressed. Inner sections are TLV, zero-padded to a multiple of 256 bytes: the size says little.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..bioprofile.profile import ProfileError, ProfileState, ProfileVersionError, validate
from .bundle import _clean
from .identity import Card, Identity, _lib, digest_of, format_id
from .slots import SLOT_COUNT, clean_name
from . import tuning as T

MAGIC, VERSION, SUITE = b"DOBS", 2, 1
HEAD_FMT = "<4sBBH16s8s8sQ32s32s32s12sI"
HEAD = struct.calcsize(HEAD_FMT)
SIG = 64
TAG = 16
# a CtlLink body is 16 KB; what is bigger (model weights) travels as a streamed `DOPK` package (updates/package.py)
MAX_FILE = 16000
PAD = 256
F_SELF = 1
S_PROFILE, S_TUNING, S_MANIFEST, S_META, S_MODEL, S_SLOT = 1, 2, 3, 4, 5, 6
KNOWN = (S_PROFILE, S_TUNING, S_MANIFEST, S_META, S_SLOT)
FLAT = (S_PROFILE, S_TUNING, S_MANIFEST)
assert HEAD == 160


class SealError(ValueError):
    """`key` is the i18n suffix the phone shows: damaged | version | unsupported | wrong_device | bad_signature | replay | plain_refused."""

    def __init__(self, key: str, msg: str = "") -> None:
        super().__init__(msg or key)
        self.key = key


# ---------------------------------------------------------------------------------------------------------------- inner sections
@dataclass
class SlotData:
    """What one hardware slot carries in a file (docs/SLOTS.md). A file without slot sections is about the receiver's ACTIVE slot."""
    n: int
    profile: Optional[ProfileState] = None
    tuning: Optional[tuple[int, int]] = None
    manifest: Optional[dict] = None
    name: str = ""


def _tlv(sections: list[tuple[int, bytes]], pad: int = PAD) -> bytes:
    out = bytes([len(sections)]) + b"".join(struct.pack("<BI", t, len(b)) + b for t, b in sections)
    return out + b"\0" * (pad * max(1, -(-len(out) // pad)) - len(out))


def _untlv(inner: bytes, pad: int = PAD) -> dict[int, bytes]:
    if not inner:
        raise SealError("damaged", "empty")
    n, pos, out = inner[0], 1, {}
    for _ in range(n):
        if pos + 5 > len(inner):
            raise SealError("damaged", "truncated section header")
        t, ln = struct.unpack_from("<BI", inner, pos)
        pos += 5
        if pos + ln > len(inner):
            raise SealError("damaged", "truncated section")
        if t in out:
            raise SealError("damaged", "duplicate section")
        out[t] = inner[pos : pos + ln]
        pos += ln
    if (pad == 1 and pos != len(inner)) or any(inner[pos:]) or len(inner) % pad:
        raise SealError("damaged", "padding")
    return out


def _sections(profile: Optional[ProfileState], tuning: Optional[tuple[int, int]], manifest: Optional[dict],
              meta: Optional[dict]) -> list[tuple[int, bytes]]:
    sections: list[tuple[int, bytes]] = []
    if profile is not None:
        sections.append((S_PROFILE, profile.pack()))
    if tuning is not None:
        sections.append((S_TUNING, json.dumps({"strength": tuning[0], "tremor": tuning[1]}, separators=(",", ":")).encode()))
    if manifest is not None:
        sections.append((S_MANIFEST, json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
    if meta is not None:
        clean = {"name": _clean(meta.get("name", "")), "created": _clean(meta.get("created", ""), 10)}
        sections.append((S_META, json.dumps(clean, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
    return sections


def encode_inner(profile: Optional[ProfileState] = None, tuning: Optional[tuple[int, int]] = None, manifest: Optional[dict] = None,
                 meta: Optional[dict] = None, slots: Optional[list[SlotData]] = None) -> bytes:
    sections = _sections(profile, tuning, manifest, meta)
    if slots:
        if profile is not None or tuning is not None or manifest is not None:
            raise SealError("damaged", "a file is either about the active slot or about named slots")
        body = bytes([len(slots)])
        seen: set[int] = set()
        for sd in slots:
            if not 0 <= sd.n < SLOT_COUNT or sd.n in seen:
                raise SealError("damaged", "slot number")
            seen.add(sd.n)
            inner = _tlv(_sections(sd.profile, sd.tuning, sd.manifest, {"name": clean_name(sd.name)} if sd.name else None), pad=1)
            body += struct.pack("<BI", sd.n, len(inner)) + inner
        sections.append((S_SLOT, body))
    return _tlv(sections)


def _decode(sec: dict[int, bytes]) -> tuple[Optional[ProfileState], Optional[tuple[int, int]], Optional[dict], dict]:
    """The profile / levels / manifest / meta of one set of sections (shared by the flat file and by every slot inside a file)."""
    profile = tuning = manifest = None
    meta: dict = {}
    if S_PROFILE in sec:
        try:
            validate(sec[S_PROFILE])
            profile = ProfileState.unpack(sec[S_PROFILE])
        except ProfileVersionError:
            raise SealError("version", "the profile inside is newer than this device understands") from None
        except ProfileError as e:
            raise SealError("damaged", f"the profile inside: {e}") from None
    try:
        if S_TUNING in sec:
            j = json.loads(sec[S_TUNING].decode("utf-8"))
            s, t = j["strength"], j["tremor"]
            for v in (s, t):
                if isinstance(v, bool) or not isinstance(v, int) or not T.LEVEL_MIN <= v <= T.LEVEL_MAX:
                    raise ValueError("levels out of range")
            tuning = (s, t)
        if S_MANIFEST in sec:
            manifest = json.loads(sec[S_MANIFEST].decode("utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")
        if S_META in sec:
            m = json.loads(sec[S_META].decode("utf-8"))
            meta = {"name": _clean(m.get("name", "")), "created": _clean(m.get("created", ""), 10)} if isinstance(m, dict) else {}
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
        raise SealError("damaged", f"section content: {e}") from None
    return profile, tuning, manifest, meta


def _decode_slots(body: bytes) -> list[SlotData]:
    if not body or body[0] == 0:
        raise SealError("damaged", "no slots")
    n, pos, out, seen = body[0], 1, [], set()
    for _ in range(n):
        if pos + 5 > len(body):
            raise SealError("damaged", "truncated slot header")
        k, ln = struct.unpack_from("<BI", body, pos)
        pos += 5
        if pos + ln > len(body):
            raise SealError("damaged", "truncated slot")
        if k >= SLOT_COUNT or k in seen:
            raise SealError("damaged", "slot number")
        seen.add(k)
        sec = _untlv(body[pos : pos + ln], pad=1)
        pos += ln
        if any(t not in (S_PROFILE, S_TUNING, S_MANIFEST, S_META) for t in sec) or not sec:
            raise SealError("unsupported" if sec else "damaged", "a section that does not belong in a slot")
        profile, tuning, manifest, meta = _decode(sec)
        out.append(SlotData(k, profile, tuning, manifest, clean_name(meta.get("name", ""))))
    if pos != len(body):
        raise SealError("damaged", "trailing bytes after the slots")
    return out


# ---------------------------------------------------------------------------------------------------------------- keys
def _key(shared: bytes, eph_pub: bytes, recipient_x: bytes, sender_digest: bytes, recipient_digest: bytes, bundle_id: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=eph_pub + recipient_x,
                info=b"DOBS v2 key" + sender_digest + recipient_digest + bundle_id).derive(shared)


def seal(sender: Identity, recipient: Card, seq: int, *, profile: Optional[ProfileState] = None, tuning: Optional[tuple[int, int]] = None,
         manifest: Optional[dict] = None, meta: Optional[dict] = None, slots: Optional[list[SlotData]] = None,
         rng: Callable[[int], bytes] = os.urandom) -> bytes:
    """Seal what the arguments carry to `recipient` (a verified Card). `rng` is injectable for the fixed test vectors only."""
    ser, _, x25519 = _lib()
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    if not recipient.verify():
        raise SealError("bad_signature", "the recipient's card is not valid")
    inner = encode_inner(profile, tuning, manifest, meta, slots)
    bundle_id, nonce, eph_seed = rng(16), rng(12), rng(32)
    eph = x25519.X25519PrivateKey.from_private_bytes(eph_seed)
    eph_pub = eph.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)
    shared = eph.exchange(x25519.X25519PublicKey.from_public_bytes(recipient.x))
    key = _key(shared, eph_pub, recipient.x, sender.digest, recipient.digest, bundle_id)
    flags = F_SELF if recipient.digest == sender.digest else 0
    header = struct.pack(HEAD_FMT, MAGIC, VERSION, SUITE, flags, bundle_id, recipient.digest[:8], sender.fp8, seq,
                         sender.ed_pub, sender.x_pub, eph_pub, nonce, len(inner) + TAG)
    ct = ChaCha20Poly1305(key).encrypt(nonce, inner, header)
    out = header + ct + sender.sign(b"DOBS-v2-sig" + header + ct)
    if len(out) > MAX_FILE:
        raise SealError("damaged", "too large for one message")
    return out


@dataclass
class Opened:
    sender_ed: bytes
    sender_x: bytes
    sender_digest: bytes
    seq: int
    flags: int
    bundle_id: bytes
    is_self: bool
    profile: Optional[ProfileState]
    tuning: Optional[tuple[int, int]]
    manifest: Optional[dict]
    meta: dict
    slots: list[SlotData] = field(default_factory=list)

    @property
    def sender_id(self) -> str:
        return format_id(self.sender_digest)


def open_sealed(raw: bytes, me: Identity) -> Opened:
    """Everything is checked before anything is returned; the caller applies nothing until this has succeeded."""
    _, ed25519, _ = _lib()
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    if len(raw) < HEAD + TAG + SIG or len(raw) > MAX_FILE + 1024:
        raise SealError("damaged", "size")
    (magic, ver, suite, flags, bundle_id, rfp, sfp, seq, s_ed, s_x, eph_pub, nonce, ct_len) = struct.unpack_from(HEAD_FMT, raw)
    if magic != MAGIC:
        raise SealError("damaged", "not a sealed settings file")
    if ver != VERSION:
        raise SealError("version", f"format {ver}")
    if suite != SUITE:
        raise SealError("unsupported", f"suite {suite}")
    if len(raw) != HEAD + ct_len + SIG or ct_len < TAG + PAD:
        raise SealError("damaged", "length")
    header, ct, sig = raw[:HEAD], raw[HEAD : HEAD + ct_len], raw[HEAD + ct_len :]
    sdig = digest_of(s_ed, s_x)
    if sdig[:8] != sfp:
        raise SealError("bad_signature", "the sender's short ID does not match its keys")
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(s_ed).verify(sig, b"DOBS-v2-sig" + header + ct)
    except Exception:
        raise SealError("bad_signature", "changed or damaged") from None
    if rfp != me.fp8:
        raise SealError("wrong_device", "made for another device")
    key = _key(me.exchange(eph_pub), eph_pub, me.x_pub, sdig, me.digest, bundle_id)
    try:
        inner = ChaCha20Poly1305(key).decrypt(nonce, ct, header)
    except InvalidTag:
        raise SealError("wrong_device", "made for another device") from None     # signed by the sender, yet not for these keys
    sec = _untlv(inner)
    if any(t not in KNOWN for t in sec):
        raise SealError("unsupported", "a section this device does not understand (model weights travel in a package, docs/UPDATES.md)")
    if not sec:
        raise SealError("damaged", "nothing inside")
    slots: list[SlotData] = []
    if S_SLOT in sec:
        if any(t in sec for t in FLAT):
            raise SealError("damaged", "a file is either about the active slot or about named slots")
        slots = _decode_slots(sec[S_SLOT])
    profile, tuning, manifest, meta = _decode({t: v for t, v in sec.items() if t != S_SLOT})
    return Opened(s_ed, s_x, sdig, seq, flags, bundle_id, sdig == me.digest, profile, tuning, manifest, meta, slots)
