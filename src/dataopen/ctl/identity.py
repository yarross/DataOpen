"""The device's own keys and the public ID derived from them (docs/SECURITY.md).

Three secrets per device, created once at 'factory' provisioning and never leaving it:
  ed       Ed25519 signing key       what proves a card or a sealed file came from this device
  x        X25519 agreement key      what a sealed file addressed to this device is decryptable with
  storage  32 random bytes           what the device's own files on disk are encrypted with (and what 'erase' throws away)

`KeyStore` is the seam where a secure element / TPM / OP-TEE would sit on real hardware. `FileKeyStore` is the SIMULATION: a file with mode
0600 in a directory with mode 0700. Anyone with root on the module can read it; that is stated, not hidden (docs/SECURITY.md section 0).

Device Public ID. digest = SHA-256("DOID1" || ed_pub || x_pub). People see the first 80 bits in Crockford base32 as XXXX-XXXX-XXXX-XXXX
(to compare with a label); the file format binds the FULL digest, so the short form is only ever a human aid.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

CARD_VERSION = 1
B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"          # Crockford: no I, L, O, U


class CryptoMissing(RuntimeError):
    pass


class CardError(ValueError):
    pass


def _lib():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
    except ImportError as e:                                  # pragma: no cover (CI installs it)
        raise CryptoMissing("the control gateway needs the 'cryptography' package: pip install 'dataopen[secure]'") from e
    return serialization, ed25519, x25519


def digest_of(ed_pub: bytes, x_pub: bytes) -> bytes:
    return hashlib.sha256(b"DOID1" + ed_pub + x_pub).digest()


def format_id(digest: bytes) -> str:
    """First 80 bits as 16 Crockford base32 characters in groups of four."""
    n = int.from_bytes(digest[:10], "big")
    chars = "".join(B32[(n >> (75 - 5 * i)) & 31] for i in range(16))
    return "-".join(chars[i : i + 4] for i in range(0, 16, 4))


def normalize_id(s: str) -> str:
    """What a person typed or read off a label, as the canonical ID (case, dashes, spaces and the usual look-alikes tolerated)."""
    t = "".join(c for c in s.upper() if c.isalnum()).translate(str.maketrans("ILOU", "1100"))
    if len(t) != 16 or any(c not in B32 for c in t):
        raise ValueError("a device ID is 16 characters, e.g. 7K2M-9QX4-ABCD-WXY3")
    return "-".join(t[i : i + 4] for i in range(0, 16, 4))


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s, n: int, what: str) -> bytes:
    try:
        b = base64.b64decode(s, validate=True)
    except (ValueError, TypeError):
        raise CardError(f"{what}: not base64") from None
    if len(b) != n:
        raise CardError(f"{what}: must be {n} bytes")
    return b


# ---------------------------------------------------------------------------------------------------------------- the public card
@dataclass(frozen=True)
class Card:
    """What a device shows the world: its two public keys, signed by itself. Safe to hand to anyone; a sealed file made from it can only be
    opened by the device it describes."""
    ed: bytes
    x: bytes
    label: str = ""
    created: str = ""
    sig: bytes = b""

    @property
    def digest(self) -> bytes:
        return digest_of(self.ed, self.x)

    @property
    def id(self) -> str:
        return format_id(self.digest)

    def _body(self) -> bytes:
        body = {"v": CARD_VERSION, "ed": _b64(self.ed), "x": _b64(self.x), "id": self.id, "label": self.label, "created": self.created}
        return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def verify(self) -> bool:
        _, ed25519, _ = _lib()
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(self.ed).verify(self.sig, b"DOCARD1" + self._body())
            return True
        except Exception:                                     # InvalidSignature and any malformed key
            return False

    def to_json(self) -> dict:
        return {"v": CARD_VERSION, "ed": _b64(self.ed), "x": _b64(self.x), "id": self.id, "label": self.label,
                "created": self.created, "sig": _b64(self.sig)}

    @staticmethod
    def from_json(o) -> "Card":
        """Strict. A bad signature, or an `id` that is not the one the keys give, is refused: that is the substitution check."""
        if not isinstance(o, dict) or o.get("v") != CARD_VERSION:
            raise CardError("not a device card (version)")
        for k in ("ed", "x", "id", "sig"):
            if not isinstance(o.get(k), str):
                raise CardError(f"missing {k}")
        label, created = o.get("label", ""), o.get("created", "")
        if not isinstance(label, str) or not isinstance(created, str) or len(label) > 40 or len(created) > 10:
            raise CardError("bad label or date")
        c = Card(_unb64(o["ed"], 32, "ed"), _unb64(o["x"], 32, "x"), label, created, _unb64(o["sig"], 64, "sig"))
        if o["id"] != c.id:
            raise CardError("the ID does not match the keys")
        if not c.verify():
            raise CardError("the signature does not match the card")
        return c


# ---------------------------------------------------------------------------------------------------------------- key storage
class KeyStore(Protocol):
    def load(self) -> Optional[dict]: ...
    def save(self, d: dict) -> None: ...
    def wipe(self) -> None: ...


class FileKeyStore:
    """SIMULATION of a hardware key store: base64 JSON, file mode 0600 in a 0700 directory, atomic replace."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Optional[dict]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return d if isinstance(d, dict) and d.get("v") == 1 else None

    def save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"v": 1, **d}))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def wipe(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


class Identity:
    def __init__(self, store: KeyStore, d: dict) -> None:
        ser, ed25519, x25519 = _lib()
        self.store = store
        self._ed = ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(d["ed"]))
        self._x = x25519.X25519PrivateKey.from_private_bytes(base64.b64decode(d["x"]))
        raw = ser.Encoding.Raw, ser.PublicFormat.Raw
        self.ed_pub = self._ed.public_key().public_bytes(*raw)
        self.x_pub = self._x.public_key().public_bytes(*raw)
        self.storage_key = base64.b64decode(d["storage"])
        self.created = d.get("created", "")
        self.export_seq = int(d.get("seq", 0))
        ep = d.get("epochs")
        self.slot_epochs = [int(x) for x in ep] if isinstance(ep, list) and all(isinstance(x, int) and x >= 0 for x in ep) else []
        self.digest = digest_of(self.ed_pub, self.x_pub)
        self.id = format_id(self.digest)
        self.fp8 = self.digest[:8]

    # -- what only this device can do
    def sign(self, msg: bytes) -> bytes:
        return self._ed.sign(msg)

    def exchange(self, peer_x_pub: bytes) -> bytes:
        _, _, x25519 = _lib()
        return self._x.exchange(x25519.X25519PublicKey.from_public_bytes(peer_x_pub))

    def card(self, label: str = "DataOpen") -> Card:
        c = Card(self.ed_pub, self.x_pub, label, self.created)
        return Card(c.ed, c.x, c.label, c.created, self.sign(b"DOCARD1" + c._body()))

    # -- state that must survive restarts
    def _dump(self) -> dict:
        ser, _, _ = _lib()
        raw = ser.Encoding.Raw, ser.PrivateFormat.Raw, ser.NoEncryption()
        return {"ed": _b64(self._ed.private_bytes(*raw)), "x": _b64(self._x.private_bytes(*raw)),
                "storage": _b64(self.storage_key), "created": self.created, "seq": self.export_seq,
                "epochs": self.slot_epochs}

    def next_seq(self) -> int:
        """The counter every sealed file this device makes carries (receivers refuse a number they have already seen from it)."""
        self.export_seq += 1
        self.store.save(self._dump())
        return self.export_seq

    def slot_epoch(self, k: int) -> int:
        return self.slot_epochs[k] if k < len(self.slot_epochs) else 0

    def slot_key(self, k: int) -> bytes:
        """The key one hardware slot's files are encrypted with: HKDF(storage key, slot number, slot epoch). Another slot's key (or the same
        slot's key after `bump_slot_epoch`) opens nothing of it."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        info = b"DataOpen slot key" + bytes([k]) + self.slot_epoch(k).to_bytes(4, "big")
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=b"DOSLOT1", info=info).derive(self.storage_key)

    def bump_slot_epoch(self, k: int) -> None:
        """Crypto-erase ONE slot: its files can never be read again, the other slots are untouched."""
        n = max(len(self.slot_epochs), k + 1)
        self.slot_epochs = (self.slot_epochs + [0] * n)[:n]
        self.slot_epochs[k] += 1
        self.store.save(self._dump())

    def rotate_storage_key(self) -> bytes:
        """Crypto-erase: files written under the old key can never be read again."""
        self.storage_key = os.urandom(32)
        self.slot_epochs = []
        self.store.save(self._dump())
        return self.storage_key


def provision(store: KeyStore, rng=os.urandom) -> Identity:
    """'Factory' initialisation: fresh keys, new ID. Replaces whatever the store held."""
    d = {"ed": _b64(rng(32)), "x": _b64(rng(32)), "storage": _b64(rng(32)), "created": time.strftime("%Y-%m-%d", time.gmtime()), "seq": 0}
    store.save(d)
    return Identity(store, d)


def load_identity(store: KeyStore) -> Identity:
    d = store.load()
    if d is None or any(k not in d for k in ("ed", "x", "storage")):
        return provision(store)
    try:
        return Identity(store, d)
    except (ValueError, TypeError):
        return provision(store)                              # unreadable keys: the device starts over (and says so: its ID changes)
