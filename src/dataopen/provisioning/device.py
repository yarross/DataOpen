"""The device's side of provisioning: write-once storage, a secure element that never lets its key out, and the agent that follows the rules.

SIMULATION. On a board `OtpSim` is OTP / eFuse / a locked secure-element data slot, `SecureElementSim` is a secure element that generates and
holds the DAK, and `DeviceAgent` is a small service in the first-stage firmware. The files here are 0600 JSON and anyone with root on the
host can read them: stated, not hidden (docs/PROVISIONING.md section 0).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Optional

from ..ctl.identity import FileKeyStore, Identity, _lib, load_identity
from . import records as R


class DeviceError(RuntimeError):
    """`key`: locked | exists | lifecycle | binding | signature | already_provisioned | not_provisioned | token | presence | domain"""

    def __init__(self, key: str, msg: str = "") -> None:
        super().__init__(msg or key)
        self.key = key


def _write(path: Path, d: dict, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600 if private else 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(d, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


class OtpSim:
    """Fields that can be written ONCE, counters that only grow, append-only logs, and a lock after which nothing can be written at all."""

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / "otp" / "otp.json"

    def _d(self) -> dict:
        return _read(self.path)

    @property
    def locked(self) -> bool:
        return bool(self._d().get("_locked"))

    def read(self, name: str, default=None):
        return self._d().get(name, default)

    def write_once(self, name: str, value) -> None:
        d = self._d()
        if d.get("_locked"):
            raise DeviceError("locked", f"{name}: the OTP is locked")
        if name in d:
            raise DeviceError("exists", f"{name} is already written")
        d[name] = value
        _write(self.path, d)

    def counter(self, name: str, value: int) -> None:
        d = self._d()
        if d.get("_locked") and name != "floor":
            raise DeviceError("locked", f"{name}: the OTP is locked")
        if value < int(d.get(name, 0)):
            raise DeviceError("exists", f"{name} only grows")
        d[name] = int(value)
        _write(self.path, d)

    def append(self, name: str, value: str) -> None:
        d = self._d()
        d.setdefault(name, []).append(value)
        _write(self.path, d)

    def lock(self) -> None:
        d = self._d()
        d["_locked"] = True
        _write(self.path, d)


class SecureElementSim:
    """Holds the DAK. There is no method that returns the secret, and it will only sign messages of the two domains it is meant for
    (a proof of possession and a device certificate), never anything that could pass for the manufacturer's attestation or a service token."""

    SIGNS = (R.DEV_DOMAIN, R.POP_DOMAIN)

    def __init__(self, directory: Path) -> None:
        self._path = Path(directory) / "se" / "se.json"

    def _key(self):
        _, ed25519, _ = _lib()
        d = _read(self._path)
        if "dak" not in d:
            return None
        return ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(d["dak"]))

    @property
    def has_dak(self) -> bool:
        return self._key() is not None

    @property
    def dak_pub(self) -> Optional[bytes]:
        k = self._key()
        if k is None:
            return None
        ser, _, _ = _lib()
        return k.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw)

    def generate_dak(self) -> bytes:
        if self.has_dak:
            raise DeviceError("exists", "the device key already exists and can not be replaced")
        _write(self._path, {"dak": base64.b64encode(os.urandom(32)).decode()}, private=True)
        return self.dak_pub

    def sign(self, body: bytes) -> bytes:
        if not any(body.startswith(d) for d in self.SIGNS):
            raise DeviceError("domain", "the secure element signs only proofs of possession and device certificates")
        k = self._key()
        if k is None:
            raise DeviceError("not_provisioned", "no device key")
        return k.sign(body)


class DeviceAgent:
    """What a device does during provisioning and service. One directory is one device (the same one the gateway runs on)."""

    def __init__(self, directory: str | Path, *, hw_id: bytes, vendor_pub: bytes, chip_uid: Optional[bytes] = None) -> None:
        self.dir = Path(directory)
        self.hw_id = hw_id.ljust(R.HW_ID_LEN, b"\0")[:R.HW_ID_LEN]
        self.vendor_pub = vendor_pub
        self.otp, self.se = OtpSim(self.dir), SecureElementSim(self.dir)
        bpath = self.dir / "board.json"
        if not _read(bpath).get("chip"):
            _write(bpath, {"chip": (chip_uid or secrets.token_bytes(16)).hex()})
        self._board_uid = bytes.fromhex(_read(bpath)["chip"])
        self._svc = self.dir / "otp" / "service.json"

    # -- identity of the board
    @property
    def board_id(self) -> bytes:
        return hashlib.sha256(b"DOBRD1" + self._board_uid).digest()

    def owner(self) -> Identity:
        return load_identity(FileKeyStore(self.dir / "keys" / "keys.json"))

    # -- lifecycle
    @property
    def lifecycle(self) -> str:
        log = self.otp.read("lifecycle_log", [])
        return log[-1] if log else R.BLANK

    def _advance(self, to: str) -> None:
        if to not in R.TRANSITIONS.get(self.lifecycle, ()):
            raise DeviceError("lifecycle", f"{self.lifecycle} -> {to} is not allowed")
        self.otp.append("lifecycle_log", to)

    def mark_tested(self) -> None:
        self._advance(R.DEV_TESTED)

    # -- provisioning
    def begin(self) -> dict:
        """The request the station sends to the manufacturer's HSM: public keys only, plus a proof that this device holds the DAK."""
        if self.lifecycle != R.DEV_TESTED:
            raise DeviceError("lifecycle", "provisioning starts after the board passed its tests")
        if self.otp.read("record"):
            raise DeviceError("already_provisioned")
        dak = self.se.dak_pub or self.se.generate_dak()
        nonce = secrets.token_bytes(16)
        owner = self.owner()
        pop = self.se.sign(R.POP_DOMAIN + nonce + dak + self.board_id + self.hw_id)
        return {"hw": self.hw_id.hex(), "dak": base64.b64encode(dak).decode(), "board": base64.b64encode(self.board_id).decode(),
                "owner": owner.digest.hex(), "nonce": nonce.hex(), "pop": base64.b64encode(pop).decode()}

    def accept(self, att_json: dict, *, sku: str, factory_fw: int) -> R.DeviceRecord:
        if self.lifecycle != R.DEV_TESTED:
            raise DeviceError("lifecycle", "not waiting for an attestation")
        if self.otp.read("record"):
            raise DeviceError("already_provisioned")
        try:
            att = R.Attestation.from_json(att_json)
        except R.RecordError as e:
            raise DeviceError("signature", str(e)) from None
        if not att.verify(self.vendor_pub):
            raise DeviceError("signature", "not signed by the manufacturer this device trusts")
        if att.dak != self.se.dak_pub or att.board != self.board_id or att.hw_id != self.hw_id:
            raise DeviceError("binding", "the attestation is for another device")
        rec = R.DeviceRecord(att.serial, sku, self.hw_id, att, factory_fw, "", R.PROVISIONED)
        self.otp.write_once("record", rec.to_json())
        self.otp.write_once("vendor", R.vendor_id(self.vendor_pub))
        self._advance(R.PROVISIONED)
        return rec

    def finalize(self, report_hash: str) -> None:
        """The last, irreversible step: the report hash goes in, the OTP and the key slot lock, the lifecycle moves to SHIPPED."""
        if self.lifecycle != R.PROVISIONED:
            raise DeviceError("lifecycle")
        rec = self.record
        if rec is None:
            raise DeviceError("not_provisioned")
        self.otp.write_once("report", report_hash)
        self._advance(R.SHIPPED)
        self.otp.lock()

    @property
    def record(self) -> Optional[R.DeviceRecord]:
        d = self.otp.read("record")
        if not d:
            return None
        rec = R.DeviceRecord.from_json(d)
        rec.report_hash, rec.lifecycle = self.otp.read("report", ""), self.lifecycle
        return rec

    def first_boot(self) -> None:
        if self.lifecycle == R.SHIPPED:
            self._advance(R.IN_FIELD)

    # -- the certificate that ties the owner card to this device
    def cert(self, owner_digest: Optional[bytes] = None) -> R.DeviceCert:
        rec = self.record
        if rec is None:
            raise DeviceError("not_provisioned")
        owner_digest = owner_digest or self.owner().digest
        return R.DeviceCert(rec.att, self.se.sign(R.DeviceCert.body(rec.serial, owner_digest)))

    # -- service (RMA): a token from the manufacturer for THIS serial, a nonce this device issued, and a hand on the device
    def service_challenge(self) -> bytes:
        nonce = secrets.token_bytes(16)
        _write(self._svc, {"nonce": nonce.hex()})
        return nonce

    def apply_service(self, token_json: dict, *, presence: bool) -> str:
        rec = self.record
        if rec is None:
            raise DeviceError("not_provisioned")
        if not presence:
            raise DeviceError("presence", "a service action needs a hand on the device (the recovery chord or the RECOVERY button)")
        try:
            tok = R.ServiceToken.from_json(token_json)
        except R.RecordError as e:
            raise DeviceError("token", str(e)) from None
        _, ed25519, _ = _lib()
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(self.vendor_pub).verify(tok.sig, tok.body())
        except Exception:
            raise DeviceError("token", "not signed by the manufacturer") from None
        if tok.serial != rec.serial:
            raise DeviceError("token", "made for another device")
        pending = _read(self._svc).get("nonce")
        if not pending or pending != tok.nonce.hex():
            raise DeviceError("token", "not the challenge this device issued (or already used)")
        _write(self._svc, {})                                                 # one use
        return tok.action

    def to_rma(self) -> None:
        if self.lifecycle in (R.SHIPPED, R.IN_FIELD):
            self._advance(R.RMA)

    def back_from_rma(self) -> None:
        """After a full return to factory state: RMA -> PROVISIONED -> SHIPPED. The log stays (history is never rewritten)."""
        if self.lifecycle == R.RMA:
            self.otp.append("lifecycle_log", R.PROVISIONED)
            self.otp.append("lifecycle_log", R.SHIPPED)
