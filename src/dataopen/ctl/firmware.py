"""Safe firmware update, as a SIMULATION of the logic a real boot chain would run (docs/SECURITY.md section 6).

Nothing here writes flash or boots anything. It pins down the rules, so that a bootloader and an updater can be built against them
and tested:

  image   'DOFW' | fmt=1 | hw_id[8] | version u32 | min_version u32 | payload_len u32 | sha256(payload)[32] | signature[64] | payload
          the signature (the MANUFACTURER's key, not the device's) covers everything before it, hash included
  slots   two banks, A and B. An update is written to the bank that is NOT running; the running one is never touched
  trial   the new bank boots, and must `confirm()` after it has proved itself; N boots without a confirmation put the old bank back
  floor   anti-rollback: once an image with `min_version` m is confirmed, nothing older than m is ever accepted again (a monotonic counter
          in real hardware: eFuse, RPMB or a secure element), so a validly signed OLD image with a known hole can not be put back
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Optional

MAGIC, FMT = b"DOFW", 1
HEAD_FMT = "<4sB3x8sIII32s"
HEAD = struct.calcsize(HEAD_FMT)
SIG = 64
MAX_BOOTS = 3
assert HEAD == 60


class FirmwareError(ValueError):
    """`key`: damaged | signature | hardware | rollback | same | too_large | no_trial | interrupted"""

    def __init__(self, key: str, msg: str = "") -> None:
        super().__init__(msg or key)
        self.key = key


class WriteInterrupted(FirmwareError):
    def __init__(self) -> None:
        super().__init__("interrupted", "power lost while writing the inactive bank")


def build_image(payload: bytes, hw_id: bytes, version: int, min_version: int, signer) -> bytes:
    """`signer`: an Ed25519 private key object (the manufacturer's; held offline, never on a device)."""
    digest = hashlib.sha256(payload).digest()
    head = struct.pack(HEAD_FMT, MAGIC, FMT, hw_id.ljust(8, b"\0")[:8], version, min_version, len(payload), digest)
    return head + signer.sign(head) + payload


@dataclass(frozen=True)
class Image:
    hw_id: bytes
    version: int
    min_version: int
    payload: bytes
    digest: bytes


def verify_image(raw: bytes, vendor_pub: bytes, hw_id: bytes) -> Image:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    if len(raw) < HEAD + SIG or raw[:4] != MAGIC:
        raise FirmwareError("damaged", "not a firmware image")
    magic, fmt, hw, ver, minv, n, digest = struct.unpack_from(HEAD_FMT, raw)
    if fmt != FMT or len(raw) != HEAD + SIG + n or minv > ver:
        raise FirmwareError("damaged", "layout")
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(vendor_pub).verify(raw[HEAD : HEAD + SIG], raw[:HEAD])
    except Exception:
        raise FirmwareError("signature", "not signed by the manufacturer") from None
    payload = raw[HEAD + SIG :]
    if hashlib.sha256(payload).digest() != digest:
        raise FirmwareError("damaged", "the payload does not match its hash")
    if hw != hw_id.ljust(8, b"\0")[:8]:
        raise FirmwareError("hardware", "built for other hardware")
    return Image(hw, ver, minv, payload, digest)


@dataclass
class Slot:
    image: Optional[Image] = None
    bad: bool = False                     # failed its trial: never booted again


@dataclass
class SlotManager:
    vendor_pub: bytes
    hw_id: bytes
    slot_size: int = 1 << 20
    floor: int = 0
    active: str = "A"
    trial: Optional[str] = None           # the bank that is booting on probation
    boots: int = 0                        # boots of the trial bank so far
    slots: dict = field(default_factory=lambda: {"A": Slot(), "B": Slot()})
    log: list = field(default_factory=list)

    def install_factory(self, raw: bytes) -> None:
        """What the factory flashes into the first bank (still verified: a bad image never becomes 'active')."""
        img = verify_image(raw, self.vendor_pub, self.hw_id)
        self.slots[self.active] = Slot(img)
        self.floor = max(self.floor, img.min_version)

    @property
    def other(self) -> str:
        return "B" if self.active == "A" else "A"

    @property
    def version(self) -> int:
        img = self.slots[self.active].image
        return img.version if img else 0

    def stage(self, raw: bytes, crash_after: Optional[int] = None) -> Image:
        """Verify, then write into the bank that is not running. `crash_after` simulates power loss that many bytes into the write."""
        img = verify_image(raw, self.vendor_pub, self.hw_id)
        if img.version < self.floor:
            raise FirmwareError("rollback", f"version {img.version} is below the floor {self.floor}")
        if img.version == self.version:
            raise FirmwareError("same", "this version is already running")
        if len(img.payload) > self.slot_size:
            raise FirmwareError("too_large", "does not fit a bank")
        if self.trial is not None:
            raise FirmwareError("no_trial", "the previous update has not been confirmed yet")
        if crash_after is not None and crash_after < len(raw):
            self.slots[self.other] = Slot(None)                       # half a bank is not a bank: the running one is untouched
            raise WriteInterrupted()
        self.slots[self.other] = Slot(img)
        self.trial, self.boots = self.other, 0
        self.log.append(f"staged v{img.version} in {self.other}")
        return img

    def reboot(self) -> str:
        """The bootloader's decision: a trial bank gets MAX_BOOTS tries to be confirmed, then the old bank is put back. Returns the bank."""
        if self.trial is not None:
            self.boots += 1
            if self.boots > MAX_BOOTS:
                self.slots[self.trial].bad = True
                self.log.append(f"{self.trial} never confirmed: back to {self.active}")
                self.trial, self.boots = None, 0
            else:
                return self.trial
        return self.active

    def confirm(self) -> None:
        """Called by the NEW firmware once it has proved itself (talked to the bridge, passed its self-test): it stays for good."""
        if self.trial is None or self.slots[self.trial].image is None:
            raise FirmwareError("no_trial", "nothing to confirm")
        img = self.slots[self.trial].image
        self.active, self.trial, self.boots = self.trial, None, 0
        self.floor = max(self.floor, img.min_version)
        self.log.append(f"confirmed v{img.version}, floor {self.floor}")

    def to_dict(self) -> dict:
        return {"floor": self.floor, "active": self.active, "trial": self.trial, "boots": self.boots,
                "slots": {k: {"bad": s.bad, "image": None if s.image is None else
                              {"hw": s.image.hw_id.hex(), "v": s.image.version, "min": s.image.min_version, "digest": s.image.digest.hex()}}
                          for k, s in self.slots.items()}}
