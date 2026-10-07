"""Safe firmware update, as a SIMULATION of the logic a real boot chain would run (docs/SECURITY.md section 6).

Nothing here writes flash or boots anything. It pins down the rules, so that a bootloader and an updater can be built against them
and tested:

  image   'DOFW' | fmt=1 | hw_id[8] | version u32 | min_version u32 | payload_len u32 | sha256(payload)[32] | signature[64] | payload
          the signature (the MANUFACTURER's key, not the device's) covers everything before it, hash included
  slots   two banks, A and B. An update is written to the bank that is NOT running; the running one is never touched
  trial   the new bank boots, and must `confirm()` after it has proved itself; N boots without a confirmation put the old bank back
  stream  the image arrives over CtlLink as FW_BEGIN / FW_CHUNK* / FW_END (`Upload`), is checked as a whole, and only then staged
  floor   anti-rollback: once an image with `min_version` m is confirmed, nothing older than m is ever accepted again (a monotonic counter
          in real hardware: eFuse, RPMB or a secure element), so a validly signed OLD image with a known hole can not be put back
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MAGIC, FMT = b"DOFW", 1
HEAD_FMT = "<4sB3x8sIII32s"
HEAD = struct.calcsize(HEAD_FMT)
SIG = 64
MAX_BOOTS = 3
assert HEAD == 60


class FirmwareError(ValueError):
    """`key`: damaged | signature | hardware | rollback | same | too_large | no_trial | interrupted | sequence | no_previous"""

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
    raw: bytes = b""                      # the image as it was written (the simulation's 'flash content')


@dataclass
class SlotManager:
    vendor_pub: bytes
    hw_id: bytes
    slot_size: int = 1 << 20
    floor: int = 0
    active: str = "A"
    trial: Optional[str] = None           # the bank that is booting on probation
    boots: int = 0                        # boots of the trial bank so far
    running: str = "A"                    # the bank the last boot chose
    # True: a staged image is booted only after `approve()` (the person pressed the button), not by any restart
    require_approval: bool = False
    approved: bool = False
    slots: dict = field(default_factory=lambda: {"A": Slot(), "B": Slot()})
    log: list = field(default_factory=list)

    def install_factory(self, raw: bytes) -> None:
        """What the factory flashes into the first bank (still verified: a bad image never becomes 'active')."""
        img = verify_image(raw, self.vendor_pub, self.hw_id)
        self.slots[self.active] = Slot(img, raw=raw)
        self.running = self.active
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
        self.slots[self.other] = Slot(img, raw=raw)
        self.trial, self.boots, self.approved = self.other, 0, False
        self.log.append(f"staged v{img.version} in {self.other}")
        return img

    def reboot(self) -> str:
        """The bootloader's decision: a trial bank gets MAX_BOOTS tries to be confirmed, then the old bank is put back. Returns the bank."""
        if self.trial is not None and (self.approved or not self.require_approval):
            self.boots += 1
            if self.boots > MAX_BOOTS:
                self.slots[self.trial].bad = True
                self.log.append(f"{self.trial} never confirmed: back to {self.active}")
                self.trial, self.boots, self.approved = None, 0, False
            else:
                self.running = self.trial
                return self.trial
        self.running = self.active
        return self.active

    def confirm(self) -> None:
        """Called by the NEW firmware once it has proved itself (talked to the bridge, passed its self-test): it stays for good."""
        if self.trial is None or self.slots[self.trial].image is None:
            raise FirmwareError("no_trial", "nothing to confirm")
        img = self.slots[self.trial].image
        self.active, self.trial, self.boots, self.approved = self.trial, None, 0, False
        self.running = self.active
        self.floor = max(self.floor, img.min_version)
        self.log.append(f"confirmed v{img.version}, floor {self.floor}")

    def approve(self) -> None:
        """The person said 'now' (the button on the device): the staged image may be booted, by the next restart."""
        if self.trial is None or self.slots[self.trial].image is None:
            raise FirmwareError("no_trial", "nothing is staged")
        self.approved = True

    def check_switch_back(self) -> None:
        """Raises exactly when `switch_back` would (so a caller can ask for the person's button only for something that can happen)."""
        if self.trial is not None:
            return
        o = self.slots[self.other]
        if o.image is None or o.bad:
            raise FirmwareError("no_previous", "there is no other version on the device")
        if o.image.version < self.floor:
            raise FirmwareError("rollback", f"version {o.image.version} is below the floor {self.floor}")

    def switch_back(self) -> str:
        """The person asks for the other bank (the one the last update replaced). While an update is on trial this drops the trial. Never
        to a bank that is empty, failed its own trial, or is older than the anti-rollback floor."""
        self.check_switch_back()
        if self.trial is not None:
            self.slots[self.trial].bad = True
            self.log.append(f"{self.trial} dropped by the user: back to {self.active}")
            self.trial, self.boots, self.approved = None, 0, False
            self.running = self.active
            return self.active
        o = self.slots[self.other]
        self.log.append(f"switched back to {self.other} (v{o.image.version})")
        self.active = self.running = self.other
        return self.active

    def save(self, directory: Path) -> None:
        """The state a reboot has to find again: both banks' images and the bookkeeping. Atomic (temp file + rename)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name, sl in self.slots.items():
            p = d / f"bank-{name}.img"
            if sl.image is None:
                p.unlink(missing_ok=True)
                continue
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_bytes(sl.raw)
            os.replace(tmp, p)
        tmp = d / "state.json.tmp"
        tmp.write_text(json.dumps({"floor": self.floor, "active": self.active, "trial": self.trial, "boots": self.boots,
                                   "approved": self.approved,
                                   "bad": [k for k, s in self.slots.items() if s.bad]}))
        os.replace(tmp, d / "state.json")

    @staticmethod
    def load(directory: Path, vendor_pub: bytes, hw_id: bytes, slot_size: int = 1 << 20) -> "SlotManager":
        """Back from disk. Every image is verified again: a bank whose file was changed or damaged comes back empty."""
        d = Path(directory)
        m = SlotManager(vendor_pub, hw_id, slot_size)
        try:
            st = json.loads((d / "state.json").read_text())
        except (OSError, ValueError):
            return m
        if not isinstance(st, dict):
            return m
        for name in ("A", "B"):
            try:
                raw = (d / f"bank-{name}.img").read_bytes()
                m.slots[name] = Slot(verify_image(raw, vendor_pub, hw_id), bad=name in st.get("bad", []), raw=raw)
            except (OSError, FirmwareError):
                m.slots[name] = Slot()
        m.floor = int(st.get("floor", 0)) if isinstance(st.get("floor"), int) else 0
        m.active = st.get("active") if st.get("active") in ("A", "B") else "A"
        m.trial = st.get("trial") if st.get("trial") in ("A", "B") and m.slots[st["trial"]].image is not None else None
        m.boots = int(st.get("boots", 0)) if isinstance(st.get("boots"), int) else 0
        m.approved = st.get("approved") is True and m.trial is not None
        m.running = m.active
        return m

    def to_dict(self) -> dict:
        return {"floor": self.floor, "active": self.active, "trial": self.trial, "boots": self.boots,
                "slots": {k: {"bad": s.bad, "image": None if s.image is None else
                              {"hw": s.image.hw_id.hex(), "v": s.image.version, "min": s.image.min_version, "digest": s.image.digest.hex()}}
                          for k, s in self.slots.items()}}


class Upload:
    """An image arriving in pieces (CtlLink FW_BEGIN / FW_CHUNK / FW_END). Holds only what the sender has sent in order; it can be resumed
    from `next`, a piece that was already taken is acknowledged again without being applied twice, and a gap is refused."""

    def __init__(self, size: int, sha256: Optional[bytes], max_size: int, now_us: int = 0, name: str = "") -> None:
        """`sha256` is optional: an image carries its own signed hash, which `verify_image` checks anyway; the announced one only lets a
        sender that has a hash (a script, a test) catch damage before the longer check."""
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= max_size:
            raise FirmwareError("too_large", f"{size} bytes")
        if sha256 is not None and len(sha256) != 32:
            raise FirmwareError("damaged", "hash")
        self.size, self.sha256, self.name = size, sha256, name
        self.buf = bytearray()
        self.last_us = now_us

    @property
    def next(self) -> int:
        return len(self.buf)

    def chunk(self, offset: int, data: bytes, now_us: int = 0) -> int:
        self.last_us = now_us
        if offset + len(data) <= len(self.buf):                      # a repeat (the acknowledgement was lost)
            return len(self.buf)
        if offset != len(self.buf):
            raise FirmwareError("sequence", f"expected offset {len(self.buf)}")
        if len(self.buf) + len(data) > self.size:
            raise FirmwareError("too_large", "more than announced")
        self.buf += data
        return len(self.buf)

    def finish(self) -> bytes:
        if len(self.buf) != self.size:
            raise FirmwareError("sequence", f"{len(self.buf)} of {self.size} bytes")
        if self.sha256 is not None and hashlib.sha256(self.buf).digest() != self.sha256:
            raise FirmwareError("damaged", "the image does not match the announced hash")
        return bytes(self.buf)
