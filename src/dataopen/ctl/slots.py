"""Hardware slots: several independent contexts on one device (docs/SLOTS.md).

A person has a few places they work in (the office programme, the browser, the one or two applications whose accessibility API is bad)
and each wants its own motor profile, its own two levels and its own screen layout. Carrying files to the device every time the context
changes is not an option, so the device keeps `SLOT_COUNT` of them and the person switches with a button or from the phone.

What is IN a slot: the profile (A/B files + the previous one), the two levels, the manifest (optional), a name, and `vetted`.
What is NOT: 'assistance is on' (switching context must never turn assistance on by itself), which slot is active, the trusted senders,
the device identity. Those are the device's.

Every file of a slot is encrypted under that slot's own key (`Identity.slot_key`): another slot's key opens nothing of it, and clearing
a slot (`SlotSet.clear`) moves it to a new key, so the old ciphertext is dead even if the flash still holds it, while the other slots
are untouched. `vetted` is 'the person has kept this content in work' (a trial ended with 'keep', or it was in use when they left): a
vetted slot is switched to at once; one that just arrived from a file or a calibration goes through the same try-it-then-keep-it timer as
any other raise of help.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Optional

from ..bioprofile.profile import ProfileError
from ..bioprofile.store import ProfileStore
from . import protocol as P
from . import tuning as T
from .identity import Identity
from .vault import Vault, VaultError

SLOT_COUNT = P.SLOT_COUNT
NAME_MAX = 24
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def clean_name(s) -> str:
    return _CTRL.sub("", str(s)).strip()[:NAME_MAX]


class SettingsStore:
    """Two files, newest valid wins (same idea as ProfileStore), JSON + CRC32 inside; encrypted at rest when given a vault."""

    def __init__(self, base: Path, vault: Optional[Vault] = None) -> None:
        self.base, self.vault = base, vault
        base.parent.mkdir(parents=True, exist_ok=True)
        self.slots = [base.with_name(base.name + ".a"), base.with_name(base.name + ".b")]

    def _read(self, p: Path) -> Optional[dict]:
        try:
            raw = self.vault.read(p, self.base.name) if self.vault else p.read_bytes()
            if len(raw) < 5 or struct.unpack_from("<I", raw, len(raw) - 4)[0] != P.crc32(raw[:-4]):
                return None
            d = json.loads(raw[:-4].decode("utf-8"))
            return d if isinstance(d, dict) and isinstance(d.get("n"), int) else None
        except (OSError, ValueError, VaultError):
            return None

    def load(self) -> dict:
        best: dict = {}
        for p in self.slots:
            d = self._read(p)
            if d is not None and (not best or d["n"] >= best["n"]):
                best = d
        return best

    def save(self, data: dict) -> None:
        cur = self.load()
        data = {**data, "n": cur.get("n", 0) + 1}
        older = min(self.slots, key=lambda p: (self._read(p) or {"n": -1})["n"])
        body = json.dumps(data, separators=(",", ":")).encode()
        tmp = older.with_name(older.name + ".tmp")
        blob = body + struct.pack("<I", P.crc32(body))
        with open(tmp, "wb") as f:
            f.write(self.vault.seal(self.base.name, blob) if self.vault else blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, older)


def _lvl(v) -> int:
    try:
        return min(max(int(v), T.LEVEL_MIN), T.LEVEL_MAX)
    except (TypeError, ValueError):
        return T.LEVEL_DEFAULT


class Slot:
    """One slot's storage. Everything that goes on disk goes through this slot's own vault."""

    def __init__(self, base: Path, k: int, vault: Vault) -> None:
        self.k = k
        self.dir = base / str(k)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vault = vault
        self.profile = ProfileStore(self.dir / "profile", codec=vault)
        self.meta_store = SettingsStore(self.dir / "meta", vault)
        self.prev_path = self.dir / "profile.prev"
        self.manifest_path = self.dir / "manifest.json"
        m = self.meta_store.load()
        self.name = clean_name(m.get("name", ""))
        self.strength, self.tremor = _lvl(m.get("strength", T.LEVEL_DEFAULT)), _lvl(m.get("tremor", T.LEVEL_DEFAULT))
        self.vetted = bool(m.get("vetted", False))
        self.manifest: Optional[dict] = self._load_manifest()
        self.has = False
        self.refresh()

    # -- what is stored
    def refresh(self) -> None:
        try:
            self.has = self.profile.load() is not None
        except ProfileError:                                   # a newer format is still 'something there'
            self.has = True

    def save_meta(self) -> None:
        self.meta_store.save({"name": self.name, "strength": self.strength, "tremor": self.tremor, "vetted": self.vetted})

    def _load_manifest(self) -> Optional[dict]:
        try:
            m = json.loads(self.vault.read(self.manifest_path, "manifest.json", allow_plain=False).decode("utf-8"))
        except (OSError, ValueError, VaultError):
            return None
        return m if isinstance(m, dict) else None

    def set_manifest(self, m: Optional[dict]) -> None:
        if m is None:
            self.manifest_path.unlink(missing_ok=True)
        else:
            tmp = self.manifest_path.with_name("manifest.json.tmp")
            tmp.write_bytes(self.vault.seal("manifest.json", json.dumps(m, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
            os.replace(tmp, self.manifest_path)
        self.manifest = m

    @property
    def used(self) -> bool:
        """Does it hold anything worth carrying along (a profile, a layout of its own, a name)?"""
        return self.has or self.manifest is not None or bool(self.name)

    def info(self) -> dict:
        return {"n": self.k, "name": self.name, "has": self.has, "vetted": self.vetted, "manifest": self.manifest is not None,
                "strength": self.strength, "tremor": self.tremor}


class SlotSet:
    """All the slots of one device, under `<dir>/slots/<k>/`."""

    def __init__(self, directory: Path, identity: Identity) -> None:
        self.base = Path(directory) / "slots"
        self.identity = identity
        self.slots: list[Slot] = []
        self._open()

    def _open(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        self.slots = [Slot(self.base, k, Vault(self.identity.slot_key(k))) for k in range(SLOT_COUNT)]

    def __getitem__(self, k: int) -> Slot:
        return self.slots[k]

    def __iter__(self):
        return iter(self.slots)

    def mask(self) -> int:
        return sum(1 << s.k for s in self.slots if s.has)

    def clear(self, k: int) -> None:
        """Crypto-erase one slot: a new key for it, its files deleted, an empty slot in its place."""
        self.identity.bump_slot_epoch(k)
        shutil.rmtree(self.base / str(k), ignore_errors=True)
        self.slots[k] = Slot(self.base, k, Vault(self.identity.slot_key(k)))

    def wipe_all(self) -> None:
        """Everything of every slot gone; the caller has changed the storage key (or the whole identity) already."""
        shutil.rmtree(self.base, ignore_errors=True)
        self._open()


def migrate_legacy(directory: Path, base_vault: Vault, slotset: SlotSet, legacy_levels: Optional[tuple[int, int]]) -> bool:
    """A device from before slots kept ONE profile at `<dir>/profile.*`, `profile.prev`, `manifest.json`. It becomes slot 0 (re-encrypted
    under slot 0's key), as something the person had in work already (`vetted`). Idempotent: nothing left to move = nothing done."""
    d = Path(directory)
    s0 = slotset[0]
    moved = False
    files = [d / "profile.a", d / "profile.b"]
    if any(p.exists() for p in files):
        for p in files:
            try:
                data = base_vault.read(p, "profile")
            except (OSError, VaultError):
                p.unlink(missing_ok=True)
                continue
            tmp = s0.dir / ("profile." + p.name[-1] + ".tmp")
            tmp.write_bytes(s0.vault.seal("profile", data))
            os.replace(tmp, s0.dir / ("profile." + p.name[-1]))
            p.unlink(missing_ok=True)
            moved = True
    prev = d / "profile.prev"
    if prev.exists():
        try:
            s0.prev_path.write_bytes(s0.vault.seal("profile.prev", base_vault.read(prev, "profile.prev")))
        except (OSError, VaultError):
            pass
        prev.unlink(missing_ok=True)
        moved = True
    man = d / "manifest.json"
    if man.exists():
        try:
            s0.set_manifest(json.loads(base_vault.read(man, "manifest.json").decode("utf-8")))
        except (OSError, ValueError, VaultError):
            pass
        man.unlink(missing_ok=True)
        moved = True
    if moved:
        s0.refresh()
        s0.vetted = True
        if legacy_levels is not None:
            s0.strength, s0.tremor = legacy_levels
        s0.save_meta()
    return moved
