"""Safe storage and sharing of the profile.

  ProfileStore     two slot files (`<path>.a`, `<path>.b`) written alternately, each via temp file + fsync + atomic rename. A torn write
                   or a flipped bit can only damage the slot being written; `load()` returns the newest slot whose magic / CRC / version
                   check passes. A slot written by a NEWER version is never overwritten and makes `save()` raise: an old binary must
                   not destroy a newer profile.
  ProfilePublisher / ProfileReader
                   the newest profile in shared memory (seqlock + CRC32 mailbox of the inference runtime), for modules that poll it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from ..runtime.channel import ShmLatest
from .profile import SIZE, ProfileError, ProfileState, ProfileVersionError, ProfileView, validate


class ProfileStore:
    """`codec` (optional; any object with `read(path, name)` and `seal(name, data)`, e.g. `dataopen.ctl.vault.Vault`) encrypts the slot
    files at rest; without it nothing changes. A plain slot is still read when a codec is set, so an old store migrates on its next save."""

    def __init__(self, path: str | Path, codec=None) -> None:
        self.codec = codec
        self.base = Path(path)
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.slots = [self.base.with_name(self.base.name + ".a"), self.base.with_name(self.base.name + ".b")]

    def _read(self, p: Path) -> Optional[ProfileState]:
        try:
            data = self.codec.read(p, self.base.name) if self.codec else p.read_bytes()
        except (OSError, ValueError):                  # missing, or sealed under another key / damaged: the same as a bad slot
            return None
        try:
            validate(data)
        except ProfileVersionError:
            raise
        except ProfileError:
            return None
        return ProfileState.unpack(data)

    def load(self) -> Optional[ProfileState]:
        """Newest valid slot, or None (nothing stored / both slots damaged). Raises ProfileVersionError for a newer format."""
        best: Optional[ProfileState] = None
        for p in self.slots:
            st = self._read(p)
            if st is not None and (best is None or st.generation >= best.generation):
                best = st
        return best

    def save(self, state: ProfileState) -> int:
        """Write the next generation into the OLDER slot. Returns the generation written."""
        cur = self.load()
        state.generation = (cur.generation + 1) if cur else max(state.generation, 0) + 1
        gens = []
        for p in self.slots:
            st = self._read(p)                         # may raise ProfileVersionError: refuse to touch newer data
            gens.append(-1 if st is None else st.generation)
        target = self.slots[0 if gens[0] <= gens[1] else 1]
        tmp = target.with_name(target.name + ".tmp")
        data = state.pack()
        if self.codec:
            data = self.codec.seal(self.base.name, data)
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        try:
            fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        return state.generation


class ProfilePublisher:
    def __init__(self, name: str) -> None:
        self.shm = ShmLatest(name, max_bytes=max(128, SIZE))

    def publish(self, state: ProfileState) -> None:
        self.shm.publish(state.pack())

    def close(self) -> None:
        self.shm.close()


class ProfileReader:
    """What another module holds. `read()` returns a validated `ProfileView` (with its age) or None; it never raises on bad data."""

    def __init__(self, name: str) -> None:
        self.shm = ShmLatest(name, create=False)

    def read(self) -> Optional[ProfileView]:
        r = self.shm.read()
        if r is None:
            return None
        _, data, ts_us = r
        try:
            return ProfileView(data, age_ms=(time.monotonic_ns() // 1000 - ts_us) / 1000.0)
        except ProfileError:
            return None

    def close(self) -> None:
        self.shm.close()
