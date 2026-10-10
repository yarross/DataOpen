"""A sender's own identity: the keys a person (the owner, a trusted helper, a clinic) signs packages with (docs/MODELS.md, docs/UPDATES.md).

The device never makes a file itself (docs/RESIDENCY.md), so "the owner brings their own model" means: the owner has a sender identity of their
own, kept OUTSIDE the device, and the device is told once, by its button, to trust it. This module is that identity. It is the same kind of
object as a device identity (Ed25519 for the signature, X25519 for the replies, a counter for the sequence numbers), kept in a file with mode
0600 in a directory with mode 0700. It is NOT the manufacturer's key and grants nothing by itself: a package signed by it is judged like any
stranger's until the device's owner presses the button for its ID (`trust:<ID>`).

The key file is SIMULATION of a secret store. Lose it and the sender must be trusted again under a new ID; leak it and anyone can send packages
under that ID (they still need the device's card to seal to, and the button for every set of weights)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..ctl.identity import FileKeyStore, Identity, load_identity, provision

DEFAULT_NAME = "sender"


class SenderError(ValueError):
    pass


def default_dir() -> Path:
    return Path(os.environ.get("DATAOPEN_SENDER_DIR") or Path.home() / ".dataopen" / "sender")


def _store(directory, name: str) -> FileKeyStore:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise SenderError(f"bad sender name: {name!r}")
    return FileKeyStore(Path(directory) / f"{name}.json")


def load(directory=None, name: str = DEFAULT_NAME) -> Optional[Identity]:
    """The sender stored there, or None. Never creates one."""
    store = _store(directory or default_dir(), name)
    return load_identity(store) if store.load() is not None else None


def init(directory=None, name: str = DEFAULT_NAME) -> Identity:
    """Make a sender identity. Refuses to replace an existing one: a new ID would have to be trusted again on every device."""
    d = Path(directory or default_dir())
    store = _store(d, name)
    if store.load() is not None:
        raise SenderError(f"a sender already exists in {d} (name {name!r}); its ID stays the same")
    return provision(store)


def require(directory=None, name: str = DEFAULT_NAME) -> Identity:
    me = load(directory, name)
    if me is None:
        raise SenderError(f"no sender in {directory or default_dir()}: run `dataopen update sender init`")
    return me


def describe(me: Identity) -> dict:
    return {"id": me.id, "created": me.created, "packages_made": me.export_seq,
            "what_the_device_shows": f"trust:{me.id}", "kept": "outside the device, in a file only you can read"}
