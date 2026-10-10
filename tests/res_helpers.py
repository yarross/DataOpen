"""Files come IN, never out (docs/RESIDENCY.md): the device makes no sealed files, so the tests that need one get it from a sender that is
NOT a device: somebody (a clinic, a technician) with their own keys and their own data, who seals it to the device's public card."""
from __future__ import annotations

from pathlib import Path

from dataopen.ctl import seal as SL
from dataopen.ctl.identity import Card
from dataopen.updates import dev as UD


class Clinic:
    """A sender with its own keys. `file(to, ...)` seals a settings file (DOBS) to a device; `seq` only grows, as a receiver expects."""

    def __init__(self, directory, name: str = "clinic") -> None:
        self.identity = UD.sender(Path(directory) / "senders", name)
        self.seq = 0

    @property
    def id(self) -> str:
        return self.identity.id

    def file(self, to, *, profile=None, tuning=None, manifest=None, meta=None, slots=None) -> bytes:
        """`to`: a World, a SimPhone, or a card (dict)."""
        card = to if isinstance(to, dict) else (to.phone if hasattr(to, "phone") else to).get_identity()
        self.seq += 1
        return SL.seal(self.identity, Card.from_json(card), self.seq, profile=profile, tuning=tuning, manifest=manifest, meta=meta, slots=slots)

    def package(self, to, **kw) -> bytes:
        from dataopen.updates import package as K
        card = to if isinstance(to, dict) else (to.phone if hasattr(to, "phone") else to).get_identity()
        self.seq += 1
        return K.build_package(self.identity, Card.from_json(card), self.seq, **kw)


def profile_of(world):
    """The profile a simulated device holds, as the object a sender would have built (a test reaches into the device; a phone never can)."""
    return world.gw.profile_store.load()
