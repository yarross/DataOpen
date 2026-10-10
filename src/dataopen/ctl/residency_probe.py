"""The residency check on a simulated device: put a profile and a model with KNOWN bytes on it through the only door there is (a sealed
package from a sender), then ask it everything it can be asked, and look for those bytes in everything it said (docs/RESIDENCY.md).

Used by `dataopen ctl residency check` and by the tests. It is a probe, not a proof by itself: the proof is that the surface is closed
(`residency.py`) AND that nothing that was tried found a canary.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import protocol as P
from . import residency as RS

PROFILE_ID = 0x5A17C3E1                       # a number nothing else in the device has: as bytes, as decimal, as hex


class Canary:
    """Needles: runs of the bytes of things that must not leave the device, found in a blob of what the device said."""

    def __init__(self) -> None:
        self.needles: dict[str, bytes] = {}
        self.texts: dict[str, str] = {}

    def add_bytes(self, name: str, data: bytes, width: int = 12, step: int = 4, skip: int = 0, tail: int = 0) -> None:
        """Windows of `data`, from `skip` to `tail` bytes before the end (the tail of an ONNX file is its metadata: class names the device
        legitimately says, and those must not count as the weights leaking)."""
        for i in range(skip, max(skip, len(data) - tail - width + 1), step):
            w = data[i : i + width]
            if len(set(w)) >= 5:                       # a run of zeros or one repeated byte would be found anywhere
                self.needles[f"{name}@{i}"] = w

    def add_text(self, name: str, text: str) -> None:
        self.texts[name] = text

    def hits(self, blob: bytes) -> list[str]:
        b = bytes(blob)
        out = [n for n, w in self.needles.items() if w in b]
        try:
            t = b.decode("utf-8", errors="ignore")
        except Exception:                              # pragma: no cover
            t = ""
        out += [n for n, w in self.texts.items() if w in t]
        return out


def make_profile(seed: int = 7):
    """A profile with a number of its own, learned from a simulated person (the real BioProfile engine)."""
    from ..assist.sim_user import PERSONAS, build_profile
    view = build_profile(PERSONAS["tremor"], minutes=6.0, seed=seed)
    st = view._state
    st.profile_id = PROFILE_ID
    return st


def make_model(seed: int = 11, pad: int = 6000) -> bytes:
    from ..updates.dev import tiny_model
    return tiny_model(pad=pad, seed=seed)


def canary_for(profile, model: bytes, identity=None, dev_dir: Optional[Path] = None) -> Canary:
    c = Canary()
    c.add_bytes("profile", profile.pack(), width=8, step=1, skip=16)
    c.add_text("profile_id.dec", str(PROFILE_ID))
    c.add_text("profile_id.hex", f"{PROFILE_ID:08x}")
    c.add_bytes("model", model, width=16, step=16, skip=64, tail=1024)
    if identity is not None:
        c.add_bytes("storage_key", identity.storage_key, width=12, step=4)
        for k in range(P.SLOT_COUNT):
            c.add_bytes(f"slot_key{k}", identity.slot_key(k), width=12, step=4)
    if dev_dir is not None:                           # every secret in the key file (private keys, the disk key), base64 and raw
        import base64
        kf = Path(dev_dir) / "keys" / "keys.json"
        if kf.exists():
            for name, v in json.loads(kf.read_text(encoding="utf-8")).items():
                if isinstance(v, str) and len(v) >= 40:
                    c.add_text(f"keyfile.{name}", v[:32])
                    try:
                        c.add_bytes(f"keyfile.{name}", base64.b64decode(v), width=12, step=4)
                    except ValueError:
                        pass
    return c


@dataclass
class ProbeResult:
    wire: bytes
    hits: list[str]
    asked: list[str] = field(default_factory=list)
    refused_secret: dict = field(default_factory=dict)
    egress_blocked: int = 0
    storage_hits: list[str] = field(default_factory=list)


def put_resident(w, directory, profile, model: bytes, *, slot: int = 0, seq: int = 1):
    """Carry the profile and the model onto the device the only way there is: a package sealed by somebody else (the button is pressed
    for the new sender and for the weights), then apply. Returns the sender."""
    from ..updates import dev as UD
    from ..updates import package as K
    from .identity import Card
    sender = UD.sender(Path(directory) / "senders")
    card = Card.from_json(w.phone.get_identity())
    raw = K.build_package(sender, card, seq, slot=slot, profile=profile, tuning=(6, 4), name="canary",
                          model=(model, UD.card_of(model, name="canary", version=3)))
    r = w.phone.pkg_send(raw)
    assert r.type == P.T_ACK, r.body
    w.gw.physical_press(w.t)
    r = w.phone.act("pkg.apply", True)
    assert r.type == P.T_ACK, r.body
    w.phone.confirm(True)                         # a profile that arrives while help is on is a trial: keep it (an error when there is none is fine)
    return sender


def ask_everything(w) -> list[str]:
    """Every question the protocol has, plus every name somebody might try for the secret. Returns what was asked."""
    asked = []
    for kind in P.GET_KINDS:
        r = w.phone.call(P.T_GET, {"what": kind})
        asked.append(f"get {kind}: {'DATA' if r.type == P.T_DATA else 'ERR'}")
    for name in RS.ASK_FOR_SECRET + ("nothing", "", "../keys", "model.bin"):
        r = w.phone.call(P.T_GET, {"what": name})
        asked.append(f"get {name!r}: {r.json().get('key') if r.type == P.T_ERR else 'DATA!'}")
    w.gw.read_status()
    w.gw.read_info()
    return asked


def run_probe(directory: Optional[str] = None) -> ProbeResult:
    from .sim import World
    d = Path(directory or tempfile.mkdtemp(prefix="dataopen-residency-"))
    w = World(d / "dev")
    w.phone.connect()
    profile, model = make_profile(), make_model()
    put_resident(w, d, profile, model)
    w.phone.confirm(True)
    can = canary_for(profile, model, w.gw.identity, d / "dev")
    asked = ask_everything(w)
    w.run(300)
    wire = bytes(w.wire) + w.gw.read_status() + w.gw.read_info()
    refused = {}
    for name in RS.ASK_FOR_SECRET:
        r = w.phone.call(P.T_GET, {"what": name})
        refused[name] = r.json().get("code") if r.type == P.T_ERR else None
    stor = []
    for f in sorted((d / "dev").rglob("*")):
        if f.is_file() and f.suffix != ".tmp" and "senders" not in f.parts:
            stor += [f"{f.relative_to(d / 'dev')}: {h}" for h in can.hits(f.read_bytes()) if not h.startswith("keyfile")]
    return ProbeResult(wire, can.hits(wire), asked, refused, w.gw.egress_blocked, stor)
