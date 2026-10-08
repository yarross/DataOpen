"""Channel B on the device: receive a package in pieces, keep it aside, apply it on the person's say-so, take it back (docs/UPDATES.md).

    PKG_BEGIN {size}          one reception at a time; the pending package (if any) stays until a NEW one has passed every check
    PKG_CHUNK offset+data     in order, repeats acknowledged, gaps refused; the header is judged as soon as it is complete
    PKG_END                   the whole is checked (hash, every part, the UI model); the file is kept as `pkg/pending.dopk`
    ACT pkg.apply             (two steps; the button for a sender not known yet, and for model weights from anybody but the owner)
    ACT pkg.discard           forget the pending package
    ACT pkg.revert            put the active slot's previous model back

Nothing is applied by receiving. The pending file is the package as it came (sealed to this device's keys, signed by its sender); apply opens
it again from disk with the same code, so a changed file on disk is caught as a changed file on the wire would be. Applying is roll-forward:
a marker is written first, every part is written atomically, and a restart in the middle finishes the job (the button was already spent)
instead of leaving half a package. The firmware banks are never touched from here."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from ..ctl import protocol as P
from ..ctl.manifest import validate_manifest
from ..ctl.seal import SlotData
from ..ctl.vault import VaultError
from . import models as MD
from .channels import PKG_SCHEMA, UpdateError
from .package import MAX_PACKAGE, PREFIX, Opened, PackageReader, open_package

PKG_IDLE_US = 60_000_000          # a reception nobody touches for a minute is dropped (as for firmware)
MAX_TRUSTED = 16                  # the same limit as for sealed files (gateway.MAX_TRUSTED)


class Refusal(Exception):
    """A refusal with a protocol error code (BUSY, NOT_ALLOWED...): the gateway turns it into an ERR message."""

    def __init__(self, code: int, detail: str = "") -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail


def approval_detail(o: Opened, known_sender: bool) -> str:
    """What the button is needed for, in the device's own words ('' = nothing). The owner's own copies need nothing."""
    if o.is_self:
        return ""
    if not known_sender:
        return f"trust:{o.sender_id}"
    if o.model is not None:
        return f"model:{o.sender_id}"
    return ""


class ModelStore:
    """The weights of a slot: `model.bin` + `model.json` (and `.prev` of both), sealed under THE SLOT's key, so clearing the slot clears them."""

    def __init__(self, slot) -> None:
        self.slot = slot

    def _p(self, name: str) -> Path:
        return self.slot.dir / name

    def _read(self, name: str) -> Optional[bytes]:
        aad = name.replace(".prev", "")                      # a generation keeps the name it was sealed under when it is moved aside
        try:
            return self.slot.vault.read(self._p(name), aad, allow_plain=False)
        except (OSError, VaultError):
            return None

    def _write(self, name: str, data: bytes) -> Path:
        tmp = self._p(name + ".tmp")
        tmp.write_bytes(self.slot.vault.seal(name, data))
        return tmp

    def info(self, prev: bool = False) -> Optional[dict]:
        raw = self._read("model.prev.json" if prev else "model.json")
        if raw is None:
            return None
        try:
            j = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return j if isinstance(j, dict) else None

    def weights(self) -> Optional[bytes]:
        return self._read("model.bin")

    def put(self, raw: bytes, meta: dict) -> None:
        bin_tmp, json_tmp = self._write("model.bin", raw), self._write("model.json", json.dumps(meta, separators=(",", ":")).encode())
        for cur, prev in (("model.bin", "model.prev.bin"), ("model.json", "model.prev.json")):
            if self._p(cur).exists():
                os.replace(self._p(cur), self._p(prev))
        os.replace(bin_tmp, self._p("model.bin"))
        os.replace(json_tmp, self._p("model.json"))

    def revert(self) -> bool:
        """Swap the current and the previous generation (so a second revert undoes the first). False: there is no previous one."""
        if not self._p("model.prev.json").exists() or not self._p("model.prev.bin").exists():
            return False
        for a, b in (("model.bin", "model.prev.bin"), ("model.json", "model.prev.json")):
            tmp = self._p(a + ".swap")
            if self._p(a).exists():
                os.replace(self._p(a), tmp)
            os.replace(self._p(b), self._p(a))
            if tmp.exists():
                os.replace(tmp, self._p(b))
        return True

    def state(self, fw_version: int) -> str:
        m = self.info()
        if m is None:
            return "none"
        return "needs_system" if int(m.get("min_fw", 0)) > fw_version else "ok"


class PackageManager:
    def __init__(self, gw, fault: Optional[Callable[[str], None]] = None) -> None:
        self.gw = gw
        self.dir = Path(gw.dir) / "pkg"
        self.reader: Optional[PackageReader] = None
        self._spool = None
        self.fault = fault or (lambda point: None)           # tests raise from here to simulate a power cut at a named point
        self.gen = 0                                         # bumped by anything that changes what is stored (the state is cached per generation)
        self._cache: Optional[tuple] = None

    def touch(self) -> None:
        self.gen += 1

    # -- helpers the gateway provides
    def _fw_version(self) -> int:
        return self.gw._fw_version()

    def _last_seq(self, digest: bytes) -> Optional[int]:
        rec = self.gw.trust["senders"].get(digest.hex())
        return rec["last_seq"] if rec is not None else None

    @property
    def pending_path(self) -> Path:
        return self.dir / "pending.dopk"

    def summary(self) -> Optional[dict]:
        try:
            return json.loads(self.gw.vault.read(self.dir / "pending.json", "pkg.pending", allow_plain=False).decode("utf-8"))
        except (OSError, ValueError, VaultError, UnicodeDecodeError):
            return None

    def models(self) -> list[ModelStore]:
        return [ModelStore(s) for s in self.gw.slotset]

    # -- reception
    def begin(self, j: dict, now: int) -> None:
        size = j.get("size")
        if isinstance(size, bool) or not isinstance(size, int):
            raise UpdateError("B", "damaged", "the size of the package was not given")
        if not PREFIX + 16 <= size <= 2 * MAX_PACKAGE:
            raise UpdateError("B", "too_large", f"{size!r} bytes")
        self._drop_reception()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reader = PackageReader(self.gw.identity, supported_schema=PKG_SCHEMA, fw_version=self._fw_version(), last_seq=self._last_seq,
                                    size=size, now_us=now)
        self._spool = open(self.dir / "incoming.part", "wb")

    def chunk(self, body: bytes, now: int) -> int:
        r = self.reader
        if r is None:
            raise UpdateError("B", "sequence", "no reception in progress")
        if len(body) < 5:
            self._drop_reception()
            raise UpdateError("B", "damaged", "empty chunk")
        offset, data = int.from_bytes(body[:4], "little"), body[4:]
        before = r.next
        try:
            nxt = r.chunk(offset, data, now)
        except UpdateError as e:
            if e.key != "sequence":                           # a gap or a repeat from the wrong place is not fatal; everything else is
                self._drop_reception()
            raise
        if nxt > before:
            self._spool.write(data[before - offset :])
        return nxt

    def end(self, now: int) -> dict:
        r = self.reader
        if r is None:
            raise UpdateError("B", "sequence", "no reception in progress")
        try:
            o = r.finish()
            self._check_content(o)
        except UpdateError:
            self._drop_reception()
            raise
        self._spool.close()
        self._spool = None
        os.replace(self.dir / "incoming.part", self.pending_path)
        sm = self._summarize(o, r.known_sender)
        tmp = self.dir / "pending.json.tmp"
        tmp.write_bytes(self.gw.vault.seal("pkg.pending", json.dumps(sm, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
        os.replace(tmp, self.dir / "pending.json")
        self.reader = None
        self.touch()
        return sm

    def _summarize(self, o: Opened, known: bool) -> dict:
        model = None if o.model is None else {"name": o.model_card.get("name", ""), "version": o.model_card.get("version", 0), "size": len(o.model)}
        return {"id": o.pkg_id.hex(), "kinds": o.kind_names, "slot": o.slot, "from": o.sender_id, "self": o.is_self, "seq": o.seq, "min_fw": o.min_fw,
                "button": approval_detail(o, known or o.is_self).split(":")[0], "model": model, "name": o.name}

    def _check_content(self, o: Opened) -> None:
        """What the device checks of the CONTENT before it keeps the package (the structure was checked while it arrived)."""
        if o.ui_manifest is not None and validate_manifest(o.ui_manifest):
            raise UpdateError("B", "bad_part", "the layout inside is not valid")
        if o.model is not None:
            MD.check_model(o.model, o.model_card)

    def _drop_reception(self) -> None:
        self.reader = None
        if self._spool is not None:
            try:
                self._spool.close()
            except OSError:
                pass
            self._spool = None
        (self.dir / "incoming.part").unlink(missing_ok=True)

    def tick(self, now: int) -> None:
        if self.reader is not None and now - self.reader.last_us > PKG_IDLE_US:
            self._drop_reception()                           # nobody is sending any more: the half package is dropped

    # -- applying
    def apply(self, now: int) -> None:
        gw = self.gw
        sm = self.summary()
        if sm is None or not self.pending_path.exists():
            raise UpdateError("B", "no_pending", "nothing to apply")
        if gw.calibrating:
            raise Refusal(P.E.BUSY, "calibrating")
        fw = gw.fw
        if fw is not None and fw.trial is not None and fw.running == fw.trial:
            raise Refusal(P.E.BUSY, "fw_trial")               # the new system has not yet proved that it reads what is stored
        o = self._open_pending()
        known = o.is_self or self._last_seq(o.sender_digest) is not None
        if not known and len(gw.trust["senders"]) >= MAX_TRUSTED:
            raise Refusal(P.E.NOT_ALLOWED, "trust list full")
        need = approval_detail(o, known)
        if need:
            gw._need_button(now, need)
        if not o.is_self:
            rec = gw.trust["senders"].get(o.sender_digest.hex())
            if rec is None:
                rec = gw.trust["senders"][o.sender_digest.hex()] = {"id": o.sender_id}
            rec["last_seq"] = o.seq
            gw._save_trust()
        (self.dir / "applying").write_text(sm["id"])
        self._install(o, now)

    def _open_pending(self, check_replay: bool = True) -> Opened:
        try:
            raw = self.pending_path.read_bytes()
        except OSError:
            raise UpdateError("B", "no_pending", "the pending file is gone") from None
        o = open_package(raw, self.gw.identity, fw_version=self._fw_version())
        last = self._last_seq(o.sender_digest)
        if check_replay and not o.is_self and last is not None and o.seq <= last:
            raise UpdateError("B", "replay", "this sequence number was used already")
        self._check_content(o)
        return o

    def _install(self, o: Opened, now: int) -> None:
        gw = self.gw
        k = o.slot if o.slot is not None else gw.active
        if o.model is not None:
            ModelStore(gw.slotset[k]).put(o.model, {"card": o.model_card, "min_fw": o.min_fw, "from": o.sender_id, "pkg": o.pkg_id.hex(), "seq": o.seq,
                                                   "self": o.is_self})
            self.touch()
            self.fault("after_model")
        if o.profile is not None or o.tuning is not None or o.ui_manifest is not None or o.name:
            gw._install_slot(SlotData(k, o.profile, o.tuning, o.ui_manifest, o.name), now)
        self.fault("after_parts")
        self._finish()

    def _finish(self) -> None:
        for name in ("pending.dopk", "pending.json", "applying"):
            (self.dir / name).unlink(missing_ok=True)
        self.touch()

    def recover(self, now: int) -> None:
        """At start: an apply that a restart cut in the middle is finished (roll-forward); a marker without a package is just removed."""
        if not (self.dir / "applying").exists():
            return
        try:
            o = self._open_pending(check_replay=False)       # the sender's sequence number was recorded before the cut: that is not a replay
        except (UpdateError, OSError):
            self._finish()
            return
        self._install(o, now)

    # -- the rest
    def discard(self) -> None:
        self._drop_reception()
        self._finish()

    def revert(self, now: int) -> None:
        if self.gw.calibrating:
            raise Refusal(P.E.BUSY, "calibrating")
        if not ModelStore(self.gw.slot).revert():
            raise UpdateError("B", "no_previous", "there is no previous model in this slot")
        self.touch()

    def erase(self) -> None:
        """Personal data goes with everything else: the reception, the pending package and the marker (models live in the slots)."""
        self._drop_reception()
        shutil.rmtree(self.dir, ignore_errors=True)
        self.touch()

    # -- what the phone sees
    def state(self) -> dict:
        key = (self.gen, self.gw.active, self._fw_version())
        if self._cache is None or self._cache[0] != key:
            sm = self.summary()
            ms = ModelStore(self.gw.slot)
            card = (ms.info() or {}).get("card") or {}
            self._cache = (key, {"pkg.from": sm["from"] if sm else "", "pkg.kinds": ",".join(sm["kinds"]) if sm else "",
                                 "pkg.pending": sm is not None, "model.state": ms.state(self._fw_version()), "model.name": card.get("name", ""),
                                 "model.version": int(card.get("version", 0))})
        c = dict(self._cache[1])
        pending = c.pop("pkg.pending")
        return {"pkg.state": "receiving" if self.reader is not None else ("pending" if pending else "none"), **c}

    def info(self) -> dict:
        r = self.reader
        return {"supported": True, "schema": PKG_SCHEMA, "max_size": MAX_PACKAGE, "pending": self.summary(),
                "upload": None if r is None else {"next": r.next, "size": r.size},
                "models": [{"n": s.k, "model": (ModelStore(s).info() or {}).get("card"), "state": ModelStore(s).state(self._fw_version()),
                            "previous": ModelStore(s).info(prev=True) is not None} for s in self.gw.slotset]}
