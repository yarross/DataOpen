"""The detector of a model that lives on the device (docs/MODELS.md, docs/RESIDENCY.md).

A custom model arrives as a package, is kept in its slot under the slot's key, and is never handed out again. The one thing that may read
its bytes is the process that RUNS it: this function. It decrypts the weights in memory, checks them once more against the card that was
accepted with them (the file on the device could have been changed; the same check as on arrival), and builds the detector from the
bytes. No file is written. Nothing here puts the bytes, or anything made from them, into a message for the phone: a test pins that this is
the only caller of the store's weights accessor in the code.

Not connected to the video path yet: the module daemon (MOD) that would hold this detector and run it in the loop is a build step that
does not exist (docs/MODELS.md, what is not done). What is here, and tested, is that a model accepted through the package channel can
become a working detector without leaving the device."""
from __future__ import annotations

from typing import Optional

from ..updates import models as MD
from ..updates.manager import ModelStore
from .service import OrtUiDetector


def detector_for_slot(gw, k: Optional[int] = None, conf: float = 0.35, threads: int = 2) -> Optional[OrtUiDetector]:
    """The detector for slot `k`'s own model (the active slot by default); None when the slot has no model of its own, or the system
    running is older than the model needs (`needs_system`), or the model is heavier than the latency budget allows (`over_budget`: it is kept,
    not run; docs/LATENCY.md)."""
    store = ModelStore(gw.slotset[gw.active if k is None else k])
    if store.state(gw._fw_version()) != "ok":
        return None
    meta, raw = store.info(), store.weights()
    if meta is None or raw is None:
        return None
    MD.check_model(raw, meta["card"])                       # the bytes are still the bytes that were accepted
    return OrtUiDetector(raw, conf=conf, threads=threads)
