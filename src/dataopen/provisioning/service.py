"""Service operations on a device directory, done by the support tool from the recovery system or through the service port:
the full return to factory state (L4) and its checks (docs/PROVISIONING.md sections 4 and 5).

L4 is the only level that touches the firmware banks. It needs the manufacturer's single-use token (made for THIS serial and for the
challenge THIS device issued) AND a hand on the device. It never lowers the anti-rollback floor, never changes the serial or the DAK, and
never writes an image that is not signed by the manufacturer.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..ctl import firmware as FW
from ..ctl.identity import FileKeyStore, provision
from .device import DeviceAgent, DeviceError


class ServiceError(RuntimeError):
    """`key`: token | presence | golden | floor | lifecycle | action"""

    def __init__(self, key: str, msg: str = "") -> None:
        super().__init__(msg or key)
        self.key = key


def wipe_personal(directory: Path) -> None:
    """L3 on disk: every file of the owner, a new storage key and new owner keys (the DAK and the OTP are not here, they are not touched)."""
    d = Path(directory)
    for pattern in ("settings.*", "trust.*", "profile.*", "manifest.json*"):
        for p in d.glob(pattern):
            p.unlink(missing_ok=True)
    shutil.rmtree(d / "slots", ignore_errors=True)
    provision(FileKeyStore(d / "keys" / "keys.json"))


def full_return(directory: str | Path, agent: DeviceAgent, token_json: dict, *, presence: bool) -> dict:
    """L4. Returns what was done. Raises ServiceError and leaves the device as it was when anything is wrong."""
    d = Path(directory)
    rec = agent.record
    if rec is None:
        raise ServiceError("lifecycle", "not a provisioned device")
    golden, golden_mcu = d / "golden" / "golden.img", d / "golden" / "golden-mcu.img"
    try:
        g = FW.verify_image(golden.read_bytes(), agent.vendor_pub, agent.hw_id)
        m = FW.verify_image(golden_mcu.read_bytes(), agent.vendor_pub, agent.hw_id)
    except (OSError, FW.FirmwareError):
        raise ServiceError("golden", "the golden image is missing or damaged: use the support tool through the service port") from None
    mgr = FW.SlotManager.load(d / "fw", agent.vendor_pub, agent.hw_id)
    if g.version < mgr.floor:
        raise ServiceError("floor", "the golden image is older than the anti-rollback floor: download a newer one")
    try:
        action = agent.apply_service(token_json, presence=presence)         # the token is spent here, after everything that can fail cheaply
    except DeviceError as e:
        raise ServiceError("presence" if e.key == "presence" else "token", str(e)) from None
    if action != "factory_return":
        raise ServiceError("action", f"this token is for {action}")
    agent.to_rma()
    mgr.slots = {"A": FW.Slot(g, raw=golden.read_bytes()), "B": FW.Slot(g, raw=golden.read_bytes())}
    mgr.active, mgr.running, mgr.trial, mgr.boots, mgr.approved = "A", "A", None, 0, False
    mgr.save(d / "fw")                                                      # the floor stays what it was: `save` writes mgr.floor as loaded
    for bank in "AB":
        (d / "mcu" / f"bank-{bank}.img").write_bytes(golden_mcu.read_bytes())
    wipe_personal(d)
    agent.back_from_rma()
    return {"serial": rec.serial, "som": g.version, "mcu": m.version, "floor": mgr.floor, "lifecycle": agent.lifecycle}
