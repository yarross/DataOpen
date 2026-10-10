"""The rule that keeps self-recovery alive across updates (docs/V1.md, docs/UPDATES.md).

The golden recovery image sits in a write-protected region and can not be replaced by an ordinary update. The anti-rollback floor, on the other
hand, rises with every confirmed update whose `min_version` is higher. If the floor ever gets above the golden image's version, the recovery
system refuses to restore from it (and so does the full return), and the unit then needs a support tool on the service port. So a release must
never set a floor the golden image can not meet."""
from __future__ import annotations

from ..ctl.firmware import FirmwareError, verify_image


def check_release(update_raw: bytes, golden_raw: bytes, vendor_pub: bytes, hw_id: bytes, floor_now: int = 0) -> list[str]:
    """Problems with putting `update_raw` into the field next to the golden image `golden_raw` (empty list: fine)."""
    problems: list[str] = []
    try:
        upd = verify_image(update_raw, vendor_pub, hw_id)
    except FirmwareError as e:
        return [f"the update image is not acceptable: {e.key}"]
    try:
        gold = verify_image(golden_raw, vendor_pub, hw_id)
    except FirmwareError as e:
        return [f"the golden image is not acceptable: {e.key}"]
    floor_after = max(floor_now, upd.min_version)
    if floor_after > gold.version:
        problems.append(f"this release raises the anti-rollback floor to {floor_after}, above the golden image's version {gold.version}: "
                        "after it neither the recovery system nor the full return can restore the unit (a support tool would be needed)")
    if upd.version < floor_now:
        problems.append(f"this release (version {upd.version}) is below the floor already in the field ({floor_now}) and would be refused")
    return problems
