"""`.dobundle`: one file that carries a profile and the two knob levels between devices or back up from the phone (docs/PWA.md section 6).

    'DOBN' | ver u8 | kind u8 | flags u8 (0) | rsvd u8 | header_len u16 | profile_len u16 | header (JSON, UTF-8) | profile | crc32 u32

`profile` is the 98-byte BioProfileV1 (validated with the same function every reader uses; it may be empty: a bundle with settings only).
The CRC32 catches damage, it is NOT authentication: a bundle is trusted exactly as much as the bonded phone that sends it, and everything
it can set is limited by the bridge (assistance only ever subtracts).
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from typing import Optional

from ..bioprofile.profile import SIZE as PROFILE_SIZE
from ..bioprofile.profile import ProfileError, ProfileState, ProfileVersionError, validate
from . import protocol as P
from . import tuning as T

MAGIC = b"DOBN"
VERSION = 1
MAX_BUNDLE = 4096
_HEAD = "<4sBBBBHH"
_HEAD_SIZE = struct.calcsize(_HEAD)
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


class BundleError(ValueError):
    def __init__(self, msg: str, key: str = "bad_bundle") -> None:
        super().__init__(msg)
        self.key = key


@dataclass
class Bundle:
    name: str = ""
    created: str = ""
    strength: int = T.LEVEL_DEFAULT
    tremor: int = T.LEVEL_DEFAULT
    profile: Optional[ProfileState] = None

    def pack(self) -> bytes:
        tuning = {"strength": self.strength, "tremor": self.tremor}
        head = {"name": _clean(self.name), "created": _clean(self.created, 10), "tuning": tuning}
        header = json.dumps(head, ensure_ascii=False, separators=(",", ":")).encode()
        prof = self.profile.pack() if self.profile is not None else b""
        body = struct.pack(_HEAD, MAGIC, VERSION, 1, 0, 0, len(header), len(prof)) + header + prof
        out = body + struct.pack("<I", P.crc32(body))
        if len(out) > MAX_BUNDLE:
            raise BundleError("bundle too large", "too_big")
        return out


def _clean(s: str, limit: int = 40) -> str:
    return _CTRL.sub("", str(s))[:limit]


def unpack(raw: bytes) -> Bundle:
    """Strict: anything off raises BundleError (its `key` is the i18n suffix the phone shows)."""
    if len(raw) > MAX_BUNDLE:
        raise BundleError("bundle too large", "too_big")
    if len(raw) < _HEAD_SIZE + 4 or raw[:4] != MAGIC:
        raise BundleError("not a settings file")
    if struct.unpack_from("<I", raw, len(raw) - 4)[0] != P.crc32(raw[:-4]):
        raise BundleError("the file is damaged (CRC mismatch)", "damaged")
    _, ver, kind, flags, _, hlen, plen = struct.unpack_from(_HEAD, raw)
    if ver > VERSION:
        raise BundleError(f"file version {ver} is newer than this device understands ({VERSION})", "version")
    if ver != VERSION or kind != 1 or flags != 0 or plen not in (0, PROFILE_SIZE) or _HEAD_SIZE + hlen + plen + 4 != len(raw):
        raise BundleError("unsupported file layout")
    try:
        head = json.loads(raw[_HEAD_SIZE : _HEAD_SIZE + hlen].decode("utf-8"))
        tun = head["tuning"]
        strength, tremor = tun["strength"], tun["tremor"]
        name, created = head.get("name", ""), head.get("created", "")
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
        raise BundleError(f"bad header: {e}") from None
    for v in (strength, tremor):
        if isinstance(v, bool) or not isinstance(v, int) or not T.LEVEL_MIN <= v <= T.LEVEL_MAX:
            raise BundleError("knob levels out of range")
    if not isinstance(name, str) or not isinstance(created, str):
        raise BundleError("bad header")
    prof = None
    if plen:
        try:
            validate(raw[_HEAD_SIZE + hlen : _HEAD_SIZE + hlen + plen])
            prof = ProfileState.unpack(raw[_HEAD_SIZE + hlen : _HEAD_SIZE + hlen + plen])
        except ProfileVersionError:
            raise BundleError("the profile inside is newer than this device understands", "version") from None
        except ProfileError as e:
            raise BundleError(f"the profile inside is damaged: {e}", "damaged") from None
    return Bundle(_clean(name), _clean(created, 10), strength, tremor, prof)
