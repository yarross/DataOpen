"""Game profiles: everything that differs between games, as data (TOML), not code.

A profile says where the mod's mailbox lives, the render size, how frames are captured, what to pass
to the mod (population/camera modes, bone-name overrides) and sensible session defaults, so adding or
tuning a game does not require touching Python or recompiling a mod.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


class ProfileError(ValueError):
    pass


@dataclass
class GameProfile:
    id: str
    name: str
    engine: str = ""
    archetype: str = ""
    mailbox_dir: str = ""
    width: int = 1280
    height: int = 720
    capture_mode: str = "auto"
    grabber_region: Optional[dict[str, int]] = None
    mod_options: dict[str, Any] = field(default_factory=dict)
    bone_map: dict[str, list[list[Any]]] = field(default_factory=dict)
    session: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GameProfile":
        if "id" not in d or "name" not in d:
            raise ProfileError("a profile needs `id` and `name`")
        r, c = d.get("render", {}), d.get("capture", {})
        return GameProfile(
            id=d["id"], name=d["name"], engine=d.get("engine", ""), archetype=d.get("archetype", ""),
            mailbox_dir=d.get("mailbox_dir", ""), width=int(r.get("width", 1280)), height=int(r.get("height", 720)),
            capture_mode=c.get("mode", "auto"), grabber_region=c.get("region"),
            mod_options=dict(d.get("mod_options", {})), bone_map=dict(d.get("bones", {})),
            session=dict(d.get("session", {})), server=dict(d.get("server", {})),
            provenance=dict(d.get("provenance", {})), status=dict(d.get("status", {})))

    def mailbox(self, override: Optional[str] = None) -> Optional[Path]:
        """Resolve the mailbox directory; None means "run in-process" (mock only)."""
        raw = override if override is not None else self.mailbox_dir
        if not raw:
            return None
        expanded = os.path.expanduser(os.path.expandvars(raw))
        if "${" in expanded or (os.name == "nt" and "%" in expanded):
            var = raw[raw.index("${") + 2:raw.index("}")] if "${" in raw else "the variable"
            raise ProfileError(f"mailbox_dir {raw!r} uses {var}, which is not set. Set the environment variable "
                               f"or pass --mailbox DIR")
        return Path(expanded)


def _parse(text: str) -> GameProfile:
    try:
        return GameProfile.from_dict(tomllib.loads(text))
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"invalid TOML: {e}") from e


def list_profiles() -> list[GameProfile]:
    out = []
    for f in sorted((resources.files("dataopen") / "game_profiles").iterdir(), key=lambda p: p.name):
        if f.name.endswith(".toml"):
            out.append(_parse(f.read_text(encoding="utf-8")))
    return out


def load_profile(name_or_path: str) -> GameProfile:
    p = Path(name_or_path)
    if p.suffix == ".toml" and p.exists():
        return _parse(p.read_text(encoding="utf-8"))
    for prof in list_profiles():
        if prof.id == name_or_path:
            return prof
    raise ProfileError(f"unknown game {name_or_path!r}; known: {', '.join(p.id for p in list_profiles())} "
                       f"(or pass a path to a .toml profile)")
