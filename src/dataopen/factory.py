"""Build a ready-to-use adapter from a game profile."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .adapters.rcon import CompositeAdapter, RconClient, RconController
from .adapters.remote import MssGrabber, RemoteGameAdapter, RemoteOptions, ScreenGrabber
from .core.interfaces import IGameAdapter
from .core.transport import FileMailboxTransport
from .profiles import GameProfile, ProfileError


def build_adapter(profile: GameProfile, mailbox: Optional[str] = None, connect: bool = True,
                  grabber: Optional[ScreenGrabber] = None) -> IGameAdapter:
    """In-process mock when the profile has no mailbox, otherwise a RemoteGameAdapter that talks to the mod."""
    box = profile.mailbox(mailbox)
    if box is None:
        if profile.engine != "mock":
            raise ValueError(f"{profile.id}: no mailbox directory configured (use --mailbox)")
        from .adapters.mock import MockGameAdapter
        return MockGameAdapter(profile.width, profile.height, variant=profile.mock_variant)
    opts = RemoteOptions(capture_mode=profile.capture_mode, image_size=(profile.width, profile.height),
                         bone_map=profile.bone_map, mod_options=profile.mod_options)
    grabber = grabber or MssGrabber(profile.grabber_region)
    adapter: IGameAdapter = RemoteGameAdapter(FileMailboxTransport(Path(box)), opts, grabber=grabber)
    if profile.server.get("rcon"):
        env_name = profile.server.get("password_env", "RCON_PASSWORD")
        password = os.environ.get(env_name)
        if not password:
            raise ProfileError(f"{profile.id}: the server password is read from the environment variable "
                               f"{env_name}, which is not set")
        client = RconClient(profile.server["rcon"], password)
        adapter = CompositeAdapter(adapter, RconController(
            client, profile.server.get("scene_begin", []), profile.server.get("scene_end", []),
            float(profile.server.get("settle_s", 1.0))))
    if connect:
        adapter.connect()
    return adapter
