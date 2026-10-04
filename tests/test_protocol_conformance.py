"""The core parses specific JSON keys; every mod implementation (C#, Lua) must produce them. We cannot run the C#
here, but a typo in a key name is exactly the kind of bug that survives a syntax check, so verify the literals."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "adapters"

HELLO_KEYS = ["protocol", "game", "engine", "game_version", "mod_version", "capabilities", "image", "schema_errors",
              "parameter_space"]
CAPTURE_KEYS = ["frame_token", "tick", "camera", "entities", "probes", "warnings", "width", "height", "pos", "forward",
                "right", "up", "fov_v_deg", "near", "entity_id", "rig_id", "skeleton_world", "joint_valid",
                "engine_visibility", "hull_points", "meta", "world", "screen"]
METHODS = ["hello", "begin_scene", "capture_frame", "release", "commit", "discard", "end_scene", "selftest", "health",
           "shutdown", "peek"]
SELFTEST_KEYS = ["checks", "name", "ok", "detail", "hint", "data", "unmapped", "bones_found"]
REQUEST_KEYS = ["frame_id", "frame", "image_mode", "frame_token", "dest", "options", "bone_map", "schema", "keypoints"]
PEEK_KEYS = ["transport", "width", "height", "staged"]
SHM_KEYS = ["shm", "capacity"]          # C# only: Lua has no shared memory and answers peek with a staged file
# `max_side` is a hint a mod may ignore (neither implementation downscales yet)


@pytest.mark.parametrize("keys", [HELLO_KEYS, CAPTURE_KEYS, METHODS, SELFTEST_KEYS, REQUEST_KEYS])
def test_csharp_sdk_uses_the_protocol_keys(keys):
    src = "".join(p.read_text() for p in (ROOT / "unity" / "DataOpen.UnitySdk").glob("*.cs"))
    missing = [k for k in keys if f'"{k}"' not in src]
    assert not missing, f"C# SDK never mentions these protocol keys: {missing}"


def test_peek_keys_are_present_in_every_mod_implementation():
    cs = "".join(p.read_text() for p in (ROOT / "unity" / "DataOpen.UnitySdk").glob("*.cs"))
    lua = (ROOT / "lua" / "runtime" / "dataopen_rpc.lua").read_text()
    for k in PEEK_KEYS:
        assert f'"{k}"' in cs, f"C# peek never mentions {k}"
        assert k in lua, f"Lua peek never mentions {k}"
    for k in SHM_KEYS:
        assert f'"{k}"' in cs
    for cap in ("image_peek", "image_shm"):
        assert f'"{cap}"' in cs
    assert '"image_peek"' in lua


def test_lua_runtime_uses_the_protocol_keys():
    src = (ROOT / "lua" / "runtime" / "dataopen_rpc.lua").read_text()
    # Lua writes keys as identifiers (frame_token = ...) or quoted; accept both spellings
    missing = [k for k in HELLO_KEYS + CAPTURE_KEYS + SELFTEST_KEYS + REQUEST_KEYS
               if f"{k} =" not in src and f'"{k}"' not in src and f"p.{k}" not in src and f".{k}" not in src]
    assert not missing, f"Lua runtime never mentions: {missing}"
    for m in METHODS:
        assert f"H.{m}" in src or f"function H.{m}" in src, f"Lua runtime has no handler for {m}"


def test_core_and_mods_agree_on_the_method_list():
    from dataopen.core.protocol import METHODS as CORE
    assert list(CORE) == METHODS
