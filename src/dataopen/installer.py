"""`dataopen install`: put the in-game mod where the game will load it.

  gmod             copies the addon (Lua) into <GarrysMod>/garrysmod/addons/dataopen
  valheim | rust   builds the BepInEx plugin with `dotnet build` (needs the .NET SDK and BepInEx installed in the
                   game) and copies it to <game>/BepInEx/plugins/DataOpen/

Works from a source checkout (the mods live in ./adapters).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

ADAPTERS = Path(__file__).resolve().parents[2] / "adapters"


class InstallError(RuntimeError):
    pass


def _check_checkout() -> None:
    if not (ADAPTERS / "lua" / "runtime" / "dataopen_rpc.lua").exists():
        raise InstallError("the mods are not part of the installed wheel: run `pip install -e .` from a source checkout "
                           f"(expected {ADAPTERS})")


def install_gmod(game_dir: Path) -> list[Path]:
    _check_checkout()
    game_dir = Path(game_dir)
    if not (game_dir / "garrysmod").is_dir():
        raise InstallError(f"{game_dir} does not look like a Garry's Mod folder (no 'garrysmod' subfolder inside). "
                           f"Point --dir at the folder that contains hl2.exe / gmod.exe")
    dest = game_dir / "garrysmod" / "addons" / "dataopen"
    src = ADAPTERS / "gmod" / "addon" / "dataopen"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    runtime_dest = dest / "lua" / "dataopen" / "dataopen_rpc.lua"
    shutil.copy2(ADAPTERS / "lua" / "runtime" / "dataopen_rpc.lua", runtime_dest)
    return [dest, runtime_dest]


_UNITY = {"valheim": "DataOpen.Valheim", "rust": "DataOpen.Rust"}


def build_unity_plugin(game: str, game_dir: Path, dotnet: str = "dotnet",
                       run: Callable = subprocess.run) -> Path:
    _check_checkout()
    if game not in _UNITY:
        raise InstallError(f"no Unity plugin for {game!r}; known: {sorted(_UNITY)}")
    game_dir = Path(game_dir)
    bepinex = game_dir / "BepInEx" / "core" / "BepInEx.dll"
    if not bepinex.exists():
        raise InstallError(f"BepInEx is not installed in {game_dir} (missing BepInEx/core/BepInEx.dll). Install BepInEx 5 "
                           f"(Mono build) for this game, start the game once, close it, then run this again")
    if shutil.which(dotnet) is None:
        raise InstallError("the .NET SDK is required to build the plugin but `dotnet` was not found. Install the .NET SDK "
                           "(6.0 or newer) and try again")
    proj = ADAPTERS / "unity" / _UNITY[game] / f"{_UNITY[game]}.csproj"
    out = ADAPTERS / "unity" / _UNITY[game] / "bin" / "release-out"
    res = run([dotnet, "build", str(proj), "-c", "Release", f"-p:GameDir={game_dir}", "-o", str(out)],
              capture_output=True, text=True)
    if res.returncode != 0:
        tail = "\n".join((res.stdout + res.stderr).strip().splitlines()[-25:])
        raise InstallError(f"dotnet build failed:\n{tail}\n\nCompile errors usually mean a different game/BepInEx/Unity "
                           f"version than the plugin was written against: send this output and the game version.")
    dll = out / f"{_UNITY[game]}.dll"
    if not dll.exists():
        raise InstallError(f"build succeeded but {dll} was not produced")
    return dll


def install_unity_plugin(game: str, game_dir: Path, dotnet: str = "dotnet",
                         run: Callable = subprocess.run) -> list[Path]:
    dll = build_unity_plugin(game, game_dir, dotnet, run)
    dest_dir = Path(game_dir) / "BepInEx" / "plugins" / "DataOpen"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dll.name
    shutil.copy2(dll, dest)
    return [dest]


def install(game: str, game_dir: Path, dotnet: str = "dotnet", run: Optional[Callable] = None) -> list[Path]:
    if game == "gmod":
        return install_gmod(game_dir)
    return install_unity_plugin(game, game_dir, dotnet, run or subprocess.run)


NEXT_STEPS = {
    "gmod": ("Start Garry's Mod with the launch option  -insecure  , load a map (gm_flatgrass), then:\n"
             "  set GMOD_DIR={dir}\n  dataopen doctor --game gmod\n"
             "Keep the game window focused and do not open the pause menu while collecting."),
    "valheim": ("Start Valheim, load into a world, then:\n  set VALHEIM_DIR={dir}\n  dataopen doctor --game valheim"),
    "rust": ("Start your OWN local server with RCON enabled, join it with the client, then:\n"
             "  set RUST_DIR={dir}\n  set RUST_RCON_PASSWORD=<your rcon password>\n  dataopen doctor --game rust"),
}
