import stat
from pathlib import Path

import pytest

from dataopen.cli import EXIT_FAILED_CHECK, EXIT_OK, main
from dataopen.installer import ADAPTERS, InstallError, build_unity_plugin, install, install_gmod


def test_gmod_install_copies_addon_and_the_shared_runtime(tmp_path):
    (tmp_path / "garrysmod").mkdir()
    paths = install_gmod(tmp_path)
    dest = tmp_path / "garrysmod" / "addons" / "dataopen"
    assert (dest / "lua" / "autorun" / "dataopen_init.lua").exists() and (dest / "lua" / "dataopen" / "gmod_host.lua").exists()
    assert (dest / "lua" / "dataopen" / "dataopen_rpc.lua").read_bytes() == \
        (ADAPTERS / "lua" / "runtime" / "dataopen_rpc.lua").read_bytes()
    assert dest in paths
    install_gmod(tmp_path)                                    # re-install is idempotent
    assert (dest / "lua" / "dataopen" / "dataopen_rpc.lua").exists()


def test_gmod_install_rejects_the_wrong_folder(tmp_path):
    with pytest.raises(InstallError, match="garrysmod"):
        install_gmod(tmp_path)


def fake_game(tmp_path, bepinex=True) -> Path:
    g = tmp_path / "game"
    (g / "BepInEx" / "core").mkdir(parents=True, exist_ok=True)
    if bepinex:
        (g / "BepInEx" / "core" / "BepInEx.dll").write_bytes(b"x")
    return g


def test_unity_install_requires_bepinex_and_dotnet(tmp_path, monkeypatch):
    with pytest.raises(InstallError, match="BepInEx is not installed"):
        install("valheim", fake_game(tmp_path, bepinex=False))
    with pytest.raises(InstallError, match="dotnet"):
        install("valheim", fake_game(tmp_path), dotnet="definitely-not-dotnet")


def test_unity_install_runs_dotnet_with_the_right_arguments_and_copies_the_dll(tmp_path):
    game = fake_game(tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = Path(cmd[cmd.index("-o") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "DataOpen.Valheim.dll").write_bytes(b"dll")

        class R:
            returncode, stdout, stderr = 0, "", ""
        return R()

    dotnet = tmp_path / "dotnet"
    dotnet.write_text("#!/bin/sh\n")
    dotnet.chmod(dotnet.stat().st_mode | stat.S_IEXEC)
    paths = install("valheim", game, dotnet=str(dotnet), run=fake_run)
    cmd = calls[0]
    assert cmd[1] == "build" and cmd[2].endswith("DataOpen.Valheim.csproj") and f"-p:GameDir={game}" in cmd and "Release" in cmd
    assert paths == [game / "BepInEx" / "plugins" / "DataOpen" / "DataOpen.Valheim.dll"] and paths[0].read_bytes() == b"dll"


def test_build_failure_is_reported_with_the_compiler_output(tmp_path):
    dotnet = tmp_path / "dotnet"
    dotnet.write_text("#!/bin/sh\n")
    dotnet.chmod(dotnet.stat().st_mode | stat.S_IEXEC)

    def failing(cmd, **kw):
        class R:
            returncode, stdout, stderr = 1, "Foo.cs(3,1): error CS1002: ; expected", ""
        return R()

    with pytest.raises(InstallError, match="CS1002"):
        build_unity_plugin("rust", fake_game(tmp_path), dotnet=str(dotnet), run=failing)


def test_cli_install_reports_failures_and_success(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["install", "--game", "gmod", "--dir", str(tmp_path)])
    assert e.value.code == EXIT_FAILED_CHECK and "install failed" in capsys.readouterr().out
    (tmp_path / "garrysmod").mkdir()
    with pytest.raises(SystemExit) as e:
        main(["install", "--game", "gmod", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert e.value.code == EXIT_OK and "-insecure" in out and "GMOD_DIR" in out
