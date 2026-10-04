import json
import threading
from pathlib import Path

import pytest

from dataopen.adapters.mock.server import MockServerOptions, serve_mock
from dataopen.cli import EXIT_ABORTED, EXIT_FAILED_CHECK, EXIT_OK, EXIT_USAGE, build_parser, main
from dataopen.profiles import ProfileError, list_profiles, load_profile


def run(*argv) -> int:
    with pytest.raises(SystemExit) as e:
        main(list(argv))
    return e.value.code


class Game:
    def __init__(self, directory: Path, **opts):
        self.stop = threading.Event()
        self.t = threading.Thread(target=serve_mock, args=(directory, MockServerOptions(**opts), self.stop), daemon=True)
        self.t.start()

    def close(self):
        self.stop.set()
        self.t.join(3)


def test_profiles_load_and_describe_the_chosen_games():
    ids = {p.id for p in list_profiles()}
    assert {"mock", "gmod", "valheim", "rust"} <= ids
    rust = load_profile("rust")
    assert rust.server["rcon"].startswith("127.0.0.1") and rust.mod_options["camera_mode"] == "player_relative"
    assert "UNSPECIFIED" in load_profile("gmod").provenance["assets"]
    with pytest.raises(ProfileError, match="unknown game"):
        load_profile("nope")


def test_mailbox_env_expansion_and_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("GMOD_DIR", raising=False)
    with pytest.raises(ProfileError, match="GMOD_DIR"):
        load_profile("gmod").mailbox()
    monkeypatch.setenv("GMOD_DIR", str(tmp_path))
    assert load_profile("gmod").mailbox() == tmp_path / "garrysmod" / "data" / "dataopen"
    assert load_profile("gmod").mailbox("/x/y") == Path("/x/y")


def test_custom_toml_profile(tmp_path):
    p = tmp_path / "g.toml"
    p.write_text('id="g"\nname="G"\n[render]\nwidth=640\nheight=360\n[bones]\nhead=[["Head",1.0]]\n')
    prof = load_profile(str(p))
    assert (prof.width, prof.height) == (640, 360) and prof.bone_map["head"] == [["Head", 1.0]]
    (tmp_path / "bad.toml").write_text("id = ")
    with pytest.raises(ProfileError):
        load_profile(str(tmp_path / "bad.toml"))


def test_games_doctor_collect_verify_preview_on_the_in_process_mock(tmp_path, capsys):
    assert run("games") == EXIT_OK and "gmod" in capsys.readouterr().out
    assert run("doctor", "--game", "mock", "--out", str(tmp_path / "doc")) == EXIT_OK
    assert "ready to collect" in capsys.readouterr().out
    ds = tmp_path / "ds"
    assert run("collect", "--game", "mock", "--out", str(ds), "--frames", "30", "--seed", "3") == EXIT_OK
    assert run("verify", str(ds)) == EXIT_OK and (ds / "qa_report.md").exists()
    assert run("preview", str(ds), "--n", "6") == EXIT_OK and (ds / "preview.png").stat().st_size > 1000


def test_collect_over_the_wire_with_shards_then_merge(tmp_path, capsys):
    mb = tmp_path / "mb"
    g = Game(mb)
    try:
        for i in range(2):
            assert run("collect", "--game", "mock", "--mailbox", str(mb), "--out", str(tmp_path / f"s{i}"),
                       "--frames", "20", "--shard", f"{i}/2", "--seed", "5", "--frames-per-scene", "5") == EXIT_OK
    finally:
        g.close()
    assert run("merge", str(tmp_path / "all"), str(tmp_path / "s0"), str(tmp_path / "s1")) == EXIT_OK
    assert run("verify", str(tmp_path / "all")) == EXIT_OK
    card = json.loads((tmp_path / "all" / "DATASET_CARD.json").read_text())
    assert card["counts"]["frames"] == 40 and len(card["merged_from"]) == 2
    # merging the same shard twice must be refused (frame ids collide)
    assert run("merge", str(tmp_path / "dup"), str(tmp_path / "s0"), str(tmp_path / "s0")) == EXIT_FAILED_CHECK


def test_collect_refuses_when_preflight_doctor_fails(tmp_path, capsys):
    mb = tmp_path / "mb"
    g = Game(mb, flip_probe_y=True)
    try:
        assert run("collect", "--game", "mock", "--mailbox", str(mb), "--out", str(tmp_path / "ds"),
                   "--frames", "10") == EXIT_FAILED_CHECK
        assert "Refusing to collect" in capsys.readouterr().out
        assert not (tmp_path / "ds").exists()                      # nothing was written
        # --no-doctor still stops, via the in-session calibration guard
        assert run("collect", "--game", "mock", "--mailbox", str(mb), "--out", str(tmp_path / "ds2"),
                   "--frames", "10", "--no-doctor") == EXIT_ABORTED
        assert "CALIBRATION ERROR" in capsys.readouterr().out
    finally:
        g.close()


def test_unset_game_dir_is_a_usage_error(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("VALHEIM_DIR", raising=False)
    assert run("doctor", "--game", "valheim", "--out", str(tmp_path)) == EXIT_USAGE
    assert "VALHEIM_DIR" in capsys.readouterr().out


def test_serve_mock_rejects_unknown_option(tmp_path, capsys):
    assert run("serve-mock", "--mailbox", str(tmp_path), "--option", "bogus=1") == EXIT_USAGE


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
