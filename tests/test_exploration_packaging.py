"""An installed CLI locates release assets without importing a robot backend."""
import pytest

from homebody.real2sim.exploration import cli


def tree(path):
    (path / "configs").mkdir(parents=True)
    (path / "configs/exploration.toml").write_text("[exploration]\n")
    return path


def installed_cli(monkeypatch, tmp_path):
    module = tmp_path / "site-packages/homebody/real2sim/exploration/cli.py"
    monkeypatch.setattr(cli, "__file__", str(module))
    monkeypatch.delenv("HOMEBODY_ROOT", raising=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)


def test_installed_cli_finds_release_from_nested_working_directory(monkeypatch, tmp_path):
    installed_cli(monkeypatch, tmp_path)
    release = tree(tmp_path / "release")
    nested = release / "runs/example"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert cli.release_root() == release


def test_explicit_root_wins_and_missing_explicit_root_refuses(monkeypatch, tmp_path):
    installed_cli(monkeypatch, tmp_path)
    explicit = tree(tmp_path / "explicit")
    assert cli.release_root(explicit) == explicit
    with pytest.raises(ValueError, match="--root"):
        cli.release_root(tmp_path / "missing")


def test_missing_release_is_clear_but_help_remains_available(monkeypatch, tmp_path, capsys):
    installed_cli(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="--root"):
        cli.release_root()
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "G1 + LiDAR + iPhone" in capsys.readouterr().out


def test_editable_install_keeps_source_tree_default(monkeypatch, tmp_path):
    installed_cli(monkeypatch, tmp_path)
    release = tree(tmp_path / "editable")
    monkeypatch.setattr(cli, "__file__", str(release / "src/homebody/real2sim/exploration/cli.py"))
    assert cli.release_root() == release
