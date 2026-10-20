"""The proposer's bubblewrap jail (evolve/astra.py)."""
import shutil
import subprocess

import pytest

from evolve import astra


def test_the_jail_binds_only_the_named_folders(tmp_path):
    cd, out, home, extra = tmp_path / "export", tmp_path / "round", tmp_path / "home", tmp_path / "tools"
    for folder in (cd, out, home, extra):
        folder.mkdir()
    command = astra.jailed(["ls", "-la"], cd, [out], [extra], home)
    assert command[0] == "bwrap" and "--unshare-all" in command and "--share-net" in command
    binds = {command[i + 1]: command[i] for i, word in enumerate(command) if word in ("--bind", "--ro-bind")}
    assert binds[str(cd)] == "--bind" and binds[str(out)] == "--bind" and binds[str(home)] == "--bind"
    assert binds[str(extra)] == "--ro-bind" and binds["/usr"] == "--ro-bind"
    assert not any(path == "/home" or path == str(astra.CODEX_HOME) for path in binds)
    assert command[command.index("CODEX_HOME") + 1] == str(home)
    assert command[command.index("--") + 1].endswith("/ls") and command[-1] == "-la"


def test_a_fresh_home_holds_only_the_credentials(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    for name in ("auth.json", "config.toml", "history.jsonl"):
        (real / name).write_text(name)
    monkeypatch.setattr(astra, "CODEX_HOME", real)
    home = astra.fresh_home(tmp_path / "copy")
    assert sorted(p.name for p in home.iterdir()) == ["auth.json", "config.toml"]
    (home / "auth.json").write_text("refreshed")
    astra.return_home(home)
    assert (real / "auth.json").read_text() == "refreshed" and not home.exists()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
def test_inside_the_jail_home_is_invisible_and_the_export_writable(tmp_path):
    cd, home = tmp_path / "export", tmp_path / "home"
    cd.mkdir()
    home.mkdir()
    script = f"ls {astra.HERE} 2>/dev/null && echo SEEN; echo ok > {cd}/written; cat {cd}/written"
    command = astra.jailed(["sh", "-c", script], cd, [], [], home)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 and "Operation not permitted" in result.stderr:
        pytest.skip("user namespaces are not available here")
    assert "SEEN" not in result.stdout and result.stdout.strip().endswith("ok")


def test_sandboxed_candidate_checks_run_under_bubblewrap(monkeypatch):
    from evolve import remote

    commands = []
    monkeypatch.setattr(remote.Remote, "ssh", lambda self, command, timeout=0, check=True: commands.append(command)
                        or "PYTEST=0 RUFF=0")
    plain = remote.Remote("h", "/r", "/r/base", "/r/work", 1)
    jailed = remote.Remote("h", "/r", "/r/base", "/r/work", 1, sandbox=True)
    assert plain.static_checks("c")[0] and jailed.static_checks("c")[0]
    assert "bwrap" not in commands[0] and f"{remote.SANDBOX} /r/base/.venv/bin/python -m pytest" in commands[1]
