"""The proposer: Codex (astra) as a diagnostician (read-only) and as a patch writer (in its own
git worktree)."""
import subprocess
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def settings(path=HERE / "config.toml"):
    return tomllib.loads(path.read_text())["astra"]


def codex(prompt, cd, sandbox, out, extra_dirs=()):
    s = settings()
    command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--sandbox", sandbox,
               "--cd", str(cd), "--model", s["model"], "-c", f'model_reasoning_effort="{s["effort"]}"',
               "-c", 'web_search="disabled"', "--output-last-message", str(out), "--color", "never"]
    for directory in extra_dirs:
        command += ["--add-dir", str(directory)]
    result = subprocess.run(command + ["-"], check=False, input=prompt.encode(), capture_output=True, timeout=s["timeout"])
    (Path(out).parent / (Path(out).stem + ".stderr.txt")).write_bytes(result.stdout[-20000:] + result.stderr[-20000:])
    if result.returncode != 0:
        raise RuntimeError(f"codex exited {result.returncode}; see {out}")
    return Path(out).read_text()


def diagnose(worktree, workspace, out):
    prompt = (HERE / "prompts" / "diagnose.md").read_text().format(workspace=workspace)
    return codex(prompt, worktree, "read-only", out)


def propose(worktree, workspace, diagnosis, mechanism, out):
    prompt = (HERE / "prompts" / "propose.md").read_text().format(
        workspace=workspace, diagnosis=diagnosis, mechanism=mechanism, ruff=settings()["ruff"])
    return codex(prompt, worktree, "workspace-write", out)
