"""The proposer: Codex (astra) as a diagnostician (read-only) and as a patch writer (in its own
git worktree)."""
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Codex credits per 1M tokens (input, cached input, output); unknown models are priced as Astra.
RATES = {"gpt-6-astra": (250.0, 25.0, 1250.0), "gpt-6.1-sol": (50.0, 2.5, 250.0), "gpt-6-luna": (2.5, 0.25, 12.5)}
TIMED_OUT_CREDITS = 50.0  # charged for a call that reported no finished turn (killed at its time limit, failed or cut)
CODEX_HOME = Path.home() / ".codex"  # the real one: only auth.json and config.toml are copied into a jailed call's
SYSTEM_DIRS = ("/usr", "/etc", "/mnt/wsl")  # read-only in the jail (/mnt/wsl holds WSL's resolv.conf)


class BudgetExceeded(RuntimeError):
    """A credit cap was reached: the call or evaluation was not started; resume deliberately."""


def settings(path=HERE / "config.toml"):
    return tomllib.loads(path.read_text())["astra"]


def tool_dirs(s):
    """The folders the proposer's tools need read-only in a jail: ruff's, the test environment and its Python."""
    python = Path(s["python"])
    venv = python.parents[1]
    home = next((line.split("=", 1)[1].strip() for line in (venv / "pyvenv.cfg").read_text().splitlines()
                 if line.startswith("home")), None) if (venv / "pyvenv.cfg").exists() else None
    return [Path(s["ruff"]).parent, venv] + ([Path(home).parent] if home else [])


def jailed(command, cd, writable, readable, codex_home):
    """COMMAND (a Codex exec) inside bubblewrap: the system read-only, CD and WRITABLE writable, READABLE read-only,
    CODEX_HOME (a fresh copy holding only the credentials) as Codex's home, and nothing else of /home: no other
    checkout, no results, no archive, no Codex history. The network stays (Codex reaches its model)."""
    binary = Path(shutil.which(command[0]) or command[0]).resolve()
    jail = ["bwrap", "--die-with-parent", "--unshare-all", "--share-net", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin"]
    for directory in SYSTEM_DIRS:
        if Path(directory).exists():
            jail += ["--ro-bind", directory, directory]
    jail += ["--ro-bind", str(binary.parent), str(binary.parent)]
    for directory in readable:
        jail += ["--ro-bind", str(directory), str(directory)]
    for directory in (cd, *writable, codex_home):
        jail += ["--bind", str(directory), str(directory)]
    jail += ["--setenv", "CODEX_HOME", str(codex_home), "--setenv", "HOME", str(codex_home), "--chdir", str(cd)]
    return jail + ["--", str(binary), *command[1:]]


def fresh_home(folder):
    """A Codex home in FOLDER holding only the real one's auth.json and config.toml."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in ("auth.json", "config.toml"):
        if (CODEX_HOME / name).exists():
            shutil.copy2(CODEX_HOME / name, folder / name)
    return folder


def return_home(folder):
    """Keep credentials the jailed call refreshed (a rotated refresh token would otherwise be lost), then delete the
    copy."""
    auth = folder / "auth.json"
    if auth.exists() and (CODEX_HOME / "auth.json").exists() and auth.read_bytes() != (CODEX_HOME / "auth.json").read_bytes() \
            and auth.stat().st_mtime > (CODEX_HOME / "auth.json").stat().st_mtime:
        shutil.copy2(auth, CODEX_HOME / "auth.json")
    shutil.rmtree(folder, ignore_errors=True)


def codex(prompt, cd, sandbox, out, extra_dirs=(), model=None, effort=None, jail=None):
    """Run one Codex exec with PROMPT in CD; MODEL and EFFORT override the [astra] settings. JAIL (a list of folders
    it may read, besides CD and its tools) runs it inside bubblewrap (jailed); None runs it as is."""
    s = settings()
    command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--sandbox", sandbox,
               "--cd", str(cd), "--model", model or s["model"], "-c", f'model_reasoning_effort="{effort or s["effort"]}"',
               "-c", 'web_search="disabled"', "--output-last-message", str(out), "--color", "never", "--json"]
    for directory in extra_dirs:
        command += ["--add-dir", str(directory)]
    usage = Path(out).parent / (Path(out).stem + ".usage.json")
    home = None
    command = command + ["-"]
    if jail is not None:
        home = fresh_home(Path(out).parent / (Path(out).stem + ".codex_home"))
        command = jailed(command, Path(cd).resolve(), [Path(out).parent.resolve(), *extra_dirs],
                         [*tool_dirs(s), *jail], home)
    try:
        result = subprocess.run(command, check=False, input=prompt.encode(), capture_output=True, timeout=s["timeout"])
    except subprocess.TimeoutExpired as error:  # the call was paid for: record what it reported, never nothing
        stdout, stderr = error.stdout or b"", error.stderr or b""
        (Path(out).parent / (Path(out).stem + ".stderr.txt")).write_bytes(stdout[-20000:] + stderr[-20000:])
        record_usage(stdout, model or s["model"], usage, unreported=TIMED_OUT_CREDITS, timed_out=True)
        raise RuntimeError(f"codex timed out after {s['timeout']} s; see {out}") from error
    finally:
        if home is not None:
            return_home(home)
    (Path(out).parent / (Path(out).stem + ".stderr.txt")).write_bytes(result.stdout[-20000:] + result.stderr[-20000:])
    record_usage(result.stdout, model or s["model"], usage, unreported=TIMED_OUT_CREDITS)
    if result.returncode != 0:
        raise RuntimeError(f"codex exited {result.returncode}; see {out}")
    return Path(out).read_text()


def record_usage(stdout, model, path, unreported=0.0, timed_out=False):
    """Write the call's tokens and credits (from Codex's turn.completed events) to PATH. A call that reported no
    finished turn (failed, cut off) is charged at least UNREPORTED, never nothing; a TIMED_OUT one too, since its
    last turn never reports."""
    turns = []
    for line in stdout.decode(errors="replace").splitlines():
        if '"turn.completed"' in line:
            try:
                turns.append(json.loads(line)["usage"])
            except (json.JSONDecodeError, KeyError):
                continue
    total_in = sum(t.get("input_tokens", 0) for t in turns)
    cached = sum(t.get("cached_input_tokens", 0) for t in turns)
    output = sum(t.get("output_tokens", 0) for t in turns)
    rate_in, rate_cached, rate_out = RATES.get(model, RATES["gpt-6-astra"])
    credits = ((total_in - cached) * rate_in + cached * rate_cached + output * rate_out) / 1e6
    if timed_out or not turns:
        credits = max(credits, unreported)
    path.write_text(json.dumps({"model": model, "input": total_in, "cached": cached, "output": output,
                                "credits": round(credits, 2), "metered": bool(turns) and not timed_out}))


def credits_in(folder):
    """Total credits of the Codex calls recorded under FOLDER (*.usage.json)."""
    return round(sum(json.loads(p.read_text()).get("credits", 0) for p in Path(folder).glob("*.usage.json")), 2)


def call_credits(out, default):
    """The credits the call writing OUT recorded beside it (OUT's .usage.json), or DEFAULT when it recorded none."""
    path = Path(out).parent / (Path(out).stem + ".usage.json")
    try:
        record = json.loads(path.read_text())
        value, metered = record.get("credits"), record.get("metered", True)
    except (OSError, ValueError, AttributeError):
        return float(default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not value >= 0 or value == float("inf"):
        return float(default)
    return float(value) if metered is True else max(float(value), float(default))  # unknown cost: the whole reservation


def diagnose(worktree, workspace, out):
    prompt = (HERE / "prompts" / "diagnose.md").read_text().format(workspace=workspace)
    return codex(prompt, worktree, "read-only", out)


def propose(worktree, workspace, diagnosis, mechanism, out):
    prompt = (HERE / "prompts" / "propose.md").read_text().format(
        workspace=workspace, diagnosis=diagnosis, mechanism=mechanism, ruff=settings()["ruff"],
        python=settings()["python"])
    return codex(prompt, worktree, "workspace-write", out)
