"""The proposer: Codex (astra) as a diagnostician (read-only) and as a patch writer (in its own
git worktree)."""
import json
import subprocess
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Codex credits per 1M tokens (input, cached input, output); unknown models are priced as Astra.
RATES = {"gpt-6-astra": (250.0, 25.0, 1250.0), "gpt-6.1-sol": (50.0, 2.5, 250.0), "gpt-6-luna": (2.5, 0.25, 12.5)}


def settings(path=HERE / "config.toml"):
    return tomllib.loads(path.read_text())["astra"]


def codex(prompt, cd, sandbox, out, extra_dirs=(), model=None, effort=None):
    """Run one Codex exec with PROMPT in CD; MODEL and EFFORT override the [astra] settings."""
    s = settings()
    command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--sandbox", sandbox,
               "--cd", str(cd), "--model", model or s["model"], "-c", f'model_reasoning_effort="{effort or s["effort"]}"',
               "-c", 'web_search="disabled"', "--output-last-message", str(out), "--color", "never", "--json"]
    for directory in extra_dirs:
        command += ["--add-dir", str(directory)]
    result = subprocess.run(command + ["-"], check=False, input=prompt.encode(), capture_output=True, timeout=s["timeout"])
    (Path(out).parent / (Path(out).stem + ".stderr.txt")).write_bytes(result.stdout[-20000:] + result.stderr[-20000:])
    record_usage(result.stdout, model or s["model"], Path(out).parent / (Path(out).stem + ".usage.json"))
    if result.returncode != 0:
        raise RuntimeError(f"codex exited {result.returncode}; see {out}")
    return Path(out).read_text()


def record_usage(stdout, model, path):
    """Write the call's tokens and credits (from Codex's turn.completed events) to PATH."""
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
    path.write_text(json.dumps({"model": model, "input": total_in, "cached": cached, "output": output,
                                "credits": round(credits, 2), "metered": bool(turns)}))


def credits_in(folder):
    """Total credits of the Codex calls recorded under FOLDER (*.usage.json)."""
    return round(sum(json.loads(p.read_text()).get("credits", 0) for p in Path(folder).glob("*.usage.json")), 2)


def diagnose(worktree, workspace, out):
    prompt = (HERE / "prompts" / "diagnose.md").read_text().format(workspace=workspace)
    return codex(prompt, worktree, "read-only", out)


def propose(worktree, workspace, diagnosis, mechanism, out):
    prompt = (HERE / "prompts" / "propose.md").read_text().format(
        workspace=workspace, diagnosis=diagnosis, mechanism=mechanism, ruff=settings()["ruff"],
        python=settings()["python"])
    return codex(prompt, worktree, "workspace-write", out)
