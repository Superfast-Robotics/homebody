"""The proposer's workspace for one round: brief, train results, traces and history.

Only train runs are shown. Held-out (val) results decide acceptance but never appear here
beyond an accepted/rejected verdict.
"""
import json
import shutil
from pathlib import Path

from . import score

HERE = Path(__file__).resolve().parent
MAX_FAILURES_PER_RUN = 6
KEEP_EVENT_KINDS = {"skill"}  # stages, plans, results, traces, blocked arms


def build(round_dir, best, train_local, history, repo_tasks):
    """Write ROUND_DIR/workspace and return its path. BEST is the archive entry being improved,
    TRAIN_LOCAL its collected train runs, HISTORY prior archive entries, REPO_TASKS the train
    tasks' instruction text by name."""
    ws = round_dir / "workspace"
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "traces").mkdir(parents=True)
    (ws / "BRIEF.md").write_text((HERE / "prompts" / "brief.md").read_text())
    runs = score.read_split(train_local)
    lines = [f"# Train results of version {best['id']} (commit {best['commit'][:10]})", "",
             (f"Score {best['train']['score']:.3f} (mean stage / 3), success rate {best['train']['success_rate']:.2f}. "
              "Stage: 0 nothing, 1 pick verified, 2 place reported done, 3 physics-verified success."), "",
             "| task | instruction | stages | failures (label: code: message) |", "|---|---|---|---|"]
    for task, stages in best["train"]["by_task"].items():
        task_runs = [r for r in runs if r.task == task]
        failures = "<br>".join(f"{label}: {code}: {message}" for r in task_runs
                               for label, code, message in r.failures[:MAX_FAILURES_PER_RUN])
        lines.append(f"| {task} | {repo_tasks.get(task, '')} | {stages} | {failures or '-'} |")
    (ws / "results.md").write_text("\n".join(lines) + "\n")
    for task_dir in sorted(p for p in Path(train_local).iterdir() if p.is_dir()):
        for repeat_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            out = ws / "traces" / task_dir.name / repeat_dir.name
            out.mkdir(parents=True)
            log = repeat_dir / "run.log"
            if log.exists():
                kept = [line for line in log.read_text(errors="replace").splitlines()
                        if line.startswith(("RESULT", "PHYSICS_SCORE", "FINAL_OBJECT", "FINISHED", "Traceback"))
                        or "Error" in line]
                (out / "run_summary.log").write_text("\n".join(kept) + "\n")
            events = next(repeat_dir.rglob("events.jsonl"), None)
            if events:
                skill_events = []
                for raw in events.read_text(errors="replace").splitlines():
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if record.get("kind") in KEEP_EVENT_KINDS:
                        skill_events.append(raw[:2000])
                (out / "skill_events.jsonl").write_text("\n".join(skill_events) + "\n")
    (ws / "history.md").write_text(render_history(history))
    return ws


def render_history(history):
    if not history:
        return "# History\n\nNo earlier proposals.\n"
    lines = ["# History of proposals (most recent last)", ""]
    for entry in history:
        lines.append(f"## {entry['id']} (parent {entry.get('parent')}, round {entry.get('round')}): {entry['status']}")
        if entry.get("reason"):
            lines.append(f"Verdict: {entry['reason']}")
        if entry.get("tier") is not None:
            lines.append(f"Tier {entry['tier']}, {entry.get('lines_changed', 0)} lines changed in "
                         f"{', '.join(entry.get('files', []))}")
        if entry.get("train"):
            lines.append(f"Train score {entry['train']['score']:.3f}, success rate {entry['train']['success_rate']:.2f}")
        if entry.get("notes"):
            lines.append("Proposer's notes:\n" + entry["notes"].strip()[:1500])
        lines.append("")
    return "\n".join(lines) + "\n"
