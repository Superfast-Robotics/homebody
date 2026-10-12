"""The meta-proposer's workspace for one round: brief, train results, every inner-loop transcript
(the programs astra wrote, the harness's reports back, the executed run) and the history.

Only train runs are shown. Held-out results decide acceptance but never appear here beyond an
accepted/rejected verdict.
"""
import shutil
from pathlib import Path

from . import score

HERE = Path(__file__).resolve().parent


def build(round_dir, best, train_local, history, instructions):
    ws = round_dir / "workspace"
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "runs").mkdir(parents=True)
    (ws / "BRIEF.md").write_text((HERE / "prompts" / "brief.md").read_text())
    runs = score.read_split(train_local)
    lines = [f"# Train results with harness {best['id']} (commit {best['commit'][:10]})", "",
             (f"Score {best['train']['score']:.3f} (mean stage / 3), success rate {best['train']['success_rate']:.2f}. "
              "Object stage: 0 untouched, 1 lifted, 2 ended in the target region, 3 physics-verified placement."), "",
             "| task | instruction | stages per run | dry runs | failures |", "|---|---|---|---|---|"]
    for task, stages in best["train"]["by_task"].items():
        task_runs = [r for r in runs if r.task == task]
        failures = "<br>".join(f"{r.repeat}: {f}" for r in task_runs for f in r.failures[:4])
        lines.append(f"| {task} | {instructions.get(task, '')} | {stages} | {[r.attempts for r in task_runs]} | "
                     f"{failures or '-'} |")
    (ws / "results.md").write_text("\n".join(lines) + "\n")
    for task_dir in sorted(p for p in Path(train_local).iterdir() if p.is_dir()):
        for repeat_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            out = ws / "runs" / task_dir.name / repeat_dir.name
            out.mkdir(parents=True)
            transcript = repeat_dir / "transcript"
            if transcript.exists():
                for item in sorted(transcript.iterdir()):
                    if not item.name.endswith("_prompt.md"):  # prompts are prompt.md plus the reports copied here
                        shutil.copy(item, out / item.name)
            result = repeat_dir / "result.json"
            if result.exists():
                shutil.copy(result, out / "result.json")
            log = repeat_dir / "run.log"
            if log.exists():
                errors = [line for line in log.read_text(errors="replace").splitlines()
                          if "Traceback" in line or "Error" in line][-20:]
                if errors:
                    (out / "runner_errors.log").write_text("\n".join(errors) + "\n")
    (ws / "history.md").write_text(render_history(history))
    return ws


def render_history(history):
    if not history:
        return "# History\n\nNo earlier proposals.\n"
    lines = ["# History of harness proposals (most recent last)", ""]
    for entry in history:
        lines.append(f"## {entry['id']} (parent {entry.get('parent')}, round {entry.get('round')}): {entry['status']}")
        if entry.get("reason"):
            lines.append(f"Verdict: {entry['reason'][:600]}")
        if entry.get("tier") is not None:
            lines.append(f"Tier {entry['tier']}, {entry.get('lines_changed', 0)} lines changed in "
                         f"{', '.join(entry.get('files', []))}")
        if entry.get("train"):
            lines.append(f"Train score {entry['train']['score']:.3f}, success rate {entry['train']['success_rate']:.2f}")
        if entry.get("notes"):
            lines.append("Proposer's notes:\n" + entry["notes"].strip()[:1500])
        lines.append("")
    return "\n".join(lines) + "\n"
