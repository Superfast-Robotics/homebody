"""The meta-proposer's workspace for one round: brief, train results, every session transcript
(the agent's decisions and results, and each coding-assistant call) and the history.

Only train runs are shown. Held-out results decide acceptance but never appear here beyond an
accepted/rejected verdict.
"""
import difflib
import json
import shutil
import subprocess
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
             "| task | instruction | stages per run | coder calls per run | failures |", "|---|---|---|---|---|"]
    for task, stages in best["train"]["by_task"].items():
        task_runs = [r for r in runs if r.task == task]
        failures = "<br>".join(f"{r.repeat}: {f}" for r in task_runs for f in r.failures[:4])
        lines.append(f"| {task} | {instructions.get(task, '')} | {stages} | {[r.attempts for r in task_runs]} | "
                     f"{failures or '-'} |")
    (ws / "results.md").write_text("\n".join(lines) + "\n")
    copy_runs(Path(train_local), ws / "runs")
    versions(ws, Path(train_local).parents[1], best, history)
    (ws / "history.md").write_text(render_history(history))
    notes = round_dir.parents[1] / "operator_notes.md"  # the operator's analysis of train runs, if any
    if notes.exists():
        shutil.copy(notes, ws / "NOTES.md")
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


def coder_call(call):
    """One coding-assistant session as Markdown: request, actions, robot tool calls, the change to
    the skill and the final answer."""
    def read(path):
        return path.read_text(errors="replace") if path.is_file() else ""
    lines = [f"# Coder {call.name}", "", read(call / "workspace" / "REQUEST.md"), "## What the assistant did"]
    for raw in read(call / "events.jsonl").splitlines():
        try:
            item = json.loads(raw).get("item", {})
        except json.JSONDecodeError:
            continue
        if raw.find('"item.completed"') < 0:
            continue
        if item.get("type") == "agent_message":
            lines.append(f"- says: {item.get('text', '')[:600]}")
        elif item.get("type") == "command_execution":
            lines.append(f"- runs `{item.get('command', '')[:200]}` (exit {item.get('exit_code')}): "
                         f"{str(item.get('aggregated_output', ''))[:300]!r}")
        elif item.get("type") == "file_change":
            lines.append(f"- edits {', '.join(change.get('path', '').split('/workspace/')[-1] for change in item.get('changes', []))}")
    lines.append("\n## Robot tool calls")
    for raw in read(call / "tools.jsonl").splitlines():
        entry = json.loads(raw)
        lines.append(f"- {entry['request'].get('tool')}({json.dumps(entry['request'].get('arguments'))}) -> "
                     f"{json.dumps(entry['answer'], default=str)[:900]}")
    start = next(call.glob("start_*.py"), None)
    if start is not None:
        final = call / "workspace" / "skills" / start.name.removeprefix("start_")
        diff = difflib.unified_diff(read(start).splitlines(), read(final).splitlines(), "before", "after", lineterm="")
        lines += ["\n## Change to the skill", "```diff", "\n".join(diff) or "(unchanged)", "```"]
    for library in sorted((call / "workspace" / "lib").glob("*.py")):
        lines += [f"\n## lib/{library.name}", f"```python\n{read(library)}\n```"]
    lines += ["\n## Result", f"```json\n{read(call / 'result.json')}\n```"]
    return "\n".join(lines) + "\n"


def transcript(events_path):
    """The agent's turns as Markdown: what it saw and chose and every result, in order."""
    lines = []
    for raw in events_path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("kind") == "decision":
            d = event["decision"]
            lines.append(f"## Step {event['step']}: {d['skill']} {json.dumps(d.get('arguments', {}))}\n{d.get('text', '')}")
        elif event.get("kind") == "result":
            r = event["result"]
            details = {k: v for k, v in r.get("details", {}).items() if k != "traceback"}
            lines.append(f"Result: {r['code']}: {r['message'][:400]}")
            if details.get("log"):
                lines.append("Printed:\n" + "\n".join(str(x) for x in details["log"][-30:]))
            elif details:
                lines.append("Details: " + json.dumps(details, default=str)[:600])
        elif event.get("kind") == "task_finished":
            lines.append(f"## Task finished: {json.dumps(event.get('outcome', {}), default=str)[:400]}")
    return "\n\n".join(lines) + "\n"


def copy_runs(source, dest, prefix=""):
    """Each run of a collected split under SOURCE as DEST/<task>/<prefix><repeat>/: its result, the
    agent's transcript, one write-up per coder call and the runner's errors."""
    for task_dir in sorted(p for p in Path(source).iterdir() if p.is_dir()):
        for repeat_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            out = dest / task_dir.name / f"{prefix}{repeat_dir.name}"
            out.mkdir(parents=True)
            result = repeat_dir / "result.json"
            if result.exists():
                shutil.copy(result, out / "result.json")
            events = next((path for path in repeat_dir.rglob("events.jsonl") if "coder" not in path.parts), None)
            if events:
                (out / "transcript.md").write_text(transcript(events))
            for call in sorted(repeat_dir.rglob("coder/call-*")):
                (out / f"coder-{call.name}.md").write_text(coder_call(call))
            log = repeat_dir / "run.log"
            if log.exists():
                errors = [line for line in log.read_text(errors="replace").splitlines()
                          if "Traceback" in line or "Error" in line][-20:]
                if errors:
                    (out / "runner_errors.log").write_text("\n".join(errors) + "\n")


def short_name(task):
    """train_04_shelf_return -> 04 shelf; other scenes keep their name."""
    parts = task.split("_")
    return f"{parts[1]} {parts[2]}" if len(parts) > 2 and parts[1].isdigit() else task


LABELS = {"plain": "homebody_baseline: the original HomeBody skills and prompt, no coder",
          "v000-old": "the starting harness measured under the old conditions (physics ran during coder calls)"}


def versions(ws, results_dir, best, history):
    """Every earlier version's train evidence for the diagnosis: versions.md compares them with the
    current harness scene by scene, and versions/<id>/ holds each one's change and its train runs.
    Held-out results stay out, as everywhere in the workspace."""
    entries = {e["id"]: e for e in history}
    current = best["train"]
    lines = ["# Every version tried so far, on the train scenes",
             "",
             f"The current harness is {best['id']}. Each row is an earlier version: what it changed and how its",
             "train runs went per scene (mean stage over its runs; 0 untouched, 1 lifted, 2 in the target",
             "region, 3 placed), against the current harness. Candidates r002* to r004* were built on r001c2, so",
             "their diffs add to r001c2's change. Candidates evaluated before the current conditions ran 2 runs per",
             "scene with the physics running during coder calls. Use this to learn what helped where, what hurt",
             "and what was already tried; versions/<id>/ has each one's change.diff, note.md and train runs.",
             "",
             "| version | status | change | " + " | ".join(short_name(t) for t in current["by_task"])
             + " | success | credits/run | min/run |",
             "|---|---|---|" + "---|" * (len(current["by_task"]) + 3)]
    mean = lambda stages: f"{sum(stages) / len(stages):.2f}" if stages else "-"
    lines.append(f"| **{best['id']}** (current) | {best['status']} | - | "
                 + " | ".join(mean(v) for v in current["by_task"].values())
                 + f" | {current['success_rate']:.2f} | {current.get('credits', '-')} | {current.get('minutes', '-')} |")
    for vdir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        vid = vdir.name
        if vid in (best["id"], "pilot"):
            continue
        splits = sorted(d for d in vdir.glob("train*") if d.is_dir())
        runs = [run for d in splits for run in score.read_split(d)]
        if not runs:
            continue
        out = ws / "versions" / vid
        for d in splits:
            copy_runs(d, out / "runs", prefix=d.name.removeprefix("train").strip("_") + ("_" if d.name != "train" else ""))
        entry = entries.get(vid, {})
        if entry.get("notes"):
            (out / "note.md").write_text(entry["notes"])
        if entry.get("commit"):
            diff = subprocess.run(["git", "diff", f"{entry['commit']}~1", entry["commit"], "--", "src", "tests"],
                                  cwd=HERE.parent, capture_output=True, text=True, check=False).stdout
            (out / "change.diff").write_text(diff)
        summary = score.summary(runs)
        change = LABELS.get(vid) or (entry.get("notes", "").splitlines() or ["-"])[0].removeprefix("Mechanism:").strip()
        lines.append(f"| {vid} | {entry.get('status', 'reference')} | {change[:90]} | "
                     + " | ".join(mean(summary["by_task"].get(t, [])) for t in current["by_task"])
                     + f" | {summary['success_rate']:.2f} | {summary['credits'] or '-'} | {summary['minutes']} |")
    (ws / "versions.md").write_text("\n".join(lines) + "\n")
