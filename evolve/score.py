"""Scores from collected scripted runs: success, progress stage and the failures met.

A run's stage is 0 (nothing), 1 (a pick verified by the skill), 2 (a place the skill reported
done) or 3 (the physics scorer's pass: skills completed, pick and placement verified).
A split's score is the mean stage over its tasks and repeats, divided by 3.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

RESULT = re.compile(r"^RESULT (\S+) (\{.*\})$")


@dataclass
class Run:
    task: str
    repeat: str
    stage: int
    success: bool
    status: str
    wall: float
    failures: list = field(default_factory=list)  # [(label, code, message)]


def read_run(task_dir, repeat_dir):
    result_path = next(repeat_dir.rglob("result.json"), None)
    log = (repeat_dir / "run.log").read_text(errors="replace") if (repeat_dir / "run.log").exists() else ""
    results = []
    for line in log.splitlines():
        match = RESULT.match(line)
        if match:
            try:
                results.append((match.group(1), json.loads(match.group(2))))
            except json.JSONDecodeError:
                continue
    picked = any(r["skill"] == "pick" and r["code"] == "OK" for _, r in results)
    placed = any(r["skill"] == "place" and r["code"] == "OK" for _, r in results)
    outcome = json.loads(result_path.read_text()) if result_path else {}
    success = (outcome.get("status") == "skills_completed" and outcome.get("independent_physical_pick_verified") is True
               and outcome.get("local_placement_score", {}).get("success") is True)
    stage = 3 if success else 2 if placed else 1 if picked else 0
    failures = [(label, r["code"], r["message"][:200]) for label, r in results if r["code"] != "OK"]
    if not result_path:
        tail = log.strip().splitlines()[-3:] if log.strip() else ["no output"]
        failures.append(("run", "CRASHED", " | ".join(tail)[:300]))
    return Run(task_dir.name, repeat_dir.name, stage, success, outcome.get("status", "crashed"),
               float(outcome.get("wall_seconds", 0.0)), failures)


def read_split(local):
    runs = []
    for task_dir in sorted(p for p in Path(local).iterdir() if p.is_dir()):
        for repeat_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            runs.append(read_run(task_dir, repeat_dir))
    return runs


def summary(runs):
    if not runs:
        return {"score": 0.0, "success_rate": 0.0, "runs": 0, "by_task": {}}
    by_task = {}
    for run in runs:
        by_task.setdefault(run.task, []).append(run.stage)
    return {"score": round(sum(r.stage for r in runs) / (3 * len(runs)), 4),
            "success_rate": round(sum(r.success for r in runs) / len(runs), 4), "runs": len(runs),
            "by_task": {task: stages for task, stages in by_task.items()}}


def accept(parent, child):
    """Self-Harness's rule on (train, val) summaries: neither split worse, at least one better."""
    d_train = child["train"]["score"] - parent["train"]["score"]
    d_val = child["val"]["score"] - parent["val"]["score"]
    return d_train >= 0 and d_val >= 0 and max(d_train, d_val) > 0, round(d_train, 4), round(d_val, 4)
