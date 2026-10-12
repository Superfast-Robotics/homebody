"""Scores from collected inner-loop runs.

Each run's result.json (written by inner.py) holds a stage per targeted object: 0 untouched,
1 lifted by the robot, 2 ended inside the target region, 3 physics-verified placement. A run's
stage is the mean over its objects; a split's score is the mean stage over runs, divided by 3.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Run:
    task: str
    repeat: str
    stage: float
    success: bool
    attempts: int
    wall: float
    failures: list = field(default_factory=list)  # short strings


def read_run(task_dir, repeat_dir):
    path = repeat_dir / "result.json"
    if not path.exists():
        log = (repeat_dir / "run.log").read_text(errors="replace") if (repeat_dir / "run.log").exists() else ""
        tail = " | ".join(log.strip().splitlines()[-3:])[:300] if log.strip() else "no output"
        return Run(task_dir.name, repeat_dir.name, 0.0, False, 0, 0.0, [f"inner loop crashed: {tail}"])
    result = json.loads(path.read_text())
    failures = [f"{name}: stage {o['stage']} ({o['reason']})" for name, o in result.get("objects", {}).items()
                if o.get("reason") is not None]
    execution = result.get("execution") or {}
    if execution.get("error"):
        failures.append(f"executed program error: {execution['error'][:200]}")
    dry = [a for a in result.get("attempts", []) if a.get("kind") == "dry_run"]
    return Run(task_dir.name, repeat_dir.name, float(result.get("stage", 0.0)), bool(result.get("success")), len(dry),
               float(result.get("wall_seconds", 0.0)), failures)


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
        by_task.setdefault(run.task, []).append(round(run.stage, 2))
    return {"score": round(sum(r.stage for r in runs) / (3 * len(runs)), 4),
            "success_rate": round(sum(r.success for r in runs) / len(runs), 4), "runs": len(runs),
            "by_task": by_task}


def accept(parent, child):
    """Self-Harness's rule on (train, val) summaries: neither split worse, at least one better."""
    d_train = child["train"]["score"] - parent["train"]["score"]
    d_val = child["val"]["score"] - parent["val"]["score"]
    return d_train >= 0 and d_val >= 0 and max(d_train, d_val) > 0, round(d_train, 4), round(d_val, 4)
