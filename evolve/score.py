"""Scores from collected sessions.

Each run's result.json (written by score_episode.py) holds a stage per targeted object: 0 untouched,
1 lifted by the robot, 2 ended inside the target region, 3 physics-verified placement. A run's
stage is the mean over its objects; a split's score is the mean stage over runs, divided by 3.

Acceptance compares a candidate with its parent scene by scene, with the standard error of that
difference estimated from the runs' spread, so a change is judged against the evaluation's own noise.
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
    attempts: int  # call_coder calls
    wall: float
    failures: list = field(default_factory=list)  # short strings


def read_run(task_dir, repeat_dir):
    path = repeat_dir / "result.json"
    if not path.exists():
        log = (repeat_dir / "run.log").read_text(errors="replace") if (repeat_dir / "run.log").exists() else ""
        tail = " | ".join(log.strip().splitlines()[-3:])[:300] if log.strip() else "no output"
        return Run(task_dir.name, repeat_dir.name, 0.0, False, 0, 0.0, [f"session crashed: {tail}"])
    result = json.loads(path.read_text())
    failures = [f"{name}: stage {o['stage']} ({o['reason']})" for name, o in result.get("objects", {}).items()
                if o.get("reason") is not None]
    if result.get("status") not in (None, "model_done"):
        failures.append(f"session ended {result['status']}")
    return Run(task_dir.name, repeat_dir.name, float(result.get("stage", 0.0)), bool(result.get("success")),
               int(result.get("customizations", 0)), float(result.get("wall_seconds", 0.0)), failures)


def read_split(local):
    runs = []
    for task_dir in sorted(p for p in Path(local).iterdir() if p.is_dir()):
        for repeat_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            runs.append(read_run(task_dir, repeat_dir))
    return runs


def summary(runs):
    if not runs:
        return {"score": 0.0, "success_rate": 0.0, "runs": 0, "by_task": {}}
    by_task, by_family = {}, {}
    for run in runs:
        by_task.setdefault(run.task, []).append(round(run.stage, 2))
        by_family.setdefault(family(run.task), []).append(run.success)
    return {"score": round(sum(r.stage for r in runs) / (3 * len(runs)), 4),
            "success_rate": round(sum(r.success for r in runs) / len(runs), 4), "runs": len(runs),
            "by_task": by_task,
            "success_by_family": {name: round(sum(s) / len(s), 3) for name, s in sorted(by_family.items())}}


def family(task):
    """A generated scene's family (train_04_shelf_return -> shelf_return); other scenes are their own."""
    parts = task.split("_", 2)
    return parts[2] if len(parts) == 3 and parts[1].isdigit() else task


Z_DROP = 1.0  # a split counts as worse only if it falls by more than this many standard errors
Z_RISE = 1.5  # and as better only if it rises by more than this many


def difference(parent, child):
    """(score difference, its standard error) of CHILD over PARENT for one split's summaries, scene by
    scene over the scenes both ran: each scene's runs give its mean and, pooled across both, its spread."""
    scenes = sorted(set(parent["by_task"]) & set(child["by_task"]))
    if not scenes:
        return 0.0, 0.0
    total, variance = 0.0, 0.0
    for scene in scenes:
        a, b = parent["by_task"][scene], child["by_task"][scene]
        both = a + b
        mean = sum(both) / len(both)
        spread = sum((x - mean) ** 2 for x in both) / max(1, len(both) - 1)
        total += sum(b) / len(b) - sum(a) / len(a)
        variance += spread / len(a) + spread / len(b)
    return total / (3 * len(scenes)), variance ** 0.5 / (3 * len(scenes))


def accept(parent, child):
    """The rule on (train, val) summaries: no split significantly worse (by more than Z_DROP standard
    errors) and at least one significantly better (by more than Z_RISE). Returns (ok, d_train, d_val,
    se_train, se_val)."""
    (d_train, se_train), (d_val, se_val) = (difference(parent[s], child[s]) for s in ("train", "val"))
    no_drop = d_train + Z_DROP * se_train >= 0 and d_val + Z_DROP * se_val >= 0
    rise = d_train - Z_RISE * se_train > 0 or d_val - Z_RISE * se_val > 0
    return no_drop and rise, round(d_train, 4), round(d_val, 4), round(se_train, 4), round(se_val, 4)


def verdict(d_train, d_val, se_train, se_val):
    """The archive's one-line reason for a comparison."""
    return f"train {d_train:+.3f} (se {se_train:.3f}), held-out {d_val:+.3f} (se {se_val:.3f})"


QUOTA_MARKERS = ("usage limit", "rate limit exceeded", "insufficient_quota")


def quota_hit(local):
    """Whether any collected session ended because the model's usage quota ran out."""
    for path in Path(local).rglob("*"):
        if path.name in ("outcome.json", "run.log") and path.is_file():
            text = path.read_text(errors="replace").lower()
            if any(marker in text for marker in QUOTA_MARKERS):
                return True
    return False
