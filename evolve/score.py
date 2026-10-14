"""Scores from collected sessions.

Each run's result.json (written by score_episode.py) holds a stage per targeted object: 0 untouched,
1 lifted by the robot, 2 ended inside the target region, 3 physics-verified placement. A run's
stage is the mean over its objects; a split's score is the mean stage over runs, divided by 3.

Acceptance compares a candidate with its parent scene by scene on each run's value, half its stage
(/3) and half its task success, with the standard error of that difference estimated from the runs'
spread, so a change is judged against the evaluation's own noise and a clean completion counts more
than partial progress. Accuracy comes first, then cost (credits per run), then wall time: a version
whose accuracy is unchanged within the noise band is also accepted when it costs clearly less, or,
at no higher cost, takes clearly less time.
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
    credits: float = 0.0


def read_run(task_dir, repeat_dir):
    path = repeat_dir / "result.json"
    if not path.exists():
        log = (repeat_dir / "run.log").read_text(errors="replace") if (repeat_dir / "run.log").exists() else ""
        tail = " | ".join(log.strip().splitlines()[-3:])[:300] if log.strip() else "no output"
        return Run(task_dir.name, repeat_dir.name, 0.0, False, 0, 0.0, [f"session crashed: {tail}"], 0.0)
    result = json.loads(path.read_text())
    failures = [f"{name}: stage {o['stage']} ({o['reason']})" for name, o in result.get("objects", {}).items()
                if o.get("reason") is not None]
    if result.get("status") not in (None, "model_done"):
        failures.append(f"session ended {result['status']}")
    return Run(task_dir.name, repeat_dir.name, float(result.get("stage", 0.0)), bool(result.get("success")),
               int(result.get("customizations", 0)), float(result.get("wall_seconds", 0.0)), failures,
               float((result.get("cost") or {}).get("credits", 0.0)))


def read_split(local):
    runs = []
    for task_dir in sorted(p for p in Path(local).iterdir() if p.is_dir()):
        for repeat_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            runs.append(read_run(task_dir, repeat_dir))
    return runs


def summary(runs):
    if not runs:
        return {"score": 0.0, "success_rate": 0.0, "runs": 0, "by_task": {}}
    by_task, by_family, success_by_task, credits_by_task, minutes_by_task = {}, {}, {}, {}, {}
    for run in runs:
        by_task.setdefault(run.task, []).append(round(run.stage, 2))
        success_by_task.setdefault(run.task, []).append(int(run.success))
        credits_by_task.setdefault(run.task, []).append(round(run.credits, 2))
        minutes_by_task.setdefault(run.task, []).append(round(run.wall / 60, 1))
        by_family.setdefault(family(run.task), []).append(run.success)
    return {"score": round(sum(r.stage for r in runs) / (3 * len(runs)), 4),
            "success_rate": round(sum(r.success for r in runs) / len(runs), 4), "runs": len(runs),
            "credits": round(sum(r.credits for r in runs) / len(runs), 2),
            "minutes": round(sum(r.wall for r in runs) / 60 / len(runs), 1),
            "by_task": by_task, "success_by_task": success_by_task,
            "credits_by_task": credits_by_task, "minutes_by_task": minutes_by_task,
            "success_by_family": {name: round(sum(s) / len(s), 3) for name, s in sorted(by_family.items())}}


def family(task):
    """A generated scene's family (train_04_shelf_return -> shelf_return); other scenes are their own."""
    parts = task.split("_", 2)
    return parts[2] if len(parts) == 3 and parts[1].isdigit() else task


SUCCESS_WEIGHT = 0.5  # a run's value: (1 - w) * stage / 3 + w * task success
Z_DROP = 2.0  # a split counts as worse only if it falls by more than this many standard errors (drops make
#               single runs swing, so the band is wide)
Z_RISE = 1.5  # and as better only if it rises by more than this many


def values(split, scene):
    """The runs' values on SCENE: stage and success weighed by SUCCESS_WEIGHT. A summary without
    per-run success (older archive entries) counts a stage-3 run as a success."""
    stages = split["by_task"][scene]
    successes = split.get("success_by_task", {}).get(scene) or [int(stage >= 3) for stage in stages]
    return [(1 - SUCCESS_WEIGHT) * stage / 3 + SUCCESS_WEIGHT * success for stage, success in zip(stages, successes)]


def difference(parent, child, metric=values):
    """(difference, its standard error) of CHILD over PARENT in METRIC (scene -> per-run list) for one
    split's summaries, scene by scene over the scenes both ran: each scene's runs give its mean and,
    pooled across both, its spread."""
    scenes = sorted(set(parent["by_task"]) & set(child["by_task"]))
    if not scenes:
        return 0.0, 0.0
    total, variance = 0.0, 0.0
    for scene in scenes:
        a, b = metric(parent, scene), metric(child, scene)
        both = a + b
        mean = sum(both) / len(both)
        spread = sum((x - mean) ** 2 for x in both) / max(1, len(both) - 1)
        total += sum(b) / len(b) - sum(a) / len(a)
        variance += spread / len(a) + spread / len(b)
    return total / len(scenes), variance ** 0.5 / len(scenes)


CUT = 0.20  # cost or time must fall by at least this fraction (and significantly) to count on its own
COST_SLACK = 0.10  # a time win may not cost more than this fraction extra


def per_run(key):
    """A metric reading the per-run KEY lists of a summary (credits_by_task, minutes_by_task)."""
    return lambda split, scene: split.get(key, {}).get(scene, [])


def relative(parent, child, key):
    """(relative change, its standard error) of CHILD over PARENT in a per-run KEY, both splits pooled
    scene by scene; None when either side lacks it (runs scored before costs were recorded)."""
    merged = [{"by_task": {f"{s}/{t}": v for s in ("train", "val") for t, v in side[s]["by_task"].items()},
               key: {f"{s}/{t}": v for s in ("train", "val") for t, v in side[s].get(key, {}).items()}}
              for side in (parent, child)]
    scenes = [t for t in merged[0]["by_task"] if merged[0][key].get(t) and merged[1][key].get(t)]
    if not scenes or not any(merged[0][key].get(t) for t in scenes):
        return None
    base = sum(sum(merged[0][key][t]) / len(merged[0][key][t]) for t in scenes) / len(scenes)
    if base <= 0:
        return None
    for side in merged:
        side["by_task"] = {t: side["by_task"][t] for t in scenes}
    diff, se = difference(merged[0], merged[1], per_run(key))
    return diff / base, se / base


def judge(parent, child):
    """The rule on (train, val) summaries, in the order accuracy > cost > time. Accuracy: a run's value
    (stage and success); no split may fall by more than Z_DROP standard errors. Accepted when one split
    rises by more than Z_RISE; otherwise, with accuracy unchanged, when credits per run fall by CUT or
    more and by more than Z_RISE standard errors; otherwise when minutes per run fall that way at no more
    than COST_SLACK extra cost."""
    (d_train, se_train), (d_val, se_val) = (difference(parent[s], child[s]) for s in ("train", "val"))
    cost, time = relative(parent, child, "credits_by_task"), relative(parent, child, "minutes_by_task")
    out = {"d_train": round(d_train, 4), "d_val": round(d_val, 4), "se_train": round(se_train, 4),
           "se_val": round(se_val, 4), "cost": cost and round(cost[0], 3), "time": time and round(time[0], 3)}
    if d_train + Z_DROP * se_train < 0 or d_val + Z_DROP * se_val < 0:
        return {**out, "ok": False, "basis": "accuracy dropped"}
    if d_train - Z_RISE * se_train > 0 or d_val - Z_RISE * se_val > 0:
        return {**out, "ok": True, "basis": "accuracy"}
    if cost and cost[0] <= -CUT and cost[0] + Z_RISE * cost[1] < 0:
        return {**out, "ok": True, "basis": "cost (accuracy unchanged)"}
    if time and time[0] <= -CUT and time[0] + Z_RISE * time[1] < 0 and (cost is None or cost[0] <= COST_SLACK):
        return {**out, "ok": True, "basis": "time (accuracy and cost unchanged)"}
    return {**out, "ok": False, "basis": "no clear gain"}


def accept(parent, child):
    """judge() as (ok, d_train, d_val, se_train, se_val)."""
    j = judge(parent, child)
    return j["ok"], j["d_train"], j["d_val"], j["se_train"], j["se_val"]


def describe(parent, child):
    """The archive's one-line reason for a comparison."""
    j = judge(parent, child)
    pct = lambda x: "n/a" if x is None else f"{x:+.0%}"
    return (f"{j['basis']}: value train {j['d_train']:+.3f} (se {j['se_train']:.3f}), held-out {j['d_val']:+.3f} "
            f"(se {j['se_val']:.3f}); credits/run {pct(j['cost'])}, minutes/run {pct(j['time'])}")


QUOTA_MARKERS = ("usage limit", "rate limit exceeded", "insufficient_quota")


def quota_hit(local):
    """Whether any collected session ended because the model's usage quota ran out."""
    for path in Path(local).rglob("*"):
        if path.name in ("outcome.json", "run.log") and path.is_file():
            text = path.read_text(errors="replace").lower()
            if any(marker in text for marker in QUOTA_MARKERS):
                return True
    return False
