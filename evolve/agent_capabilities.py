"""The capability scores of an offline bench run (evolve/agent_bench.py), each measured on its own cases against the
reference model's recorded turn:

format   the reply parses (format.parsed), offers an allowed decision (format.allowed) and its arguments have the
         skill's shape (format.args_ok)
plan     on plan turns the tree is one planned.Plan accepts (plan.accepted); its goal count over the reference's
choose   on turns acting inside a goal, the same skill as the reference (choose.same_skill)
judge    on goal turns, claiming goal_done exactly when the reference's claim passed the plan check: accuracy, and
         how often a done goal is claimed (judge.recall) and an unfinished one is not (judge.specificity)
recover  after a refused or failed result, not repeating that decision unchanged (recover.changed)
pick     on picks the reference made successfully, the click on the object it picked (pick.hit), on no object
         (pick.background); pick.asked counts the cases where the model also chose pick
place    on point placements the reference made successfully, the point on the same support (place.same_support)
navigate on navigations the reference made successfully, the goal and facing points' distance from the reference's
         in metres (median) and the share within 0.5 m
point    (asked alone) the pointer's click on the object: point.hit, point.background, point.unparsed
verify   (asked alone) yes/no on done and unfinished goals: verify.accuracy, verify.recall, verify.specificity"""
import json
import statistics
from pathlib import Path

import numpy as np

from homebody.session.planned import GOAL_DONE, Plan, PlanError

ACTING = ("navigate", "pick", "place")
NEAR_M = 0.5


def pixel(point, shape):
    """observations.normalized_pixel: (u, v) of point_normalized_1000 on an image of SHAPE, or None."""
    if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(
            type(value) in (int, float) and 0 <= value <= 1000 for value in point):
        return None
    height, width = shape
    return tuple(np.floor(np.asarray(point) * [width - 1, height - 1] / 1000).astype(int).tolist())


def label_at(point, labels):
    """The label under a point_normalized_1000: an object id, 0 for background, None for no valid point."""
    uv = pixel(point, labels.shape)
    return None if uv is None else int(labels[uv[1], uv[0]])


def args_ok(skill: str, arguments: dict) -> bool:
    """The arguments have the shape the skill's prompt line gives (values are the skill's own business)."""
    def point(name):
        return pixel(arguments.get(name), (1001, 1001)) is not None

    if skill == "pick":
        return point("point_normalized_1000")
    if skill == "navigate":
        return (point("goal_map_1000") and point("facing_map_1000")) or (
            "goal_xy_m" in arguments and "facing_xy_m" in arguments)
    if skill == "place":
        return arguments.get("hand") in ("left", "right") and (
            point("release_point_1000") or point("point_normalized_1000")
            or ("x_ahead_m" in arguments and "y_left_m" in arguments))
    if skill == "plan":
        return isinstance(arguments.get("tree"), dict)
    return True


def map_click(arguments: dict, prefix: str, observation) -> tuple[float, float] | None:
    """A navigate point as a map click: its *_map_1000, or its *_xy_m through map_click_from_world."""
    if pixel(arguments.get(f"{prefix}_map_1000"), (1001, 1001)) is not None:
        return tuple(arguments[f"{prefix}_map_1000"])
    world = arguments.get(f"{prefix}_xy_m")
    if observation and isinstance(world, list) and len(world) == 2:
        matrix = np.asarray(observation["map_click_from_world"])
        return tuple((matrix @ [*world, 1.0]).tolist())
    return None


def metres(a, b, observation) -> float:
    scale = abs(float(np.asarray(observation["map_click_from_world"])[0][0]))  # map clicks per metre
    return float(np.hypot(a[0] - b[0], a[1] - b[1]) / scale)


def plan_goals(tree) -> int | None:
    try:
        return len(Plan(tree).status)
    except (PlanError, TypeError, KeyError, AttributeError):
        return None


def judged(meta) -> bool | None:
    """agent_bench.verify_label: True for an accepted goal_done, False for acting inside a goal."""
    if meta["goal"] is None:
        return None
    if meta["reference"]["skill"] == GOAL_DONE:
        return True if meta["result"] == "OK" else None
    return False if meta["reference"]["skill"] in ACTING else None


def share(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "-"


class Tally:
    def __init__(self):
        self.counts: dict[str, list[int]] = {}

    def add(self, key: str, hit: bool):
        got, out_of = self.counts.get(key, [0, 0])
        self.counts[key] = [got + bool(hit), out_of + 1]

    def row(self) -> dict:
        return {key: share(*value) for key, value in self.counts.items()}


def score_turns(rows, cases: Path) -> dict:
    tally, goals, nav = Tally(), [], {"goal": [], "facing": []}
    for row in rows:
        meta = json.loads((cases / row["case"] / "case.json").read_text())
        if not meta["offered"]:
            continue  # an older record without its offered skills: left out for every model alike
        reference, ok = meta["reference"], "error" not in row
        tally.add("format.parsed", ok)
        if ok:
            tally.add("format.allowed", row["skill"] in meta["allowed"])
            tally.add("format.args_ok", args_ok(row["skill"], row["arguments"]))
        if reference["skill"] == "plan":
            count = plan_goals(row["arguments"].get("tree")) if ok and row["skill"] == "plan" else None
            tally.add("plan.accepted", count is not None)
            if count is not None and (wanted := plan_goals(reference["arguments"].get("tree"))):
                goals.append(count / wanted)
        if meta["goal"] is not None and reference["skill"] in ACTING:
            tally.add("choose.same_skill", ok and row["skill"] == reference["skill"])
        if (truth := judged(meta)) is not None:
            claimed = ok and row["skill"] == GOAL_DONE
            tally.add("judge.accuracy", claimed == truth)
            tally.add("judge.recall" if truth else "judge.specificity", claimed == truth)
        previous = meta.get("previous") or {}
        if previous.get("decision") and previous.get("code") not in (None, "OK"):
            same = ok and row["skill"] == previous["decision"]["skill"] and (
                row["arguments"] == previous["decision"]["arguments"])
            tally.add("recover.changed", not same)
        if meta["target"] is not None and reference["skill"] == "pick":
            asked = ok and row["skill"] == "pick"
            tally.add("pick.asked", asked)
            if asked:
                labels = np.load(cases / row["case"] / "labels.npz")["labels"]
                got = label_at(row["arguments"].get("point_normalized_1000"), labels)
                tally.add("pick.hit", got == meta["target"])
                tally.add("pick.background", got == 0)
        if meta["target"] is not None and reference["skill"] == "place":
            labels = np.load(cases / row["case"] / "labels.npz")["labels"]
            got = ok and row["skill"] == "place" and label_at(row["arguments"].get("point_normalized_1000"), labels)
            tally.add("place.same_support", got == meta["target"])
        if reference["skill"] == "navigate" and meta["result"] == "OK" and meta["observation"]:
            for prefix in nav:
                mine = ok and row["skill"] == "navigate" and map_click(row["arguments"], prefix, meta["observation"])
                theirs = map_click(reference["arguments"], prefix, meta["observation"])
                if theirs is not None:
                    nav[prefix].append(metres(mine, theirs, meta["observation"]) if mine else float("inf"))
    result = tally.row()
    if goals:
        result["plan.goals_vs_ref"] = round(statistics.median(goals), 2)
    for prefix, distances in nav.items():
        if distances:
            median = statistics.median(distances)
            result[f"navigate.{prefix}_median_m"] = round(median, 2) if median != float("inf") else "inf"
            result[f"navigate.{prefix}_within"] = share(sum(d <= NEAR_M for d in distances), len(distances))
    return result


def score_point(rows, cases: Path) -> dict:
    tally = Tally()
    for row in rows:
        meta = json.loads((cases / row["case"] / "case.json").read_text())
        got = None
        if "point" in row:
            got = label_at(row["point"], np.load(cases / row["case"] / "labels.npz")["labels"])
        tally.add("point.hit", got == meta["target"])
        tally.add("point.background", got == 0)
        tally.add("point.unparsed", "point" not in row)
    return tally.row()


def score_verify(rows, _cases: Path) -> dict:
    tally = Tally()
    for row in rows:
        if "achieved" not in row:
            tally.add("verify.answered", False)
            continue
        tally.add("verify.answered", True)
        tally.add("verify.accuracy", row["achieved"] == row["truth"])
        tally.add("verify.recall" if row["truth"] else "verify.specificity", row["achieved"] == row["truth"])
    return tally.row()


def score_run(out: Path) -> dict:
    run = json.loads((out / "run.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*/bench.json"))]
    scorer = {"turns": score_turns, "point": score_point, "verify": score_verify}[run["kind"]]
    seconds = [r["seconds"] for r in rows]
    return {"model": run["model"], "kind": run["kind"], "cases": len(rows),
            "seconds": round(statistics.mean(seconds), 1) if seconds else "-",
            **scorer(rows, Path(run["cases"]))}
