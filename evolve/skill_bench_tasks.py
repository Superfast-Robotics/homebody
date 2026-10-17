"""The new-skill benchmark's driver, loaded by skill_bench.py as DRIVERS["newskill"].

An episode is a task no built-in skill does (tools/scenegen_tasks.py). The robot stands at the stance (and for
a hand-over first picks the object with the built-in pick: setup, not scored), then the coding assistant gets
the request for a new skill (its name, goal, success test) and must create it and run it on the robot: one
call, nothing tried before. Scored from physics by the task's kind: 2 done, 1 part-way, 0 not.
"""
import math
import os

import numpy as np

bench = None  # skill_bench, set by its loader


def run_newskill(cm, session, episode, step):
    task = episode["step"]["task"]
    port, execute = session.port, session.execute
    if not execute("navigate", cm.navigate_arguments(episode["stance"], step.facing, step.hand), port.snapshot(),
                   evidence_name="stance"):
        return {"setup_failed": "stance navigation refused"}
    if task.get("setup") == "pick":
        frame = port.snapshot()
        click = cm.target_click(frame, step.label)
        if click is None or not execute("pick", {"point_normalized_1000": cm.normalized(frame, *click), "hand": step.hand},
                                        frame) or step.hand not in session.context.carried:
            return {"setup_failed": "the built-in pick did not hold the object"}
    req = task["request"]
    ask = {"skill": req["name"], "goal": req["goal"], "success": req["success"], "constraints": req["constraints"],
           "tried": "nothing: no skill does this yet, so it has to be created"}
    if os.environ.get("HOMEBODY_BENCH_CODER") != "1":  # screening: setup and scene only, no skill created
        coder = {"code": "NOT_CALLED", "message": "coder disabled", "seconds": 0.0, "credits": 0.0, "executes": []}
    else:
        coder = bench.consult(session, ask, [{"operator_task": episode["step"].get("task_text") or req["goal"]}])
    for _ in range(int(bench.SETTLE_SECONDS / 0.02)):
        port.advance(port.epoch, 0.02)
    return {"call": {"skill": req["name"], "code": "NEW", "message": "a new skill: the assistant creates it"},
            "coder": coder, "kind": task["kind"], "timing": {"call_wall": 0.0, "coder_wall": coder["seconds"],
                                                             "skill_wall": coder["seconds"]}}


def score_newskill(rows, episode, record):
    task, name = episode["step"]["task"], episode["step"]["fixture_id"]
    score = SCORERS[task["kind"]](rows, task, name)
    return {**score, "kind": task["kind"]}


# --- per-kind scores from the recorded physics ---

def state(rows, name):
    return rows[-1]["objects"][name]


def upright(obj, limit_deg=25.0):
    w, x, y, z = obj["quaternion_wxyz"]
    up = 1 - 2 * (x * x + y * y)  # z of the body's z axis
    return up >= math.cos(math.radians(limit_deg))


def yaw(obj):
    w, x, y, z = obj["quaternion_wxyz"]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def still(rows, name, seconds=0.5):
    end = rows[-1]
    earlier = next((r for r in reversed(rows) if end["time"] - r["time"] >= seconds), rows[0])
    moved = np.linalg.norm(np.subtract(end["objects"][name]["position"], earlier["objects"][name]["position"]))
    return moved / max(1e-6, end["time"] - earlier["time"]) < 0.03


def lowest(obj):
    return min(c[2] for c in obj["corners"])


def on_table(obj, task):
    return lowest(obj) > task["table_top"] - 0.05 and not obj["hand_contacts"]


def within(xy, centre, half, margin=0.0):
    return bool(np.all(np.abs(np.subtract(xy, centre)) <= np.asarray(half) + margin))


def to_zone(rows, task, name):
    """push and slide_in: 2 the centre over the zone, upright, not held and resting on the table; 1 half-way there."""
    obj = state(rows, name)
    xy = obj["position"][:2]
    done = within(xy, task["zone_xy"], task["zone_half"], 0.01) and upright(obj) and on_table(obj, task)
    start = np.linalg.norm(np.subtract(task["start_xy"], task["zone_xy"]))
    left = np.linalg.norm(np.subtract(xy, task["zone_xy"]))
    stage = 2 if done else 1 if left <= start / 2 and on_table(obj, task) else 0
    return {"stage": stage, "distance_to_zone_m": round(float(left), 3), "upright": upright(obj)}


def rotate(rows, task, name):
    """2 turned by the requested angle within 15 degrees (a box looks the same turned by 180), moved under 10 cm,
    upright and not held; 1 turned at least half-way the right way."""
    obj = state(rows, name)
    turned = math.remainder(yaw(obj) - task["start_yaw"], math.pi)  # box symmetry
    want = math.remainder(task["turn"], math.pi)
    error = abs(math.remainder(turned - want, math.pi))
    moved = np.linalg.norm(np.subtract(obj["position"][:2], task["start_xy"]))
    done = error <= math.radians(15) and moved <= 0.1 and upright(obj) and on_table(obj, task)
    half = turned * math.copysign(1, want) >= abs(want) / 2 and on_table(obj, task)
    return {"stage": 2 if done else 1 if half else 0, "turned_deg": round(math.degrees(turned), 1),
            "error_deg": round(math.degrees(error), 1), "moved_m": round(float(moved), 3)}


def stack(rows, task, name):
    """2 resting on the base object (touching it, bottom at its top), upright, released and still; 1 released near it."""
    obj, under = state(rows, name), state(rows, task["under"])
    resting = task["under"] in obj["object_contacts"] and lowest(obj) >= task["under_top"] - 0.015
    done = resting and upright(obj) and not obj["hand_contacts"] and still(rows, name)
    near = np.linalg.norm(np.subtract(obj["position"][:2], under["position"][:2])) <= 0.12 and not obj["hand_contacts"]
    return {"stage": 2 if done else 1 if near else 0, "resting": bool(resting)}


def lift_high(rows, task, name):
    """2 held with its bottom at least the requested height above the table and touching nothing else; 1 lifted."""
    obj = state(rows, name)
    rise = lowest(obj) - task["table_top"]
    held = bool(obj["hand_contacts"]) and not obj["support_contacts"]
    stage = 2 if held and rise >= task["height"] else 1 if held and rise >= 0.05 else 0
    return {"stage": stage, "height_m": round(float(rise), 3)}


def sweep(rows, task, name):
    """2 every object's centre off the zone and still on the table, not held; 1 at least one moved off."""
    states = [state(rows, o) for o in task["objects"]]
    off = [not within(s["position"][:2], task["zone_xy"], task["zone_half"]) and on_table(s, task) for s in states]
    return {"stage": 2 if all(off) else 1 if any(off) else 0, "cleared": sum(off), "objects": len(off)}


def line_up(rows, task, name):
    """2 upright on the table at the target beside the reference (within tolerance), released and still; 1 within
    15 cm of it."""
    obj = state(rows, name)
    xy = obj["position"][:2]
    done = within(xy, task["target_xy"], task["tolerance"]) and upright(obj) and on_table(obj, task) and still(rows, name)
    near = np.linalg.norm(np.subtract(xy, task["target_xy"])) <= 0.15 and on_table(obj, task)
    return {"stage": 2 if done else 1 if near else 0,
            "offset_m": round(float(np.linalg.norm(np.subtract(xy, task["target_xy"]))), 3)}


def hand_over(rows, task, name):
    """2 held by the receiving hand only, off any support; 1 still held off the table by either hand."""
    obj = state(rows, name)
    hands = set(obj["hand_contacts"])
    free = not obj["support_contacts"]
    stage = 2 if hands == {task["to"]} and free else 1 if hands and free else 0
    return {"stage": stage, "hands": sorted(hands)}


SCORERS = {"push": to_zone, "slide_in": to_zone, "rotate": rotate, "stack": stack, "lift_high": lift_high,
           "sweep": sweep, "line_up": line_up, "hand_over": hand_over}
