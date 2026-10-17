"""Scenes for the new-skill benchmark: tasks no built-in skill does, each a request for a skill the coding
assistant must create (push, rotate, stack, lift_high, sweep, line_up, slide_in, hand_over).

    scenegen_tasks.py --out DIR --split nstrain --count 60 --seed 0 --kinds push rotate stack lift_high

Every scene is an open table in front of the robot with the task's objects (and a marker zone or tray where
the task names one); its sequence.json step carries the robot's stance, the object, and a `task` block: the
kind, the skill request the agent would make (name, goal, success) and the numeric target the benchmark
scores from physics. Packages use tools/scenegen.py's writer.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenegen import Box, Item, Scene, bottle, colour, container, packet, stance, table, write_package  # noqa: E402


def base(name, rng, kind):
    """A table in front of the robot; returns (scene, top height, table half depth)."""
    scene = Scene(name, f"task_{kind}")
    half_y = rng.uniform(0.3, 0.36)
    top = table(scene, "worktable", (0.0, 0.0), rng.uniform(0.74, 0.88), (rng.uniform(0.7, 0.9), half_y))
    scene.descriptions["worktable"] = ("Work table", "Table in front of the robot.")
    scene.robot_start = [float(rng.uniform(-0.3, 0.3)), float(-rng.uniform(1.2, 1.6)), math.pi / 2]
    return scene, top, half_y


def item(scene, rng, xy, top, used, kind=None, yaw=0.0, half=None):
    """A labelled bottle or box on the table at XY; returns it."""
    cname, rgba = colour(rng, used)
    used.add(cname)
    kind = kind or rng.choice(("bottle", "packet"))
    label = f"{cname.capitalize()} {'bottle' if kind == 'bottle' else 'box'}"
    identity = f"{cname}_{kind}"
    if kind == "bottle":
        made = bottle(identity, xy, top, rng, rgba, label)
    elif half is not None:
        made = Item(identity, "box", tuple(half), np.array([*xy, top + half[2] + 0.002]), yaw, rng.uniform(0.1, 0.25),
                    rgba, label)
    else:
        made = packet(identity, xy, top, rng, rgba, label, yaw=yaw)
    scene.items.append(made)
    scene.descriptions[identity] = (label, "On the work table.")
    return made


def zone(scene, rng, xy, half, top, used, entity="zone"):
    """A flat coloured marker on the table (static, 4 mm high); returns its label."""
    cname, rgba = colour(rng, used)
    used.add(cname)
    scene.statics.append(Box(entity, "mat", np.array([*xy, top + 0.002]), np.array([*half, 0.002]), 0.0, rgba))
    label = f"{cname} zone"
    scene.descriptions[entity] = (label.capitalize(), "Flat marker on the work table.")
    return label


def step(scene, half_y, focus_xy, obj, task, hand="right"):
    scene.steps.append({"goal": [*stance((0.0, -half_y), (0, 1), focus_xy), 0.0],
                        "facing": np.round(focus_xy, 4).tolist(), "hand": hand, "label": obj.identity,
                        "fixture_id": obj.identity, "task": task})


def request(name, goal, success, constraints=""):
    return {"name": name, "goal": goal, "success": success,
            "constraints": constraints or "Create the skill and run it on the robot; do not put other objects out of place."}


# --- kinds (the robot faces +y; its left is -x) ---

def push(name, rng):
    scene, top, half_y = base(name, rng, "push")
    used = set()
    obj = item(scene, rng, (rng.uniform(-0.15, 0.15), -half_y + rng.uniform(0.15, 0.2)), top, used)
    direction = rng.choice(("left", "right", "away"))
    offset = {"left": (-1, 0), "right": (1, 0), "away": (0, 1)}[direction]
    distance = rng.uniform(0.15, 0.25) if direction == "away" else rng.uniform(0.2, 0.32)
    centre = obj.position[:2] + np.array(offset) * distance
    half = np.array([rng.uniform(0.07, 0.1)] * 2)
    label = zone(scene, rng, centre, half, top, used)
    step(scene, half_y, (obj.position[:2] + centre) / 2, obj, {
        "kind": "push", "zone_xy": centre.round(4).tolist(), "zone_half": half.round(4).tolist(),
        "start_xy": obj.position[:2].round(4).tolist(), "table_top": round(top, 4),
        "request": request("push", f"Push the {obj.label.lower()} {direction} along the table onto the {label} without "
                                   f"picking it up.",
                           f"The {obj.label.lower()} rests on the {label} (its centre over the zone), upright and not held.")})
    scene.task = f"Push the {obj.label.lower()} onto the {label}."
    return scene


def rotate(name, rng):
    scene, top, half_y = base(name, rng, "rotate")
    used = set()
    yaw0 = rng.uniform(-math.pi / 2, math.pi / 2)
    half = (rng.uniform(0.045, 0.06), rng.uniform(0.02, 0.028), rng.uniform(0.04, 0.07))
    obj = item(scene, rng, (rng.uniform(-0.15, 0.15), -half_y + rng.uniform(0.15, 0.2)), top, used, "packet", yaw0, half)
    turn = math.radians(rng.uniform(60, 120)) * rng.choice((-1, 1))
    way = "counter-clockwise" if turn > 0 else "clockwise"
    step(scene, half_y, obj.position[:2], obj, {
        "kind": "rotate", "start_yaw": round(yaw0, 4), "turn": round(turn, 4), "start_xy": obj.position[:2].round(4).tolist(),
        "table_top": round(top, 4),
        "request": request("rotate", f"Turn the {obj.label.lower()} in place by {abs(math.degrees(turn)):.0f} degrees "
                                     f"{way} as seen from above, leaving it standing where it is.",
                           f"The {obj.label.lower()} has turned by that angle (within 15 degrees), moved less than 10 cm, "
                           f"and stands upright on the table, not held.")})
    scene.task = f"Rotate the {obj.label.lower()} {abs(math.degrees(turn)):.0f} degrees {way}."
    return scene


def stack(name, rng):
    scene, top, half_y = base(name, rng, "stack")
    used = set()
    side = rng.choice((-1, 1))
    y = -half_y + rng.uniform(0.15, 0.2)
    base_half = (rng.uniform(0.06, 0.08), rng.uniform(0.05, 0.07), rng.uniform(0.03, 0.05))
    under = item(scene, rng, (side * rng.uniform(0.05, 0.15), y), top, used, "packet", 0.0, base_half)
    under.mass = rng.uniform(0.5, 0.8)  # a steady base
    obj = item(scene, rng, (-side * rng.uniform(0.15, 0.25), y), top, used, "packet")
    step(scene, half_y, (obj.position[:2] + under.position[:2]) / 2, obj, {
        "kind": "stack", "under": under.identity, "under_top": round(top + 2 * base_half[2], 4), "table_top": round(top, 4),
        "request": request("stack", f"Put the {obj.label.lower()} on top of the {under.label.lower()} and leave it there.",
                           f"The {obj.label.lower()} rests on the {under.label.lower()}, upright, released and still.")})
    scene.task = f"Stack the {obj.label.lower()} on the {under.label.lower()}."
    return scene


def lift_high(name, rng):
    scene, top, half_y = base(name, rng, "lift_high")
    used = set()
    obj = item(scene, rng, (rng.uniform(-0.15, 0.15), -half_y + rng.uniform(0.12, 0.16)), top, used)
    height = rng.uniform(0.25, 0.35)
    step(scene, half_y, obj.position[:2], obj, {
        "kind": "lift_high", "height": round(height, 3), "table_top": round(top, 4),
        "request": request("lift_high", f"Pick up the {obj.label.lower()} and raise it so its bottom is at least "
                                        f"{height:.2f} m above the table, then hold it there.",
                           f"At the end the {obj.label.lower()} is held in a hand with its bottom at least {height:.2f} m "
                           f"above the table and touching nothing else.")})
    scene.task = f"Lift the {obj.label.lower()} {height:.2f} m above the table and hold it."
    return scene


def sweep(name, rng):
    scene, top, half_y = base(name, rng, "sweep")
    used = set()
    centre = np.array([rng.uniform(-0.1, 0.1), -half_y + rng.uniform(0.2, 0.25)])
    half = np.array([rng.uniform(0.12, 0.16), rng.uniform(0.08, 0.1)])
    label = zone(scene, rng, centre, half, top, used, "clear_zone")
    objects = []
    for _ in range(int(rng.integers(2, 4))):
        for _ in range(100):
            xy = centre + rng.uniform(-1, 1, 2) * (half - 0.04)
            if all(np.linalg.norm(xy - o.position[:2]) > 0.08 for o in objects):
                break
        objects.append(item(scene, rng, xy, top, used, "packet"))
    names = ", ".join(o.label.lower() for o in objects)
    step(scene, half_y, centre, objects[0], {
        "kind": "sweep", "zone_xy": centre.round(4).tolist(), "zone_half": half.round(4).tolist(),
        "objects": [o.identity for o in objects], "table_top": round(top, 4),
        "request": request("sweep", f"Clear the {label}: move every object on it ({names}) off the zone, keeping them on "
                                    f"the table.",
                           "No object's centre is over the zone, and all of them are still on the table, not held.")})
    scene.task = f"Clear the {label}."
    return scene


def line_up(name, rng):
    scene, top, half_y = base(name, rng, "line_up")
    used = set()
    y = -half_y + rng.uniform(0.15, 0.2)
    ref = item(scene, rng, (rng.uniform(-0.05, 0.05), y), top, used)
    side = rng.choice((-1, 1))
    obj = item(scene, rng, (side * -rng.uniform(0.22, 0.32), y + rng.uniform(-0.03, 0.05)), top, used)
    gap = rng.uniform(0.05, 0.1)
    reach = (ref.size[0] if ref.shape == "cylinder" else max(ref.size[:2])) + gap + (
        obj.size[0] if obj.shape == "cylinder" else max(obj.size[:2]))
    target = ref.position[:2] + np.array([side * reach, 0.0])
    words = "left" if side < 0 else "right"
    step(scene, half_y, (obj.position[:2] + target) / 2, obj, {
        "kind": "line_up", "target_xy": target.round(4).tolist(), "tolerance": [0.04, 0.05], "table_top": round(top, 4),
        "request": request("line_up", f"Put the {obj.label.lower()} beside the {ref.label.lower()}, on its {words} as the "
                                      f"robot sees it, about {gap * 100:.0f} cm apart and level with it.",
                           f"The {obj.label.lower()} stands upright on the table on that side, about {gap * 100:.0f} cm from "
                           f"the {ref.label.lower()} (within 4 cm), released and still.")})
    scene.task = f"Line up the {obj.label.lower()} beside the {ref.label.lower()}."
    return scene


def slide_in(name, rng):
    scene, top, half_y = base(name, rng, "slide_in")
    used = set()
    side = rng.choice((-1, 1))
    y = -half_y + rng.uniform(0.16, 0.2)
    tray_xy = np.array([side * rng.uniform(0.12, 0.22), y])
    inner = np.array([rng.uniform(0.09, 0.12), rng.uniform(0.08, 0.1)])
    tcol, trgba = colour(rng, used)
    used.add(tcol)
    container(scene, "tray", tray_xy, top, inner, rng.uniform(0.012, 0.022), trgba)
    scene.descriptions["tray"] = (f"Shallow {tcol} tray", "On the work table.")
    obj = item(scene, rng, (tray_xy[0] - side * rng.uniform(0.25, 0.32), y), top, used, "packet")
    step(scene, half_y, (obj.position[:2] + tray_xy) / 2, obj, {
        "kind": "slide_in", "zone_xy": tray_xy.round(4).tolist(), "zone_half": inner.round(4).tolist(),
        "start_xy": obj.position[:2].round(4).tolist(), "table_top": round(top, 4),
        "request": request("slide_in", f"Slide the {obj.label.lower()} along the table into the shallow {tcol} tray "
                                       f"without lifting it off the table.",
                           f"The {obj.label.lower()} rests inside the tray, upright and not held.")})
    scene.task = f"Slide the {obj.label.lower()} into the {tcol} tray."
    return scene


def hand_over(name, rng):
    scene, top, half_y = base(name, rng, "hand_over")
    used = set()
    obj = item(scene, rng, (rng.uniform(0.0, 0.12), -half_y + rng.uniform(0.12, 0.16)), top, used, "packet")
    step(scene, half_y, obj.position[:2], obj, {
        "kind": "hand_over", "from": "right", "to": "left", "table_top": round(top, 4), "setup": "pick",
        "request": request("hand_over", f"The right hand holds the {obj.label.lower()}. Pass it to the left hand without "
                                        f"putting it down.",
                           f"At the end only the left hand holds the {obj.label.lower()}, off the table and touching "
                           f"nothing else.")}, hand="right")
    scene.task = f"Pass the {obj.label.lower()} from the right hand to the left."
    return scene


KINDS = {"push": push, "rotate": rotate, "stack": stack, "lift_high": lift_high, "sweep": sweep, "line_up": line_up,
         "slide_in": slide_in, "hand_over": hand_over}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", required=True, help="Name prefix ending in train, val or test")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kinds", nargs="+", required=True, choices=list(KINDS))
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    for index in range(args.count):
        kind = args.kinds[index % len(args.kinds)]
        name = f"{args.split}_{index:02d}_{kind}"
        write_package(KINDS[kind](name, rng), args.out / name)
        print(f"{name}: {name.split('_', 2)[2]}")


if __name__ == "__main__":
    main()
