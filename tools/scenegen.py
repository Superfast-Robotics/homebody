"""Procedural HomeBody scene packages for developing skills away from the evaluation scenes.

    scenegen.py --out assets/generated --split train --count 8 --seed 0

Three families, each randomised in layout, sizes, heights, colours and robot start:
  bin_to_carton  objects standing in an open bin on a table, to load into an open carton
  shelf_return   cups and plates on a counter or tray, to return to a shelf unit's levels
  counter_clutter objects close to a backsplash and to each other, to move into a side basket
  place_variety  one easy pick and a short walk, then a varied place (carton, tray, mat or raised stand,
                 random size, depth, height and clutter, release anywhere over the opening): place robustness
  move_variety   one easy pick and an easy place on a second table 1.8-3.2 m away in any direction, with floor
                 obstacles and sometimes a narrow gap on the way (routable at the robot's clearance): walk robustness
Every scene folder is a full package (scene.xml, map.npz, semantics.json, physics.json,
task_scene.json, source.json) plus sequence.json: one transfer step per moved object in the
format of configs/continuous_four.json, with the evaluator target it is scored against.
"""
import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

RESOLUTION = 0.02
OBSTACLE_BAND = (0.03, 1.9)
ROOM_HALF = 3.5
CYLINDER_SEGMENTS = 24
STANCE_BACK = 0.55  # m from a table's front edge to a scripted stance
PALETTE = {"red": (0.8, 0.15, 0.12), "blue": (0.15, 0.3, 0.8), "green": (0.2, 0.6, 0.25),
           "yellow": (0.95, 0.8, 0.15), "orange": (0.95, 0.5, 0.1), "white": (0.92, 0.92, 0.9),
           "black": (0.08, 0.08, 0.09), "brown": (0.45, 0.28, 0.12), "purple": (0.45, 0.2, 0.6),
           "teal": (0.1, 0.55, 0.55), "grey": (0.5, 0.5, 0.52), "pink": (0.9, 0.5, 0.65)}


@dataclass
class Box:
    entity: str
    name: str
    center: np.ndarray
    half: np.ndarray
    yaw: float = 0.0
    rgba: tuple = (0.6, 0.6, 0.6, 1.0)

    def corners(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * self.half
        return local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + self.center


@dataclass
class Item:
    identity: str
    shape: str  # "box" or "cylinder"
    size: tuple  # box half sizes (x, y, z) or cylinder (radius, half height)
    position: np.ndarray  # body centre
    yaw: float
    mass: float
    rgba: tuple
    label: str

    @property
    def half_height(self):
        return self.size[2] if self.shape == "box" else self.size[1]

    def vertices(self):
        if self.shape == "box":
            return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * self.size
        a = np.linspace(0, 2 * math.pi, CYLINDER_SEGMENTS, endpoint=False)
        ring = np.stack([self.size[0] * np.cos(a), self.size[0] * np.sin(a)], axis=1)
        return np.vstack([np.c_[ring, np.full(len(a), -self.size[1])], np.c_[ring, np.full(len(a), self.size[1])]])

    def corners(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.vertices() @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + self.position


@dataclass
class Scene:
    name: str
    family: str
    statics: list = field(default_factory=list)
    items: list = field(default_factory=list)
    descriptions: dict = field(default_factory=dict)  # entity -> (label, location)
    robot_start: list | None = None
    task: str = ""
    steps: list = field(default_factory=list)


def fmt(values):
    return " ".join(f"{float(v):.6g}" for v in values)


def colour(rng, exclude=()):
    name = rng.choice([n for n in PALETTE if n not in exclude])
    return name, (*PALETTE[name], 1.0)


# --- furniture ---

def table(scene, entity, center_xy, top_z, half_xy, yaw=0.0, rgba=(0.55, 0.42, 0.3, 1.0)):
    """A slab top on four legs; returns the top's height."""
    thickness = 0.02
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    scene.statics.append(Box(entity, "top", np.array([*center_xy, top_z - thickness]), np.array([*half_xy, thickness]),
                             yaw, rgba))
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        offset = rot @ (np.array([sx, sy]) * (np.asarray(half_xy) - 0.04))
        scene.statics.append(Box(entity, f"leg_{i}", np.array([*(np.asarray(center_xy) + offset), (top_z - 0.04) / 2]),
                                 np.array([0.025, 0.025, (top_z - 0.04) / 2]), yaw, (0.3, 0.3, 0.3, 1.0)))
    return top_z


def container(scene, entity, center_xy, floor_z, inner_half, wall_height, rgba, yaw=0.0, thickness=0.008):
    """An open box: a floor and four walls. Returns the interior's (low, high) in its own frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    hx, hy = inner_half
    scene.statics.append(Box(entity, "floor", np.array([*center_xy, floor_z + thickness / 2]),
                             np.array([hx + thickness, hy + thickness, thickness / 2]), yaw, rgba))
    for name, off, half in (("wall_front", (-(hx + thickness / 2), 0), (thickness / 2, hy + thickness)),
                            ("wall_back", (hx + thickness / 2, 0), (thickness / 2, hy + thickness)),
                            ("wall_left", (0, hy + thickness / 2), (hx, thickness / 2)),
                            ("wall_right", (0, -(hy + thickness / 2)), (hx, thickness / 2))):
        xy = np.asarray(center_xy) + rot @ np.array(off)
        scene.statics.append(Box(entity, name, np.array([*xy, floor_z + thickness + wall_height / 2]),
                                 np.array([*half, wall_height / 2]), yaw, rgba))
    return floor_z + thickness


def shelf_unit(scene, prefix, center_xy, width, depth, levels, rgba=(0.85, 0.85, 0.82, 1.0)):
    """Open toward -y: side panels at +-x, a back panel, LEVELS board heights; each board is its
    own entity PREFIX_i."""
    top = max(levels) + 0.25
    cx, cy = center_xy
    for side in (-1, 1):
        scene.statics.append(Box(f"{prefix}_frame", f"panel_{side + 1}", np.array([cx + side * (width / 2 + 0.01), cy, top / 2]),
                                 np.array([0.01, depth / 2, top / 2]), 0.0, rgba))
    scene.statics.append(Box(f"{prefix}_frame", "back", np.array([cx, cy + depth / 2 + 0.01, top / 2]),
                             np.array([width / 2 + 0.02, 0.01, top / 2]), 0.0, rgba))
    for i, height in enumerate(levels):
        scene.statics.append(Box(f"{prefix}_{i}", "board", np.array([cx, cy, height - 0.01]),
                                 np.array([width / 2, depth / 2, 0.01]), 0.0, rgba))


# --- objects ---

def bottle(identity, xy, base_z, rng, rgba, label):
    r, h = rng.uniform(0.022, 0.034), rng.uniform(0.08, 0.15)
    return Item(identity, "cylinder", (r, h / 2), np.array([*xy, base_z + h / 2 + 0.002]), 0.0,
                rng.uniform(0.12, 0.35), rgba, label)


def packet(identity, xy, base_z, rng, rgba, label, yaw=0.0):
    half = (rng.uniform(0.025, 0.045), rng.uniform(0.02, 0.035), rng.uniform(0.04, 0.08))
    return Item(identity, "box", half, np.array([*xy, base_z + half[2] + 0.002]), yaw, rng.uniform(0.1, 0.3), rgba, label)


def footprint(item):
    return item.size[0] if item.shape == "cylinder" else math.hypot(item.size[0], item.size[1])


def pack(rng, lo, hi, makers, gap=(0.006, 0.03), tries=400):
    """Place items made by MAKERS (xy -> Item) without overlap inside the rectangle [lo, hi]."""
    placed = []
    for make in makers:
        for _ in range(tries):
            probe = make(np.zeros(2))
            r = footprint(probe)
            xy = rng.uniform(np.asarray(lo) + r, np.asarray(hi) - r)
            if all(np.linalg.norm(xy - p.position[:2]) >= r + footprint(p) + rng.uniform(*gap) for p in placed):
                placed.append(make(xy))
                break
        else:
            raise RuntimeError("could not place every item")
    return placed


# --- tasks ---

def stance(front_xy, normal_xy, target_xy):
    """A stance STANCE_BACK in front of a table edge, opposite TARGET_XY along the edge normal."""
    normal = np.asarray(normal_xy) / np.linalg.norm(normal_xy)
    along = np.asarray(target_xy) - np.asarray(front_xy)
    along -= normal * (along @ normal)
    return (np.asarray(front_xy) + along - normal * STANCE_BACK).round(4).tolist()


def transfer(scene, item, pick_stance, place_stance, facing_place, destination, target):
    scene.steps.append({
        "goal": [*pick_stance, 0.0], "facing": item.position[:2].round(4).tolist(), "hand": "right",
        "label": item.identity, "fixture_id": item.identity,
        "carry_goal": [*place_stance, 0.0], "carry_facing": np.round(facing_place, 4).tolist(),
        "destination_label": destination, "release_map_xy_m": np.round(facing_place, 4).tolist(),
        "release_delta_z_m": 0.0, "evaluator_target": target})


def interior_target(entity, center_xy, inner_half, floor_top, height):
    lo = np.array([*(np.asarray(center_xy) - inner_half), floor_top - 0.01])
    hi = np.array([*(np.asarray(center_xy) + inner_half), floor_top + height])
    return {"bounds": [lo.round(4).tolist(), hi.round(4).tolist()], "support_contacts": [entity]}


# --- families (every table's front edge faces -y, the robot starts on -y) ---

def bin_to_carton(name, rng):
    scene = Scene(name, "bin_to_carton")
    half_y = rng.uniform(0.28, 0.36)
    top = table(scene, "worktable", (0.0, 0.0), rng.uniform(0.72, 0.88), (rng.uniform(0.65, 0.85), half_y))
    scene.descriptions["worktable"] = ("Work table", "Table in front of the robot.")
    side = rng.choice((-1, 1))
    bin_colour, bin_rgba = colour(rng)
    bin_half = np.array([rng.uniform(0.1, 0.17), rng.uniform(0.09, 0.14)])
    bin_xy = np.array([side * rng.uniform(0.2, 0.4), -half_y + bin_half[1] + rng.uniform(0.03, 0.08)])
    bin_wall = rng.uniform(0.06, 0.13)
    bin_floor = container(scene, "bin", bin_xy, top, bin_half, bin_wall, bin_rgba)
    scene.descriptions["bin"] = (f"Open {bin_colour} bin", f"{'Right' if side < 0 else 'Left'} side of the work table.")
    carton_half = np.array([rng.uniform(0.12, 0.17), rng.uniform(0.1, 0.15)])
    carton_xy = np.array([-side * rng.uniform(0.25, 0.42), -half_y + carton_half[1] + rng.uniform(0.03, 0.1)])
    carton_wall = rng.uniform(0.1, 0.18)
    carton_floor = container(scene, "carton", carton_xy, top, carton_half, carton_wall, (0.72, 0.55, 0.36, 1.0))
    scene.descriptions["carton"] = ("Open cardboard carton", f"{'Left' if side < 0 else 'Right'} side of the work table.")
    count = int(rng.integers(2, 5))
    used = {bin_colour}
    makers = []
    for i in range(count):
        cname, rgba = colour(rng, used)
        used.add(cname)
        kind = rng.choice(("bottle", "packet"))
        identity = f"{cname}_{kind}"
        label = f"{cname.capitalize()} {'bottle' if kind == 'bottle' else 'box'}"
        maker = bottle if kind == "bottle" else packet
        makers.append(lambda xy, m=maker, i=identity, r=rgba, lab=label: m(i, xy, bin_floor, rng, r, lab))
        scene.descriptions[identity] = (label, f"Standing in the {bin_colour} bin.")
    scene.items = pack(rng, bin_xy - bin_half, bin_xy + bin_half, makers)
    reachable = sorted(scene.items, key=lambda i: i.position[1])  # nearest the robot's edge first
    movers = reachable[:int(rng.integers(1, min(2, count) + 1))]
    front = (0.0, -half_y)
    for item in movers:
        transfer(scene, item, stance(front, (0, 1), item.position[:2]), stance(front, (0, 1), carton_xy), carton_xy,
                 "carton", interior_target("carton", carton_xy, carton_half, carton_floor, carton_wall + 0.15))
    names = " and ".join(scene.descriptions[i.identity][0].lower() for i in movers)
    scene.task = f"Put the {names} from the {bin_colour} bin into the open carton."
    scene.robot_start = [float(rng.uniform(-0.3, 0.3)), float(-rng.uniform(1.2, 1.7)), math.pi / 2]
    return scene


def shelf_return(name, rng):
    scene = Scene(name, "shelf_return")
    top = table(scene, "counter", (0.0, 0.0), rng.uniform(0.82, 0.95), (rng.uniform(0.6, 0.85), 0.32),
                rgba=(0.35, 0.35, 0.38, 1.0))
    scene.descriptions["counter"] = ("Counter", "Long counter in front of the robot.")
    side = rng.choice((-1, 1))
    shelf_x = side * rng.uniform(1.35, 1.7)
    levels = sorted(rng.uniform(0.3, 1.15, size=int(rng.integers(2, 4))))
    levels = [levels[0]] + [max(h, levels[i] + 0.28) for i, h in enumerate(levels[1:])]
    width = rng.uniform(0.45, 0.65)
    depth = 0.34
    shelf_unit(scene, "shelf", (shelf_x, 0.0), width, depth, levels)
    for i, h in enumerate(levels):
        scene.descriptions[f"shelf_{i}"] = (f"Return shelf, level {i + 1} of {len(levels)}",
                                            f"Shelf unit {'right' if side < 0 else 'left'} of the counter.")
    has_tray = rng.random() < 0.6
    base = top
    if has_tray:
        tray = Item("tray", "box", (0.17, 0.12, 0.008), np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.18, -0.12), top + 0.01]),
                    0.0, 0.4, (0.15, 0.15, 0.17, 1.0), "Dark serving tray")
        scene.items.append(tray)
        scene.descriptions["tray"] = ("Dark serving tray", "On the counter.")
        base = top + 0.016
    cup_colour, cup_rgba = colour(rng, ("black",))
    r, h = rng.uniform(0.03, 0.042), rng.uniform(0.08, 0.12)
    cup_xy = (scene.items[0].position[:2] + rng.uniform(-0.08, 0.08, 2)) if has_tray else np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.24, -0.1)])
    cup = Item(f"{cup_colour}_cup", "cylinder", (r, h / 2), np.array([*cup_xy, base + h / 2 + 0.002]), 0.0,
               rng.uniform(0.1, 0.2), cup_rgba, f"{cup_colour.capitalize()} cup")
    scene.items.append(cup)
    scene.descriptions[cup.identity] = (cup.label, "On the tray on the counter." if has_tray else "On the counter.")
    if rng.random() < 0.6:
        px = cup_xy[0] + (0.22 if cup_xy[0] < 0 else -0.22)
        plate = Item("plate", "cylinder", (rng.uniform(0.08, 0.1), 0.008), np.array([px, rng.uniform(-0.18, -0.08), top + 0.01]),
                     0.0, 0.25, (0.95, 0.95, 0.95, 1.0), "White plate")
        scene.items.append(plate)
        scene.descriptions["plate"] = ("White plate", "On the counter.")
    level = int(rng.integers(0, len(levels)))
    shelf_entity = f"shelf_{level}"
    above = levels[level + 1] - levels[level] - 0.02 if level + 1 < len(levels) else 0.35
    target = {"bounds": [[shelf_x - width / 2, -depth / 2, levels[level] - 0.02],
                         [shelf_x + width / 2, depth / 2, levels[level] + above]], "support_contacts": [shelf_entity]}
    shelf_stance = [round(shelf_x, 4), round(-depth / 2 - STANCE_BACK, 4)]
    transfer(scene, cup, stance((0.0, -0.32), (0, 1), cup.position[:2]), shelf_stance, (shelf_x, 0.0), shelf_entity, target)
    ordinal = ("bottom", "middle", "top") if len(levels) == 3 else ("lower", "upper")
    scene.task = f"Return the {cup.label.lower()} to the {ordinal[level]} return shelf."
    scene.robot_start = [float(rng.uniform(-0.3, 0.3)), float(-rng.uniform(1.2, 1.6)), math.pi / 2]
    return scene


def counter_clutter(name, rng):
    scene = Scene(name, "counter_clutter")
    depth = rng.uniform(0.2, 0.26)
    top = table(scene, "counter", (0.0, 0.0), rng.uniform(0.8, 0.92), (rng.uniform(0.6, 0.8), depth),
                rgba=(0.8, 0.78, 0.72, 1.0))
    scene.statics.append(Box("backsplash", "wall", np.array([0.0, depth + 0.02, top + 0.3]), np.array([0.9, 0.02, 0.3]),
                             0.0, (0.9, 0.9, 0.88, 1.0)))
    scene.descriptions["counter"] = ("Counter", "Counter against the wall.")
    side = rng.choice((-1, 1))
    side_top = table(scene, "side_table", (side * rng.uniform(1.25, 1.5), -0.1), rng.uniform(0.6, 0.75), (0.28, 0.25))
    basket_colour, basket_rgba = colour(rng)
    basket_xy = np.array([side * 1.38, -0.1])
    basket_half = np.array([0.13, 0.11])
    basket_wall = rng.uniform(0.06, 0.1)
    basket_floor = container(scene, "basket", basket_xy, side_top, basket_half, basket_wall, basket_rgba)
    scene.descriptions["side_table"] = ("Side table", f"{'Right' if side < 0 else 'Left'} of the counter.")
    scene.descriptions["basket"] = (f"{basket_colour.capitalize()} basket", "On the side table.")
    used = {basket_colour}
    makers = []
    for i in range(int(rng.integers(3, 6))):
        cname, rgba = colour(rng, used)
        used.add(cname)
        kind = rng.choice(("bottle", "packet"))
        identity = f"{cname}_{kind}"
        label = f"{cname.capitalize()} {'bottle' if kind == 'bottle' else 'box'}"
        maker = bottle if kind == "bottle" else packet
        makers.append(lambda xy, m=maker, i=identity, r=rgba, lab=label: m(i, xy, top, rng, r, lab))
        scene.descriptions[identity] = (label, "On the counter near the wall.")
    scene.items = pack(rng, (-0.45, depth - 0.2), (0.45, depth - 0.005), makers, gap=(0.004, 0.02))
    near = [i for i in scene.items if i.position[1] + depth <= 0.3] or [min(scene.items, key=lambda i: i.position[1])]
    mover = near[int(rng.integers(0, len(near)))]
    transfer(scene, mover, stance((0.0, -depth), (0, 1), mover.position[:2]),
             [round(basket_xy[0], 4), round(-0.1 - 0.25 - STANCE_BACK, 4)], basket_xy, "basket",
             interior_target("basket", basket_xy, basket_half, basket_floor, basket_wall + 0.15))
    scene.task = f"Put the {mover.label.lower()} from the counter into the {basket_colour} basket."
    scene.robot_start = [float(rng.uniform(-0.3, 0.3)), float(-rng.uniform(1.2, 1.6)), math.pi / 2]
    return scene


def place_variety(name, rng):
    """One easy pick (a lone object near the front edge of an open table) and a short walk along the edge,
    then a varied place: a carton, low tray, flat mat or raised stand of random size, depth and height,
    sometimes with clutter beside it, and a release point anywhere over its opening."""
    scene = Scene(name, "place_variety")
    half_y = rng.uniform(0.3, 0.36)
    top = table(scene, "worktable", (0.0, 0.0), rng.uniform(0.74, 0.9), (rng.uniform(0.8, 1.0), half_y))
    scene.descriptions["worktable"] = ("Work table", "Table in front of the robot.")
    side = rng.choice((-1, 1))
    item_colour, item_rgba = colour(rng)
    kind = rng.choice(("bottle", "packet"))
    identity = f"{item_colour}_{kind}"
    label = f"{item_colour.capitalize()} {'bottle' if kind == 'bottle' else 'box'}"
    maker = bottle if kind == "bottle" else packet
    item = maker(identity, (side * -rng.uniform(0.0, 0.15), -half_y + rng.uniform(0.11, 0.15)), top, rng, item_rgba, label)
    scene.items.append(item)
    scene.descriptions[identity] = (label, "Alone near the front edge of the work table.")
    dest_kind = rng.choice(("carton", "tray", "mat", "stand"))
    inner = np.array([rng.uniform(0.07, 0.16), rng.uniform(0.07, 0.13)])
    dest_xy = np.array([side * rng.uniform(0.35, 0.6), -half_y + inner[1] + rng.uniform(0.04, 0.16)])
    dest_colour, dest_rgba = colour(rng, (item_colour,))
    if dest_kind in ("carton", "tray"):
        wall = rng.uniform(0.1, 0.22) if dest_kind == "carton" else rng.uniform(0.02, 0.045)
        floor = container(scene, "destination", dest_xy, top, inner, wall, dest_rgba)
        target = interior_target("destination", dest_xy, inner, floor, wall + 0.15)
        text = f"Open {dest_colour} {'carton' if dest_kind == 'carton' else 'tray'}"
    else:
        height = 0.006 if dest_kind == "mat" else rng.uniform(0.08, 0.25)
        scene.statics.append(Box("destination", "surface", np.array([*dest_xy, top + height / 2]),
                                 np.array([*inner, height / 2]), 0.0, dest_rgba))
        target = interior_target("destination", dest_xy, inner, top + height, 0.25)
        text = f"{dest_colour.capitalize()} {'mat' if dest_kind == 'mat' else 'stand'}"
    scene.descriptions["destination"] = (text, f"{'Left' if side > 0 else 'Right'} part of the work table.")
    if rng.random() < 0.5:  # clutter beside the destination, on its far side from the object
        used, makers = {item_colour, dest_colour}, []
        for _ in range(int(rng.integers(1, 4))):
            cname, rgba = colour(rng, used)
            used.add(cname)
            ckind = rng.choice(("bottle", "packet"))
            cmaker = bottle if ckind == "bottle" else packet
            makers.append(lambda xy, m=cmaker, i=f"{cname}_{ckind}", r=rgba,
                          lab=f"{cname.capitalize()} {'bottle' if ckind == 'bottle' else 'box'}": m(i, xy, top, rng, r, lab))
        outer = dest_xy[0] + side * (inner[0] + 0.06)
        lo = np.array([min(outer, outer + side * 0.22), -half_y + 0.04])
        hi = np.array([max(outer, outer + side * 0.22), half_y - 0.04])
        for clutter in pack(rng, lo, hi, makers):
            scene.items.append(clutter)
            scene.descriptions[clutter.identity] = (clutter.label, f"Beside the {text.lower()}.")
    release = dest_xy + rng.uniform(-0.6, 0.6, 2) * inner
    front = (0.0, -half_y)
    scene.steps.append({
        "goal": [*stance(front, (0, 1), item.position[:2]), 0.0], "facing": item.position[:2].round(4).tolist(),
        "hand": "right", "label": identity, "fixture_id": identity,
        "carry_goal": [*stance(front, (0, 1), dest_xy), 0.0], "carry_facing": dest_xy.round(4).tolist(),
        "destination_label": "destination", "release_map_xy_m": release.round(4).tolist(), "release_delta_z_m": 0.0,
        "evaluator_target": target})
    scene.task = f"Put the {label.lower()} {'in' if dest_kind in ('carton', 'tray') else 'on'} the {text.lower()}."
    scene.robot_start = [float(rng.uniform(-0.3, 0.3)), float(-rng.uniform(1.2, 1.6)), math.pi / 2]
    return scene


ROUTE_CLEARANCE = 0.34  # m: the robot's radius (0.28) and a margin, for the route check of move_variety


def routable(scene, start_xy, goal_xy, clearance=ROUTE_CLEARANCE):
    """Whether a path at CLEARANCE from all static geometry joins START_XY and GOAL_XY on the floor grid."""
    from collections import deque

    from scipy.ndimage import binary_dilation
    occupied, T = occupancy(scene)
    radius = int(math.ceil(clearance / RESOLUTION))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    blocked = binary_dilation(occupied, structure=xx ** 2 + yy ** 2 <= radius ** 2)
    inv = np.linalg.inv(T)

    def cell(xy):
        u, v, _ = inv @ np.array([*xy, 1.0])
        return int(round(v)), int(round(u))
    start, goal = cell(start_xy), cell(goal_xy)
    if blocked[start] or blocked[goal]:
        return False
    seen, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1] and not blocked[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return False


def move_variety(name, rng):
    """One easy pick (a lone object near the front edge of table A) and an easy place (a large open tray on
    table B), with a varied walk between them: B 1.8-3.2 m away in any direction and facing any way, and
    floor obstacles on the way, sometimes leaving only a narrow gap. Rejected unless a route exists."""
    scene = Scene(name, "move_variety")
    half_a = rng.uniform(0.3, 0.36)
    top = table(scene, "pick_table", (0.0, 0.0), rng.uniform(0.74, 0.88), (rng.uniform(0.45, 0.6), half_a))
    scene.descriptions["pick_table"] = ("Pick table", "Table in front of the robot.")
    item_colour, item_rgba = colour(rng)
    kind = rng.choice(("bottle", "packet"))
    identity, label = f"{item_colour}_{kind}", f"{item_colour.capitalize()} {'bottle' if kind == 'bottle' else 'box'}"
    item = (bottle if kind == "bottle" else packet)(identity, (rng.uniform(-0.1, 0.1), -half_a + rng.uniform(0.11, 0.15)),
                                                   top, rng, item_rgba, label)
    scene.items.append(item)
    scene.descriptions[identity] = (label, "Alone near the front edge of the pick table.")
    pick_stance = np.array(stance((0.0, -half_a), (0, 1), item.position[:2]))
    angle = rng.uniform(-math.pi, math.pi)
    distance = rng.uniform(1.8, 3.2)
    b_xy = pick_stance + distance * np.array([math.cos(angle), math.sin(angle)])
    if np.any(np.abs(b_xy) > ROOM_HALF - 0.7):
        raise RuntimeError("place table outside the room")
    yaw = angle - math.pi / 2 + rng.uniform(-2.2, 2.2)  # front edge toward the pick table, or turned up to ~125 deg
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    half_b = np.array([rng.uniform(0.35, 0.45), rng.uniform(0.28, 0.33)])
    top_b = table(scene, "place_table", b_xy, rng.uniform(0.72, 0.86), half_b, yaw=yaw, rgba=(0.5, 0.5, 0.55, 1.0))
    tray_colour, tray_rgba = colour(rng, (item_colour,))
    tray_xy = b_xy + rot @ np.array([0.0, -half_b[1] + 0.17])
    tray_inner = np.array([0.15, 0.12])
    tray_floor = container(scene, "tray", tray_xy, top_b, tray_inner, 0.04, tray_rgba, yaw=yaw)
    scene.descriptions["place_table"] = ("Place table", "Second table across the room.")
    scene.descriptions["tray"] = (f"Large {tray_colour} tray", "On the place table.")
    front_b, normal_b = b_xy + rot @ np.array([0.0, -half_b[1]]), rot @ np.array([0.0, 1.0])
    place_stance = np.array(stance(front_b, normal_b, tray_xy))
    route = place_stance - pick_stance
    across = np.array([-route[1], route[0]]) / np.linalg.norm(route)
    for i in range(int(rng.integers(1, 4))):  # obstacles near the straight route
        at = pick_stance + route * rng.uniform(0.3, 0.7) + across * rng.uniform(-0.6, 0.6)
        if min(np.linalg.norm(at), np.linalg.norm(at - b_xy)) < 0.75:  # not on or against a table
            continue
        half = np.array([rng.uniform(0.12, 0.3), rng.uniform(0.12, 0.3), rng.uniform(0.2, 0.5)])
        scene.statics.append(Box(f"obstacle_{i}", "body", np.array([*at, half[2]]), half, rng.uniform(0, math.pi),
                                 (0.4, 0.35, 0.3, 1.0)))
        scene.descriptions[f"obstacle_{i}"] = ("Floor obstacle", "On the floor between the tables.")
    mid = pick_stance + route * rng.uniform(0.4, 0.6)
    clear_of_tables = min(np.linalg.norm(mid), np.linalg.norm(mid - b_xy)) > 1.1
    if clear_of_tables and rng.random() < 0.5:  # a narrow gap across the route, away from both tables
        gap = rng.uniform(0.75, 0.95)
        for j, sign in enumerate((-1, 1)):
            centre = mid + sign * across * (gap / 2 + 0.6)
            scene.statics.append(Box(f"partition_{j}", "panel", np.array([*centre, 0.6]), np.array([0.6, 0.04, 0.6]),
                                     math.atan2(across[1], across[0]), (0.85, 0.85, 0.82, 1.0)))
            scene.descriptions[f"partition_{j}"] = ("Partition", "Panel standing on the floor.")
    if not routable(scene, pick_stance, place_stance):
        raise RuntimeError("no route at the robot's clearance")
    target = interior_target("tray", tray_xy, tray_inner, tray_floor, 0.2)
    scene.steps.append({
        "goal": [*pick_stance.round(4), 0.0], "facing": item.position[:2].round(4).tolist(), "hand": "right",
        "label": identity, "fixture_id": identity,
        "carry_goal": [*place_stance.round(4), 0.0], "carry_facing": tray_xy.round(4).tolist(),
        "destination_label": "tray", "release_map_xy_m": tray_xy.round(4).tolist(), "release_delta_z_m": 0.0,
        "evaluator_target": target})
    scene.task = f"Carry the {label.lower()} to the {tray_colour} tray on the other table."
    scene.robot_start = [float(pick_stance[0] + rng.uniform(-0.2, 0.2)), float(pick_stance[1] - rng.uniform(0.4, 0.7)),
                         math.pi / 2]
    return scene


WH_STANCE_BACK = 0.48  # m from a product's centre to its pick stance in the aisle, as the warehouse scenes' stances
WH_PLACE_BACK = 0.4  # m from the shelf's front edge to a place stance, as the scenes' carry goals (0.38-0.40)
WH_GRIP_LIMIT = 0.08  # m: the widest face the Dex1-1 jaws close on with clearance (calibrated aperture 0.095)
# Products of the two scanned warehouse scenes: (label, shape, size, mass range, colours). Sizes are box half
# sizes (x along the rack, y into it, z) or cylinder (radius, half height); every pickable kind has a face under
# WH_GRIP_LIMIT. Trays and bowls are wider than the open jaws: they stand on shelves but are never asked for.
WH_PRODUCTS = {
    "can": ("food can", "cylinder", ((0.033, 0.05), (0.038, 0.06)), (0.35, 0.46), ("yellow", "red", "green")),
    "jam_jar": ("jam jar", "cylinder", ((0.021, 0.018), (0.028, 0.03)), (0.1, 0.2), ("purple", "red", "orange")),
    "spice_jar": ("spice jar", "cylinder", ((0.031, 0.07), (0.035, 0.08)), (0.15, 0.25), ("green", "brown", "white")),
    "sweetener_box": ("sweetener box", "box", ((0.04, 0.013, 0.1), (0.046, 0.016, 0.11)), (0.15, 0.3),
                      ("white", "blue", "pink")),
    "sauce_bottle": ("sauce bottle", "cylinder", ((0.025, 0.07), (0.03, 0.09)), (0.2, 0.35), ("red", "orange", "brown")),
    "tea_box": ("tea box", "box", ((0.035, 0.03, 0.06), (0.04, 0.034, 0.07)), (0.1, 0.2), ("teal", "green", "black")),
}
WH_WIDE = {
    "food_tray": ("film-wrapped food tray", "box", ((0.115, 0.045, 0.02), (0.115, 0.045, 0.02)), (0.25, 0.35), ("white",)),
    "steel_bowl": ("wrapped steel bowl", "cylinder", ((0.056, 0.035), (0.056, 0.035)), (0.2, 0.3), ("grey",)),
}
WH_FRAME_RGBA = (0.82, 0.78, 0.62, 1.0)  # beige steel uprights, as rack bay 13
WH_BOARD_RGBA = (0.7, 0.7, 0.68, 1.0)


def wh_product(kind, identity, xy, base_z, rng, colour_name, catalogue):
    label, shape, (lo, hi), mass, _ = catalogue[kind]
    size = tuple(rng.uniform(a, b) for a, b in zip(lo, hi))
    return Item(identity, shape, size, np.array([*xy, base_z + size[-1] + 0.002]), 0.0, rng.uniform(*mass),
                (*PALETTE[colour_name], 1.0), f"{colour_name.capitalize()} {label}")


def wh_grip_width(item):
    return 2 * item.size[0] if item.shape == "cylinder" else 2 * min(item.size[0], item.size[1])


def wh_section(scene, rng, kind, catalogue, x_lo, front_y, depth, base_z, prefix, max_cols=3):
    """A block of one product kind on a shelf from X_LO: up to 4 rows deep and 3 wide, the front row near the
    shelf's front edge. Returns the items row by row, front-left first, the block's right edge and its width
    in items."""
    colour_name = rng.choice(catalogue[kind][4])
    probe = wh_product(kind, "probe", (0, 0), base_z, rng, colour_name, catalogue)
    half_x = probe.size[0]
    half_y = probe.size[0] if probe.shape == "cylinder" else probe.size[1]
    gap = rng.uniform(0.01, 0.06)
    inset = rng.uniform(0.03, 0.06)
    cols = int(rng.integers(1, max_cols + 1))
    rows = int(min(rng.integers(1, 5), max(1, (depth - inset - 0.03) // (2 * half_y + gap))))
    items = []
    for r in range(rows):
        for c in range(cols):
            xy = (x_lo + half_x + c * (2 * half_x + gap) + rng.uniform(-0.004, 0.004),
                  front_y + inset + half_y + r * (2 * half_y + gap) + rng.uniform(-0.004, 0.004))
            items.append(Item(f"{prefix}_{r * cols + c + 1}", probe.shape, probe.size, np.array([*xy, probe.position[2]]),
                              0.0, probe.mass, probe.rgba, probe.label))  # one product, so one size per block
    return items, x_lo + cols * (2 * half_x + gap) - gap, cols


def warehouse_rack(name, rng):
    """A warehouse rack row along x (front edge toward the aisle at -y) with 2-3 bays of 2-4 shelf levels,
    after the scanned warehouse scenes: one bay's product shelf (about elbow height, 0.65-0.92 m) holds 2-4
    pickable kinds in blocks left to right, sometimes beside a block of wide products; the destination is
    the same shelf of the bay to the left or right (empty, or with a free section next to the products) or,
    rarely, another level of the same bay. Products are named by their colour and kind, one per kind is
    taken (the front-left one of its block), the robot starts in the aisle, and floor boxes may stand there."""
    scene = Scene(name, "warehouse_rack")
    bays = int(rng.integers(2, 4))
    widths = rng.uniform(0.9, 1.6, size=bays)
    depth = rng.uniform(0.5, 0.62)
    front_y = -depth / 2
    product_z = rng.uniform(0.65, 0.92)
    levels = [rng.uniform(0.08, 0.16), product_z]
    if product_z - levels[0] > 0.9 and rng.random() < 0.5:
        levels.insert(1, rng.uniform(levels[0] + 0.4, product_z - 0.42))
    if rng.random() < 0.8:
        levels.append(rng.uniform(product_z + 0.42, min(product_z + 0.6, 1.5)))
    levels = levels[:4]
    product_level = levels.index(product_z)
    height = max(levels) + rng.uniform(0.4, 0.8)
    edges = np.concatenate([[0.0], np.cumsum(widths)]) - widths.sum() / 2 + rng.uniform(-0.3, 0.3)
    if edges[0] < -ROOM_HALF + 0.4 or edges[-1] > ROOM_HALF - 0.4:
        raise RuntimeError("rack outside the room")
    bay_names = ("left", "right") if bays == 2 else ("left", "middle", "right")
    for i, x in enumerate(edges):
        for j, y in enumerate((front_y + 0.025, -front_y - 0.025)):
            scene.statics.append(Box("rack_frame", f"upright_{i}_{j}", np.array([x, y, height / 2]),
                                     np.array([0.025, 0.025, height / 2]), 0.0, WH_FRAME_RGBA))
    for b in range(bays):
        centre = (edges[b] + edges[b + 1]) / 2
        scene.statics.append(Box("rack_frame", f"back_{b}", np.array([centre, -front_y + 0.005, height / 2]),
                                 np.array([widths[b] / 2, 0.005, height / 2]), 0.0, (0.45, 0.45, 0.45, 1.0)))
        for k, z in enumerate(levels):
            entity = f"bay_{bay_names[b]}_level_{k + 1}"
            scene.statics.append(Box(entity, "board", np.array([centre, 0.0, z - 0.015]),
                                     np.array([widths[b] / 2 - 0.025, depth / 2, 0.015]), 0.0, WH_BOARD_RGBA))
            scene.descriptions[entity] = (f"Shelf, {bay_names[b]} bay, level {k + 1} of {len(levels)}",
                                          f"The rack's {bay_names[b]} bay, {z:.2f} m above the floor.")
    scene.descriptions["rack_frame"] = ("Steel rack frame", f"Uprights and mesh backs of the rack's {bays} bays.")

    source = int(rng.integers(0, bays))
    neighbours = [b for b in (source - 1, source + 1) if 0 <= b < bays]
    other_levels = [k for k in range(len(levels)) if k != product_level and 0.5 <= levels[k] <= 1.05]
    if other_levels and rng.random() < 0.15:
        dest_bay, dest_level = source, int(rng.choice(other_levels))
    else:
        dest_bay, dest_level = int(rng.choice(neighbours)), product_level
    count = int(rng.integers(2, 5))
    kinds = list(rng.choice(list(WH_PRODUCTS), size=count, replace=False))
    wide = rng.choice(list(WH_WIDE)) if rng.random() < 0.4 else None
    lo, hi = edges[source] + 0.05, edges[source + 1] - 0.05
    x, targets, used = lo + rng.uniform(0.0, 0.08), [], {}
    order = kinds + ([wide] if wide is not None else [])
    rng.shuffle(order)
    for kind in order:
        catalogue = WH_PRODUCTS if kind in WH_PRODUCTS else WH_WIDE
        items, right, cols = wh_section(scene, rng, kind, catalogue, x, front_y, depth, levels[product_level], kind,
                                        max_cols=3 if len(order) <= 2 else 2)
        if right > hi:
            raise RuntimeError("products do not fit the bay")
        scene.items.extend(items)
        if kind in WH_PRODUCTS:
            targets.append(items[0])
        used[kind] = (items, cols)
        x = right + rng.uniform(0.04, 0.12)
    if any(wh_grip_width(item) > WH_GRIP_LIMIT for item in targets):
        raise RuntimeError("a named product is wider than the jaws")
    bay_text = f"the {bay_names[source]} bay's shelf at level {product_level + 1}"
    for kind, (items, cols) in used.items():
        for n, item in enumerate(items):
            row = ("front", "second", "third", "back")[n // cols]
            scene.descriptions[item.identity] = (item.label, f"On {bay_text}, in the {kind.replace('_', ' ')} block, {row} row.")

    # the destination: the whole board when that shelf is empty, else a free section beside its own stock
    dest_entity = f"bay_{bay_names[dest_bay]}_level_{dest_level + 1}"
    d_lo, d_hi = edges[dest_bay] + 0.05, edges[dest_bay + 1] - 0.05
    if dest_bay != source and rng.random() < 0.5:
        free = rng.uniform(0.3, 0.5)
        near_left = dest_bay > source  # the free section is the side nearer the products
        kind = rng.choice(list(WH_PRODUCTS) + list(WH_WIDE))
        catalogue = WH_PRODUCTS if kind in WH_PRODUCTS else WH_WIDE
        # the stock stands on the far side, 5 cm clear of the free section
        stock_lo = d_lo + free + 0.05 if near_left else d_lo
        stock_hi = d_hi if near_left else d_hi - free - 0.05
        items, right, _ = wh_section(scene, rng, kind, catalogue, stock_lo, front_y, depth, levels[dest_level],
                                     f"stock_{kind}")
        if right > stock_hi:
            raise RuntimeError("stock does not fit beside the free section")
        for item in items:
            scene.descriptions[item.identity] = (item.label, f"On the {bay_names[dest_bay]} bay's shelf, beside its free space.")
        scene.items.extend(items)
        free_lo, free_hi = (d_lo, d_lo + free) if near_left else (d_hi - free, d_hi)
        side_text = "left" if near_left else "right"
        dest_text = f"the free space at the front {side_text} of the {bay_names[dest_bay]} bay's shelf at the same height"
        scene.descriptions[dest_entity] = (f"Shelf, {bay_names[dest_bay]} bay, level {dest_level + 1} (free space at the "
                                           f"{side_text})", f"The rack's {bay_names[dest_bay]} bay, beside the products' bay.")
    else:
        free_lo, free_hi = d_lo, d_hi
        if dest_bay == source:
            dest_text = f"the empty shelf at level {dest_level + 1} of the same bay"
        else:
            dest_text = f"the empty shelf of the adjoining bay to the {'left' if dest_bay < source else 'right'}"
        scene.descriptions[dest_entity] = (f"Empty shelf, {bay_names[dest_bay]} bay, level {dest_level + 1}",
                                           f"The rack's {bay_names[dest_bay]} bay, {levels[dest_level]:.2f} m above the floor.")
    top_z = levels[dest_level]
    above = (levels[dest_level + 1] - 0.03 - top_z) if dest_level + 1 < len(levels) else 0.33
    target = {"bounds": [[round(free_lo, 4), round(front_y, 4), round(top_z - 0.02, 4)],
                         [round(free_hi, 4), round(-front_y, 4), round(top_z + min(above, 0.33), 4)]],
              "support_contacts": [dest_entity]}
    release_y = front_y + rng.uniform(0.12, 0.17)
    for n, item in enumerate(targets):
        release_x = free_lo + (free_hi - free_lo) * (n + 1) / (len(targets) + 1)
        scene.steps.append({
            "goal": [round(float(item.position[0]), 4), round(float(item.position[1]) - WH_STANCE_BACK, 4), round(math.pi / 2, 4)],
            "facing": item.position[:2].round(4).tolist(), "hand": "right", "label": item.identity,
            "fixture_id": item.identity,
            "carry_goal": [round(release_x, 4), round(front_y - WH_PLACE_BACK, 4), round(math.pi / 2, 4)],
            "carry_facing": [round(release_x, 4), round(release_y, 4)], "destination_label": dest_entity,
            "release_map_xy_m": [round(release_x, 4), round(release_y, 4)], "release_delta_z_m": 0.0,
            "evaluator_target": target})

    # the aisle: sometimes the opposite rack row behind the robot, and floor boxes away from every stance
    aisle = rng.uniform(1.9, 2.4)
    if rng.random() < 0.5 and front_y - aisle - 0.6 > -ROOM_HALF + 0.05:
        scene.statics.append(Box("opposite_rack", "body", np.array([(edges[0] + edges[-1]) / 2, front_y - aisle - 0.3, 1.0]),
                                 np.array([(edges[-1] - edges[0]) / 2, 0.3, 1.0]), 0.0, WH_FRAME_RGBA))
        scene.descriptions["opposite_rack"] = ("Opposite rack row", "The rack row across the aisle, behind the robot.")
    stances = [np.array(s["goal"][:2]) for s in scene.steps] + [np.array(s["carry_goal"][:2]) for s in scene.steps]
    start = np.array([rng.uniform(edges[0] + 0.3, edges[-1] - 0.3), front_y - rng.uniform(0.75, 1.1)])
    for i in range(int(rng.integers(0, 3))):
        at = np.array([rng.uniform(edges[0] - 0.6, edges[-1] + 0.6), front_y - rng.uniform(0.35, 0.6)])
        if min(np.linalg.norm(at - p) for p in [*stances, start]) < 0.75:
            continue
        half = np.array([0.25, 0.25, rng.uniform(0.25, 0.5)])
        scene.statics.append(Box(f"cardboard_boxes_{i}", "stack", np.array([*at, half[2]]), half, rng.uniform(-0.3, 0.3),
                                 (0.6, 0.45, 0.3, 1.0)))
        scene.descriptions[f"cardboard_boxes_{i}"] = ("Stack of cardboard boxes", "On the floor of the aisle by the rack.")
    if not all(routable(scene, start, p) for p in stances) or not all(
            routable(scene, s["goal"][:2], s["carry_goal"][:2]) for s in scene.steps):
        raise RuntimeError("a stance cannot be reached from the start, or a place stance from its pick stance")

    names = [f"the front-left {item.label.lower()}" for item in targets]
    listing = ", then ".join(names)
    scene.task = (f"Pick one product of each kind from {bay_text}, in left-to-right order: first {listing}. After each "
                  f"pick, place it on {dest_text}. Keep all other products in place. If a product cannot be picked "
                  f"after two reasonable attempts, record that failure and continue to the next kind. Finish only "
                  f"after attempting all {len(targets)} kinds. Report actual outcomes, including failed grasps.")
    scene.robot_start = [float(start[0]), float(start[1]), math.pi / 2]
    return scene


FAMILIES = {"bin_to_carton": bin_to_carton, "shelf_return": shelf_return, "counter_clutter": counter_clutter,
            "place_variety": place_variety, "move_variety": move_variety, "warehouse_rack": warehouse_rack}


# --- package writer ---

def write_package(scene, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "meshes").mkdir(exist_ok=True)
    floor_z = 0.0
    world = [(f'<geom name="floor__ground" type="plane" pos="0 0 {floor_z}" size="0 0 0.05" rgba="0 0 0 0" '
              f'group="3" friction="1 0.005 0.0001" contype="2" conaffinity="3"/>'),
             (f'<geom name="floor__visual" type="box" pos="0 0 -0.01" size="{ROOM_HALF} {ROOM_HALF} 0.01" '
              f'rgba="0.62 0.6 0.56 1" group="2" contype="0" conaffinity="0"/>')]
    for b in scene.statics:
        quat = (math.cos(b.yaw / 2), 0, 0, math.sin(b.yaw / 2))
        world.append(f'<geom name="{b.entity}__{b.name}" type="box" pos="{fmt(b.center)}" quat="{fmt(quat)}" '
                     f'size="{fmt(b.half)}" rgba="{fmt(b.rgba)}" friction="1 0.005 0.0001" contype="2" '
                     f'conaffinity="3" group="2"/>')
    asset = []
    for item in scene.items:
        hull = ConvexHull(item.vertices())
        lines = [f"v {fmt(v)}" for v in item.vertices()] + ["f " + " ".join(str(i + 1) for i in s) for s in hull.simplices]
        (out / "meshes" / f"{item.identity}_collision.obj").write_text("\n".join(lines) + "\n")
        asset.append(f'<mesh name="{item.identity}_collision" file="meshes/{item.identity}_collision.obj" inertia="convex"/>')
        quat = (math.cos(item.yaw / 2), 0, 0, math.sin(item.yaw / 2))
        size = item.size if item.shape == "box" else item.size[:2]
        world.append(
            f'<body name="{item.identity}" pos="{fmt(item.position)}" quat="{fmt(quat)}">'
            f'<freejoint name="{item.identity}_joint"/>'
            f'<geom name="{item.identity}__collision" type="mesh" mesh="{item.identity}_collision" mass="{item.mass:.4g}" '
            f'friction="1.2 0.01 0.0002" solref="0.01 1" solimp="0.95 0.99 0.001" priority="1" condim="6" '
            f'contype="2" conaffinity="3" group="3" rgba="0 0 0 0"/>'
            f'<geom name="{item.identity}__visual" type="{item.shape}" size="{fmt(size)}" rgba="{fmt(item.rgba)}" '
            f'contype="0" conaffinity="0" group="2" mass="0"/></body>')
    lights = ('<light pos="0 -1 3" dir="0 0.3 -1" diffuse="0.55 0.55 0.55"/>'
              '<light pos="-3 -3 2.5" dir="0.6 0.6 -0.5" diffuse="0.25 0.25 0.25"/>')
    (out / "scene.xml").write_text(f'<mujoco model="{scene.name}">\n<asset>\n' + "\n".join(asset) +
                                   "\n</asset>\n<worldbody>\n" + lights + "\n" + "\n".join(world) + "\n</worldbody>\n</mujoco>\n")
    write_map(scene, out, floor_z)
    write_semantics(scene, out)
    frame = {"name": "world", "units": "metres", "up_axis": "Z", "handedness": "right", "origin": "generated scene"}
    (out / "physics.json").write_text(json.dumps({"frame": frame, "collision": "convex hull of the primitive", "objects": {
        i.identity: {"initial_position": i.position.tolist(), "mass_kg": i.mass, "mass_source": "generated",
                     "source_bounds": [i.corners().min(0).tolist(), i.corners().max(0).tolist()],
                     "collision": "convex hull of the primitive"} for i in scene.items}}, indent=1))
    (out / "task_scene.json").write_text(json.dumps(
        {"name": f"{scene.family.replace('_', ' ').capitalize()} ({scene.name})", "frame": "world",
         "robot_start": scene.robot_start, "movable_entities": [i.identity for i in scene.items],
         "default_task": scene.task}, indent=1))
    (out / "source.json").write_text(json.dumps({"origin": "tools/scenegen.py", "family": scene.family}, indent=1))
    (out / "sequence.json").write_text(json.dumps({"schema_version": 1, "steps": scene.steps}, indent=1))


def occupancy(scene, floor_z=0.0):
    """The static geometry's floor occupancy grid and its pixel-to-map transform."""
    cols = rows = math.ceil(2 * ROOM_HALF / RESOLUTION)
    T = np.array([[RESOLUTION, 0, -ROOM_HALF + RESOLUTION / 2], [0, -RESOLUTION, ROOM_HALF - RESOLUTION / 2], [0, 0, 1]])
    gx, gy = np.meshgrid(T[0, 0] * np.arange(cols) + T[0, 2], T[1, 1] * np.arange(rows) + T[1, 2])
    occupied = np.zeros((rows, cols), bool)
    occupied[[0, -1], :] = occupied[:, [0, -1]] = True
    for b in scene.statics:
        corners = b.corners()
        if corners[:, 2].max() - floor_z < OBSTACLE_BAND[0] or corners[:, 2].min() - floor_z > OBSTACLE_BAND[1]:
            continue
        hull = ConvexHull(corners[:, :2])
        inside = np.all(hull.equations[:, :2] @ np.stack([gx.ravel(), gy.ravel()]) + hull.equations[:, 2:3]
                        <= RESOLUTION, axis=0)
        occupied |= inside.reshape(rows, cols)
    return occupied, T


def write_map(scene, out, floor_z):
    occupied, T = occupancy(scene, floor_z)
    raster = np.where(occupied[..., None], 60, 220).astype(np.uint8).repeat(3, axis=2)
    np.savez(out / "map.npz", occupancy=occupied, T_map_px=T, raster=raster,
             provenance=json.dumps({"floor_z": floor_z, "metres_per_pixel": RESOLUTION, "frame": "world",
                                    "source": "rasterised from generated static geometry"}))


def write_semantics(scene, out):
    groups = {}
    for b in scene.statics:
        groups.setdefault(b.entity, []).append(b.corners())
    entities = []
    for identity, corners, portable in ([(i.identity, i.corners(), True) for i in scene.items] +
                                        [(e, np.vstack(c), False) for e, c in groups.items() if e in scene.descriptions]):
        label, location = scene.descriptions[identity]
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        entities.append({"id": identity, "label": label, "visual_description": label + ".",
                         "location_description": location, "portable_candidate": portable,
                         "bounds_ws_map_m": [lo.tolist(), hi.tolist()], "reference_anchor_ws_map_m": ((lo + hi) / 2).tolist()})
    frame = {"name": "world", "units": "metres", "up_axis": "Z", "handedness": "right", "origin": "generated scene"}
    (out / "semantics.json").write_text(json.dumps(
        {"schema": "homebody.generated.semantics", "schema_version": "1.0", "scene_id": scene.name,
         "coordinate_frame": frame, "robot_role": "service robot", "entities": entities}, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", required=True, help="Name prefix, e.g. train or val")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--first", type=int, default=0, help="Index of the first scene (its name's NN)")
    parser.add_argument("--plain-names", action="store_true", help="Name scenes SPLIT_NN, without the family")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    for index in range(args.first, args.first + args.count):
        family = args.families[index % len(args.families)]
        name = f"{args.split}_{index:02d}" + ("" if args.plain_names else f"_{family}")
        for _ in range(200):
            try:
                scene = FAMILIES[family](name, rng)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(f"{name}: no valid layout in 20 draws")
        write_package(scene, args.out / name)
        print(f"{name}: {len(scene.items)} objects, {len(scene.steps)} scored steps: {scene.task}")


if __name__ == "__main__":
    main()
