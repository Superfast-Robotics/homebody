"""warehouse_variants.py SUITE [--only NAME ...] [--settle]: task variants of the two real warehouse scene packages.

    warehouse_variants.py /lambda/nfs/group-volume/dex1-demos/2026-10-20/warehouse_suite \\
        --source /lambda/nfs/group-volume/dex1-demos/2026-10-20

Each variant in VARIANTS is a full copy of warehouse_shelf or warehouse_picking (the handoff's look untouched)
with its own task: other products of the kinds the Dex1 gripper can hold, another destination that exists in
the scene, another robot start in the aisle and, for some, the targets nudged a centimetre or two on their
shelf. It writes SUITE/<split>/<name>/ with task_scene.json (the task, in the originals' style), semantics.json
(the nudged targets' bounds; a destination shelf that had none gets its entry), physics.json and sequence.json
(one scripted step per target with the evaluator target it is scored against, the format of
configs/warehouse_shelf_four.json). A destination shelf in the picking scene that had no entity is renamed
from structure__<shelf> to <shelf>__shelf, so the scorer reads its contacts by name. The sources are only read.

--settle loads each variant's scene alone in MuJoCo and records how far each product moves in 2 s
(screening: none should move more than 1 cm), with each stance's clearance on map.npz."""
import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STANCE_RADIUS = 0.25   # m of free map around a scripted stance (the picking scene's own stances have this much)
NUDGE = 0.015          # m, the largest shift of a nudged target
NUDGE_GAP = 0.005      # m left between a nudged target and anything beside it
COUNTS = {2: "both products", 3: "all three products"}
PATCH = 0.22           # m, side of the clutter-free square a picking-scene destination offers
ENDING = ("Keep all other products in place. If a product cannot be picked after two reasonable attempts, record "
          "that failure and continue to the next one. Finish only after attempting {count}. Report "
          "actual outcomes, including failed grasps.")

# What each scene's graspable products are called in a task (the steel bowls, wider than the Dex1 jaws open,
# are left out); the scripted pick stance's x offset from the product for the right hand.
NAMES = {
    "shelf": {"can_1": "the front-left sweet corn can", "can_2": "the front-right sweet corn can",
              "jar_1": "the front-left blueberry jam jar", "jar_2": "the front-right blueberry jam jar",
              "tray_1": "the front wrapped food tray"},
    "picking": {"basil_03": "the front basil jar", "jam_08": "the front-left jam jar", "jam_07": "the front-right jam jar",
                "allulose_01": "the left front allulose box", "allulose_02": "the middle front allulose box",
                "allulose_03": "the right front allulose box"},
}
PICK_OFFSET = {"can": -0.08, "jar": -0.08, "tray": -0.11}
SHELF_STANCE_BACK = 0.475   # warehouse_shelf's stances (configs/warehouse_shelf_four.json)
PICKING_STANCE_BACK = 0.48  # warehouse_picking's (tools/package_warehouse_picking.py)
PLACE_REACH = 0.56          # m from a picking-scene carry stance to its release point (its own stances)


@dataclass(frozen=True)
class Destination:
    """Where a variant's products go: the support the scorer requires, the target box's footprint and shelf top,
    and how a task names it. A picking-scene destination is found as a clutter-free square on SHELF from its
    CORNER. Every destination is on the product rack's side of the aisle: the racks across the aisle and the
    fourth bay could not be reached (screening 2026-10-20: the scripted place lost sight of the shelf across the
    aisle, and the aisle's clutter kept the stance at the fourth bay 0.76 m from it)."""
    support: str
    phrase: str
    lo: tuple = ()
    hi: tuple = ()
    top: float = 0.74
    shelf: str = ""
    corner: str = "front-left"
    label: str = ""


DESTINATIONS = {
    "shelf": {
        "left": Destination("staging_area", "the empty adjoining shelf to the left", (-1.5, -0.3), (-0.9, 0.3)),
        "right": Destination("neighbor_rack_right", "the empty shelf at the same height in the adjoining bay to "
                             "the right", (0.96, -0.3), (1.56, 0.3)),
        "far_left": Destination("neighbor_rack_left", "the empty shelf at the same height at the far left end of "
                                "the adjoining bay to the left", (-2.3, -0.3), (-1.7, 0.3)),
    },
    "picking": {
        "bay2_left": Destination("product_section1_bay2_shelf1", "the free space at the front left of the adjoining "
                                 "shelf bay to the right", shelf="product_section1_bay2_shelf1",
                                 label="Product shelf, right bay (free space)"),
        "bay2_right": Destination("product_section1_bay2_shelf1", "the free space at the front right of the "
                                  "adjoining shelf bay to the right", shelf="product_section1_bay2_shelf1",
                                  corner="front-right", label="Product shelf, right bay (free space at its front right)"),
    },
}


@dataclass(frozen=True)
class Variant:
    name: str
    scene: str
    split: str
    targets: tuple
    destination: str
    seed: int
    nudge: bool = False


# Each (targets, destination) pair is used once; val and test pairs are ones train never uses.
VARIANTS = (
    Variant("wh_shelf_train_00", "shelf", "train", ("can_2", "jar_2", "tray_1"), "right", 11),
    Variant("wh_shelf_train_01", "shelf", "train", ("jar_1", "can_1"), "left", 12, nudge=True),
    Variant("wh_shelf_train_02", "shelf", "train", ("tray_1", "jar_2"), "far_left", 13),
    Variant("wh_picking_train_00", "picking", "train", ("jam_08", "allulose_02"), "bay2_right", 21),
    Variant("wh_picking_train_01", "picking", "train", ("basil_03", "allulose_01"), "bay2_right", 22, nudge=True),
    Variant("wh_picking_train_02", "picking", "train", ("allulose_01", "jam_07"), "bay2_left", 23),
    Variant("wh_shelf_val_00", "shelf", "val", ("can_2", "jar_1"), "far_left", 31, nudge=True),
    Variant("wh_shelf_val_01", "shelf", "val", ("jar_1", "tray_1", "can_2"), "left", 32),
    Variant("wh_picking_val_00", "picking", "val", ("allulose_02", "jam_07"), "bay2_left", 41, nudge=True),
    Variant("wh_picking_val_01", "picking", "val", ("basil_03", "jam_08"), "bay2_right", 42),
    Variant("wh_shelf_test_00", "shelf", "test", ("can_1", "jar_2"), "right", 51),
    Variant("wh_shelf_test_01", "shelf", "test", ("can_2", "tray_1", "jar_1"), "far_left", 52, nudge=True),
    Variant("wh_picking_test_00", "picking", "test", ("allulose_03", "jam_08", "basil_03"), "bay2_right", 61),
)
ORIGINALS = {"shelf": ("warehouse_shelf", "configs/warehouse_shelf_four.json"),
             "picking": ("warehouse_picking", "configs/warehouse_picking_three.json")}


def static_boxes(world):
    """(name, centre, half-size) of each colliding static box, the picking scene's hidden clutter included."""
    return [(g.get("name"), np.array([float(v) for v in g.get("pos").split()]),
             np.array([float(v) for v in g.get("size").split()]))
            for g in world.findall("geom") if g.get("type") == "box" and g.get("contype") != "0"]


def shelf_geom(world, shelf):
    """SHELF's geom, by its task name or its structure name."""
    for geom in world.findall("geom"):
        if geom.get("name") in (f"{shelf}__shelf", f"structure__{shelf}"):
            return geom
    raise ValueError(f"no shelf {shelf} in the scene")


def free_square(static, geom, corner):
    """The clutter-free PATCH square on GEOM's shelf nearest the aisle (its low-y edge), from CORNER's end:
    (lo, hi, top)."""
    centre = np.array([float(v) for v in geom.get("pos").split()])
    half = np.array([float(v) for v in geom.get("size").split()])
    top = centre[2] + half[2]
    above = [(q, r) for name, q, r in static if name != geom.get("name") and q[2] + r[2] > top + 0.005
             and q[2] - r[2] < top + 0.3 and np.all(q[:2] + r[:2] > centre[:2] - half[:2])
             and np.all(q[:2] - r[:2] < centre[:2] + half[:2])]
    ys = np.arange(centre[1] - half[1] + 0.03, centre[1] + half[1] - PATCH, 0.02)
    xs = np.arange(centre[0] - half[0] + 0.03, centre[0] + half[0] - PATCH, 0.02)
    for y in ys:
        for x in xs if corner == "front-left" else xs[::-1]:
            lo, hi = np.array([x, y]), np.array([x + PATCH, y + PATCH])
            if not any(np.all(q[:2] + r[:2] > lo) and np.all(q[:2] - r[:2] < hi) for q, r in above):
                return lo, hi, top
    raise ValueError(f"no clutter-free {PATCH} m square on {geom.get('name')}")


class FreeMap:
    """map.npz's occupancy: whether a disc on the floor is free of anything at robot height."""

    def __init__(self, path):
        data = np.load(path)
        self.occupancy, transform = data["occupancy"], data["T_map_px"]
        rows, cols = np.mgrid[0:self.occupancy.shape[0], 0:self.occupancy.shape[1]]
        self.x = (cols + 0.5) * transform[0, 0] + transform[0, 2]
        self.y = (rows + 0.5) * transform[1, 1] + transform[1, 2]

    def clearance(self, x, y):
        """Distance from (x, y) to the nearest occupied cell."""
        taken = self.occupancy
        return float(np.min(np.hypot(self.x[taken] - x, self.y[taken] - y))) if taken.any() else math.inf

    def free(self, x, y, radius=STANCE_RADIUS):
        return self.clearance(x, y) >= radius


def aisle_stance(free_map, x, y):
    """Y moved back from the rack into the aisle in steps of 2 cm until the disc at (x, y) is free."""
    for step in range(40):
        candidate = y - 0.02 * step
        if free_map.free(x, candidate):
            return round(candidate, 3)
    raise ValueError(f"no free stance near ({x:.2f}, {y:.2f})")


def product_boxes(semantics):
    """{id: (lo, hi)} of each portable product's bounds."""
    return {e["id"]: (np.array(e["bounds_ws_map_m"][0]), np.array(e["bounds_ws_map_m"][1]))
            for e in semantics["entities"] if e.get("portable_candidate")}


def nudges(rng, variant, boxes, static):
    """{target: (dx, dy)}: each target shifted up to NUDGE where it keeps NUDGE_GAP from every other product and
    from the scene's clutter, else left in place."""
    shifts = {}
    for target in variant.targets:
        lo, hi = boxes[target]
        others = [b for name, b in boxes.items() if name != target] + [
            (q - r, q + r) for _, q, r in static if q[2] + r[2] > lo[2] + 0.01 and q[2] - r[2] < hi[2]]
        for _ in range(20):
            shift = rng.uniform(-NUDGE, NUDGE, 2)
            moved = (lo[:2] + shift - NUDGE_GAP, hi[:2] + shift + NUDGE_GAP)
            if not any(np.all(moved[1] > olo[:2]) and np.all(moved[0] < ohi[:2]) for olo, ohi in others):
                shifts[target] = tuple(round(float(v), 4) for v in shift)
                break
    return shifts


def task_text(variant, destination):
    names = [NAMES[variant.scene][t] for t in variant.targets]
    order = "first " + names[0] + "".join(f", then {n}" for n in names[1:])
    return (f"Pick these products from the product shelf in this order: {order}. After each pick, place it on "
            f"{destination.phrase}. " + ENDING.format(count=COUNTS[len(names)]))


def releases(destination, count):
    """Release points spread across the destination's footprint, in a row near its aisle edge."""
    lo, hi = np.array(destination.lo), np.array(destination.hi)
    inset = min(0.12, (hi[0] - lo[0]) / 4)
    xs = np.linspace(lo[0] + inset, hi[0] - inset, count) if count > 1 else [(lo[0] + hi[0]) / 2]
    y = lo[1] + 0.1 if (hi[1] - lo[1]) > 0.3 else (lo[1] + hi[1]) / 2
    return [(round(float(x), 3), round(float(y), 3)) for x in xs]


def steps(variant, destination, positions, free_map):
    """The scripted transfer steps and evaluator targets, one per target."""
    out = []
    target_box = {"bounds": [[*destination.lo, round(destination.top - 0.02, 3)],
                             [*destination.hi, round(destination.top + 0.31, 3)]],
                  "support_contacts": [destination.support]}
    for target, (rx, ry) in zip(variant.targets, releases(destination, len(variant.targets))):
        x, y = positions[target][:2]
        if variant.scene == "shelf":
            gx = round(x + PICK_OFFSET[target.split("_")[0]], 3)
            goal, facing = [gx, round(y - SHELF_STANCE_BACK, 3), round(math.pi / 2, 4)], [gx, round(y + 0.005, 3)]
        else:
            goal = [round(x, 3), round(y - PICKING_STANCE_BACK, 3), round(math.pi / 2, 4)]
            facing = [round(x, 3), round(y, 3)]
        reach = SHELF_STANCE_BACK if variant.scene == "shelf" else PLACE_REACH
        carry_goal = [rx, aisle_stance(free_map, rx, min(goal[1], ry - reach)), round(math.pi / 2, 4)]
        out.append({"fixture_id": target, "label": target, "hand": "right", "goal": goal, "facing": facing,
                    "carry_goal": carry_goal, "carry_facing": [rx, ry], "destination_label": destination.support,
                    "release_map_xy_m": [rx, ry], "release_delta_z_m": 0.0, "evaluator_target": target_box})
    return {"schema_version": 1, "steps": out}


def place_destination(variant, root, semantics):
    """The variant's Destination with its footprint; in the picking scene its shelf found free space and was
    named for the scorer, and the semantics' entry for it written."""
    destination = DESTINATIONS[variant.scene][variant.destination]
    if variant.scene == "shelf":
        return destination
    world = root.find("worldbody")
    geom = shelf_geom(world, destination.shelf)
    lo, hi, top = free_square(static_boxes(world), geom, destination.corner)
    geom.set("name", f"{destination.shelf}__shelf")
    placed = Destination(destination.support, destination.phrase, tuple(lo.round(3).tolist()),
                         tuple(hi.round(3).tolist()), round(float(top), 3), destination.shelf, destination.corner,
                         destination.label)
    entry = {"id": destination.support, "label": destination.label, "portable_candidate": False,
             "visual_description": "Metal shelf with free space for products.",
             "location_description": destination.phrase[0].upper() + destination.phrase[1:] + ".",
             "bounds_ws_map_m": [[*placed.lo, placed.top], [*placed.hi, round(placed.top + 0.3, 3)]],
             "reference_anchor_ws_map_m": [*((lo + hi) / 2).round(3).tolist(), placed.top]}
    semantics["entities"] = [e for e in semantics["entities"] if e["id"] != destination.support] + [entry]
    return placed


def apply_nudges(root, semantics, physics, shifts):
    """Move each nudged product's body, its semantic bounds and anchor, and its physics start."""
    bodies = {b.get("name"): b for b in root.find("worldbody").findall("body")}
    for target, (dx, dy) in shifts.items():
        body = bodies[target]
        pos = np.array([float(v) for v in body.get("pos").split()]) + [dx, dy, 0.0]
        body.set("pos", " ".join(f"{v:.6g}" for v in pos))
        for entity in semantics["entities"]:
            if entity["id"] == target:
                entity["bounds_ws_map_m"] = [[round(v + d, 4) for v, d in zip(corner, (dx, dy, 0.0))]
                                             for corner in entity["bounds_ws_map_m"]]
                entity["reference_anchor_ws_map_m"] = [round(v + d, 4) for v, d in
                                                       zip(entity["reference_anchor_ws_map_m"], (dx, dy, 0.0))]
        start = physics["objects"].get(target, {})
        if "initial_position" in start:
            start["initial_position"] = [round(v + d, 4) for v, d in zip(start["initial_position"], (dx, dy, 0.0))]
        if "source_bounds" in start:
            start["source_bounds"] = [[round(v + d, 4) for v, d in zip(c, (dx, dy, 0.0))] for c in start["source_bounds"]]


def robot_start(rng, variant, positions, free_map):
    """A start in the aisle within reach of the targets, facing the product rack."""
    xs = [positions[t][0] for t in variant.targets]
    band = (-1.25, -0.95) if variant.scene == "shelf" else (-0.85, -0.6)
    for _ in range(200):
        x, y = rng.uniform(min(xs) - 0.4, max(xs) + 0.4), rng.uniform(*band)
        if free_map.clearance(x, y) >= 0.3:
            return [round(float(x), 3), round(float(y), 3), round(math.pi / 2, 4)]
    raise ValueError(f"no free start for {variant.name}")


def make(variant, source, suite):
    """Write VARIANT's scene package under SUITE; returns its folder."""
    original = source / ORIGINALS[variant.scene][0]
    out = suite / variant.split / variant.name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(original, out, symlinks=False)
    tree = ET.parse(out / "scene.xml")
    root = tree.getroot()
    semantics = json.loads((out / "semantics.json").read_text())
    physics = json.loads((out / "physics.json").read_text())
    rng = np.random.default_rng(variant.seed)
    destination = place_destination(variant, root, semantics)
    shifts = nudges(rng, variant, product_boxes(semantics), static_boxes(root.find("worldbody"))) if variant.nudge else {}
    apply_nudges(root, semantics, physics, shifts)
    positions = {b.get("name"): [float(v) for v in b.get("pos").split()] for b in root.find("worldbody").findall("body")}
    free_map = FreeMap(out / "map.npz")
    task = json.loads((out / "task_scene.json").read_text())
    task.update(name=f"{task['name']}: {variant.name}", robot_start=robot_start(rng, variant, positions, free_map),
                default_task=task_text(variant, destination))
    sequence = steps(variant, destination, positions, free_map)
    tree.write(out / "scene.xml", encoding="unicode")
    for name, data in (("semantics.json", semantics), ("physics.json", physics), ("task_scene.json", task),
                       ("sequence.json", sequence)):
        (out / name).write_text(json.dumps(data, indent=1) + "\n")
    (out / "variant.json").write_text(json.dumps({"from": str(original), "targets": list(variant.targets),
                                                  "destination": variant.destination, "nudges": shifts,
                                                  "seed": variant.seed}, indent=1) + "\n")
    return out


def originals(source, suite, repo):
    """test/wh_<scene>_original: each original package, linked file by file, with its scoring targets as
    sequence.json (a symlink to the folder itself could not hold that file without writing into the source)."""
    made = []
    for scene, (folder, targets) in ORIGINALS.items():
        out = suite / "test" / f"wh_{scene}_original"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for item in (source / folder).iterdir():
            (out / item.name).symlink_to(item)
        shutil.copy(repo / targets, out / "sequence.json")
        made.append(out)
    return made


def settle(folder, seconds=2.0):
    """{product: metres moved} after SECONDS of the scene alone, and each scripted stance's clearance."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = [model.body(i).name for i in range(model.nbody) if model.body_jntnum[i]]
    start = {name: data.body(name).xpos.copy() for name in bodies}
    while data.time < seconds:
        mujoco.mj_step(model, data)
    moved = {name: round(float(np.linalg.norm(data.body(name).xpos - start[name])), 4) for name in bodies}
    free_map = FreeMap(folder / "map.npz")
    sequence = json.loads((folder / "sequence.json").read_text())
    task = json.loads((folder / "task_scene.json").read_text())
    stances = {"start": round(free_map.clearance(*task["robot_start"][:2]), 3)}
    for step in sequence["steps"]:
        stances[f"{step['fixture_id']}:pick"] = round(free_map.clearance(*step["goal"][:2]), 3)
        stances[f"{step['fixture_id']}:place"] = round(free_map.clearance(*step["carry_goal"][:2]), 3)
    return {"max_moved_m": max(moved.values()), "moved": {k: v for k, v in moved.items() if v > 0.002},
            "stance_clearance_m": stances}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suite", type=Path, help="the suite folder (its train/, val/ and test/ are written)")
    parser.add_argument("--source", type=Path, required=True, help="the folder holding the two original packages")
    parser.add_argument("--only", nargs="+", help="make only these variants")
    parser.add_argument("--settle", action="store_true", help="also screen each variant alone in MuJoCo")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    chosen = [v for v in VARIANTS if not args.only or v.name in args.only]
    report = {}
    for variant in chosen:
        folder = make(variant, args.source, args.suite)
        report[variant.name] = {"split": variant.split, "folder": str(folder)}
        if args.settle:
            report[variant.name].update(settle(folder))
        print(variant.name, json.dumps(report[variant.name]))
    if not args.only:
        for folder in originals(args.source, args.suite, repo):
            report[folder.name] = {"split": "test", "folder": str(folder)}
            if args.settle:
                report[folder.name].update(settle(folder))
            print(folder.name, json.dumps(report[folder.name]))
    (args.suite / "variants_made.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
