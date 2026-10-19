"""package_warehouse_picking.py PACKAGE OUT [--targets CONFIG]: the simulation team's warehouse picking handoff
(Warehouse_Picking_G1_Dex1_1_*, unpacked) as a HomeBody scene package.

From the handoff's environment-only visual scene (scenes/warehouse_visual.xml: the floor, the scanned textured
surface, the estimated rack slabs, the clutter as hidden collision boxes, 15 free products) it builds scene.xml with its mesh and texture closure (HomeBody adds its own G1), map.npz
(the boxes overlapping robot height, projected onto a 2 cm grid), semantics.json (each product and the shelves
the task names), physics.json and task_scene.json (the robot in the aisle facing the products, a task over one
product of each kind), then packages them with homebody.scene_package. --targets writes the scripted steps the
episode scorer reads: each product's pick stance in the aisle, a carry to the next bay, a release over the
clutter-free part of that shelf, and the evaluator's target box with its support.

Nothing here runs the handoff's own scripts; its files are read as data."""
import argparse
import json
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from homebody.backends.scene import DEPTH_ONLY_GROUP
from homebody.scene_package import package

RESOLUTION = 0.02          # m per map cell
ROBOT_BAND = (0.05, 1.80)  # heights that block the walking robot
STANCE_BACK = 0.48         # m from a product's front row to the stance, as warehouse_shelf's stances
SOURCE_SHELF = "product_section1_bay1_shelf1"
DESTINATION_SHELF = "product_section1_bay2_shelf1"
KINDS = {
    "basil": ("Basil jar", "Green jar of dried basil (50 g) with a dark cap and a printed label."),
    "jam": ("Jam jar", "Small round jam jar with a lid and a printed label."),
    "allulose": ("Allulose box", "Tall thin box of allulose sweetener (350 g) standing upright."),
}
SEQUENCE = ("basil", "jam", "allulose")


def boxes(world):
    """(name, centre, half-size) of each static box geom."""
    return [(g.get("name"), np.array([float(v) for v in g.get("pos").split()]),
             np.array([float(v) for v in g.get("size").split()]))
            for g in world.findall("geom") if g.get("type") == "box"]


def products(world):
    """{id: (centre of its collision box, half-size, mass)}."""
    found = {}
    for body in world.findall("body"):
        geom = body.find("geom")
        base = np.array([float(v) for v in body.get("pos").split()])
        offset = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
        half = np.array([float(v) for v in geom.get("size").split()])
        if len(half) == 2:  # a cylinder: radius, half height
            half = np.array([half[0], half[0], half[1]])
        found[body.get("name")] = (base + offset, half, float(geom.get("mass")))
    return found


def occupancy(static):
    """The navigation map: True where a box overlapping robot height stands. Returns (grid, T_map_px)."""
    tall = [(p, s) for _, p, s in static if p[2] + s[2] > ROBOT_BAND[0] and p[2] - s[2] < ROBOT_BAND[1]]
    lo = np.min([p[:2] - s[:2] for p, s in tall], axis=0) - 0.5
    hi = np.max([p[:2] + s[:2] for p, s in tall], axis=0) + 0.5
    width, height = (np.ceil((hi - lo) / RESOLUTION)).astype(int)
    grid = np.zeros((height, width), dtype=bool)
    for p, s in tall:
        (x0, y0), (x1, y1) = p[:2] - s[:2], p[:2] + s[:2]
        u0, u1 = int(math.floor((x0 - lo[0]) / RESOLUTION)), int(math.ceil((x1 - lo[0]) / RESOLUTION))
        v0, v1 = int(math.floor((hi[1] - y1) / RESOLUTION)), int(math.ceil((hi[1] - y0) / RESOLUTION))
        grid[max(v0, 0):v1, max(u0, 0):u1] = True
    transform = np.array([[RESOLUTION, 0, lo[0]], [0, -RESOLUTION, hi[1]], [0, 0, 1.0]])
    return grid, transform


def free_patch(static, shelf, size=0.22):
    """The clutter-free square of side SIZE on SHELF nearest its front-left: (lo, hi) in x, y."""
    _, p, s = next(b for b in static if b[0] == shelf)
    top = p[2] + s[2]
    above = [(q, r) for name, q, r in static if name != shelf and q[2] + r[2] > top + 0.005 and q[2] - r[2] < top + 0.3
             and np.all(q[:2] + r[:2] > p[:2] - s[:2]) and np.all(q[:2] - r[:2] < p[:2] + s[:2])]
    for y in np.arange(p[1] - s[1] + 0.03, p[1] + s[1] - size, 0.02):
        for x in np.arange(p[0] - s[0] + 0.03, p[0] + s[0] - size, 0.02):
            lo, hi = np.array([x, y]), np.array([x + size, y + size])
            if not any(np.all(q[:2] + r[:2] > lo) and np.all(q[:2] - r[:2] < hi) for q, r in above):
                return lo, hi, top
    raise ValueError(f"no clutter-free {size} m square on {shelf}")


def front_of_kind(items, kind):
    """The kind's product nearest the aisle (smallest y), then leftmost."""
    return min((i for i in items if i.startswith(kind)), key=lambda i: (round(items[i][0][1], 2), items[i][0][0]))


COLLIDE = {"contype": "2", "conaffinity": "3"}  # HomeBody's scene and object collision bits
SILENT = {"contype": "0", "conaffinity": "0"}


def box_mesh(path, half):
    """An OBJ box of half-extents HALF centred on the origin: a movable object's collision hull."""
    corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    lines = [f"v {x * half[0]:.6f} {y * half[1]:.6f} {z * half[2]:.6f}" for x, y, z in corners]
    lines += [f"f {a} {b} {c}\nf {a} {c} {d}" for a, b, c, d in faces]
    path.write_text("\n".join(lines) + "\n")


def homebody_names(root, world, assets):
    """HomeBody's conventions: the ground a colliding group-3 plane "floor__ground" with a group-2 "floor__visual"
    (where the robot is stood), scenery colliding in group 2 (the task's shelves "<name>__shelf", the rest
    "structure__<name>", unlabelled), and each product a group-3 mesh hull "<id>__collision" with group-2 visuals
    "<id>__<part>"."""
    floor = world.find("geom[@name='floor']")
    floor.attrib.update({"name": "floor__ground", "group": "3", **COLLIDE})
    size = floor.get("size").split()
    ET.SubElement(world, "geom", {"name": "floor__visual", "type": "box", "group": "2", **SILENT,
                                  "size": f"{size[0]} {size[1]} 0.0005", "pos": "0 0 -0.0005",
                                  "rgba": floor.get("rgba", ".48 .49 .47 1")})
    for index, geom in enumerate(world.findall("geom")):
        name = geom.get("name") or f"observed_{index}"
        if name in ("floor__ground", "floor__visual"):
            continue
        visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"  # the scanned surface, drawn only
        hidden = geom.get("group") == "3"  # clutter: collides, the scan draws it, the depth camera must meet it
        geom.attrib.update({"group": str(DEPTH_ONLY_GROUP) if hidden else "2", **(SILENT if visual else COLLIDE),
                            "name": f"{name}__shelf" if name in (SOURCE_SHELF, DESTINATION_SHELF) else f"structure__{name}"})
    asset = root.find("asset")
    hulls = set()
    for body in world.findall("body"):
        identity, kind = body.get("name"), body.get("name").split("_")[0]
        for geom in body.findall("geom"):
            part = geom.get("name").removeprefix(identity + "_")
            if part == "collision":
                half = [float(v) for v in geom.get("size").split()]
                if len(half) == 2:  # a cylinder's radius and half height, as its bounding box
                    half = [half[0], half[0], half[1]]
                if kind not in hulls:
                    box_mesh(assets / f"{kind}_collision.obj", half)
                    ET.SubElement(asset, "mesh", name=f"{kind}_collision", file=f"assets/{kind}_collision.obj")
                    hulls.add(kind)
                geom.attrib.pop("size")
                geom.attrib.update({"name": f"{identity}__collision", "type": "mesh", "mesh": f"{kind}_collision",
                                    "group": "3", **COLLIDE})
            else:
                geom.attrib.update({"name": f"{identity}__{part}", "group": "2", **SILENT})


def build(source, work):
    """The unpackaged scene in WORK: scene.xml with its assets beside it, and the JSON and map files."""
    tree = ET.parse(source / "scenes/warehouse_visual.xml")
    root = tree.getroot()
    root.remove(root.find("compiler"))  # radians are not used: no euler or axis-angle in the scene
    # The scene's <option> and <default> would reach HomeBody's robot once merged (its feet slip on the
    # default's soft contact and the walking controller drifts): only the scene's own geoms get the defaults.
    defaults = root.find("default").find("geom").attrib
    for tag in ("option", "size", "default"):
        root.remove(root.find(tag))
    # The handoff's look as delivered: its light, its sky and MuJoCo's default headlight, which its own viewer
    # used; HomeBody's robot model brings a brighter headlight, a black sky and its own light, so the package
    # declares the headlight and HomeBody lets a scene with <visual> own its look (backends/scene.py).
    if root.find("visual") is None:
        visual = ET.Element("visual")
        ET.SubElement(visual, "headlight", ambient=".1 .1 .1", diffuse=".4 .4 .4", specular=".5 .5 .5")
        root.insert(0, visual)
    for geom in root.find("worldbody").iter("geom"):
        for key, value in defaults.items():
            geom.attrib.setdefault(key, value)
    assets = work / "assets"
    assets.mkdir()
    for node in root.iter():
        name = node.get("file")
        if name is not None:
            path = (source / "scenes" / name).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError(f"scene dependency outside the handoff: {name}")
            shutil.copy(path, assets / path.name)
            node.set("file", f"assets/{path.name}")
    world = root.find("worldbody")
    static, items = boxes(world), products(world)  # by the handoff's names, before HomeBody's are given
    homebody_names(root, world, assets)
    tree.write(work / "scene.xml", encoding="unicode")
    grid, transform = occupancy(static)
    np.savez_compressed(work / "map.npz", occupancy=grid, T_map_px=transform)
    _, shelf_centre, shelf_half = next(b for b in static if b[0] == SOURCE_SHELF)
    release_lo, release_hi, top = free_patch(static, DESTINATION_SHELF)
    frame = {"name": "world", "units": "metres", "up_axis": "Z", "handedness": "right",
             "origin": "the product shelf's left-front corner on the floor; +x along the shelf row, +y into the rack"}
    entities = []
    for identity, (centre, half, _) in sorted(items.items()):
        kind = identity.split("_")[0]
        label, description = KINDS[kind]
        entities.append({"id": identity, "label": f"{label} {int(identity.split('_')[1])}",
                         "visual_description": description,
                         "location_description": "On the product shelf's left bay, 0.74 m above the floor.",
                         "portable_candidate": True,
                         "bounds_ws_map_m": [(centre - half).round(4).tolist(), (centre + half).round(4).tolist()],
                         "reference_anchor_ws_map_m": centre.round(4).tolist()})
    entities += [
        {"id": SOURCE_SHELF, "label": "Product shelf, left bay", "portable_candidate": False,
         "visual_description": "Metal shelf holding the basil jars, jam jars and allulose boxes.",
         "location_description": "Left bay of the product rack, 0.74 m above the floor.",
         "bounds_ws_map_m": [(shelf_centre - shelf_half).round(3).tolist(), (shelf_centre + shelf_half).round(3).tolist()],
         "reference_anchor_ws_map_m": [*shelf_centre[:2].round(3).tolist(), round(top, 3)]},
        {"id": DESTINATION_SHELF, "label": "Product shelf, right bay (free space)", "portable_candidate": False,
         "visual_description": "The adjoining shelf bay to the right, with free space at its front left.",
         "location_description": "Right of the products' bay, same 0.74 m shelf height.",
         "bounds_ws_map_m": [[*release_lo.round(3).tolist(), round(top, 3)], [*release_hi.round(3).tolist(), round(top + 0.3, 3)]],
         "reference_anchor_ws_map_m": [*((release_lo + release_hi) / 2).round(3).tolist(), round(top, 3)]},
    ]
    json.dump({"schema": "workcell.homebody.semantics", "schema_version": "1.0", "scene_id": "warehouse_picking",
               "coordinate_frame": frame, "robot_role": "warehouse picker", "entities": entities},
              (work / "semantics.json").open("w"), indent=1)
    json.dump({"frame": frame, "collision": "the handoff's primitive collision boxes",
               "objects": {identity: {"initial_position": centre.round(4).tolist(), "mass_kg": mass}
                           for identity, (centre, _, mass) in items.items()}},
              (work / "physics.json").open("w"), indent=1)
    picks = [front_of_kind(items, kind) for kind in SEQUENCE]
    start_x = float(np.mean([items[p][0][0] for p in picks]))
    task = ("Pick one product of each kind from the left bay of the product shelf, in left-to-right order: first a "
            "front basil jar, then a front jam jar, then a front allulose box. After each pick, place it on the free "
            "space at the front left of the adjoining shelf bay to the right. Keep all other products in place. If a "
            "product cannot be picked after two reasonable attempts, record that failure and continue to the next "
            "kind. Finish only after attempting all three kinds. Report actual outcomes, including failed grasps.")
    json.dump({"name": "Warehouse picking, left bay", "frame": "world",
               "robot_start": [round(start_x, 3), -0.85, math.pi / 2],
               "movable_entities": sorted(items), "default_task": task},
              (work / "task_scene.json").open("w"), indent=1)
    return items, picks, (release_lo, release_hi, top)


def targets(items, picks, release):
    """The scripted steps the scorer reads, one per picked kind."""
    release_lo, release_hi, top = release
    centre = (release_lo + release_hi) / 2
    steps = []
    for identity in picks:
        position = items[identity][0]
        stance_y = round(float(position[1]) - STANCE_BACK, 3)
        steps.append({
            "fixture_id": identity, "label": identity, "hand": "right",
            "goal": [round(float(position[0]), 3), stance_y, round(math.pi / 2, 4)],
            "facing": [round(float(position[0]), 3), round(float(position[1]), 3)],
            "carry_goal": [round(float(centre[0]), 3), stance_y, round(math.pi / 2, 4)],
            "carry_facing": [round(float(centre[0]), 3), round(float(centre[1]), 3)],
            "destination_label": DESTINATION_SHELF,
            "release_map_xy_m": [round(float(centre[0]), 3), round(float(centre[1]), 3)],
            "release_delta_z_m": 0.0,
            "evaluator_target": {"bounds": [[*release_lo.round(3).tolist(), round(top - 0.02, 3)],
                                            [*release_hi.round(3).tolist(), round(top + 0.3, 3)]],
                                 "support_contacts": [DESTINATION_SHELF]},
        })
    return {"schema_version": 1, "steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path, help="the unpacked Warehouse_Picking_G1_Dex1_1_* folder")
    parser.add_argument("out", type=Path, help="the scene package to create (must not exist)")
    parser.add_argument("--targets", type=Path, help="write the episode scorer's steps here")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as work:
        items, picks, release = build(args.handoff, Path(work))
        files = package(Path(work), args.out)
    print(f"packaged {len(files)} files into {args.out}; picks {picks}; release box {release[0].round(2).tolist()}-"
          f"{release[1].round(2).tolist()} at {release[2]:.2f} m")
    if args.targets:
        args.targets.write_text(json.dumps(targets(items, picks, release), indent=1) + "\n")
        print(f"targets: {args.targets}")


if __name__ == "__main__":
    main()
