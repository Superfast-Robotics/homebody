"""package_robot_demo.py ROOT OUT [--scene NAME ...]: the 2026-10-22 robot demo environments (Robot_Demo_Environments_*,
unpacked at ROOT) as HomeBody scene packages, one folder per scene under OUT.

Two scenes, from each package's environment-only MJCF and its TASK_CONTRACT.json task seeds:

- kitchen_baskets (Kitchen_G1_Dex1_1): the apple into the round basket, then a coffee capsule into the rectangular one;
- warehouse_cart (Warehouse_Picking_Cart_G1_Dex1_1): a jam jar, a basil jar and an allulose box into the picking cart.

Each package holds scene.xml with its mesh and texture closure (HomeBody adds its own G1), map.npz (static colliding
geoms overlapping robot height, projected onto a 2 cm grid), semantics.json, physics.json, task_scene.json,
settings.toml (the Dex1-1 hands) and sequence.json (the scripted steps and the scorer's targets, read by
tools/check_manipulation.py, tools/run_wave.py and the evolve loops). The conversion follows HomeBody's conventions
(tools/package_warehouse_picking.py): one ground plane "floor__ground"; scenery colliding in group 2; the scene's
hidden collision proxies in the depth-only group; each movable object a free body with one convex mesh hull
"<id>__collision" (built from its collision geoms) and its visuals "<id>__<n>"; containers and other named things
static, their geoms "<id>__<n>" so the cameras' segmentation and the scorer's support contacts name them. Task
markers (sites) are left out: a camera must not show where an object belongs.

Nothing here runs the packages' own scripts; their files are read as data."""
import argparse
import json
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from homebody.backends.scene import DEPTH_ONLY_GROUP
from homebody.scene_package import package

RESOLUTION = 0.02          # m per map cell
ROBOT_BAND = (0.05, 1.80)  # heights that block the walking robot
CLEARANCE = 0.31           # m of free floor around a stance (navigation's robot_radius 0.28 and a margin)
RELEASE_DROP = -0.05       # m below the container's interior centre the place releases
RELEASE_INSET = 0.06       # m the release point keeps from a container's walls
RELEASE_REACH = 0.7        # how far from the interior's centre toward the stance the release point lies (of the way)
WALL = 0.02                # m a container's walls take from its footprint
BASKET_DEPTH = 0.25        # m a container's interior reaches below its interior marker at most
APPROACH_CONE = math.radians(50)  # how far from the preferred side a stance may face from
COLLIDE = {"contype": "2", "conaffinity": "3"}  # HomeBody's scene and object collision bits
SILENT = {"contype": "0", "conaffinity": "0"}


@dataclass(frozen=True)
class Thing:
    label: str
    visual: str
    location: str = ""


@dataclass(frozen=True)
class Spec:
    package: str
    scene: str
    scene_id: str
    name: str
    role: str
    support: str            # entity id of the surface the task's objects stand on
    movable: dict           # id -> Thing
    containers: dict        # id -> (interior site, Thing)
    labelled: dict = field(default_factory=dict)  # other static things the cameras should name
    steps: tuple = ()       # (object, container); an object "front:<kind>" is that kind's item nearest the robot's side
    task: str = ""
    approach: tuple | None = None  # the side stances face from (a unit xy), else whichever side is nearest
    carry_approach: tuple | None = None  # the same for the stance at the container
    placements: dict = field(default_factory=dict)  # id -> (x, y): a body moved from its delivered place
    why_moved: str = ""


KITCHEN = Spec(
    package="Kitchen_G1_Dex1_1", scene="kitchen.xml", scene_id="kitchen_baskets", role="kitchen helper",
    name="Kitchen counter: fruit and capsules into baskets", support="kitchen_table",
    movable={
        "apple": Thing("Apple", "A red apple."),
        "kiwi": Thing("Kiwi", "A brown, fuzzy kiwi fruit."),
        "banana_1": Thing("Banana 1", "A single yellow banana."),
        "banana_2": Thing("Banana 2", "A single yellow banana."),
        "banana_3": Thing("Banana 3", "A single yellow banana."),
        "pepper_plush": Thing("Pepper toy", "A soft red pepper toy."),
        "carrot_plush": Thing("Carrot toy", "A soft orange carrot toy."),
        "pear_with_foam": Thing("Pear", "A pear in a white foam net."),
        "capsule_1": Thing("Coffee capsule 1", "A small round coffee capsule."),
        "capsule_2": Thing("Coffee capsule 2", "A small round coffee capsule."),
        "capsule_3": Thing("Coffee capsule 3", "A small round coffee capsule."),
    },
    containers={
        "round_basket": ("round_basket_interior", Thing("Round basket", "A shallow round woven basket.")),
        "rectangular_basket": ("rectangular_basket_interior",
                               Thing("Rectangular basket", "An open white rectangular basket.")),
    },
    labelled={
        "coffee_machine": Thing("Coffee machine", "A white capsule coffee machine."),
        "gold_tray": Thing("Gold tray", "A round gold tray with a tiered stand."),
        "clear_capsule_box": Thing("Capsule box", "A clear box of coffee capsules."),
    },
    steps=(("apple", "round_basket"), ("capsule_1", "rectangular_basket")),
    # The table is 2.40 x 1.00 m and the task's objects stand 0.23-0.84 m in from its edges, beyond the Dex1's
    # ~0.45 m reach from a stance; the island blocks the front edge beyond x 1.9. The task's objects and baskets
    # move to the open front edge, about 0.1 m in; the room and everything else stay as delivered.
    placements={"round_basket": (0.45, 0.14), "rectangular_basket": (0.85, 0.11),
                "apple": (1.15, 0.10), "capsule_1": (1.55, 0.09)},
    why_moved=("The delivered table puts the task's objects 0.23-0.84 m from its edges, beyond the G1 Dex1-1's "
               "reach from where it can stand; they and the baskets stand at the open front edge instead."),
    task=("On the kitchen table: put the red apple into the round woven basket, then put one loose coffee capsule "
          "into the open rectangular basket. Keep everything else on the table in place. If an object cannot be "
          "picked after two reasonable attempts, record that failure and continue. Finish after attempting both. "
          "Report actual outcomes, including failed grasps."),
)

WAREHOUSE = Spec(
    package="Warehouse_Picking_Cart_G1_Dex1_1", scene="warehouse.xml", scene_id="warehouse_cart",
    role="warehouse picker", name="Warehouse picking: products into the cart", support="product_shelf",
    movable={**{f"basil_{i:02d}": Thing(f"Basil jar {i}", "Green jar of dried basil with a dark cap and a printed label.")
                for i in range(1, 4)},
             **{f"jam_{i:02d}": Thing(f"Jam jar {i}", "Small round jam jar with a lid and a printed label.")
                for i in range(1, 9)},
             **{f"allulose_{i:02d}": Thing(f"Allulose box {i}", "Tall thin box of allulose sweetener standing upright.")
                for i in range(1, 5)}},
    containers={"picking_cart": ("cart_basket_interior", Thing("Picking cart", "A blue wire shopping cart."))},
    # the contract names jam_01, basil_01 and allulose_01, but they stand at the back, beyond the Dex1's reach
    steps=(("front:jam", "picking_cart"), ("front:basil", "picking_cart"), ("front:allulose", "picking_cart")),
    approach=(0.0, -1.0),  # from the aisle in front of the shelf, as the package's own robot stands
    carry_approach=(0.0, -1.0),
    # In its 0.9 m aisle the navigate skill's edge fit takes in both shelf rows and finds no stance at the cart;
    # the cart moves to the open floor past the aisle's left end. Everything else stays as delivered.
    placements={"picking_cart": (-1.2, -0.63)},
    why_moved=("The delivered cart stands mid-aisle, where HomeBody's navigate skill finds no square stance facing "
               "it; it stands on the open floor past the aisle's left end instead."),
    task=("Pick products from the shelf into the blue picking cart: first a jam jar, then a basil jar, then an "
          "allulose box. Keep all other products in place. If a product cannot be picked after two reasonable "
          "attempts, record that failure and continue to the next. Finish only after attempting all three. Report "
          "actual outcomes, including failed grasps."),
)

SCENES = {spec.scene_id: spec for spec in (KITCHEN, WAREHOUSE)}


def world_box(model, data, geom):
    """GEOM's axis-aligned bounds in the world: (lo, hi)."""
    centre, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    middle = data.geom_xpos[geom] + rotation @ centre
    reach = np.abs(rotation) @ half
    return middle - reach, middle + reach


def colliding(model, geom):
    return bool(model.geom_contype[geom] or model.geom_conaffinity[geom])


def top_body(model, geom):
    """The name of GEOM's top-level body, or None for the world."""
    root = model.body_rootid[model.geom_bodyid[geom]]
    return None if root == 0 else model.body(root).name


def occupancy(model, data, movable, solid=()):
    """The navigation map: True where a static colliding geom overlaps robot height. Returns (grid, T_map_px).
    Each SOLID body (a wire cart, a basket) is drawn as its whole footprint: navigation fits straight edges to
    what it faces, and a ragged outline of wires does not give one."""
    boxes = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE or not colliding(model, geom):
            continue
        if top_body(model, geom) in movable or top_body(model, geom) in solid:
            continue
        lo, hi = world_box(model, data, geom)
        if hi[2] > ROBOT_BAND[0] and lo[2] < ROBOT_BAND[1]:
            boxes.append((lo, hi))
    for name in solid:
        lo, hi = body_bounds(model, data, name)
        if hi[2] > ROBOT_BAND[0] and lo[2] < ROBOT_BAND[1]:
            boxes.append((lo, hi))
    lo = np.min([b[0][:2] for b in boxes], axis=0) - 1.0
    hi = np.max([b[1][:2] for b in boxes], axis=0) + 1.0
    width, height = np.ceil((hi - lo) / RESOLUTION).astype(int)
    grid = np.zeros((height, width), dtype=bool)
    for (x0, y0, _), (x1, y1, _) in boxes:
        u0, u1 = math.floor((x0 - lo[0]) / RESOLUTION), math.ceil((x1 - lo[0]) / RESOLUTION)
        v0, v1 = math.floor((hi[1] - y1) / RESOLUTION), math.ceil((hi[1] - y0) / RESOLUTION)
        grid[max(v0, 0):v1, max(u0, 0):u1] = True
    return grid, np.array([[RESOLUTION, 0, lo[0]], [0, -RESOLUTION, hi[1]], [0, 0, 1.0]])


def free(grid, transform, xy, radius=CLEARANCE):
    """Whether the floor within RADIUS of XY is free on the map."""
    u = (xy[0] - transform[0, 2]) / RESOLUTION
    v = (transform[1, 2] - xy[1]) / RESOLUTION
    r = radius / RESOLUTION
    vs, us = np.mgrid[int(v - r):int(v + r) + 1, int(u - r):int(u + r) + 1]
    inside = (us - u) ** 2 + (vs - v) ** 2 <= r * r
    if vs.min() < 0 or us.min() < 0 or vs.max() >= grid.shape[0] or us.max() >= grid.shape[1]:
        return False
    return not grid[vs[inside], us[inside]].any()


def stance(grid, transform, target, approach=None):
    """The free standing point nearest TARGET, as (x, y, yaw) facing the target: on the APPROACH side when given
    (within APPROACH_CONE), else on whichever side is closer and open. The pick and place skills walk on to their
    own stance from there."""
    centre = np.asarray(target[:2], dtype=float)
    angles = np.radians(np.arange(0, 360, 5))
    if approach is not None:
        side = math.atan2(approach[1], approach[0])
        angles = angles[np.abs(np.angle(np.exp(1j * (angles - side)))) <= APPROACH_CONE]
    for radius in np.arange(0.3, 2.5, RESOLUTION):
        for angle in angles:
            xy = centre + radius * np.array([math.cos(angle), math.sin(angle)])
            if free(grid, transform, xy):
                yaw = math.atan2(centre[1] - xy[1], centre[0] - xy[0])
                return [round(float(xy[0]), 3), round(float(xy[1]), 3), round(yaw, 4)]
    raise ValueError(f"no free stance within 2.5 m of {np.round(centre, 2).tolist()}")


def surface_points(model, geom, count=24):
    """Points on GEOM's collision surface in its own frame: a mesh's vertices, else its primitive sampled."""
    kind, size = model.geom_type[geom], model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        return model.mesh_vert[start:start + model.mesh_vertnum[mesh]].astype(float)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
    angles = np.linspace(0, 2 * math.pi, count, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(count)], axis=1)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return np.concatenate([ring * [size[0], size[0], 0] + [0, 0, z] for z in (-size[1], size[1])])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.concatenate([ring * [size[0], size[0], 0] + [0, 0, z] for z in (-size[1], size[1])]
                              + [[[0, 0, -size[1] - size[0]], [0, 0, size[1] + size[0]]]])
    sphere = np.concatenate([np.stack([np.cos(angles) * math.cos(e), np.sin(angles) * math.cos(e),
                                       np.full(count, math.sin(e))], axis=1)
                             for e in np.linspace(-math.pi / 2, math.pi / 2, 9)])
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return sphere * size[0]
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return sphere * size
    raise ValueError(f"no surface sampling for geom type {kind}")


def hull(model, data, body):
    """The convex hull of BODY's colliding geoms in the body frame: (vertices, triangles)."""
    points = []
    body_pos, body_rot = data.xpos[body], data.xmat[body].reshape(3, 3)
    for geom in np.flatnonzero(model.geom_bodyid == body):
        if not colliding(model, geom):
            continue
        local = surface_points(model, geom)
        world = data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T
        points.append((world - body_pos) @ body_rot)
    points = np.concatenate(points)
    shape = ConvexHull(points)
    used = np.unique(shape.simplices)
    index = {old: new for new, old in enumerate(used)}
    triangles = []
    for simplex, equation in zip(shape.simplices, shape.equations):
        a, b, c = (points[i] for i in simplex)
        if np.dot(np.cross(b - a, c - a), equation[:3]) < 0:  # outward winding
            simplex = simplex[[0, 2, 1]]
        triangles.append([index[i] for i in simplex])
    return points[used], np.array(triangles)


def write_obj(path, vertices, triangles):
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles]
    path.write_text("\n".join(lines) + "\n")


def numbers(values, digits=6):
    return " ".join(f"{float(v):.{digits}g}" for v in values)


def body_bounds(model, data, name):
    """World bounds of the named top-level body's geoms: (lo, hi)."""
    body = model.body(name).id
    boxes = [world_box(model, data, g) for g in range(model.ngeom)
             if model.body_rootid[model.geom_bodyid[g]] == body]
    return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)


def support_geom(model, data, xy, top):
    """The static geom right under XY just below height TOP (what the task's objects stand on)."""
    start, hit = np.array([*xy, top + 0.02]), np.array([-1], dtype=np.int32)
    groups = np.ones(6, dtype=np.uint8)
    while True:
        distance = mujoco.mj_ray(model, data, start, np.array([0., 0., -1.]), groups, 1, -1, hit)
        if distance < 0:
            raise ValueError(f"nothing under {np.round(xy, 2).tolist()}")
        geom = int(hit[0])
        if colliding(model, geom) and model.body_rootid[model.geom_bodyid[geom]] == 0:
            return geom
        start = start - [0, 0, distance + 1e-4]


def interior(model, data, container, site):
    """A container's interior: the release point at its interior site, and the volume inside its walls (its
    footprint less WALL, from its floor to above its rim) as (centre, lo, hi) in the world. The packages' interior
    sites are small markers, not volumes."""
    centre = data.site_xpos[model.site(site).id].copy()
    lo, hi = body_bounds(model, data, container)
    floor = max(lo[2], centre[2] - BASKET_DEPTH)  # a cart's basket, not its lower rack or wheels
    inner_lo = np.array([lo[0] + WALL, lo[1] + WALL, min(floor, centre[2]) - 0.02])
    inner_hi = np.array([hi[0] - WALL, hi[1] - WALL, hi[2] + 0.05])
    return centre, inner_lo, inner_hi


def resolve(spec, model, data):
    """SPEC's steps with each "front:<kind>" replaced by that kind's item nearest the approach side (the front row,
    then leftmost)."""
    side = np.asarray(spec.approach or (0.0, -1.0))
    steps = []
    for identity, container in spec.steps:
        if identity.startswith("front:"):
            kind = identity.split(":", 1)[1]
            items = [name for name in spec.movable if name.startswith(kind + "_")]
            identity = max(items, key=lambda name: (round(float(data.xpos[model.body(name).id][:2] @ side), 2),
                                                    -float(data.xpos[model.body(name).id][0])))
        steps.append((identity, container))
    return replace(spec, steps=tuple(steps))


def place(spec, model, data):
    """Move SPEC's placed bodies in the compiled model (a free body by its joint, any other by its pose):
    {id: {"from": xyz, "to": xyz}}."""
    moved = {}
    for identity, (x, y) in spec.placements.items():
        body = model.body(identity).id
        before = data.xpos[body].round(4).tolist()
        joint = model.body_jntadr[body]
        if model.body_jntnum[body] == 1 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            address = model.jnt_qposadr[joint]
            data.qpos[address:address + 2] = model.qpos0[address:address + 2] = (x, y)
        else:
            model.body_pos[body][:2] = (x, y)
        mujoco.mj_forward(model, data)
        moved[identity] = {"from": before, "to": data.xpos[body].round(4).tolist()}
    return moved


def release_point(centre, lo, hi, stance_xy):
    """Where to release in a container: from its interior centre toward the stance, RELEASE_REACH of the way to the
    inset wall, so the near side of a deep container (a cart's basket) is within reach."""
    centre = np.asarray(centre[:2], dtype=float)
    toward = np.asarray(stance_xy[:2], dtype=float) - centre
    if np.linalg.norm(toward) < 1e-6:
        return centre
    unit = toward / np.linalg.norm(toward)
    inner_lo, inner_hi = np.asarray(lo[:2]) + RELEASE_INSET, np.asarray(hi[:2]) - RELEASE_INSET
    limits = [((inner_hi if u > 0 else inner_lo)[i] - centre[i]) / u for i, u in enumerate(unit) if abs(u) > 1e-9]
    travel = max(0.0, min(limits))
    return centre + RELEASE_REACH * travel * unit


def build(spec, source, work):
    """The unpackaged scene in WORK and its scripted steps."""
    scene_path = source / spec.package / "scenes" / spec.scene
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    spec = resolve(spec, model, data)
    moved = place(spec, model, data)
    movable, containers = set(spec.movable), set(spec.containers)
    named = {**spec.containers, **spec.labelled}
    missing = (movable | containers | set(spec.labelled)) - {model.body(i).name for i in range(model.nbody)}
    if missing:
        raise ValueError(f"{spec.package} has no body named {sorted(missing)}")
    grid, transform = occupancy(model, data, movable, containers)
    first = model.body(spec.steps[0][0]).id
    shelf = support_geom(model, data, data.xpos[first][:2], float(body_bounds(model, data, spec.steps[0][0])[0][2]))

    tree = ET.parse(scene_path)
    root = tree.getroot()
    defaults = root.find("default")
    if defaults is not None and (defaults.findall("default") or any(n.get("class") for n in root.iter())):
        raise ValueError("default classes are not supported; only one top-level <default><geom>")
    geom_defaults = defaults.find("geom").attrib if defaults is not None and defaults.find("geom") is not None else {}
    for tag in ("compiler", "option", "size", "default", "keyframe"):
        for node in root.findall(tag):
            root.remove(node)
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(0, visual)
    if len(visual) == 0:  # the scene's lights, not the robot model's (backends/scene.py own_appearance)
        ET.SubElement(visual, "headlight", ambient=".1 .1 .1", diffuse=".4 .4 .4", specular=".5 .5 .5")
    world = root.find("worldbody")
    for node in list(root.iter()):
        for site in node.findall("site"):
            node.remove(site)
    for geom in world.iter("geom"):
        for key, value in geom_defaults.items():
            geom.attrib.setdefault(key, value)

    assets = work / "assets"
    assets.mkdir()
    asset = root.find("asset")
    for node in asset:
        name = node.get("file")
        if name is not None:
            path = (scene_path.parent / name).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError(f"scene dependency outside the package: {name}")
            target = f"{node.tag}_{len(list(assets.iterdir())):04d}{path.suffix.lower()}"
            shutil.copy(path, assets / target)
            node.set("file", f"assets/{target}")

    floor = next(g for g in world.findall("geom") if g.get("type") == "plane")
    size = floor.get("size").split()
    floor.attrib.update({"name": "floor__ground", "group": "3", **COLLIDE})
    look = {k: floor.attrib.pop(k) for k in ("material", "rgba") if k in floor.attrib}
    ET.SubElement(world, "geom", {"name": "floor__visual", "type": "box", "group": "2", **SILENT,
                                  "size": f"{size[0]} {size[1]} 0.0005", "pos": "0 0 -0.0005", **look})
    shelf_name = model.geom(shelf).name

    counters = {}

    def rename(geom, owner):
        visual_only = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        hidden = geom.get("group") == "3"
        index = counters[owner] = counters.get(owner, 0) + 1
        original = geom.get("name") or f"g{index}"
        if owner is None:
            name = f"{spec.support}__surface" if original == shelf_name else f"structure__{original}"
        else:
            name = f"{owner}__{original}" if owner in named else f"structure__{owner}_{original}"
        geom.attrib.update({"name": name, "group": str(DEPTH_ONLY_GROUP) if hidden and not visual_only else "2",
                            **(SILENT if visual_only else COLLIDE)})

    for geom in world.findall("geom"):
        if geom.get("name") not in ("floor__ground", "floor__visual"):
            rename(geom, None)
    for body in world.findall("body"):
        if body.get("name") in moved:
            body.set("pos", numbers(moved[body.get("name")]["to"]))
    for body in world.findall("body"):
        identity = body.get("name")
        if identity in movable:
            continue
        for joint in body.findall("freejoint") + body.findall("joint"):
            body.remove(joint)  # containers, plants and the rest stand still: only the task's objects move
        for inertial in body.findall("inertial"):
            body.remove(inertial)
        for geom in body.iter("geom"):
            rename(geom, identity)
    for body in world.findall("body"):
        identity = body.get("name")
        if identity not in movable:
            continue
        b = model.body(identity).id
        vertices, triangles = hull(model, data, b)
        write_obj(assets / f"{identity}_hull.obj", vertices, triangles)
        ET.SubElement(asset, "mesh", name=f"{identity}_hull", file=f"assets/{identity}_hull.obj")
        collision = next(g for g in body.iter("geom") if not (g.get("contype") == "0" and g.get("conaffinity") == "0"))
        contact = {k: collision.get(k) for k in ("friction", "solref", "solimp", "condim") if collision.get(k)}
        for parent in [body, *body.iter("body")]:
            for geom in list(parent.findall("geom")):
                if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                    index = counters[identity] = counters.get(identity, 0) + 1
                    geom.attrib.update({"name": f"{identity}__v{index}", "group": "2", **SILENT, "mass": "0"})
                    geom.attrib.pop("density", None)
                else:
                    parent.remove(geom)
        for inertial in body.findall("inertial"):
            body.remove(inertial)
        body.insert(0, ET.Element("inertial", {
            "pos": numbers(model.body_ipos[b]), "quat": numbers(model.body_iquat[b]),
            "mass": f"{model.body_mass[b]:.6g}", "diaginertia": numbers(np.maximum(model.body_inertia[b], 1e-7))}))
        ET.SubElement(body, "geom", {"name": f"{identity}__collision", "type": "mesh", "mesh": f"{identity}_hull",
                                     "group": "3", **COLLIDE, "mass": "0", **contact})
    tree.write(work / "scene.xml", encoding="unicode")
    np.savez_compressed(work / "map.npz", occupancy=grid, T_map_px=transform)

    frame = {"name": "world", "units": "metres", "up_axis": "Z", "handedness": "right",
             "origin": f"the {spec.package} package's world frame"}
    shelf_lo, shelf_hi = world_box(model, data, shelf)
    top = round(float(shelf_hi[2]), 3)
    entities = []
    for identity, thing in spec.movable.items():
        lo, hi = body_bounds(model, data, identity)
        entities.append({"id": identity, "label": thing.label, "visual_description": thing.visual,
                         "location_description": thing.location or f"On the {spec.support.replace('_', ' ')}, "
                                                                   f"{top:.2f} m above the floor.",
                         "portable_candidate": True, "bounds_ws_map_m": [lo.round(4).tolist(), hi.round(4).tolist()],
                         "reference_anchor_ws_map_m": data.xpos[model.body(identity).id].round(4).tolist()})
    for identity, value in named.items():
        thing = value[1] if isinstance(value, tuple) else value
        lo, hi = body_bounds(model, data, identity)
        entities.append({"id": identity, "label": thing.label, "visual_description": thing.visual,
                         "location_description": thing.location or "In the room, as the images show.",
                         "portable_candidate": False, "bounds_ws_map_m": [lo.round(3).tolist(), hi.round(3).tolist()],
                         "reference_anchor_ws_map_m": ((lo + hi) / 2).round(3).tolist()})
    entities.append({"id": spec.support, "label": spec.support.replace("_", " ").capitalize(), "portable_candidate": False,
                     "visual_description": "The surface the task's objects stand on.",
                     "location_description": f"Its top is {top:.2f} m above the floor.",
                     "bounds_ws_map_m": [shelf_lo.round(3).tolist(), shelf_hi.round(3).tolist()],
                     "reference_anchor_ws_map_m": [*((shelf_lo + shelf_hi) / 2)[:2].round(3).tolist(), top]})
    json.dump({"schema": "workcell.homebody.semantics", "schema_version": "1.0", "scene_id": spec.scene_id,
               "coordinate_frame": frame, "robot_role": spec.role, "entities": entities},
              (work / "semantics.json").open("w"), indent=1)
    json.dump({"frame": frame, "collision": "convex hulls of the package's collision geoms for movable objects",
               "objects": {identity: {"initial_position": data.xpos[model.body(identity).id].round(4).tolist(),
                                      "mass_kg": round(float(model.body_mass[model.body(identity).id]), 4)}
                           for identity in spec.movable}},
              (work / "physics.json").open("w"), indent=1)
    (work / "settings.toml").write_text('[robot]\nhands = "dex1"\n')
    if moved:
        json.dump({"source": f"{spec.package}/scenes/{spec.scene}", "why": spec.why_moved, "moved": moved},
                  (work / "variant.json").open("w"), indent=1)

    steps = []
    for identity, container in spec.steps:
        position = data.xpos[model.body(identity).id]
        site = spec.containers[container][0]
        centre, lo, hi = interior(model, data, container, site)
        goal = stance(grid, transform, position, spec.approach)
        carry = stance(grid, transform, centre, spec.carry_approach)
        release = release_point(centre, lo, hi, carry)
        steps.append({
            "fixture_id": identity, "label": identity, "hand": "right",
            "goal": goal, "facing": position[:2].round(3).tolist(),
            "carry_goal": carry, "carry_facing": release.round(3).tolist(),
            "destination_label": container,
            "release_map_xy_m": release.round(3).tolist(), "release_delta_z_m": RELEASE_DROP,
            "evaluator_target": {"bounds": [lo.round(3).tolist(), hi.round(3).tolist()],
                                 "support_contacts": [container]},
        })
    start = stance(grid, transform, data.xpos[first], spec.approach)
    json.dump({"name": spec.name, "frame": "world", "robot_start": start,
               "movable_entities": sorted(spec.movable), "default_task": spec.task},
              (work / "task_scene.json").open("w"), indent=1)
    return {"schema_version": 1, "steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="the unpacked Robot_Demo_Environments_* folder")
    parser.add_argument("out", type=Path, help="the folder to create the packages in (each must not exist)")
    parser.add_argument("--scene", choices=tuple(SCENES), action="append", help="only this scene (repeatable)")
    args = parser.parse_args()
    for name in args.scene or SCENES:
        spec = SCENES[name]
        destination = args.out / name
        with tempfile.TemporaryDirectory() as work:
            sequence = build(spec, args.root, Path(work))
            files = package(Path(work), destination)
            if (Path(work) / "variant.json").exists():
                shutil.copy(Path(work) / "variant.json", destination / "variant.json")
        (destination / "sequence.json").write_text(json.dumps(sequence, indent=1) + "\n")
        print(f"{name}: {len(files)} files in {destination}; steps "
              + ", ".join(f"{s['label']} -> {s['destination_label']}" for s in sequence["steps"]))


if __name__ == "__main__":
    main()
