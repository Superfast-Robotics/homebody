"""Synthetic reconstructed rooms exercise the map builder and real navigation consumer."""
import json
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh
from trimesh.visual.material import PBRMaterial

from homebody.helpers.navigation import pixel, route
from homebody.primitives.observations import NavigationMap
from tools import build_kitchen, build_scene_map

FRAME = {"name": "synthetic_room", "units": "metres", "up_axis": "Z", "handedness": "right"}
OFFSET = np.array([10., -7., .4])


def floor_mesh(hole=False):
    outer = [[-3, -2, 0], [3, -2, 0], [3, 2, 0], [-3, 2, 0]]
    if not hole:
        return trimesh.Trimesh(np.array(outer) + OFFSET, [[0, 1, 2], [0, 2, 3]], process=False)
    inner = [[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]]
    faces = [[i, (i + 1) % 4, 4 + (i + 1) % 4] for i in range(4)]
    faces += [[i, 4 + (i + 1) % 4, 4 + i] for i in range(4)]
    return trimesh.Trimesh(np.array(outer + inner) + OFFSET, faces, process=False)


def room(tmp_path, gap=1.4, hole=False):
    folder = tmp_path / "synthetic_room"
    folder.mkdir()
    parts = {"floor": floor_mesh(hole)}
    if not hole:
        for name, sign in [("wall_north", 1), ("wall_south", -1)]:
            length = 2 - gap / 2
            wall = trimesh.creation.box([.1, length, 1.8])
            wall.apply_translation(OFFSET + [0, sign * (1 + gap / 4), .9])
            parts[name] = wall
    scene = trimesh.Scene()
    for name, geometry in parts.items():
        geometry.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(baseColorFactor=[150, 150, 150, 255]))
        scene.add_geometry(geometry, node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    (folder / "room.glb").write_bytes(data)
    length = struct.unpack("<I", data[12:16])[0]
    nodes = json.loads(data[20:20 + length])["nodes"]
    entities = [{"id": node["name"], "glb_node_indices": [index]}
                for index, node in enumerate(nodes) if "mesh" in node]
    (folder / "semantics.json").write_text(json.dumps({"coordinate_frame": FRAME,
                                                       "entities": entities}))
    (folder / "task_scene.json").write_text(json.dumps({"frame": FRAME["name"],
        "robot_start": [8., -7., 0.], "movable_entities": []}))
    build_kitchen.build(folder)
    return folder


def load_map(folder):
    with np.load(folder / "map.npz", allow_pickle=False) as artifact:
        return NavigationMap(artifact["occupancy"], artifact["T_map_px"],
                             float(np.linalg.norm(artifact["T_map_px"][:2, 0])))


def occupied(nav, xy):
    return bool(nav.occupied[pixel(nav, xy)])


def test_second_room_compiles_loads_and_open_door_remains_traversable(tmp_path):
    folder = room(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    assert model.ngeom > 1
    report = build_scene_map.build(folder, .05, [.05, 1.6])
    nav = load_map(folder)
    assert not occupied(nav, [10., -7.])
    assert occupied(nav, [10., -5.5])
    assert route(nav, [8., -7.], [12., -7.], radius=.25) is not None
    assert report["frame"] == FRAME
    assert report["robot_radius_inflated"] is False
    assert np.allclose(nav.T_map_px @ [0, 0, 1], [7.025, -5.025, 1])
    assert "meshes/part_001.obj" in report["source_sha256"]
    with np.load(folder / "map.npz", allow_pickle=False) as artifact:
        assert json.loads(str(artifact["metadata_json"])) == report


def test_narrow_gap_is_raw_free_but_existing_planner_blocks_robot(tmp_path):
    folder = room(tmp_path, gap=.35)
    build_scene_map.build(folder, .025, [.05, 1.6])
    nav = load_map(folder)
    assert not occupied(nav, [10., -7.])
    assert route(nav, [8., -7.], [12., -7.], radius=.25) is None


def test_finite_floor_hole_and_outside_never_become_free(tmp_path):
    folder = room(tmp_path, hole=True)
    build_scene_map.build(folder, .05, [.05, 1.6], [6., -10., 14., -4.])
    nav = load_map(folder)
    assert occupied(nav, [10., -7.])
    assert occupied(nav, [6.5, -7.])
    assert not occupied(nav, [8., -7.])
    assert route(nav, [8., -7.], [12., -7.], radius=.25) is not None


def test_subcell_floor_hole_is_blocked_even_when_cell_center_misses_it(tmp_path):
    folder = room(tmp_path, hole=True)
    geometry = floor_mesh(True)
    geometry.vertices[4:, :2] = OFFSET[:2] + [[.01, .01], [.02, .01], [.02, .02], [.01, .02]]
    geometry.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(baseColorFactor=[150, 150, 150, 255]))
    scene = trimesh.Scene()
    scene.add_geometry(geometry, node_name="floor", geom_name="floor")
    (folder / "room.glb").write_bytes(scene.export(file_type="glb"))
    build_kitchen.build(folder)
    build_scene_map.build(folder, .1, [.05, 1.6])
    assert occupied(load_map(folder), [10.05, -6.95])


def test_height_band_ignores_only_geometry_wholly_above_the_selected_band(tmp_path):
    folder = room(tmp_path)
    tree = ET.parse(folder / "scene.xml")
    ET.SubElement(tree.find("worldbody"), "geom", name="overhead", type="box",
                  pos="10 -7 3", size=".1 .7 .1", contype="2", conaffinity="3")
    tree.write(folder / "scene.xml")
    build_scene_map.build(folder, .05, [.05, 1.6])
    assert not occupied(load_map(folder), [10., -7.])
    build_scene_map.build(folder, .05, [.05, 3.], replace=True)
    assert occupied(load_map(folder), [10., -7.])


def test_collision_box_rotation_and_vertical_clipping_preserve_clear_space(tmp_path):
    folder = room(tmp_path, hole=True)
    tree = ET.parse(folder / "scene.xml")
    ET.SubElement(tree.find("worldbody"), "geom", name="tilted_box", type="box",
                  pos="8 -7 1.4", size=".1 .1 1", euler="0 45 0", contype="2", conaffinity="3")
    tree.write(folder / "scene.xml")
    build_scene_map.build(folder, .025, [.05, .5])
    nav = load_map(folder)
    assert occupied(nav, [7.45, -7.])
    assert not occupied(nav, [8.55, -7.])


def test_initial_movable_collision_is_included(tmp_path):
    folder = room(tmp_path)
    tree = ET.parse(folder / "scene.xml")
    body = ET.SubElement(tree.find("worldbody"), "body", name="cart", pos="8 -8 .9")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", name="cart_collision", type="box", size=".3 .3 .5")
    tree.write(folder / "scene.xml")
    build_scene_map.build(folder, .05, [.05, 1.6])
    nav = load_map(folder)
    assert occupied(nav, [8., -8.])
    assert not occupied(nav, [8., -7.])


def test_partial_cells_at_explicit_bounds_remain_blocked(tmp_path):
    folder = room(tmp_path)
    build_scene_map.build(folder, .1, [.05, 1.6], [7.5, -8.23, 9.03, -5.5])
    nav = load_map(folder)
    assert nav.occupied[-1].all() and nav.occupied[:, -1].all()
    assert not occupied(nav, [8., -7.])


@pytest.mark.parametrize("resolution,band,bounds", [
    (0, [.05, 1.6], None), (.05, [0, 1.6], None), (.05, [2, 1], None),
    (.05, [float("nan"), 1.6], None), (1e-8, [.05, 1.6], None),
    (.05, [.05, 1.6], [0, 0, 0, 1]), (.05, [.05, 1.6], [0, 0, float("inf"), 1]),
])
def test_invalid_parameters_never_write_output(tmp_path, resolution, band, bounds):
    folder = room(tmp_path)
    with pytest.raises(ValueError):
        build_scene_map.build(folder, resolution, band, bounds)
    assert not (folder / "map.npz").exists()


@pytest.mark.parametrize("problem", ["frame", "unsupported", "no_floor", "margin"])
def test_unsupported_scene_is_refused_without_writing(tmp_path, problem):
    folder = room(tmp_path)
    if problem == "frame":
        path = folder / "task_scene.json"
        task = json.loads(path.read_text())
        task["frame"] = "other"
        path.write_text(json.dumps(task))
    else:
        tree = ET.parse(folder / "scene.xml")
        world = tree.find("worldbody")
        if problem == "unsupported":
            ET.SubElement(world, "geom", name="ball", type="sphere", size=".1", pos="8 -7 1")
        elif problem == "no_floor":
            for geom in list(world):
                if geom.get("name", "").startswith("floor__") and geom.get("type") == "mesh":
                    world.remove(geom)
        else:
            next(g for g in world if g.get("name", "").startswith("wall_")).set("margin", ".02")
        tree.write(folder / "scene.xml")
    with pytest.raises(ValueError):
        build_scene_map.build(folder, .05, [.05, 1.6])
    assert not (folder / "map.npz").exists()


def test_existing_map_is_preserved_without_explicit_replace(tmp_path):
    folder = room(tmp_path)
    target = folder / "map.npz"
    target.write_bytes(b"previous result")
    with pytest.raises(ValueError, match="--replace"):
        build_scene_map.build(folder, .05, [.05, 1.6])
    assert target.read_bytes() == b"previous result"
    build_scene_map.build(folder, .05, [.05, 1.6], replace=True)
    assert load_map(folder).occupied.ndim == 2


def test_preexisting_temporary_symlink_cannot_clobber_an_outside_file(tmp_path):
    folder = room(tmp_path)
    outside = tmp_path / "keep.bin"
    outside.write_bytes(b"outside sentinel")
    link = folder / "map.npz.tmp"
    link.symlink_to(outside)
    original_files = set(folder.iterdir())
    build_scene_map.build(folder, .05, [.05, 1.6])
    assert outside.read_bytes() == b"outside sentinel"
    assert link.is_symlink()
    assert not (folder / "map.npz").is_symlink()
    assert load_map(folder).occupied.ndim == 2
    assert set(folder.iterdir()) == original_files | {folder / "map.npz"}


def test_nested_include_cannot_pull_unhashed_dependencies_into_the_map(tmp_path):
    folder = room(tmp_path)
    outside = tmp_path / "outside.xml"
    original = ('<mujocoinclude><geom name="outside_collision" type="box" '
                'pos="8 -7 1" size=".1 .1 .1"/></mujocoinclude>')
    outside.write_text(original)
    (folder / "nested.xml").write_text(
        '<mujocoinclude><include file="../outside.xml"/></mujocoinclude>')
    tree = ET.parse(folder / "scene.xml")
    ET.SubElement(tree.find("worldbody"), "include", file="nested.xml")
    tree.write(folder / "scene.xml")
    with pytest.raises(ValueError, match="without includes"):
        build_scene_map.build(folder, .05, [.05, 1.6])
    assert outside.read_text() == original
    assert not (folder / "map.npz").exists()
