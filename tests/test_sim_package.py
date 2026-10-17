"""The runtime package keeps the compiled scene and its semantic prior."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from test_build_scene import room as room  # noqa: PLC0414 -- pytest fixture re-export

from homebody import scene_package
from homebody.vlm.observations import semantic_prior
from tools import build_kitchen


@pytest.fixture
def built(room):
    task = json.loads((room / "task_scene.json").read_text())
    task.update(robot_start=[0, 0, 0], frame="room")
    (room / "task_scene.json").write_text(json.dumps(task))
    semantics = json.loads((room / "semantics.json").read_text())
    semantics["entities"][2].update(label="Cup", portable_candidate=True,
                                    visual_description="A cup", reference_anchor_ws_map_m=[1, 0, .75])
    (room / "semantics.json").write_text(json.dumps(semantics))
    build_kitchen.build(room)
    np.savez(room / "map.npz", occupancy=np.zeros((4, 4), bool), T_map_px=np.eye(3))
    return room


def test_package_keeps_model_map_and_prior(built, tmp_path):
    out = tmp_path / "runtime"
    files = scene_package.package(built, out)
    assert {p for p in files if "/" not in p} == {
        "scene.xml", "semantics.json", "task_scene.json", "physics.json", "map.npz"}
    assert semantic_prior(built) == semantic_prior(out)
    with np.load(out / "map.npz") as data:
        assert set(data.files) == {"occupancy", "T_map_px"}
    before = mujoco.MjModel.from_xml_path(str(built / "scene.xml"))
    after = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    for name in ("geom_pos", "geom_quat", "geom_size", "geom_friction", "body_mass",
                 "body_inertia", "mesh_vert", "mesh_face", "mat_rgba"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


def test_package_refuses_outside_dependencies_and_existing_destinations(built, tmp_path):
    out = tmp_path / "runtime"
    (built / "settings.toml").write_text('[camera]\nwidth = 640\n')
    scene_package.package(built, out)
    assert "640" in (out / "settings.toml").read_text()
    with pytest.raises(ValueError, match="already exists"):
        scene_package.package(built, out)
    tree = ET.parse(built / "scene.xml")
    tree.find("asset/mesh").set("file", "../outside.obj")
    tree.write(built / "scene.xml")
    with pytest.raises(ValueError, match="relative path"):
        scene_package.package(built, tmp_path / "second")
    assert not (tmp_path / "second").exists()
