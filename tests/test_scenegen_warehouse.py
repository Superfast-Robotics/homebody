"""tools/scenegen.py's warehouse_rack family: what each generated task promises its scorer and the robot."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("scenegen", ROOT / "tools/scenegen.py")
scenegen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scenegen)


def scenes(count=12, seed=5):
    rng = np.random.default_rng(seed)
    made = []
    while len(made) < count:
        try:
            made.append(scenegen.warehouse_rack(f"wh_{len(made):02d}", rng))
        except RuntimeError:
            continue
    return made


def test_every_named_product_fits_the_jaws_and_is_named_in_order():
    for scene in scenes():
        targets = {item.identity: item for item in scene.items}
        named = [targets[step["fixture_id"]] for step in scene.steps]
        assert 2 <= len(named) <= 4
        assert all(scenegen.wh_grip_width(item) <= scenegen.WH_GRIP_LIMIT for item in named)
        assert [item.position[0] for item in named] == sorted(item.position[0] for item in named)
        positions = [scene.task.index(item.label.lower()) for item in named]
        assert positions == sorted(positions)


def test_releases_land_inside_their_target_on_a_described_support():
    for scene in scenes():
        for step in scene.steps:
            (lo, hi), support = step["evaluator_target"]["bounds"], step["evaluator_target"]["support_contacts"][0]
            assert lo[0] <= step["release_map_xy_m"][0] <= hi[0] and lo[1] <= step["release_map_xy_m"][1] <= hi[1]
            assert support in scene.descriptions and support == step["destination_label"]


def test_every_stance_is_reachable_from_the_start(tmp_path):
    scene = scenes(count=3, seed=11)[-1]
    start = scene.robot_start[:2]
    for step in scene.steps:
        assert scenegen.routable(scene, start, step["goal"][:2])
        assert scenegen.routable(scene, start, step["carry_goal"][:2])
    scenegen.write_package(scene, tmp_path / scene.name)
    written = json.loads((tmp_path / scene.name / "sequence.json").read_text())
    assert written["steps"] == scene.steps
    assert json.loads((tmp_path / scene.name / "task_scene.json").read_text())["default_task"] == scene.task
