"""tools/warehouse_variants.py: the variants' splits stay disjoint and their geometry helpers find free space."""
import xml.etree.ElementTree as ET

import numpy as np

from tools import warehouse_variants as wv


def test_variant_pairs_are_unique_and_held_out_pairs_unseen_in_train():
    pairs = [(v.scene, frozenset(v.targets), v.destination) for v in wv.VARIANTS]
    assert len(pairs) == len(set(pairs))
    train = {(v.scene, frozenset(v.targets), v.destination) for v in wv.VARIANTS if v.split == "train"}
    assert not train & {(v.scene, frozenset(v.targets), v.destination) for v in wv.VARIANTS if v.split != "train"}
    for variant in wv.VARIANTS:
        assert set(variant.targets) <= set(wv.NAMES[variant.scene])
        assert variant.destination in wv.DESTINATIONS[variant.scene]


def test_task_names_every_target_in_order():
    variant = wv.Variant("v", "shelf", "train", ("tray_1", "jar_2"), "right", 0)
    text = wv.task_text(variant, wv.DESTINATIONS["shelf"]["right"])
    assert text.index("wrapped food tray") < text.index("front-right blueberry jam jar")
    assert "adjoining bay to the right" in text and "both products" in text


def test_free_square_skips_clutter_from_the_chosen_corner():
    world = ET.fromstring(
        '<worldbody><geom name="structure__s" type="box" pos="0.5 0.3 0.7" size="0.5 0.3 0.01" contype="2"/>'
        '<geom name="structure__c" type="box" pos="0.15 0.15 0.8" size="0.15 0.15 0.09" contype="2" group="4"/>'
        '</worldbody>')
    static = wv.static_boxes(world)
    lo, hi, top = wv.free_square(static, wv.shelf_geom(world, "s"), "front-left")
    assert top == 0.71 and lo[0] >= 0.3 and abs(hi[0] - lo[0] - wv.PATCH) < 1e-9
    lo, _, _ = wv.free_square(static, wv.shelf_geom(world, "s"), "front-right")
    assert lo[0] > 0.7


def test_nudges_keep_a_gap_to_neighbours():
    boxes = {"a": (np.array([0., 0., 0.]), np.array([.1, .1, .1])),
             "b": (np.array([.106, 0., 0.]), np.array([.2, .1, .1]))}
    variant = wv.Variant("v", "shelf", "train", ("a",), "left", 0, nudge=True)
    shifts = wv.nudges(np.random.default_rng(0), variant, boxes, [])
    for dx, _ in shifts.values():
        assert .1 + dx + wv.NUDGE_GAP <= .106 + 1e-9


def test_releases_spread_inside_the_footprint():
    points = wv.releases(wv.DESTINATIONS["shelf"]["left"], 3)
    assert len(set(points)) == 3
    assert all(-1.5 < x < -0.9 and -0.3 < y < 0.3 for x, y in points)
