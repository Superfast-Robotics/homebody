"""The feasible-suite copy: dropped targets leave the scorer steps and the wording together."""
import json

import pytest

from evolve.suite_feasible import make, prune_text

PROC = ("Pick one product of each kind from the left bay's shelf at level 2, in left-to-right order: first the "
        "front-left red jam jar, then the front-left yellow food can, then the front-left blue sweetener box, then the "
        "front-left teal tea box. After each pick, place it on the empty shelf of the adjoining bay to the right. Keep "
        "all other products in place. Finish only after attempting all 4 kinds. Report actual outcomes.")
SHELF = ("Pick these products from the product shelf in this order: first the front-left blueberry jam jar, then the "
         "front wrapped food tray, then the front-right sweet corn can. After each pick, place it on the empty adjoining "
         "shelf to the left. Finish only after attempting all three products. Report actual outcomes.")


def test_the_wording_loses_exactly_the_dropped_products_and_recounts():
    text = prune_text(PROC, {"sweetener_box"}, 3)
    assert "sweetener" not in text and "jam jar" in text and "tea box" in text and "all 3 kinds" in text
    text = prune_text(SHELF, {"tray"}, 2)
    assert "tray" not in text and text.count(", then ") == 1 and "all two products" in text


def test_a_wording_that_disagrees_with_the_scorer_is_refused():
    with pytest.raises(ValueError):
        prune_text(PROC, {"sweetener_box"}, 2)


def test_the_copy_drops_steps_rewrites_the_task_and_widens_the_staging_box(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    step = {"evaluator_target": {"bounds": [[-1.5, -0.3, 0.72], [-0.9, 0.3, 1.05]], "support_contacts": ["staging_area"]}}
    steps = [{**step, "fixture_id": f} for f in ("jar_1", "tray_1", "can_2")]
    (source / "sequence.json").write_text(json.dumps({"schema_version": 1, "steps": steps}))
    (source / "task_scene.json").write_text(json.dumps({"default_task": SHELF, "movable_entities": ["jar_1"]}))
    note = make(source, tmp_path / "dst", {"tray"}, widen_staging=True)
    assert note["dropped"] == ["tray_1"] and note["kept"] == ["jar_1", "can_2"] and note["widened_staging"]
    kept = json.loads((tmp_path / "dst" / "sequence.json").read_text())["steps"]
    assert kept[0]["evaluator_target"]["bounds"][0][0] == -1.7
    assert kept[0]["evaluator_target"]["support_contacts"] == ["staging_area", "neighbor_rack_left"]
    assert "tray" not in json.loads((tmp_path / "dst" / "task_scene.json").read_text())["default_task"]
