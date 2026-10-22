import json

from PIL import Image

from evolve import nav_bench

OBS = {"map_click_from_world": [[100.0, 0.0, 500.0], [0.0, -100.0, 500.0]], "base_pose": [0, 0, 0]}  # 0.01 m/click


def case(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    image = Image.new("RGB", (1001, 1001), (233, 233, 228))
    for x in range(400, 601):  # a dark table band across y 300-340
        for y in range(300, 341):
            image.putpixel((x, y), (64, 72, 82))
    image.save(folder / "map.png")
    reference = {"goal_map_1000": [500, 400], "facing_map_1000": [500, 320]}
    (folder / "case.json").write_text(json.dumps({"observation": OBS, "reference": {"arguments": reference}}))
    return folder


def test_a_stance_is_usable_with_the_same_work_spot_side_and_free_floor(tmp_path):
    folder = case(tmp_path)
    good = nav_bench.score_case(folder, {"skill": "navigate",
                                         "arguments": {"goal_map_1000": [520, 420], "facing_map_1000": [530, 320]}})
    assert good["usable"] and good["facing_m"] == 0.3
    other_side = nav_bench.score_case(folder, {"skill": "navigate",
                                               "arguments": {"goal_map_1000": [500, 200], "facing_map_1000": [500, 320]}})
    assert not other_side["usable"] and other_side["side_deg"] == 180.0
    on_table = nav_bench.score_case(folder, {"skill": "navigate",
                                             "arguments": {"goal_map_1000": [500, 330], "facing_map_1000": [500, 320]}})
    assert not on_table["goal_free"]
    assert nav_bench.score_case(folder, {"skill": "place", "arguments": {}}) == {"navigated": False}
