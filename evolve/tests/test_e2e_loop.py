"""The end-to-end loop's pure parts: session rows, missing sessions, summaries, layout clusters, judging by placed
objects, the editable screen and the prompts."""
import json
import string
import tomllib
from pathlib import Path

from evolve import skill_score
from evolve.e2e_loop import E2EMixin, layout_cluster, read_split, session_row, summarise
from evolve.skill_loop import SkillLoop

HERE = Path(__file__).resolve().parents[1]
PREFIXES = {"wh_shelf": "wh_shelf", "wh_picking": "wh_picking"}


def write_result(folder, stages, status="model_done", credits=2.5, wall=600.0):
    folder.mkdir(parents=True)
    objects = {f"o{i}": {"stage": s, "reason": None if s == 3 else "outside_target", "dropped": False}
               for i, s in enumerate(stages)}
    (folder / "result.json").write_text(json.dumps({
        "success": all(s == 3 for s in stages), "stage": sum(stages) / len(stages), "objects": objects,
        "status": status, "steps": 12, "customizations": 1, "wall_seconds": wall, "cost": {"credits": credits}}))


def test_a_session_scores_its_placed_share_and_a_missing_one_is_a_crash(tmp_path):
    write_result(tmp_path / "train" / "wh_proc_00" / "r0", [3, 1, 0, 3])
    row = session_row("wh_proc_00", "r0", tmp_path / "train" / "wh_proc_00" / "r0")
    assert row["placed"] == 0.5 and abs(row["stage"] - 1.75) < 1e-9 and not row["crashed"] and row["credits"] == 2.5
    rows = read_split(tmp_path / "train", [("wh_proc_00", "r0"), ("wh_proc_00", "r1"), ("wh_shelf_02", "r0")])
    assert sorted(rows) == ["wh_proc_00__r0", "wh_proc_00__r1", "wh_shelf_02__r0"]
    assert rows["wh_proc_00__r1"]["crashed"] and rows["wh_proc_00__r1"]["placed"] == 0.0


def test_variants_of_one_scanned_layout_are_one_cluster():
    assert layout_cluster("train/wh_shelf_03__r1", PREFIXES) == "wh_shelf"
    assert layout_cluster("wh_picking_original__r0", PREFIXES) == "wh_picking"
    assert layout_cluster("val/wh_proc_07__r1", PREFIXES) == "wh_proc_07"


def test_summary_has_what_the_judge_and_the_search_read(tmp_path):
    write_result(tmp_path / "wh_proc_00" / "r0", [3, 3])
    write_result(tmp_path / "wh_shelf_01" / "r0", [0, 3])
    rows = read_split(tmp_path, [("wh_proc_00", "r0"), ("wh_shelf_01", "r0"), ("wh_shelf_01", "r1")])
    s = summarise(rows, lambda e: layout_cluster(e, PREFIXES))
    assert s["episodes"] == 3 and s["errors"] == 1 and s["placed"] == round(1.5 / 3, 3)
    assert s["placed_values"]["wh_proc_00__r0"] == 1.0 and s["held_values"]["wh_proc_00__r0"] == 1.0
    assert s["placed_by_layout"] == {"wh_proc_00": "2/2", "wh_shelf": "1/3"}
    assert s["untimed"] == ["wh_shelf_01__r1"] and set(s["by_task"]) == {"wh_proc_00", "wh_shelf_01"}


def test_placed_objective_counts_shared_layout_variants_as_one_sample():
    def side(placed):
        return {"placed_values": placed, "values": dict(placed), "held_values": {}, "seconds": {}, "credit_values": {},
                "episodes": len(placed), "errors": 0}
    episodes = [f"wh_shelf_{i:02d}__r0" for i in range(4)] + [f"wh_proc_{i:02d}__r0" for i in range(2)]
    parent = {"train": side(dict.fromkeys(episodes, 0.0)), "val": side({"wh_proc_09__r0": 0.0, "wh_proc_10__r0": 0.0})}
    child = {"train": side({**dict.fromkeys(episodes, 0.0), **{e: 1.0 for e in episodes[:4]}}),
             "val": side({"wh_proc_09__r0": 0.0, "wh_proc_10__r0": 0.0})}
    by_task = skill_score.judge(parent, child, "placed_values", speed=False)
    by_layout = skill_score.judge(parent, child, "placed_values", speed=False,
                                  clusters=lambda e: layout_cluster(e, PREFIXES))
    assert by_task["metric"] == "placed"
    assert by_layout["train"][1] > by_task["train"][1]  # four shelf variants are one sample, not four


def test_lost_sessions_count_as_failures_for_the_placed_objective():
    parent = {"placed_values": {"a": 1.0, "b": 1.0}}
    child = skill_score.lost_as_failures(parent, {"placed_values": {"a": 1.0}})
    assert child["placed_values"] == {"a": 1.0, "b": 0.0} and child["lost"] == ["b"]


class Screen(E2EMixin, SkillLoop):
    def __init__(self, editable):
        self.editable, self.skill = tuple(editable), "x"


def test_editable_screen_also_refuses_frozen_files():
    screen = Screen(["src/homebody/skills/dex1/*", "src/homebody/harness/*", "tests/*"])
    assert screen.allowed(["src/homebody/harness/env.py", "src/homebody/skills/dex1/pick.py"]) == []
    assert screen.allowed(["tests/test_session.py", "src/homebody/session/coder.py"]) == \
        ["src/homebody/session/coder.py", "tests/test_session.py"]


def test_the_profile_and_its_prompts_fit_together():
    profile = tomllib.loads((HERE / "config.toml").read_text())["e2e"]["demo-wh"]
    assert set(profile["suite"]) == {"train", "val", "test"} and profile["objective"] == "placed"
    operators = {"workspace", "skill", "source", "diagnosis", "mechanism", "parent", "parent_results", "other",
                 "other_results", "other_diff", "ruff", "python", "editable"}
    supplied = {"brief": {"objective", "editable", "repeats"}, "diagnose": {"workspace", "skill", "source"},
                "lessons": {"skill", "history", "lessons"}}
    for name in ("brief", "diagnose", "lessons", "propose", "diff", "crossover", "rewrite"):
        text = (HERE / "prompts" / f"{profile['prompts']}_{name}.md").read_text()
        used = {field for _, field, _, _ in string.Formatter().parse(text) if field}
        assert used <= supplied.get(name, operators), (name, used - supplied.get(name, operators))
