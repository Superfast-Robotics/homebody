"""The end-to-end loop's pure parts: session rows, missing sessions, summaries, layout clusters, judging by placed
objects, the editable screen and the prompts."""
import json
import string
import tomllib
from pathlib import Path

from evolve import skill_score
from evolve.e2e_loop import (E2EMixin, host_of, layout_cluster, read_split, session_row, summarise,
                              unsafe)
from evolve.skill_loop import SkillLoop

HERE = Path(__file__).resolve().parents[1]
PREFIXES = {"wh_shelf": "wh_shelf", "wh_picking": "wh_picking"}


def write_result(folder, stages, status="model_done", credits=2.5, wall=600.0, exit_code="0", disturbed=()):
    folder.mkdir(parents=True)
    objects = {f"o{i}": {"stage": s, "reason": None if s == 3 else "outside_target", "dropped": False}
               for i, s in enumerate(stages)}
    (folder / "result.json").write_text(json.dumps({
        "success": all(s == 3 for s in stages), "stage": sum(stages) / len(stages), "objects": objects,
        "status": status, "steps": 12, "customizations": 1, "wall_seconds": wall, "cost": {"credits": credits},
        "disturbed": list(disturbed), "invalid": []}))
    (folder / "usage.json").write_text(json.dumps({"credits": credits}))
    (folder / "exit_code").write_text(exit_code + "\n")


def test_a_session_scores_its_placed_share_and_a_missing_one_is_infrastructure(tmp_path):
    write_result(tmp_path / "train" / "wh_proc_00" / "r0", [3, 1, 0, 3])
    row = session_row("wh_proc_00", "r0", tmp_path / "train" / "wh_proc_00" / "r0", ["o0", "o1", "o2", "o3"])
    assert row["placed"] == 0.5 and abs(row["stage"] - 1.75) < 1e-9 and not row["crashed"] and row["credits"] == 2.5
    rows = read_split(tmp_path / "train", [("wh_proc_00", "r0"), ("wh_proc_00", "r1"), ("wh_shelf_02", "r0")])
    assert sorted(rows) == ["wh_proc_00__r0", "wh_proc_00__r1", "wh_shelf_02__r0"]
    assert rows["wh_proc_00__r1"]["infra"] and not rows["wh_proc_00__r1"]["crashed"]


def test_a_crashed_session_counts_as_a_crash_but_keeps_its_credits(tmp_path):
    folder = tmp_path / "t" / "r0"
    write_result(folder, [3, 0], status="error", credits=4.0)
    row = session_row("t", "r0", folder)
    assert row["crashed"] and not row["infra"] and row["credits"] == 4.0 and row["placed"] == 0.5
    (folder / "result.json").unlink()
    (folder / "exit_code").write_text("124\n")
    row = session_row("t", "r0", folder)
    assert row["crashed"] and row["status"] == "timed out" and row["credits"] == 4.0 and row["seconds"] == 7200


def test_results_that_cannot_be_judged_are_infrastructure(tmp_path):
    folder = tmp_path / "t" / "r0"
    write_result(folder, [3, 3])
    assert session_row("t", "r0", folder, ["o0", "other"])["infra"]  # scores objects the task does not name
    write_result(tmp_path / "p" / "r0", [3], status="provider_error")
    assert session_row("p", "r0", tmp_path / "p" / "r0")["infra"]
    (tmp_path / "s" / "r0").mkdir(parents=True)
    (tmp_path / "s" / "r0" / "skipped").write_text("stopped: budget")
    assert session_row("s", "r0", tmp_path / "s" / "r0")["infra"]
    (tmp_path / "u" / "r0").mkdir(parents=True)
    (tmp_path / "u" / "r0" / "exit_code").write_text("0\n")  # ran, but the scorer failed
    assert session_row("u", "r0", tmp_path / "u" / "r0")["status"] == "unscored"


def test_disturbing_other_products_costs_a_share_of_an_object(tmp_path):
    write_result(tmp_path / "t" / "r0", [3, 3, 0, 0], disturbed=["can_2", "jar_5"])
    assert session_row("t", "r0", tmp_path / "t" / "r0", penalty=0.25)["placed"] == (2 - 0.5) / 4
    write_result(tmp_path / "z" / "r0", [0, 0], disturbed=["a", "b", "c"])
    assert session_row("z", "r0", tmp_path / "z" / "r0", penalty=0.25)["placed"] == 0.0


def test_every_version_runs_a_task_repeat_on_the_same_host_and_repeats_split_hosts():
    hosts = [host_of(task, repeat, 2) for task in range(16) for repeat in range(2)]
    assert hosts.count(0) == hosts.count(1) == 16
    assert all(host_of(t, 0, 2) != host_of(t, 1, 2) for t in range(16))


def test_the_safety_screen_flags_system_access_and_settings_writes():
    found = unsafe(["import subprocess", "    open(path, 'w').write(x)", "settings.grasp.squeeze = 2",
                    "    x = settings.grasp.squeeze == 2  # compare", "    pose = env.observe()"])
    assert set(found) == {"imports a system module", "opens a file for writing", "writes the settings"}
    assert not unsafe(["    reply = env.plan_path(pres, goals)  # import os is only a comment"])


def test_variants_of_one_scanned_layout_are_one_cluster():
    assert layout_cluster("train/wh_shelf_03__r1", PREFIXES) == "wh_shelf"
    assert layout_cluster("wh_picking_original__r0", PREFIXES) == "wh_picking"
    assert layout_cluster("val/wh_proc_07__r1", PREFIXES) == "wh_proc_07"


def test_summary_has_what_the_judge_and_the_search_read(tmp_path):
    write_result(tmp_path / "wh_proc_00" / "r0", [3, 3])
    write_result(tmp_path / "wh_shelf_01" / "r0", [0, 3])
    write_result(tmp_path / "wh_shelf_01" / "r1", [0, 0], status="error")
    rows = read_split(tmp_path, [("wh_proc_00", "r0"), ("wh_shelf_01", "r0"), ("wh_shelf_01", "r1")])
    s = summarise(rows, lambda e: layout_cluster(e, PREFIXES))
    assert s["episodes"] == 3 and s["errors"] == 1 and s["placed"] == round(1.5 / 3, 3) and not s["infra"]
    assert s["placed_values"]["wh_proc_00__r0"] == 1.0 and s["held_values"]["wh_proc_00__r0"] == 1.0
    assert s["placed_by_layout"] == {"wh_proc_00": "2/2", "wh_shelf": "1/4"}
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
    assert set(profile["suite"]) == {"train", "val", "test", "legacy"} and profile["objective"] == "placed"
    assert profile["parallel"] <= 16 and profile["val_margin"] > 0 and profile["service_checks"]
    operators = {"workspace", "skill", "source", "diagnosis", "mechanism", "parent", "parent_results", "other",
                 "other_results", "other_diff", "ruff", "python", "editable"}
    supplied = {"brief": {"objective", "editable", "repeats"}, "diagnose": {"workspace", "skill", "source"},
                "lessons": {"skill", "history", "lessons"}}
    for name in ("brief", "diagnose", "lessons", "propose", "diff", "crossover", "rewrite"):
        text = (HERE / "prompts" / f"{profile['prompts']}_{name}.md").read_text()
        used = {field for _, field, _, _ in string.Formatter().parse(text) if field}
        assert used <= supplied.get(name, operators), (name, used - supplied.get(name, operators))


def test_a_preregistered_val_margin_needs_val_shown_non_inferior():
    def side(values):
        return {"placed_values": values, "values": dict(values), "held_values": {}, "seconds": {},
                "credit_values": {}, "episodes": len(values), "errors": 0}
    train = [f"wh_proc_{i:02d}__r0" for i in range(12)]
    val = [f"wh_proc_v{i:02d}__r0" for i in range(6)]
    parent = {"train": side(dict.fromkeys(train, 0.0)), "val": side(dict.fromkeys(val, 0.5))}
    child = {"train": side(dict.fromkeys(train, 0.6)),
             "val": side({**dict.fromkeys(val, 0.5), val[0]: 0.0, val[1]: 0.25})}
    assert skill_score.judge(parent, child, "placed_values", speed=False)["ok"]  # train carries the pooled gain
    held = skill_score.judge(parent, child, "placed_values", speed=False, val_margin=0.05, val_z=1.0)
    assert not held["ok"] and held["basis"].startswith("val not shown non-inferior")


def test_an_unfinished_round_resumes_and_a_barren_first_round_stays_stopped(tmp_path):
    from types import SimpleNamespace
    ran, archive = [], []
    loop = SimpleNamespace(work=tmp_path, log=lambda m: None, archive=lambda: archive,
                           completed_rounds=lambda: SkillLoop.completed_rounds(loop),
                           barren=lambda entries: E2EMixin.barren(loop, entries))
    loop.round = lambda n: (ran.append(n), archive.append({"id": f"e{n:03d}c1", "round": n, "status": "rejected"}))
    archive.append({"id": "b000", "round": 0, "status": "baseline"})
    SkillLoop.run(loop, 3)
    assert ran == [1] and (tmp_path / "STOP.json").exists()  # no accepted candidate in round 1
    SkillLoop.run(loop, 3)
    assert ran == [1]  # the stop decision persists
    (tmp_path / "STOP.json").unlink()
    (tmp_path / "rounds.json").write_text("[1]")
    archive.append({"id": "e002c1", "round": 2, "status": "rejected"})  # round 2 started, never finished
    SkillLoop.run(loop, 1)
    assert ran == [1, 2]  # round 2 is resumed, not skipped
