"""The loop's own logic: tiers, scoring of collected runs, and the acceptance rule."""
import json

from evolve import score, tiers


def test_tiers_follow_the_first_matching_rule():
    assert tiers.tier_of("src/homebody/harness/env.py") == 0
    assert tiers.tier_of("src/homebody/toolkit/reach.py") == 0
    assert tiers.tier_of("tests/harness/test_harness_api.py") == 0
    assert tiers.tier_of("src/homebody/skills/pick.py") == 1
    assert tiers.tier_of("src/homebody/motion/arm.py") == 1
    assert tiers.tier_of("src/homebody/backends/mujoco.py") == 2
    assert tiers.tier_of("evolve/loop.py") == "frozen"
    assert tiers.tier_of("tools/check_manipulation.py") == "frozen"
    assert tiers.tier_of("assets/generated/train_00/scene.xml") == "frozen"


def test_a_patch_takes_its_highest_tier_and_lists_frozen_files():
    patch = tiers.classify(["src/homebody/harness/env.py", "src/homebody/motion/arm.py", "evaluation/evaluate.py"])
    assert patch.tier == 1 and patch.frozen == ("evaluation/evaluate.py",)
    assert tiers.classify(["src/homebody/harness/API.md"]).tier == 0
    assert tiers.tier_of("evolve/score_episode.py") == "frozen"
    assert tiers.tier_of("src/homebody/session/custom_skills.py") == "frozen"
    assert tiers.tier_of("src/homebody/harness/agent_prompt.md") == 0


def run_dir(root, task, repeat, result=None, log=""):
    path = root / task / repeat
    path.mkdir(parents=True)
    (path / "run.log").write_text(log)
    if result is not None:
        (path / "result.json").write_text(json.dumps(result))


def test_stages_from_inner_results(tmp_path):
    run_dir(tmp_path, "a", "r0", {"success": True, "stage": 3.0, "objects": {"cup": {"stage": 3, "reason": None}},
                                  "customizations": 1, "status": "model_done", "wall_seconds": 90})
    run_dir(tmp_path, "a", "r1", {"success": False, "stage": 1.0, "objects": {"cup": {"stage": 1, "reason": "outside_target"}},
                                  "customizations": 4, "status": "step_limit"})
    run_dir(tmp_path, "b", "r0", log="Traceback (most recent call last):\nboom\n")
    runs = score.read_split(tmp_path)
    assert [(r.task, r.stage, r.success, r.attempts) for r in runs] == [("a", 3.0, True, 1), ("a", 1.0, False, 4),
                                                                          ("b", 0.0, False, 0)]
    assert runs[1].failures == ["cup: stage 1 (outside_target)", "session ended step_limit"]
    assert runs[2].failures[0].startswith("session crashed")
    s = score.summary(runs)
    assert s["score"] == round(4 / 9, 4) and s["success_rate"] == round(1 / 3, 4) and s["by_task"]["a"] == [3.0, 1.0]


def split(**scenes):
    runs = [stage for stages in scenes.values() for stage in stages]
    return {"score": sum(runs) / (3 * len(runs)), "by_task": scenes}


def test_acceptance_weighs_each_change_against_its_standard_error():
    parent = {"train": split(a=[1, 1, 1, 1], b=[0, 0, 0, 0]), "val": split(c=[3, 3, 2, 3])}
    clear = {"train": split(a=[3, 3, 3, 3], b=[0, 0, 0, 0]), "val": split(c=[3, 3, 3, 2])}
    assert score.accept(parent, clear)[0]
    noisy = {"train": split(a=[1, 3, 1, 1], b=[0, 0, 0, 0]), "val": split(c=[3, 3, 2, 3])}
    assert not score.accept(parent, noisy)[0]  # one lucky run is within the noise
    worse = {"train": split(a=[3, 3, 3, 3], b=[0, 0, 0, 0]), "val": split(c=[0, 0, 1, 0])}
    assert not score.accept(parent, worse)[0]
    _, d_train, d_val, se_train, _ = score.accept(parent, clear)
    assert d_train > 0.3 and se_train > 0 and d_val == 0.0


def test_only_scenes_both_ran_are_compared():
    parent = {"train": split(a=[1, 1]), "val": split(c=[3, 3])}
    child = {"train": split(a=[3, 3], new=[0, 0]), "val": split(c=[3, 3])}
    assert score.difference(parent["train"], child["train"])[0] > 0.6



def test_quota_exhaustion_is_detected_in_collected_sessions(tmp_path):
    run_dir(tmp_path, "a", "r0", log="fine")
    assert not score.quota_hit(tmp_path)
    outcome = tmp_path / "a" / "r0" / "session" / "tasks" / "task-000" / "outcome.json"
    outcome.parent.mkdir(parents=True)
    outcome.write_text('{"status": "provider_error", "reason": "Codex exited 1: You\u2019ve hit your usage limit."}')
    assert score.quota_hit(tmp_path)


def test_family_success_is_reported_and_every_run_counts(tmp_path):
    for task, repeat, result in (("train_00_shelf_return", "r0", {"stage": 3.0, "success": True}),
                                 ("train_00_shelf_return", "r1", {"stage": 1.0, "success": False, "dropped": 1}),
                                 ("src_kitchen", "r0", {"stage": 2.5, "success": False})):
        (tmp_path / task / repeat).mkdir(parents=True)
        (tmp_path / task / repeat / "result.json").write_text(json.dumps(result))
    summary = score.summary(score.read_split(tmp_path))
    assert summary["runs"] == 3 and summary["success_by_family"] == {"shelf_return": 0.5, "src_kitchen": 0.0}
