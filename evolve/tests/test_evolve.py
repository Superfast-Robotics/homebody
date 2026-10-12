"""The loop's own logic: tiers, scoring of collected runs, and the acceptance rule."""
import json

from evolve import score, tiers


def test_tiers_follow_the_first_matching_rule():
    assert tiers.tier_of("src/homebody/skills/pick.py") == 0
    assert tiers.tier_of("src/homebody/skills/lib/grasping.py") == 0
    assert tiers.tier_of("src/homebody/skills/contract.py") == 1
    assert tiers.tier_of("src/homebody/toolkit/reach.py") == 1
    assert tiers.tier_of("src/homebody/backends/mujoco.py") == 2
    assert tiers.tier_of("evolve/loop.py") == "frozen"
    assert tiers.tier_of("tools/check_manipulation.py") == "frozen"
    assert tiers.tier_of("assets/generated/train_00/scene.xml") == "frozen"


def test_a_patch_takes_its_highest_tier_and_lists_frozen_files():
    patch = tiers.classify(["src/homebody/skills/pick.py", "src/homebody/motion/arm.py", "evaluation/evaluate.py"])
    assert patch.tier == 1 and patch.frozen == ("evaluation/evaluate.py",)
    assert tiers.classify(["src/homebody/skills/place.py"]).tier == 0


def run_dir(root, task, repeat, log, result=None):
    path = root / task / repeat
    (path / "attempt").mkdir(parents=True)
    (path / "run.log").write_text(log)
    if result is not None:
        (path / "attempt" / "result.json").write_text(json.dumps(result))


PASS = {"status": "skills_completed", "independent_physical_pick_verified": True,
        "local_placement_score": {"success": True}, "wall_seconds": 80.0}


def test_stages_from_results_and_logs(tmp_path):
    ok = 'RESULT pick {"skill": "pick", "code": "OK", "message": "lifted"}\n'
    blocked = 'RESULT place {"skill": "place", "code": "COLLISION", "message": "link 4"}\n'
    run_dir(tmp_path, "a", "r0", ok, PASS)
    run_dir(tmp_path, "a", "r1", ok + blocked, {"status": "skill_failed", "local_placement_score": {}})
    run_dir(tmp_path, "b", "r0", "Traceback (most recent call last):\nboom\n")
    runs = score.read_split(tmp_path)
    assert [(r.task, r.stage, r.success) for r in runs] == [("a", 3, True), ("a", 1, False), ("b", 0, False)]
    assert runs[1].failures == [("place", "COLLISION", "link 4")]
    assert runs[2].failures[0][1] == "CRASHED"
    s = score.summary(runs)
    assert s["score"] == round(4 / 9, 4) and s["success_rate"] == round(1 / 3, 4) and s["by_task"]["a"] == [3, 1]


def test_acceptance_needs_no_drop_and_one_rise():
    parent = {"train": {"score": 0.2}, "val": {"score": 0.3}}
    assert score.accept(parent, {"train": {"score": 0.3}, "val": {"score": 0.3}})[0]
    assert not score.accept(parent, {"train": {"score": 0.5}, "val": {"score": 0.25}})[0]
    assert not score.accept(parent, {"train": {"score": 0.2}, "val": {"score": 0.3}})[0]
