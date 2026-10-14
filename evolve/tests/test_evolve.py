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


def test_acceptance_needs_no_drop_beyond_noise_and_one_real_rise():
    parent = {"train": {"score": 0.2}, "val": {"score": 0.3}}
    assert score.accept(parent, {"train": {"score": 0.3}, "val": {"score": 0.3}})[0]
    assert score.accept(parent, {"train": {"score": 0.3}, "val": {"score": 0.29}})[0]  # a noise-sized dip
    assert not score.accept(parent, {"train": {"score": 0.5}, "val": {"score": 0.25}})[0]
    assert not score.accept(parent, {"train": {"score": 0.2}, "val": {"score": 0.3}})[0]
    assert not score.accept(parent, {"train": {"score": 0.21}, "val": {"score": 0.31}})[0]  # only noise


def test_quota_exhaustion_is_detected_in_collected_sessions(tmp_path):
    run_dir(tmp_path, "a", "r0", log="fine")
    assert not score.quota_hit(tmp_path)
    outcome = tmp_path / "a" / "r0" / "session" / "tasks" / "task-000" / "outcome.json"
    outcome.parent.mkdir(parents=True)
    outcome.write_text('{"status": "provider_error", "reason": "Codex exited 1: You\u2019ve hit your usage limit."}')
    assert score.quota_hit(tmp_path)
