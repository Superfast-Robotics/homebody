"""Skill-benchmark scoring: episodes read from result.json, setup failures left out, versions
compared episode by episode."""
import json

from evolve import skill_score


def write(tmp_path, split, episode, **result):
    folder = tmp_path / split / episode
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"episode": episode, "family": "bin", **result}))


def test_setup_failures_are_left_out_and_outcomes_are_counted(tmp_path):
    write(tmp_path, "train", "a", stage=2, success=True, pick={"code": "OK"}, timing={"pick_wall": 30})
    write(tmp_path, "train", "b", stage=1, success=False, pick={"code": "OK"}, timing={"pick_wall": 50})
    write(tmp_path, "train", "c", stage=0, success=False, pick={"code": "NO_GRASP"})
    write(tmp_path, "train", "d", setup_failed="object not visible from the stance")
    summary = skill_score.summary(skill_score.read(tmp_path / "train"))
    assert summary["episodes"] == 3 and summary["value"] == 0.5
    assert summary["outcomes"] == {"OK (held)": 1, "OK but dropped": 1, "NO_GRASP": 1}


def summaries(values, seconds=None):
    split = {"values": values, "seconds": seconds or dict.fromkeys(values, 30.0), "skill_seconds": 30.0}
    return {"train": split, "val": split}


def test_versions_are_compared_episode_by_episode():
    parent = summaries({f"e{i}": 0.0 for i in range(20)})
    better = summaries({f"e{i}": 1.0 if i < 12 else 0.0 for i in range(20)})
    assert skill_score.judge(parent, better)["basis"] == "accuracy"
    one_lucky = summaries({f"e{i}": 1.0 if i == 0 else 0.0 for i in range(20)})
    assert not skill_score.judge(parent, one_lucky)["ok"]
    worse = summaries({f"e{i}": 0.0 for i in range(20)})
    assert skill_score.judge(better, worse)["basis"] == "accuracy dropped"


def test_a_faster_skill_with_unchanged_accuracy_is_accepted():
    values = {f"e{i}": 0.5 for i in range(10)}
    parent = summaries(values, {f"e{i}": 40.0 + i % 3 for i in range(10)})
    faster = summaries(values, {f"e{i}": 20.0 + i % 3 for i in range(10)})
    assert skill_score.judge(parent, faster)["basis"] == "time (accuracy and cost unchanged)"


def test_a_gain_on_both_splits_counts_pooled():
    def mixed(prefix, count, gained, lost):  # GAINED episodes go 0 -> 1, LOST ones 1 -> 0
        before = {f"{prefix}{i}": 1.0 if gained <= i < gained + lost else 0.0 for i in range(count)}
        after = {f"{prefix}{i}": 1.0 if i < gained else 0.0 for i in range(count)}
        return before, after
    (t0, t1), (v0, v1) = mixed("t", 40, 6, 2), mixed("v", 20, 3, 1)
    parent = {"train": {"values": t0, "seconds": {}}, "val": {"values": v0, "seconds": {}}}
    child = {"train": {"values": t1, "seconds": {}}, "val": {"values": v1, "seconds": {}}}
    verdict = skill_score.judge(parent, child)
    assert verdict["train"][0] - 1.5 * verdict["train"][1] < 0  # too small on train alone
    assert verdict["ok"] and verdict["basis"] == "accuracy"  # but clear over all 60 episodes


def test_a_much_slower_version_needs_a_stronger_gain():
    def side(changed, seconds):
        train = {f"t{i}": 0.0 for i in range(40)}
        if changed:  # six episodes gained, one lost: about 1.9 standard errors pooled
            train.update({f"t{i}": 1.0 for i in range(6)}, t39=0.0)
        else:
            train["t39"] = 1.0
        val = {f"v{i}": 0.0 for i in range(40)}
        return {"train": {"values": train, "seconds": dict.fromkeys(train, seconds)},
                "val": {"values": val, "seconds": dict.fromkeys(val, seconds)}}
    parent = side(False, 60.0)
    assert skill_score.judge(parent, side(True, 60.0))["ok"]
    verdict = skill_score.judge(parent, side(True, 100.0))
    assert not verdict["ok"] and "too small" in verdict["basis"] and verdict["time"] > 0.6


def test_equal_accuracy_with_much_cheaper_coder_calls_is_accepted():
    def side(credit):
        values = {f"e{i}": 0.5 for i in range(20)}
        split = {"values": values, "seconds": dict.fromkeys(values, 60.0),
                 "credit_values": {f"e{i}": credit + i % 3 * 0.1 for i in range(20)}}
        return {"train": split, "val": split}
    assert skill_score.judge(side(5.0), side(2.0))["basis"] == "cost (accuracy unchanged)"
    assert not skill_score.judge(side(5.0), side(4.8))["ok"]


def test_a_round_is_promising_only_with_an_accepted_or_clearly_positive_candidate():
    from evolve.skill_score import promising
    assert promising([{"status": "accepted", "verdict": {"pooled": (0.01, 0.03)}}])
    assert promising([{"status": "rejected", "verdict": {"pooled": (0.032, 0.030)}}])  # more than one se up
    assert not promising([{"status": "rejected", "verdict": {"pooled": (0.014, 0.030)}},
                          {"status": "rejected", "reason": "edits outside the skill"}])
    assert not promising([])


def test_a_crashed_episode_counts_as_a_failure_not_a_skip(tmp_path):
    write(tmp_path, "train", "a", stage=2, success=True, call={"code": "OK"})
    write(tmp_path, "train", "b", error="TypeError: boom")
    write(tmp_path, "train", "c", setup_failed="stance navigation refused")
    results = skill_score.read(tmp_path / "train")
    assert set(results) == {"a", "b"} and results["b"]["stage"] == 0 and not results["b"]["success"]
    assert skill_score.summary(results)["errors"] == 1


def test_a_candidate_that_crashes_in_many_episodes_is_rejected():
    def side(values, errors):
        split = {"values": values, "seconds": {k: 10.0 for k in values}, "credit_values": {k: 1.0 for k in values},
                 "episodes": len(values), "errors": errors}
        return {"train": split, "val": {**split, "errors": 0}}
    parent = side({f"e{i}": 1.0 for i in range(20)}, 0)
    child = side({f"e{i}": 1.0 if i < 2 else 0.0 for i in range(20)}, 18)
    verdict = skill_score.judge(parent, child)
    assert not verdict["ok"] and "crashed" in verdict["basis"]
