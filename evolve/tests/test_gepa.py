import json
import random

import numpy as np

from evolve import gepa_eval, gepa_open


def meta(**overrides):
    base = {"scenario": "pp_x", "goal": "Pick the can.", "observation": None, "previous": None, "target": None,
            "result": "OK", "allowed": ["navigate", "pick", "goal_done"],
            "reference": {"skill": "pick", "text": "the can", "arguments": {"point_normalized_1000": [500, 500]}}}
    return {**base, **overrides}


def test_a_case_tests_the_capabilities_its_reference_turn_shows():
    assert gepa_eval.families(meta(target=3)) == ("choose", "judge", "pick")
    failed = {"decision": {"skill": "pick", "arguments": {}}, "code": "ARM_NOT_REACHED", "message": "far"}
    assert "recover" in gepa_eval.families(meta(previous=failed))
    assert gepa_eval.families(meta(reference={"skill": "plan", "text": "", "arguments": {}})) == ()


def test_judging_and_recovery_feedback_names_the_fault():
    done = meta(reference={"skill": "goal_done", "text": "", "arguments": {}})
    assert gepa_eval.score_judge(done, {"skill": "pick"})[0] == 0.0
    score, text = gepa_eval.score_judge(meta(), {"skill": "goal_done"})
    assert score == 0.0 and "not achieved" in text
    failed = {"decision": {"skill": "pick", "arguments": {"a": 1}}, "code": "ARM_NOT_REACHED", "message": "far"}
    score, text = gepa_eval.score_recover(meta(previous=failed), {"skill": "pick", "arguments": {"a": 1}})
    assert score == 0.0 and "ARM_NOT_REACHED" in text
    assert gepa_eval.score_recover(meta(previous=failed), {"skill": "pick", "arguments": {"a": 2}})[0] == 1.0


def test_a_pick_click_is_scored_on_the_target_label(tmp_path):
    labels = np.zeros((101, 101), dtype=np.int32)
    labels[40:60, 40:60] = 3
    np.savez_compressed(tmp_path / "labels.npz", labels=labels)
    hit = {"skill": "pick", "arguments": {"point_normalized_1000": [500, 500]}}
    miss = {"skill": "pick", "arguments": {"point_normalized_1000": [100, 100]}}
    assert gepa_eval.score_pick(meta(target=3), hit, tmp_path) == (1.0, None)
    score, text = gepa_eval.score_pick(meta(target=3), miss, tmp_path)
    assert score == 0.0 and "background" in text


def test_splits_follow_the_scenario_families():
    assert [gepa_open.split_of(s) for s in ("pp_a", "pa_b", "wh_proc_train_03", "hd_c", "wh_shelf_val_01",
                                            "wh_proc_test_00")] == ["train", "train", "train", "val", "val", None]


def test_a_proposal_naming_the_scenes_or_too_long_is_refused():
    banned = ["jam", "bay"]
    assert gepa_open.lint("pick_note", "Click the centre of the object's visible top.", banned) is None
    assert "jam" in gepa_open.lint("pick_note", "Click the jam jar lid.", banned)
    assert "1.3" in gepa_open.lint("place_note", "Stay within 1.3 m.", banned)
    assert "words" in gepa_open.lint("pick_note", "word " * 61, banned)
    assert gepa_open.proposal("Here:\n```text\nNew text.\n```") == "New text."


def test_scenario_words_are_those_the_general_prompts_never_use():
    banned = gepa_open.banned_words(["Move the red jam jar to the middle bay, then place it on the shelf."])
    assert {"jam", "red"} <= set(banned) and "place" not in banned and "the" not in banned


def test_the_front_keeps_every_candidate_best_somewhere(tmp_path):
    (tmp_path / "records").mkdir()
    scores = {"c000": [1, 0, 0.5], "c001": [0, 1, 0.5], "c002": [0, 0, 0.5]}
    for cid, values in scores.items():
        (tmp_path / "records" / f"{cid}.jsonl").write_text(
            "".join(json.dumps({"case": f"k{i}", "score": v}) + "\n" for i, v in enumerate(values)))
    state = {"val": ["k0", "k1", "k2"], "candidates": [{"id": c, "val_mean": 0.5} for c in scores]}
    assert gepa_open.front(state, tmp_path) == {"c000": 2, "c001": 2, "c002": 1}
    state["candidates"][2]["dropped"] = True
    assert "c002" not in gepa_open.front(state, tmp_path)


def test_a_minibatch_draws_the_texts_capabilities():
    train = {f"p{i}": ["pick"] for i in range(30)} | {f"n{i}": ["navigate"] for i in range(30)}
    names = gepa_open.minibatch({"train": train}, "pick_note", random.Random(0))
    assert len(names) == 30 and all(n.startswith("p") for n in names)
