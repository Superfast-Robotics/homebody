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


def test_scenario_words_are_the_scene_words_the_tasks_use():
    banned = gepa_open.banned_words(["Move the red jam jar to the middle bay, then the spice-jar; verify the front."])
    assert banned == ["bay", "jam", "jar", "red", "spice"]
    assert "spice" in gepa_open.lint("place_note", "Avoid the spice-jar.", banned)


def test_the_front_keeps_every_candidate_best_somewhere(tmp_path):
    (tmp_path / "records").mkdir()
    scores = {"c000": [1, 0, 0.5], "c001": [0, 1, 0.5], "c002": [0, 0, 0.5]}
    for cid, values in scores.items():
        (tmp_path / "records" / f"{cid}.jsonl").write_text(
            "".join(json.dumps({"case": f"k{i}", "score": v}) + "\n" for i, v in enumerate(values)))
    state = {"val": ["k0", "k1", "k2"], "candidates": [{"id": c, "val_mean": 0.5, "parent": None} for c in scores]}
    # c002 is best only on k2, where c000 and c001 are too: dominated, removed (gepa's remove_dominated_programs)
    assert gepa_open.front(state, tmp_path) == {"c000": 2, "c001": 2}
    state["candidates"][1]["dropped"] = True
    assert gepa_open.front(state, tmp_path) == {"c000": 3}  # on k1 c000 ties c002 at 0; c002 is dominated


def test_a_minibatch_draws_the_texts_capabilities():
    train = {f"p{i}": ["pick"] for i in range(30)} | {f"n{i}": ["navigate"] for i in range(30)}
    names = gepa_open.minibatch({"train": train}, "pick_note", random.Random(0))
    assert len(names) == 30 and all(n.startswith("p") for n in names)


def test_a_pointer_note_child_inherits_its_parents_turns_of_other_skills(tmp_path):
    (tmp_path / "records").mkdir()
    rows = [{"case": "a", "score": 1, "decision": {"skill": "navigate"}},
            {"case": "b", "score": 0, "decision": {"skill": "pick"}},
            {"case": "c", "score": 0, "decision": {"error": "bad"}}]
    (tmp_path / "records" / "c000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gepa_open.inherit(tmp_path, "c001", "c000", "pick_note", ["a", "b", "c"])
    assert set(gepa_open.records(tmp_path, "c001")) == {"a"}
    gepa_open.inherit(tmp_path, "c002", "c000", "guidance", ["a", "b", "c"])
    assert gepa_open.records(tmp_path, "c002") == {}


def test_cases_go_to_executors_in_proportion_to_their_weights_and_always_the_same():
    executors = ({"host": "a", "weight": 3}, {"host": "b", "weight": 1})
    names = [f"case-{i}" for i in range(4000)]
    share = sum(gepa_open.executor_of(n, executors)["host"] == "a" for n in names) / len(names)
    assert 0.72 < share < 0.78
    assert gepa_open.executor_of("case-7", executors) is gepa_open.executor_of("case-7", executors)


def test_a_dropped_candidates_descendants_wait_for_a_check_of_their_own():
    state = {"candidates": [{"id": "c000", "parent": None, "val_mean": 0.7, "e2e": {"val": 0.2}},
                            {"id": "c001", "parent": "c000", "val_mean": 0.8, "dropped": True, "e2e": {"val": 0.1}},
                            {"id": "c002", "parent": "c001", "val_mean": 0.9},
                            {"id": "c003", "parent": "c002", "val_mean": 0.9, "e2e": {"val": 0.3}},
                            {"id": "c004", "parent": "c003", "val_mean": 0.9},
                            {"id": "c005", "parent": "c000", "val_mean": None}]}
    assert [c["id"] for c in gepa_open.eligible(state)] == ["c000", "c003", "c004"]


def test_domination_keeps_a_best_candidate_on_every_case():
    fronts = {"a": {"x", "y"}, "b": {"y"}, "c": {"x", "z"}}
    kept = gepa_open.dominated_removed(fronts, {"x": 0.5, "y": 0.6, "z": 0.4})
    assert all(kept[k] for k in fronts) and kept["c"] == {"x"} and "z" not in kept["c"]


def test_the_pointer_group_asks_only_the_pointer_on_a_frozen_planner_turn(tmp_path, monkeypatch):
    import numpy as np
    from evolve import gepa_eval
    case = tmp_path / "cases" / "k1"
    case.mkdir(parents=True)
    labels = np.zeros((101, 101), dtype=np.int32)
    labels[40:60, 40:60] = 3
    np.savez_compressed(case / "labels.npz", labels=labels)
    (case / "prompt.txt").write_text("Current task:\nMove the can.\n")
    meta = {"scenario": "pp_x", "goal": "Pick the can.", "observation": None, "previous": None, "target": 3,
            "result": "OK", "allowed": ["pick"],
            "reference": {"skill": "pick", "text": "can", "arguments": {"point_normalized_1000": [500, 500]}}}
    (case / "case.json").write_text(json.dumps(meta))
    from PIL import Image
    Image.new("RGB", (101, 101)).save(case / "rgb.png")
    asked = []

    class Pointer:
        format = "robobrain"

        def ask(self, question, images, **kwargs):
            asked.append(question)
            return "[(500, 500)]"

    frozen = {"decision": {"skill": "pick", "text": "t", "arguments": {"point_normalized_1000": [10, 10]}},
              "pointer": {"target": "the can's body"}}
    record = gepa_eval.point_case(case, frozen, {"pick_note": "Aim at the centre."}, Pointer(), tmp_path / "out")
    assert record["score"] == 1.0 and record["families"] == ["pick"]
    assert asked[0].startswith("the can's body.") and asked[0].endswith(" Aim at the centre.")


def test_a_plan_scores_its_mean_rubric_over_both_orders_and_names_weak_criteria():
    from evolve import gepa_groups
    verdict = {"open": {"coverage": 8, "feasibility": 6, "verifiable_checks": 9, "recovery": 2, "no_invention": 9},
               "sol": {}, "preferred": "sol", "reason": "no fallbacks"}
    judged = {"verdicts": [verdict, verdict], "preferred": "sol"}
    assert gepa_groups.plan_score(judged) == 0.68
    assert "weak on recovery" in gepa_groups.plan_feedback(judged)
    assert gepa_groups.plan_score({"verdicts": []}) is None


def test_each_group_rewrites_only_its_own_texts():
    from evolve import gepa_groups
    texts = [t for g in gepa_groups.GROUPS.values() for t in g["texts"]]
    assert sorted(texts) == sorted(gepa_open.seed_texts()) and len(texts) == len(set(texts))
    assert [g for g, spec in gepa_groups.GROUPS.items() if spec["e2e"]] == ["planner", "pointer"]
