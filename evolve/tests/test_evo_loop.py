"""The evolutionary loop's pure parts: Pareto front, parent weights, the model bandit, inert fingerprints."""
import math

from evolve.evo_loop import dominates, fingerprint, parent_weights, pareto_front, reward, ucb1


def test_the_pareto_front_keeps_versions_best_on_some_episode():
    values = {"a": {"e1": 1.0, "e2": 0.0}, "b": {"e1": 0.0, "e2": 1.0}, "c": {"e1": 0.0, "e2": 0.5}}
    assert dominates(values["b"], values["c"]) and not dominates(values["a"], values["b"])
    assert sorted(pareto_front(values)) == ["a", "b"]


def test_parent_weights_favour_value_and_few_offspring():
    weights = parent_weights({"best": 0.5, "mid": 0.4, "low": 0.1}, {"best": 3})
    assert weights["mid"] > weights["low"]
    assert weights["mid"] > weights["best"]  # three offspring already: the under-explored branch is preferred
    assert parent_weights({"best": 0.5, "mid": 0.4}, {})["best"] > parent_weights({"best": 0.5, "mid": 0.4}, {})["mid"]


def test_the_bandit_tries_every_model_then_prefers_the_one_that_pays():
    assert ucb1({}, ("astra", "sol")) == "astra"
    assert ucb1({"astra": [1, 0.0]}, ("astra", "sol")) == "sol"
    assert ucb1({"astra": [10, 0.0], "sol": [10, 3.0]}, ("astra", "sol")) == "sol"
    assert reward(-0.2) == 0.0 and reward(0.1) == math.exp(0.1) - 1


def test_a_candidate_that_changed_nothing_has_its_parents_fingerprint():
    row = {"stage": 0, "call": {"code": "ARM_NOT_REACHED", "message": "late"}, "simulation_seconds": 48.04}
    assert fingerprint(row) == fingerprint({**row, "simulation_seconds": 48.01})
    assert fingerprint(row) != fingerprint({**row, "stage": 1})


def test_hand_contacts_summarise_what_the_hand_touched_for_the_diagnosis():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("skill_bench", Path(__file__).resolve().parents[1] / "skill_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rows = [{"time": 1.0, "hand_contacts_detailed": [{"geoms": ["jar_7__collision", "right_hand_finger_1_col"],
                                                     "contact_force": [3.0, 0, 0]}]},
            {"time": 1.5, "hand_contacts_detailed": [{"geoms": ["right_hand_finger_1_col", "jar_7__collision"],
                                                     "contact_force": [-8.0, 0, 0]}]},
            {"time": 2.0, "hand_contacts_detailed": [{"geoms": ["right_hand_palm_col", "left_hand_palm_col"]}]}]
    assert bench.hand_contacts(rows) == [{"with": "jar_7__collision", "hand_part": "right_hand_finger_1_col",
                                          "first": 1.0, "last": 1.5, "samples": 2, "max_force_n": 8.0}]


def test_held_objective_with_warehouse_weight_ignores_partial_credit():
    from evolve import skill_score

    def side(stages):
        rows = {e: {"stage": s, "family": "f", "success": s == 2} for e, s in stages.items()}
        return skill_score.summary(rows)
    parent = {"train": side({"whtrain_a": 0, "whtrain_b": 0, "pk_c": 2, "pk_d": 0}),
              "val": side({"whval_e": 0, "pk_f": 2})}
    lifted = {"train": side({"whtrain_a": 1, "whtrain_b": 1, "pk_c": 2, "pk_d": 1}),
              "val": side({"whval_e": 1, "pk_f": 2})}
    assert skill_score.judge(parent, lifted)["pooled"][0] > 0  # more lifts raise stage value
    held = skill_score.judge(parent, lifted, "held_values", {"wh": 2.0})
    assert held["pooled"] == (0.0, 0.0) and not held["ok"] and held["metric"] == "held"
    assert parent["train"]["held_by_group"] == {"generated": "1/2", "warehouse": "0/2"}
    assert skill_score.weight("train/whtrain_a", {"wh": 2.0}) == 2.0 and skill_score.weight("pk_c", {"wh": 2.0}) == 1.0


def test_unweighted_paired_matches_the_plain_standard_error():
    from evolve import skill_score
    parent = {"values": {"a": 0.0, "b": 0.5, "c": 1.0}}
    child = {"values": {"a": 1.0, "b": 0.5, "c": 1.0}}
    mean, se = skill_score.paired(parent, child, "values")
    assert abs(mean - 1 / 3) < 1e-9 and abs(se - (1 / 3) ** 0.5 / 3 ** 0.5) < 1e-9


def test_the_proposer_reads_verdicts_without_validation_numbers():
    from evolve.skill_loop import public_reason
    entry = {"reason": "no clear gain: value pooled +0.01; train +0.02, val -0.30",
             "verdict": {"basis": "no clear gain", "metric": "held", "train": (0.02, 0.01), "val": (-0.3, 0.1),
                         "pooled": (0.01, 0.02), "time": 0.1}}
    text = public_reason(entry)
    assert "train +0.020" in text and "val" not in text and "-0.3" not in text


def test_a_carry_counts_only_when_every_leg_arrived():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("skill_bench", Path(__file__).resolve().parents[1] / "skill_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.physics_verdict = lambda rows, fixture: (True, True, 0.2)
    episode = {"step": {"fixture_id": "x", "carry_goal": [1, 0, 0]}}
    assert bench.score_pick([], episode, {"carry": {"code": "OK", "completed": True}})["stage"] == 2
    assert bench.score_pick([], episode, {"carry": {"code": "CARRY_CLEARED", "completed": False}})["stage"] == 1
