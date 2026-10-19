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
