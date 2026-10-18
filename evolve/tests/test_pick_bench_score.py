"""The pick benchmark's score: held counts only after the walk the episode asks for."""
from evolve import skill_bench


def rows(start_z, end_z, hands=("right",), supports=()):
    def obj(z, h, s):
        return {"corners": [[0.0, 0.0, z]], "hand_contacts": list(h), "support_contacts": list(s)}
    return [{"objects": {"cup": obj(start_z, (), ("table",))}}, {"objects": {"cup": obj(end_z, hands, supports)}}]


EPISODE = {"step": {"fixture_id": "cup", "carry_goal": [1.0, 0.0, 0.0]}}


def test_an_object_in_hand_after_the_walk_is_held():
    assert skill_bench.score_pick(rows(0.80, 0.95), EPISODE, {"carry": {"code": "OK"}})["stage"] == 2


def test_an_object_in_hand_without_the_walk_is_only_lifted():
    score = skill_bench.score_pick(rows(0.80, 0.95), EPISODE, {"carry": None})
    assert score["held"] and not score["walked"] and score["stage"] == 1


def test_a_dropped_object_after_the_walk_is_lifted_and_an_untouched_one_is_zero():
    dropped = rows(0.80, 0.80, hands=(), supports=("floor",))
    dropped[1]["objects"]["cup"]["hand_contacts"] = []
    lifted_once = [*rows(0.80, 0.95), dropped[1]]
    assert skill_bench.score_pick(lifted_once, EPISODE, {"carry": {"code": "CARRY_CLEARED"}})["stage"] == 1
    assert skill_bench.score_pick(rows(0.80, 0.80, hands=()), EPISODE, {"carry": None})["stage"] == 0
