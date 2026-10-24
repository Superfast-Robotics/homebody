"""The shown activity: the decision model's top label, switched only when another label wins votes of the window."""
import pytest

from homebody.monitor.choice import ActivityChoice
from homebody.monitor.targets import Reading
from homebody.ui.dashboard.monitor_feed import value


def test_a_new_label_takes_over_only_after_three_of_the_last_five_readings():
    choice = ActivityChoice(window=5, votes=3)
    shown = [choice.update("g1", {"reach": 0.7, "grasp": 0.3}).label]
    for top in ("grasp", "reach", "grasp", "grasp", "lift", "grasp"):
        shown.append(choice.update("g1", {top: 0.8, "other": 0.2}).label)
    assert shown == ["reach", "reach", "reach", "reach", "grasp", "grasp", "grasp"]
    held = choice.update("g1", {"grasp": 0.65, "lift": 0.35})
    assert (held.label, held.probability, held.votes, held.window) == ("grasp", 0.65, 4, 5)
    assert choice.update("g1", {}).label == "grasp"  # a refused answer keeps the choice and casts no vote
    assert choice.update("g2", {"walk": 0.9}).label == "walk"  # a new goal starts from its first reading


def test_the_votes_must_fit_the_window():
    with pytest.raises(ValueError):
        ActivityChoice(window=2, votes=3)


def test_the_feed_carries_the_held_activity():
    choice = ActivityChoice(window=5, votes=3)
    readings = [Reading("g1", float(t), activity=a) for t, a in enumerate(
        [{"reach": 0.9}, {"grasp": 0.8}, {"grasp": 0.8}, {"grasp": 0.7}])]
    shown = [value(r, None, None, 0.8, choice=choice) for r in readings]
    assert [s["activity"] for s in shown] == ["reach", "reach", "reach", "grasp"]
    assert (shown[-1]["activity_p"], shown[-1]["activity_votes"], shown[-1]["activity_window"]) == (0.7, 3, 4)
    assert value(readings[0], None, None, 0.8)["activity"] == "reach"  # without a choice: the reading's own top
