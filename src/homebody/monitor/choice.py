"""The activity the monitor shows: the decision model's most probable label, held until another label is the most
probable in `votes` of the last `window` readings of the goal (monitor.choice_votes of monitor.choice_window), so one noisy reading never switches
it. A new goal starts from its first reading; its labels are its own vocabulary's."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Choice:
    """The held label, its probability in the newest reading, and how many of the window's readings voted for it."""
    label: str | None
    probability: float | None
    votes: int
    window: int


class ActivityChoice:
    def __init__(self, window: int, votes: int):
        if not 1 <= votes <= window:
            raise ValueError("the choice needs 1 <= votes <= window")
        self.window, self.votes = window, votes
        self.goal, self.held = None, None
        self.tops: deque = deque(maxlen=window)

    def update(self, goal_id: str, activity: dict) -> Choice:
        """The choice after one reading of GOAL_ID whose activity question answered ACTIVITY (label -> probability);
        a reading without an answer keeps the choice and casts no vote."""
        if goal_id != self.goal:
            self.goal, self.held = goal_id, None
            self.tops.clear()
        if activity:
            top = max(activity, key=lambda name: activity[name])
            self.tops.append(top)
            if self.held is None or (top != self.held and self.tops.count(top) >= self.votes):
                self.held = top
        probability = activity.get(self.held) if activity and self.held is not None else None
        return Choice(self.held, probability, self.tops.count(self.held) if self.held else 0, len(self.tops))
