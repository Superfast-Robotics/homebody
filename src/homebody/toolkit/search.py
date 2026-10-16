"""Cooperative planning deadlines, usable by restricted skill modules."""
from time import monotonic


class SearchBudget:
    """A callable stop gate that reserves wall time for motion and reporting.

    Checks bound cooperative work, not a single outstanding IK/collision operation.
    A child can stop a costly branch without exhausting its parent's alternatives.
    """

    def __init__(self, seconds, *, reserve=0., cancelled=lambda: False, clock=monotonic):
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + max(0., seconds - reserve)
        self.cancelled = cancelled

    def __call__(self):
        return self.cancelled() or self.clock() >= self.deadline

    def child(self, seconds):
        return SearchBudget(min(seconds, max(0., self.deadline - self.clock())),
                            cancelled=self, clock=self.clock)

    def report(self):
        return {"elapsed_seconds": self.clock() - self.started,
                "budget_exhausted": self.clock() >= self.deadline,
                "cancelled": bool(self.cancelled())}
