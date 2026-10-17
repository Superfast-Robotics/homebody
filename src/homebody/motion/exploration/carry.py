"""Carry continuity checks shared by the lower-body skills."""
from __future__ import annotations

from homebody.primitives.navigation.observations.gripper_state import UNKNOWN


def _held_snapshot(ctx) -> dict:
    g = ctx.observe("gripper_state")
    return {"L": g.held_l, "R": g.held_r}


def _carry_check(ctx, before: dict, where: str) -> dict:
    """Log a held-to-empty gripper reading mid-drive without acting on it."""
    after = _held_snapshot(ctx)
    for hand in ("L", "R"):
        prev, now = before[hand], after[hand]
        if prev is not None and prev is not UNKNOWN and now is None:
            ctx.log("carry_lost", {"hand": hand, "object": str(prev), "at": where,
                                   "then": "walking on: the hand's verdict does not stop a drive"})
    return after
