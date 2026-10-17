"""The step inside a skill's current phase, for the status line.

`ctx.progress` names the phase (validate, plan, execute...); `stage` names the part of it the
skill is on right now (which leg, which try) and why.
"""
from __future__ import annotations


def stage(ctx, what: str, why: str = "", i: int | None = None, of: int | None = None) -> None:
    """`skill.stage`: `what` the skill is doing now, `why`, and `i` of `of` when it is one of a
    series (legs, tries)."""
    ctx.log("skill.stage", {"what": what, "why": why, "i": i, "of": of})
