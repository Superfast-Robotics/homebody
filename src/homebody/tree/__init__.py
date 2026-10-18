"""Behavior trees over HomeBody's skills: a blocking tree (an action is one whole skill call, conditions are checked
between actions; a skill's own per-tick monitors, such as navigate's carry check, guard it while it runs), skill
contracts as conditions, and templates with reactive fallbacks that recover without a model decision.

    tree = templates.pick_and_carry(...)
    ok = tree.run(Board(call=..., state=...))
"""
from .core import Action, Board, Condition, Fallback, Retry, Sequence
from .templates import pick_and_carry

__all__ = ["Action", "Board", "Condition", "Fallback", "Retry", "Sequence", "pick_and_carry"]
