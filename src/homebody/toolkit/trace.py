"""Structured decision traces: what a skill tried, why, and what it measured.

The session records every event, so these land in the run's events.jsonl next to the skill's
stages and results, where whoever revises the skill can read them back.
"""
import numpy as np

MAX_FIELD_CHARS = 400


def _plain(value):
    if isinstance(value, np.ndarray):
        return np.round(value, 4).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value if isinstance(value, (int, float, bool, str, type(None))) else str(value)[:MAX_FIELD_CHARS]


def trace(ctx, what, **data):
    """Emit a skill_trace event, e.g. trace(ctx, "grasp_search", tried=40, reasons={...})."""
    ctx.emit({"type": "skill_trace", "skill": ctx._skill_name, "what": str(what), "data": _plain(data)})
