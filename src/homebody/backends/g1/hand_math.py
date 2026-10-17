"""The Dex3-1 closure ray: hand math shared by skills and drivers; the constants live in
`_hand_tables.py`."""
from __future__ import annotations

import math
import re

import numpy as np

from homebody.backends.g1._hand_tables import (
    ARM_EFFORT,
    ARM_LIMITS,
    GAINS,
    HAND_CLOSED,
    HAND_JOINTS,
    HAND_MAX_RATE,
    HAND_OPEN,
    LIMIT_MARGIN,
    THUMB_IDX,
    finger_angles,
    grip_from_hand_q,
)
from homebody.backends.g1.assets import ROBOT_URDF

__all__ = [
    "ARM_EFFORT",
    "ARM_LIMITS",
    "GAINS",
    "HAND_CLOSED",
    "HAND_JOINTS",
    "HAND_MAX_RATE",
    "HAND_OPEN",
    "LIMIT_MARGIN",
    "finger_angles",
    "grip_from_hand_q",
    "hand_open_bound",
    "hand_q_from_grip",
]


def hand_q_from_grip(grip: float, side: str) -> np.ndarray:
    """The close pose for a grip in 0..1: `finger_angles`, the one-DOF ray at
    HAND_CLOSE_FRACTION with the thumb at THUMB_CLOSE_FRACTION."""
    return np.asarray(finger_angles(float(grip), side), dtype=float)


THUMB_OPEN_MARGIN_RAD = 0.02   # an open command stops this far inside the URDF stop
_OPEN_BOUND: dict = {}


def hand_open_bound(side: str) -> np.ndarray:
    """(7,) the open edge of a hand's command envelope: HAND_OPEN, except a thumb joint that
    closes off zero opens past zero to its URDF limit less THUMB_OPEN_MARGIN_RAD."""
    if side in _OPEN_BOUND:
        return _OPEN_BOUND[side].copy()
    urdf = ROBOT_URDF.read_text()
    out = np.asarray(HAND_OPEN, dtype=float).copy()
    closed = np.asarray(HAND_CLOSED[side], dtype=float)
    for i in THUMB_IDX:
        if closed[i] == 0.0:
            continue
        joint = re.search(rf'<joint name="{re.escape(HAND_JOINTS[side][i])}"[^>]*>(.*?)</joint>',
                          urdf, re.DOTALL)
        limit = re.search(r'<limit([^>]*)/?>', joint.group(1)) if joint else None
        if limit is None:
            continue
        bounds = dict(re.findall(r'(lower|upper)="([^"]+)"', limit.group(1)))
        stop = float(bounds["upper"] if closed[i] < 0.0 else bounds["lower"])
        if stop * closed[i] < 0.0:            # the stop lies on the open side of zero
            out[i] = math.copysign(max(0.0, abs(stop) - THUMB_OPEN_MARGIN_RAD), stop)
    _OPEN_BOUND[side] = out
    return out.copy()
