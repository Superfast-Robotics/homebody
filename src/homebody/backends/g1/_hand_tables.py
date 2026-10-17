# Dex3-1 hand tables; every float is the repr of the computed double.
"""The Dex3-1 hand tables and the one-DOF finger closure ray.

Left and right joint orders differ (left thumb/middle/index, right thumb/index/middle);
`HAND_CLOSED` is the unit ray scaled to the maximum grip.
"""
from __future__ import annotations

import numpy as np

HAND_JOINTS: dict[str, list[str]] = {
    "left": [
        'left_hand_thumb_0_joint',
        'left_hand_thumb_1_joint',
        'left_hand_thumb_2_joint',
        'left_hand_middle_0_joint',
        'left_hand_middle_1_joint',
        'left_hand_index_0_joint',
        'left_hand_index_1_joint',
    ],
    "right": [
        'right_hand_thumb_0_joint',
        'right_hand_thumb_1_joint',
        'right_hand_thumb_2_joint',
        'right_hand_index_0_joint',
        'right_hand_index_1_joint',
        'right_hand_middle_0_joint',
        'right_hand_middle_1_joint',
    ],
}
HAND_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
HAND_CLOSED: dict[str, np.ndarray] = {
    "left": np.array([-0.0, 0.9099999999999999, 0.9099999999999999, -0.78, -1.56, -0.78, -1.56], dtype=float),
    "right": np.array([0.0, -0.9099999999999999, -0.9099999999999999, 0.78, 1.56, 0.78, 1.56], dtype=float),
}
HAND_MAX_RATE = 2.0
THUMB_IDX = (0, 1, 2)
HAND_CLOSE_FRACTION = 0.8
THUMB_CLOSE_FRACTION = 0.7
GAINS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "track": (np.array([400.0, 400.0, 250.0, 250.0, 80.0, 80.0, 80.0], dtype=float),
        np.array([4.0, 4.0, 3.2, 3.2, 1.8, 1.8, 1.8], dtype=float)),
    "grasp": (np.array([150.0, 150.0, 100.0, 100.0, 40.0, 40.0, 40.0], dtype=float),
        np.array([2.5, 2.5, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=float)),
    "soft": (np.array([42.8, 42.8, 42.8, 42.8, 42.8, 50.3, 50.3], dtype=float),
        np.array([0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068], dtype=float)),
    "sim": (np.array([800.0, 800.0, 320.0, 320.0, 160.0, 160.0, 160.0], dtype=float),
        np.array([15.0, 15.0, 6.0, 6.0, 6.0, 6.0, 6.0], dtype=float)),
}
ARM_LIMITS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "left": (np.array([-3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558], dtype=float),
        np.array([2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558], dtype=float)),
    "right": (np.array([-3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558], dtype=float),
        np.array([2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558], dtype=float)),
}
LIMIT_MARGIN = 0.02
ARM_EFFORT = np.array([15.0, 15.0, 15.0, 15.0, 15.0, 3.0, 3.0], dtype=float)


def finger_angles(grip: float, side: str = "right") -> np.ndarray:
    """Normalised grip (0..1) -> the one-DOF finger closure ray, the thumb scaled by
    THUMB_CLOSE_FRACTION."""
    if side not in HAND_CLOSED:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    g = float(np.clip(grip, 0.0, 1.0)) * float(np.clip(HAND_CLOSE_FRACTION, 0.0, 1.0))
    out = HAND_OPEN + g * (HAND_CLOSED[side] - HAND_OPEN)
    f = float(np.clip(THUMB_CLOSE_FRACTION, 0.0, 1.0))
    if f < 1.0:
        idx = list(THUMB_IDX)
        out[idx] = HAND_OPEN[idx] + (g * f) * (HAND_CLOSED[side][idx] - HAND_OPEN[idx])
    return out


def grip_from_hand_q(q: np.ndarray, side: str = "right") -> float:
    """Project a measured Dex3 pose onto the closure ray: 0 open, 1 at the commanded close
    pose."""
    q = np.asarray(q, dtype=float).reshape(7)
    if side not in HAND_CLOSED:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    ray = finger_angles(1.0, side) - HAND_OPEN
    denom = float(ray @ ray)
    return float(np.clip(((q - HAND_OPEN) @ ray) / max(denom, 1e-12), 0.0, 1.0))
