"""base_pose: (x, y, yaw) in the map frame, metres and radians, from lidar registration."""
from __future__ import annotations

import numpy as np

from homebody.primitives.navigation.contracts import TermSpec

SPEC = TermSpec(
    "base_pose", (3,), "f64", "map", "m,rad", staleness_s=0.3,
    notes="base translates along its heading during a turn",
)


def measured(handle):
    """(x, y, yaw) in the map frame, from lidar registration on the G1."""
    value = np.asarray(handle.base_pose(), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"base_pose must be finite (3,), got {value}")
    return value
