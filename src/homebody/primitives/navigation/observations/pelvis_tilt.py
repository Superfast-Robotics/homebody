"""pelvis_tilt: the pelvis roll and pitch, radians, in the gravity frame; the G1's IMU low-passed
over the stride."""
from __future__ import annotations

import numpy as np

from homebody.primitives.navigation.contracts import TermSpec

SPEC = TermSpec(
    "pelvis_tilt", (2,), "f64", "base", "rad", staleness_s=0.3,
    notes="(roll, pitch) of the pelvis; the G1's IMU low-passed",
)


def measured(handle):
    """(roll, pitch) in radians from the base's own attitude."""
    value = np.asarray(handle.pelvis_tilt(), dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"pelvis_tilt is not finite: {value.tolist()}")
    return value
