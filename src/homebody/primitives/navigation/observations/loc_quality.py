"""loc_quality: localization fitness in [0, 1]; the drive reads it against its floor."""
from __future__ import annotations

from homebody.primitives.navigation.contracts import TermSpec

SPEC = TermSpec(
    "loc_quality", (), "f32 in [0,1]", "map", "-", staleness_s=0.3,
    notes="registration fitness; the drive reads it against its floor",
)


def measured(handle):
    """Localization health in [0,1]: registration fitness on the G1. Every drive gates on its floor."""
    value = float(handle.loc_quality())
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"loc_quality must be in [0,1], got {value}")
    return value
