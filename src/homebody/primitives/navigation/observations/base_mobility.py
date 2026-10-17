"""base_mobility: whether the base can move now.

True when the walking policy is deployed and past its initialization ramp; False while the
body is held, where a lower_body command is refused.
"""
from __future__ import annotations

from homebody.primitives.navigation.contracts import TermSpec

SPEC = TermSpec(
    "base_mobility", (), "bool", "base", "-", staleness_s=0.3,
    notes="whether the body policy will move the base",
)


def measured(handle):
    """Whether a lower_body command would move the base, from the body policy deployed on the G1."""
    return bool(handle.base_mobility())
