"""gripper_state: per-hand closure in [0, 1] and a held flag that can be UNKNOWN.

An outcome that depends on an UNKNOWN held state is itself unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from homebody.primitives.navigation.contracts import TermSpec


class _State(Enum):
    HELD = "HELD"
    UNKNOWN = "UNKNOWN"

    def __repr__(self) -> str:
        return self.name


HELD = _State.HELD
UNKNOWN = _State.UNKNOWN

SPEC = TermSpec(
    "gripper_state", (),
    "struct{closure:{L,R} f32 in [0,1], held:{L,R} HELD|None|UNKNOWN}",
    "joint", "-", staleness_s=0.2,
    notes="transport is physical",
)


@dataclass(frozen=True)
class GripperState:
    closure_l: float
    closure_r: float
    held_l: object  # HELD | None | UNKNOWN
    held_r: object


def measured(handle):
    """Per-hand closure in [0,1] plus an anonymous held flag that never names the object."""
    g = handle.gripper_state()
    left, right = float(g["closure_l"]), float(g["closure_r"])
    if not 0.0 <= left <= 1.0 or not 0.0 <= right <= 1.0:
        raise ValueError(f"gripper closure must be in [0,1], got left={left}, right={right}")
    return GripperState(left, right, g["held_l"], g["held_r"])
