"""qpos: whole-body joint state in radians, always accompanied by joint names.

Consumers remap by name, never by position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from homebody.primitives.navigation.contracts import TermSpec

SPEC = TermSpec(
    "qpos", (29,), "f32", "joint", "rad", staleness_s=0.1,
    notes="always accompanied by joint_names; DDS freshness at 0.1 s",
)


@dataclass(frozen=True)
class JointState:
    """(29,) joint positions plus their names — inseparable by construction."""

    q: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self):
        if self.q.shape != (len(self.names),):
            raise ValueError(f"q {self.q.shape} does not match {len(self.names)} names")


def measured(handle):
    """The 29 named body joints in the AMO profile order, from the encoders on the G1."""
    q = np.asarray(handle.qpos(), dtype=np.float32)
    names = tuple(handle.joint_names())
    if q.shape != (29,) or len(names) != 29 or len(set(names)) != 29:
        raise ValueError(f"qpos requires 29 uniquely named joints, got {q.shape}/{len(names)}")
    if not np.isfinite(q).all():
        raise ValueError("qpos contains NaN/Inf")
    return JointState(q, names)
