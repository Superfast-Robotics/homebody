"""lower_body: planar velocity (m/s, rad/s) and gait state for the walking kernel; None = unchanged.

Commands below the stepping dead-band are refused; the shared handles repeat the last command
at control rate and force zero velocity after 200 ms without a new step.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal

from homebody.primitives.navigation.contracts import ChannelSpec, Limits, Refused

# The walking policy's command envelope:
V_PLANAR_MAX = 1.0        # the largest planar speed command
VX_DEADBAND = 0.15        # below this the policy marches in place
WZ_MIN_ACCEPT = 0.15      # below this the policy under-rotates a turn to nothing
WZ_FLOOR = 0.30           # smaller accepted turn rates are clamped up to this
WZ_MAX = 1.0              # the largest turn-rate command


@dataclass(frozen=True)
class LowerBody:
    vx: float | None = None
    vy: float | None = None
    wz: float | None = None
    # "brace" freezes the stance for in-place arm work; velocities are refused in it
    state: Literal["walk", "stand", "brace"] | None = None


def clamp_lower(cmd: LowerBody, observe) -> LowerBody:
    """Pure. Refuses dead-band commands; floors clamp up; caps clamp down."""
    if cmd.state not in (None, "walk", "stand", "brace"):
        raise Refused(f"lower_body.state must be walk|stand|brace|None, got {cmd.state!r}")
    if cmd.state == "brace" and any(v not in (None, 0.0) for v in (cmd.vx, cmd.vy, cmd.wz)):
        raise Refused("lower_body.brace holds station — velocity commands are contradictory")
    vx, vy, wz = cmd.vx, cmd.vy, cmd.wz
    for name, v in (("vx", vx), ("vy", vy), ("wz", wz)):
        if v is not None and not math.isfinite(v):
            raise Refused(f"lower_body.{name} is {v} — commands must be finite")

    turning = wz is not None and wz != 0.0
    planar = [abs(v) for v in (vx, vy) if v is not None]
    moving = bool(planar) and max(planar) > 0.0

    if moving:
        if max(planar) < VX_DEADBAND and not turning:
            raise Refused(
                f"planar command below the stepping dead-band ({VX_DEADBAND} m/s): the "
                f"policy marches in place"
            )
        v = math.hypot(vx if vx is not None else 0.0, vy if vy is not None else 0.0)
        if v > V_PLANAR_MAX:  # the policy's training range
            scale = V_PLANAR_MAX / v
            vx = vx * scale if vx is not None else None
            vy = vy * scale if vy is not None else None

    if turning:
        if abs(wz) < WZ_MIN_ACCEPT:
            raise Refused(
                f"|wz|={abs(wz):.3f} below {WZ_MIN_ACCEPT} rad/s: the policy under-rotates "
                f"this to nothing"
            )
        wz = math.copysign(min(abs(wz), WZ_MAX), wz)

    return dataclasses.replace(cmd, vx=vx, vy=vy, wz=wz)


SPEC = ChannelSpec(
    "lower_body", LowerBody,
    Limits(wz=(WZ_MIN_ACCEPT, WZ_MAX)),
    clamp_lower,
)


def measured(handle, cmd: LowerBody) -> None:
    handle.apply_lower_body(cmd)
