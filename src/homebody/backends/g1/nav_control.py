"""The localization status payload the G1 backend reads its pose and health from."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

SCHEMA = "homebody-nav-status/v1"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")  # noqa: TRY004 -- status errors are ValueError
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {size} finite numbers")  # noqa: TRY004 -- status errors are ValueError
    try:
        result = tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(value))
    except TypeError:
        raise ValueError(f"{name} must contain {size} finite numbers") from None
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


@dataclass(frozen=True)
class Status:
    received_at: float
    available: bool
    state: str
    event: str
    command_id: str
    plan_token: str
    plan_generation: int
    replan_count: int
    goal_pose: tuple[float, float, float] | None
    path: tuple[tuple[float, float], ...] | None
    radius_m: float | None
    map_sha256: str
    mode: str
    nudge_velocity: tuple[float, float, float] | None
    nudge_duration_s: float | None
    nudge_remaining_s: float | None
    requires_deploy: bool
    print_only: bool
    health: str
    pose: tuple[float, float, float] | None
    localization_rmse_m: float | None
    dist_goal_m: float | None
    reason: str
    stop_reason: str
    raw: dict[str, Any]
    odom_age_s: float | None = None  # seconds since the odometry was last heard
    init_source: str = ""  # how the localizer got its registration

    def age(self) -> float:
        return max(0.0, time.monotonic() - self.received_at)

    def fresh(self, age_s: float = 1.0) -> bool:
        return self.available and self.age() <= age_s

def parse_status(payload: dict[str, Any]) -> Status:
    if not isinstance(payload, dict):
        raise ValueError("navigation status must be an object")  # noqa: TRY004 -- status errors are ValueError
    control = payload.get("control")
    if not isinstance(control, dict):
        control = {}
    goal = None if control.get("goal_pose") is None else _vector(
        control["goal_pose"], 3, "control.goal_pose")
    path = None
    if control.get("path") is not None:
        rows = tuple(_vector(row, 2, "control.path row") for row in control["path"])
        if not rows or len(rows) > 4096:
            raise ValueError("control.path must contain 1..4096 points")
        path = rows
    velocity = None if control.get("nudge_velocity") is None else _vector(
        control["nudge_velocity"], 3, "control.nudge_velocity")
    generation = int(control.get("plan_generation") or 0)
    replans = int(control.get("replan_count") or 0)
    if generation < 0 or replans < 0:
        raise ValueError("navigation generations must be non-negative")
    pose = None
    # `yaw` carries the mount offset; steering uses the body-frame heading.
    heading = "yaw_body" if payload.get("yaw_body") is not None else "yaw"
    if all(payload.get(name) is not None for name in ("x", "y", heading)):
        pose = (_finite(payload["x"], "x"), _finite(payload["y"], "y"),
                _finite(payload[heading], heading))
    return Status(
        time.monotonic(),
        bool(control.get("available") and control.get("schema") == SCHEMA),
        str(payload.get("state") or "").upper(), str(control.get("event") or ""),
        str(control.get("command_id") or ""), str(control.get("plan_token") or ""),
        generation, replans,
        goal, path,
        None if control.get("radius_m") is None else _finite(control["radius_m"], "radius"),
        str(control.get("map_sha256") or ""), str(control.get("mode") or "DRIVE").upper(),
        velocity,
        None if control.get("nudge_duration_s") is None else _finite(
            control["nudge_duration_s"], "nudge_duration_s"),
        None if control.get("nudge_remaining_s") is None else _finite(
            control["nudge_remaining_s"], "nudge_remaining_s"),
        bool(control.get("requires_deploy")), bool(payload.get("print_only", True)),
        str(payload.get("health") or "").upper(),
        pose,
        None if payload.get("rmse") is None else _finite(payload["rmse"], "rmse"),
        None if payload.get("dist_goal") is None else _finite(payload["dist_goal"], "dist_goal"),
        str(payload.get("reason") or ""),
        str(control.get("stop_reason") or ""), dict(payload),
        None if payload.get("odom_age") is None else _finite(payload["odom_age"], "odom_age"),
        str(payload.get("init_source") or ""))


