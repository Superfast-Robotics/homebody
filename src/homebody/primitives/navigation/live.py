"""Actions and task-frame identity exposed by live navigation to registered skills."""
from dataclasses import dataclass
from typing import Protocol


class LiveActions(Protocol):
    def walk(self, points, *, look=None, mps=.30, radius_m=.45, seconds=90.,
             requested_radius_m=None): ...

    def turn_hold(self, heading, hold_s, *, seconds=None): ...

    def hold_position(self): ...

    def step_to(self, xy, seconds=6.): ...

    def recover_departure(self, radius_m, seconds=30., goal_xy=None, *,
                          requested_radius_m=None): ...


@dataclass(frozen=True)
class LiveFrame:
    epoch: tuple[int, int]
    frame_id: str


LIVE_FAILURES = (
    "STOPPED", "ABORTED", "ACTION_UNAVAILABLE", "TASK_BOUNDARY", "TASK_SCOPE_UNRESOLVED",
    "TASK_SCOPE_CHANGED", "TIMEOUT", "HARDWARE_UNAVAILABLE", "NO_COLLISION_FREE_PATH",
    "NO_PROGRESS", "CONTROLLER_REFUSED", "ARRIVAL_NOT_VERIFIED", "LOCALIZATION_DEGRADED",
    "CARRY_LOST", "POSE_OFF_MAP", "OBSERVATIONS_UNAVAILABLE", "RETREAT_UNAVAILABLE",
    "RECOVERY_UNAVAILABLE", "RECOVERY_GOAL_UNAVAILABLE", "LOOK_INTERRUPTED", "LOOK_BLOCKED",
    "LOOK_REPOSITIONED", "LOOK_INCOMPLETE", "TURN_DRIFT", "TURN_UNAVAILABLE",
)
