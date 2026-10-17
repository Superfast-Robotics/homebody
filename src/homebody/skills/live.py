"""What the live skills share: the exploration they read (`LiveWorld`), and a typed motion
interruption that ends the attempt through `fail`."""
from contextlib import contextmanager
from typing import Protocol

from homebody.primitives.navigation.errors import Abort, ActionRefused, Stopped


class LiveWorld(Protocol):
    """The exploration a live skill reads as `ctx.world` (and `planning.world`); the explorer
    implements it. Motion goes through `ctx.actions`, never through this."""

    live: object  # the live map, replaced when the boundary or start frame changes
    planner: object
    boundary: object
    lock: object  # taken while reading the map
    radius: float  # the current planning radius
    pending_face: dict | None  # the facing a walk should end with
    observe_only: bool

    def pose(self) -> tuple:
        """The base pose (x, y, yaw) in the start frame."""

    def base_xy(self) -> tuple:
        """The base position (x, y)."""

    def decision_pose(self) -> tuple:
        """The pose once sustained fresh input has been seen, waiting stationary for it."""

    def check(self) -> None:
        """Raise the stop, frame reset, boundary breach or budget that ends the action."""

    def require_task_scope(self) -> None:
        """Refuse translation before the task region is defined."""

    def refresh_planning_radius(self) -> dict:
        """Resolve the planning radius; the extra motion request it implies."""

    def planning_cancelled(self) -> bool:
        """Raise when a locked search must unwind; passed to planners as `cancelled`."""

    def failed_goal_reason(self, goal, route) -> str:
        """Why GOAL along ROUTE repeats a failed destination, or ''."""

    def leg_seconds(self, points) -> float:
        """The deadline for walking POINTS within the remaining budget."""

    def action_capture(self, kind, *, route=None):
        """A context manager that retains camera evidence around one motion."""

    def capture_survey_heading(self) -> None:
        """Retain the settled view at this heading."""

    def stop_here(self, kind) -> None:
        """Record a stop of KIND at the current pose."""

    def active_monotonic(self) -> float:
        """Monotonic time with paused time removed."""

    def remaining(self) -> float:
        """Seconds left in the exploration budget."""

    def motion_epoch(self):
        """The source's motion epoch, or None."""

    def stationary_guard(self, *generation):
        """The stationary generation, or a check that GENERATION is current; None if unguarded."""

    def emit(self, kind, **data) -> None:
        """Apply a workflow event (status, route, facing, recovery, completion, help...)."""

    def region_snapshot(self) -> dict:
        """What a task-region admission compares before and after its checks."""

    def publish_region(self, candidate, growth, revision, evidence) -> dict:
        """Install an admitted task region; returns its accepted record."""

    def persist_region(self, accepted, metrics) -> None:
        """Record an accepted task region and save the map."""

    def assess_completion(self, assessment, history) -> str:
        """Why the visual evidence does not support a completion claim, or ''."""

    def ask_question(self, record) -> None:
        """Stand still and wait for the answer to RECORD's question."""


@contextmanager
def interruptions(fail):
    """Hand an ActionRefused, Stopped or Abort raised inside the block to `fail`, the context's
    (or planning's) failure hook, which records it and ends the attempt with its code."""
    try:
        yield
    except (ActionRefused, Stopped, Abort) as error:
        fail(error)
