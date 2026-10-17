"""Live navigation control flow, independent of a sensor or hardware backend."""

import math

from .contracts import Refused


class Stopped(RuntimeError):
    """Motion cannot continue without intervention."""


class Paused(RuntimeError):
    """An action was paused; it may resume once the robot is confirmed stationary."""


class ActionRefused(Stopped):
    """A stationary, healthy exploration action failed; the policy may select another action."""

    def __init__(self, message: str, *, code: str = "ACTION_UNAVAILABLE", details=None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class Abort(RuntimeError):
    """The live task frame or its permitted bounds do not support execution."""


class DecisionInterrupted(RuntimeError):
    """A cancelled decision must be rebuilt from current observations."""


class MotionBlocked(Refused):
    """Current geometry refuses this motion; observation and authority remain separate."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


class MapObstacleRefused(MotionBlocked):
    """A known live-map obstacle refused translation; not a stale or missing sensor."""


class TaskScopeRefused(Refused):
    """Semantic permission refused motion; obstacle recovery cannot relax it."""

    def __init__(self, message, *, code="TASK_BOUNDARY", details=None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {}) | {"code": code}


class ObservationUnavailable(Refused):
    """Fresh observations are missing; wait for them ahead of another motion."""

    def __init__(self, message, *, component=None, age_s=None, limit_s=None, details=None):
        super().__init__(message)
        self.details = dict(details or {})
        if component is not None:
            self.details.update(component=component,
                                age_s=age_s if age_s is not None and math.isfinite(age_s) else None,
                                limit_s=limit_s)
