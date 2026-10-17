"""The shared contract machinery: specs, limits, needs, and errors that carry the fix.

Each term owns one SPEC plus its provider (``measured``, read from the G1). Target selection
never creates another copy of the contract.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


@dataclass(frozen=True)
class TermSpec:
    """One observation term. Everything about the term lives in its file."""

    name: str
    shape: tuple[int | str, ...]  # symbolic dims allowed: ("H", "W")
    dtype: str
    frame: Literal["camera", "base", "map", "joint"]
    units: str
    staleness_s: float  # max age at point of use; observe() raises past it
    notes: str = ""  # measured divergences between the modes
    exempt: str = ""  # genuinely one-sided capability: reason, printed yellow


@dataclass(frozen=True)
class Limits:
    """Bounds a channel clamp enforces. A channel reads only the fields it names."""

    wz: tuple[float, float] | None = None  # (min effective, max) rad/s magnitude
    bounds: tuple[tuple[float, float], ...] | None = None  # per-joint (lo, hi) rad


@dataclass(frozen=True)
class ChannelSpec:
    """One action channel: frozen payload type, limits, a pure clamp, and its reach.

    ``reach`` maps each mode to what commanding this channel accomplishes there."""

    name: str
    payload: type  # frozen dataclass; None fields mean "unchanged"
    limits: Limits
    clamp: Callable[[Any, Callable[[str], Any]], Any]  # (payload, observe) -> payload
    reach: Mapping[str, str] = field(default_factory=dict)
    exempt: str = ""


@dataclass(frozen=True)
class Needs:
    """A skill's declared surface — names from the registries only."""

    obs: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    models: tuple[str, ...] = ()


class MissingProviderError(Exception):
    """A needed term/channel has no ``def <backend>()``. The message names the file."""


class UndeclaredNeed(Exception):
    """A skill touched a term/channel/model not in its NEEDS. The message names it."""


class StaleObservation(Exception):
    """Data older than the spec's staleness_s at the point of use."""


class WorldRestarted(Exception):
    """The backend's world epoch changed mid-episode."""


class Refused(Exception):
    """A clamp refused a command. Carries the reason, never silently zeroes."""


class HardwareUnavailable(RuntimeError):
    """A real feed or channel could not serve; the message names the piece. The gateway turns
    it into the HARDWARE_UNAVAILABLE failure code."""


class DriveFailure(Exception):
    """A checked-navigation failure. Callers map ``code`` by name, never substring tests
    on prose; str() is "CODE: detail"."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def contract_file(module_name: str) -> str:
    """Repo-relative path of a contract module, for errors that name the file to edit."""
    return module_name.removeprefix("homebody.").replace(".", "/") + ".py"


def jsonable(value):
    """The numpy -> canonical-JSON conversion for event streams: non-finite numbers become None,
    and every Mapping (not only dict) is converted."""
    if hasattr(value, "tolist") and hasattr(value, "shape"):  # numpy array
        return jsonable(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes, Mapping, list, tuple)):
        value = value.item()  # numpy scalar
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
