"""Guarded action dispatch for the G1 exploration motion owner."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..contracts import (
    HardwareUnavailable,
    MissingProviderError,
    Refused,
    UndeclaredNeed,
    contract_file,
)
from . import lower_body

SPECS = {"lower_body": lower_body.SPEC}
def channels(backend, needs, *, skill=None):
    if backend != "real":
        raise MissingProviderError("G1 exploration supports only the real backend")
    unknown = set(needs) - SPECS.keys()
    if unknown:
        raise MissingProviderError(f"unsupported exploration action channels: {sorted(unknown)}")
    return {name: lower_body.measured for name in needs}

class Actuator:
    """The one command path. step(name, payload) -> None, always."""

    def __init__(
        self,
        backend: str,
        handle: Any,
        needs: tuple[str, ...],
        observe: Callable[[str], Any] | None = None,
        emit: Callable[[str, dict], None] | None = None,
        skill: str | None = None,
    ):
        self._handle = handle
        self._fns = channels(backend, needs, skill=skill)
        self._observe = observe
        self._emit = emit if emit is not None else (lambda kind, payload: None)
        self._command_context: Callable[[], Mapping[str, str]] | None = None
        self.touched: set[str] = set()
        self._motion_generation = getattr(handle, "motion_generation", None)
        self._prepared_generation = (self._motion_generation()
                                     if callable(self._motion_generation) else None)

    def _check_motion_generation(self) -> None:
        if (callable(self._motion_generation)
                and self._motion_generation() != self._prepared_generation):
            reason_fn = getattr(self._handle, "motion_interruption_reason", None)
            reason = reason_fn() if callable(reason_fn) else ""
            raise HardwareUnavailable(
                "body communication was interrupted during this skill; its motion reference "
                "is invalid. Reobserve and replan from the current pose before moving"
                + (f". Interruption: {reason}" if reason else ""))

    def step(self, name: str, payload: Any) -> None:
        fn = self._fns.get(name)
        if fn is None:
            raise UndeclaredNeed(
                f"channel '{name}' was stepped but not declared in NEEDS.actions "
                f"(declared: {', '.join(sorted(self._fns))})"
            )
        spec = SPECS[name]
        if type(payload) is not spec.payload:  # admission is a type gate
            raise Refused(
                f"channel '{name}' takes {spec.payload.__name__}, got {type(payload).__name__}"
            )
        self._check_motion_generation()
        clamped = spec.clamp(payload, self._observe)  # may raise Refused, which propagates
        self._check_motion_generation()
        event_context = (
            {} if self._command_context is None else dict(self._command_context())
        )
        self.touched.add(name)
        self._emit(
            "command",
            {"channel": name, "payload": repr(clamped), **event_context},
        )
        self._check_motion_generation()
        result = fn(self._handle, clamped)
        if result is not None:
            raise TypeError(
                f"{contract_file(fn.__module__)}: def {name} provider returned "
                f"{type(result).__name__} — step() returns None; reality enters only "
                f"through observe"
            )
