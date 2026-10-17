"""Measured observations for live exploration, checked at point of use."""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ..contracts import MissingProviderError, StaleObservation, UndeclaredNeed, WorldRestarted
from . import (
    base_mobility,
    base_pose,
    gripper_state,
    lidar_obstacles,
    loc_quality,
    map,
    pelvis_tilt,
    qpos,
)

TERMS = [base_pose, loc_quality, qpos, gripper_state, lidar_obstacles, base_mobility, pelvis_tilt, map]
SPECS = {m.SPEC.name: m.SPEC for m in TERMS}
def providers(backend, needs, *, skill=None):
    if backend != "real":
        raise MissingProviderError("G1 exploration supports only real measured observations")
    modules = {m.SPEC.name: m for m in TERMS}
    unknown = set(needs) - modules.keys()
    if unknown:
        raise MissingProviderError(f"unsupported exploration observations: {sorted(unknown)}")
    return {n: modules[n].measured for n in needs}

UNPRIVILEGED_SOURCES = {
    "lidar_obstacles": {"registered_lidar"},
    "qpos": {"joint_encoder"},
    "gripper_state": {"joint_proprioceptive_gap"},
    "base_pose": {"lidar_registration"},
    "loc_quality": {"registration_health"},
    "map": {"live_lidar_map"},
    "base_mobility": {"controller_state"},
    "pelvis_tilt": {"imu_attitude"},
}

def _reject_non_finite(term: str, value: Any) -> None:
    """NaN/Inf are rejected at the provider boundary."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"observation '{term}' contains NaN/Inf — rejected at the boundary")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"observation '{term}' is {value} — rejected at the boundary")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            _reject_non_finite(term, getattr(value, f.name))
    elif isinstance(value, dict):
        for v in value.values():
            _reject_non_finite(term, v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _reject_non_finite(term, v)


class Observer:
    """Reality's one entrance. Enforces, on every observe, at the point of use:
    staleness (per-spec), world-epoch continuity, NaN/Inf rejection,
    declared-needs discipline, and the every-observe-is-an-event clause."""

    def __init__(
        self,
        backend: str,
        handle: Any,
        needs: tuple[str, ...],
        emit: Callable[[str, dict], None] | None = None,
        skill: str | None = None,
        tap: Callable[[str, Any, str | None], None] | None = None,
    ):
        self._handle = handle
        self._fns = providers(backend, needs, skill=skill)
        self._emit = emit if emit is not None else (lambda kind, payload: None)
        self._tap = tap  # recorder: sees each value after every check, changes nothing
        self._skill = skill
        self._epoch = handle.epoch()
        self.touched: set[str] = set()

    def serves(self, name: str) -> bool:
        """Whether this backend declared a provider for the term."""
        return name in self._fns

    def __call__(self, name: str) -> Any:
        fn = self._fns.get(name)
        if fn is None:
            raise UndeclaredNeed(
                f"'{name}' was touched but not declared in NEEDS.obs "
                f"(declared: {', '.join(sorted(self._fns))}) — declared == touched is the contract"
            )
        spec = SPECS[name]
        source = self._handle.observation_source(name)
        if source not in UNPRIVILEGED_SOURCES[name]:
            raise RuntimeError(
                f"observation '{name}' uses unapproved or privileged source {source!r}; "
                f"allowed: {sorted(UNPRIVILEGED_SOURCES[name])}"
            )
        epoch = self._handle.epoch()
        if epoch != self._epoch:
            raise WorldRestarted(
                f"world epoch changed {self._epoch} -> {epoch} while observing '{name}' — "
                f"fresh data from a restarted world is not an observation"
            )
        # Acquire first: providers update the term timestamp as they acquire.
        try:
            value = fn(self._handle)
        except AttributeError as exc:
            raise MissingProviderError(
                f"observation '{name}' cannot be acquired from {type(self._handle).__name__}: "
                f"{exc}. Nothing was started."
            ) from exc
        age = float(self._handle.age_s(name))
        if age > spec.staleness_s:
            raise StaleObservation(
                f"'{name}' is {age:.3f}s old > staleness_s={spec.staleness_s} — a liveness "
                f"check must demand a frame, never a ping"
            )
        _reject_non_finite(name, value)
        self.touched.add(name)
        self._emit(
            "observe",
            {"term": name, "age_s": age, "epoch": epoch, "source": source},
        )
        if self._tap is not None:
            self._tap(name, value, self._skill)
        return value
