"""A kinematic stand-in for the G1 backend surface; SOURCES name the backend's observation sources."""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class StubBody:
    def __init__(self):
        self.trace_q: list = []   # the drivers' 250 Hz joint record; empty here
        self.trace_t: list = []
        self._t = 0.0
        self._epoch = 1
        self._ages: dict[str, float] = {}
        self.commands: list[tuple[str, object]] = []

    # -- clock / epoch / staleness knobs -------------------------------------
    def now(self) -> float:
        self._t += 0.001  # strictly monotonic robot time
        return self._t

    def arm_trace(self) -> dict:
        return {"q": np.zeros((0, 29), dtype=np.float32),
                "t": np.zeros((0,), dtype=np.float64)}

    def epoch(self) -> int:
        return self._epoch

    def restart(self) -> None:
        """Simulate a world restart."""
        self._epoch += 1

    def age_s(self, term: str) -> float:
        return self._ages.get(term, 0.0)

    def set_age(self, term: str, seconds: float) -> None:
        self._ages[term] = seconds

    # Part of the handle contract: Observer refuses a term whose source is privileged.
    SOURCES: ClassVar[dict[str, str]] = {
        "rgb": "camera",
        "camera": "calibration",
        "qpos": "joint_encoder",
        "gripper_state": "joint_proprioceptive_gap",
        "base_pose": "lidar_registration",
        "loc_quality": "registration_health",
        "arm_trace": "joint_encoder",
        "map": "live_lidar_map",
        "base_mobility": "controller_state",
        "pelvis_tilt": "imu_attitude",
        "lidar_obstacles": "registered_lidar",
    }

    def observation_source(self, term: str) -> str:
        return self.SOURCES[term]

    # -- shared observation surface -------------------------------------------
    def lidar_obstacles(self) -> np.ndarray:
        return np.array([[2.0, 0.0, 0.5]], dtype=np.float32)

    def qpos(self) -> np.ndarray:
        return np.zeros(29, dtype=np.float32)

    def joint_names(self) -> tuple[str, ...]:
        return tuple(f"joint_{i:02d}" for i in range(29))

    def gripper_state(self) -> dict:
        return {"closure_l": 0.0, "closure_r": 0.0, "held_l": None, "held_r": None}

    def base_pose(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    def inject_loc_quality(self, seq) -> None:
        """Degrade localization through the provider's own input instead of monkeypatching it."""
        self._loc_seq = [float(v) for v in seq]

    def loc_quality(self) -> float:
        if getattr(self, "_loc_seq", None):
            return self._loc_seq.pop(0)
        return 1.0

    def pelvis_tilt(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def base_mobility(self) -> bool:
        return True

    def map_artifact(self) -> dict:
        return {
            "raster": np.zeros((16, 16, 3), dtype=np.uint8),
            "T_map_px": np.eye(3, dtype=np.float64),
            "occupancy": np.zeros((16, 16), dtype=bool),
            "scan_ref": "assets/scans/stub/scan.ply",
            "scan_hash": "0" * 64,
        }

    # -- shared action surface -------------------------------------------------
    def apply_lower_body(self, cmd) -> None:
        self.commands.append(("lower_body", cmd))
