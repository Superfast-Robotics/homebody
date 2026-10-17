"""Explore through RealBackend with one actuator owner and explicit motion refusals.

Every twist is checked as one circular sweep against the published map geometry and fresh
registered LiDAR in the obstacle band; overlap with a known obstacle allows only translation
that does not deepen it. Start-frame metres.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np

from homebody.backends.g1.nav_control import SCHEMA, parse_status
from homebody.backends.g1.real import (
    LOC_ODOM_AGE_MAX_S,
    RealBackend,
    RealHardwareRequired,
    _pelvis_xy,
    localization_frozen,
)
from homebody.helpers.geometry import wrap
from homebody.motion.exploration.gait import occupancy_grid
from homebody.motion.exploration.nav import swept_circle_points, twist_sweep_envelope
from homebody.primitives.navigation.actions.lower_body import LowerBody
from homebody.primitives.navigation.contracts import Refused
from homebody.primitives.navigation.errors import (
    MapObstacleRefused,
    MotionBlocked,
    ObservationUnavailable,
    TaskScopeRefused,
)
from homebody.primitives.navigation.observations.gripper_state import UNKNOWN
from homebody.skills.navigate_real.controller import CONTINUOUS_POLL_S

from .freshness import DECISION_FRESH_STABLE_S
from .livemap import OBSTACLE_BAND_M
from .planner import EXECUTION_RADIUS_M

ODOM_MOVE_MAX_S = LOC_ODOM_AGE_MAX_S  # older odometry refuses motion


class ExploreRealBackend(RealBackend):
    def __init__(self, *, source, explorer, **kwargs):
        source.require_actuation()
        super().__init__(**kwargs)
        self._explore_source = source
        self.explorer = explorer
        self._blocked: tuple = (None, None, None)  # (map generation, blocked, hit) box masks
        self._hands_note: str | None = None

    def _navigation_status(self):
        src = self._explore_source
        try:
            x, y, yaw_body = src.pose()
        except RuntimeError as exc:  # typed, so gait.pose waits standing
            raise RealHardwareRequired(f"localization unavailable: {exc}") from exc
        age = src.pose_age_s()
        age_available = math.isfinite(age) and age >= 0
        health = "LOST" if src.frame_reset or not src.zeroed() or not age_available else "OK"
        if not age_available:
            # no age: the pose is withheld too
            x = y = yaw_body = None
        payload = {
            "state": "IDLE",
            "x": x,
            "y": y,
            "yaw": yaw_body,
            "yaw_body": yaw_body,
            "health": health,
            "rmse": None,
            "odom_age": round(age, 3) if age_available else None,
            "t": time.time(),
            "init_source": "explore-odometry",
            "control": {"schema": SCHEMA, "available": True, "event": "STOPPED", "mode": "DRIVE"},
        }
        return parse_status(payload)

    def base_pose(self):
        """The driver's pelvis-frame pose, cached for the execution check."""
        src = self._explore_source
        epoch = (src.motion_epoch() if hasattr(src, "motion_epoch")
                 else getattr(src, "frame_epoch", None))
        pose = super().base_pose()
        self._execution_base_pose_sample = (
            epoch, time.monotonic(), self._term_stamp["base_pose"],
            tuple(float(v) for v in pose))
        return pose

    def footprint_pose(self):
        """The pelvis pose from fresh localization and cached body telemetry, for display."""
        src = self._explore_source
        epoch = src.motion_epoch()
        sample = self._navigation_status()
        frozen = localization_frozen(sample)
        if sample.pose is None or frozen or src.frame_reset or not src.zeroed():
            raise ObservationUnavailable(frozen or "footprint localization is unavailable",
                                         component="footprint_pose")
        now = time.monotonic()
        imu, tilt, waist = self._imu_yaw, self._tilt, self._q_waist
        age = math.inf if imu is None else now - imu[1]
        if (tilt is None or not math.isfinite(age) or not 0. <= age <= 0.5
                or not np.isfinite((*tilt, *waist)).all()
                or self._control_error is not None):
            raise ObservationUnavailable("footprint body telemetry is unavailable or stale",
                                         component="footprint_pose", age_s=age, limit_s=0.5)
        pose = np.asarray(sample.pose, dtype=float).copy()
        if not self._pose_xy_accepted(pose[:2], now):
            pose[:2] = self._pose_held[0]
        pose[2] = wrap(float(pose[2]) - waist[0])
        offset = self._yaw_offset
        if not self._imu_yaw_mirrored and offset is not None:
            pose[2] = wrap(imu[0] + offset)
        pose[:2] = _pelvis_xy(pose[:2], float(pose[2]), tilt, waist)
        if src.motion_epoch() != epoch or src.frame_reset:
            raise ObservationUnavailable("footprint frame changed during observation",
                                         component="footprint_pose")
        return tuple(float(v) for v in pose)

    def _execution_base_pose(self):
        """The cached pelvis pose for this frame; raises when older than 0.5 s."""
        src = self._explore_source
        epoch = (src.motion_epoch() if hasattr(src, "motion_epoch")
                 else getattr(src, "frame_epoch", None))
        cached = getattr(self, "_execution_base_pose_sample", None)
        if cached is None or cached[0] != epoch:
            raise ObservationUnavailable("execution pelvis pose is not prepared for this frame",
                                         component="execution_pose", limit_s=0.5)
        acquired_age, sample_age = time.monotonic() - cached[1], time.time() - cached[2]
        if (not math.isfinite(acquired_age) or not math.isfinite(sample_age)
                or acquired_age < 0. or sample_age < -0.1
                or acquired_age > 0.5 or sample_age > 0.5):
            raise ObservationUnavailable(
                f"execution pelvis pose is stale (acquired {acquired_age:.3f} s, "
                f"sample {sample_age:.3f} s; limit 0.5 s)", component="execution_pose",
                age_s=max(acquired_age, sample_age), limit_s=0.5,
                details={"acquired_age_s": acquired_age, "sample_age_s": sample_age})
        return cached[3]


    def held_observation_refusal(self):
        return self.observation_refusal(require_health_stable=False)

    def observation_refusal(self, *, require_health_stable=True):
        """Why the observations are not fresh enough for motion, or None."""
        src = self._explore_source
        odom_age, scan_age = src.odom_age_s(), time.time() - src.ring_t
        for component, age in (("registered_pose", odom_age), ("registered_scan", scan_age)):
            if age > 0.5:
                return ObservationUnavailable(
                    f"LiDAR-only check: {component.replace('_', ' ')} is stale ({age:.3f} s old; the limit is 0.5 s)",
                    component=component, age_s=age, limit_s=0.5,
                    details={"odom_age_s": odom_age if math.isfinite(odom_age) else None,
                             "scan_age_s": scan_age if math.isfinite(scan_age) else None,
                             "freshness": getattr(src, "freshness_status", dict)()})
        if not src.context_ready():
            camera = src.view_camera
            received = getattr(camera, "_received", None)
            age = None if received is None else time.monotonic() - received
            return ObservationUnavailable("camera view is unavailable or stale",
                component="camera_view", age_s=age, limit_s=getattr(camera, "max_age_s", None))
        if src.ring_fresh_m() == 0:
            return ObservationUnavailable("LiDAR near-field clearance is unavailable or zero",
                                          component="near_field_clearance", age_s=scan_age)
        with src.localization.lock:
            if not len(src.localization._last_points):
                return ObservationUnavailable("LiDAR near-field scan is empty",
                                              component="registered_scan", age_s=scan_age,
                                              details={"point_count": 0})
        odom_age = src.pose_age_s()
        if odom_age > ODOM_MOVE_MAX_S:
            return ObservationUnavailable(
                f"odometry {odom_age:.2f} s old (limit {ODOM_MOVE_MAX_S:.1f} s)",
                component="usable_pose", age_s=odom_age, limit_s=ODOM_MOVE_MAX_S)
        # an empty map refuses; a stale one does not
        if self.explorer.map_age_s() is None:
            return ObservationUnavailable("the live map is empty", component="live_map")
        if require_health_stable:
            stable = getattr(src, "health_stable_s", lambda: math.inf)()
            if stable < DECISION_FRESH_STABLE_S:
                return ObservationUnavailable("SLAM health is settling before motion",
                    component="health_continuity", details={"stable_s": stable,
                        "required_stable_s": DECISION_FRESH_STABLE_S,
                        "freshness": getattr(src, "freshness_status", dict)()})
        return None

    def refresh_execution_geometry(self):
        """Build and cache the execution checkers for the current map and task region."""
        explorer = self.explorer
        with explorer.lock:
            live = explorer.live
            key = (id(live), getattr(live, "version", explorer.map_t),
                   live.task_region_revision)
            cached = getattr(self, "_execution_cache", None)
            if cached is not None and cached[0] == key:
                return
            obstacles = live.blocking_obstacles().copy()
            art = SimpleNamespace(occupancy=live.blocked(include_boundary=False).copy(),
                T_map_px=live.T_map_px(), center_allowed=live.center_allowed.copy())
            checker, _ = occupancy_grid(art, EXECUTION_RADIUS_M,
                                         exact_footprint=True, obstacles=obstacles)
            scope_checker = None
            same_scope = cached is not None and cached[0][::2] == key[::2]
            if same_scope:
                scope_checker = cached[2]
            elif live.task_region_points is not None:
                scope_art = SimpleNamespace(occupancy=np.zeros_like(art.occupancy),
                    T_map_px=art.T_map_px, center_allowed=live._task_region_mask.copy())
                scope_checker, _ = occupancy_grid(scope_art, 0., exact_footprint=True)
            if cached is not None and not same_scope:
                # a new task region drops the last execution pose
                self._execution_pose = None
            self._execution_cache = (key, checker, scope_checker)

    def _execution_scene(self):
        cached = getattr(self, "_execution_cache", None)
        if cached is None or cached[0][0] != id(self.explorer.live):
            raise ObservationUnavailable("execution map geometry is not prepared",
                                         component="execution_map")
        if cached[0][2] != self.explorer.live.task_region_revision:
            raise TaskScopeRefused("task region changed; execution geometry must be refreshed",
                                   code="TASK_SCOPE_CHANGED")
        return cached[1]

    def execution_refusal(self, vx, vy, wz):
        """Check the swept disk of one twist against map cells and new LiDAR points."""
        src = self._explore_source
        live = self.explorer.live
        if (vx or vy) and not live.scope_ready:
            return TaskScopeRefused("translation awaits an accepted task region",
                                   code="TASK_SCOPE_UNRESOLVED")
        x, y, yaw = self._execution_base_pose()
        start = (x, y)
        # one controller poll plus the stop and settle interval
        horizon = CONTINUOUS_POLL_S + .2
        end, curve_padding = twist_sweep_envelope(x, y, yaw, vx, vy, wz, horizon)
        checker = self._execution_scene()
        epoch = (src.motion_epoch() if hasattr(src, "motion_epoch")
                 else getattr(src, "frame_epoch", None))
        prior = getattr(self, "_execution_pose", None)
        segments = [("command", start, end)]
        if prior is not None and prior[0] == epoch and prior[1] != start:
            segments.insert(0, ("measured", prior[1], start))
        with src.localization.lock:
            stamp, points = src.ring_t, src.localization._last_points
            cached = getattr(self, "_execution_points", None)
            if cached is None or cached[0] != stamp:
                keep = ((points[:, 2] >= OBSTACLE_BAND_M[0])
                        & (points[:, 2] <= OBSTACLE_BAND_M[1]))
                self._execution_points = (stamp, points[keep, :2].copy())
            points = self._execution_points[1]
        for phase, a, b in segments:
            # existing overlap may only shrink
            padding = curve_padding if phase == "command" else 0.
            egress = phase == "measured" or (bool(vx or vy) and padding == 0.)
            result = checker.check_motion(a, b, allow_egress=egress, extra_radius_m=padding)
            if not result["safe"] and self._execution_cache[2] is not None:
                scope = self._execution_cache[2].check_motion(a, b, extra_radius_m=padding)
                if not scope["safe"]:
                    return TaskScopeRefused("motion crosses the task boundary", details={
                        **scope, "guard": "task_region", "phase": phase,
                        "from_xy": list(a), "to_xy": list(b),
                        "task_region_revision": live.task_region_revision})
            evidence = "map"
            if result["safe"]:
                result = swept_circle_points(a, b, points, EXECUTION_RADIUS_M + padding,
                                             allow_egress=egress)
                evidence = "lidar"
            if not result["safe"]:
                details = {**result, "guard": "execution_footprint", "phase": phase,
                           "evidence": evidence, "from_xy": list(a), "to_xy": list(b),
                           "pose_xy_yaw": [x, y, yaw], "twist": [vx, vy, wz],
                           "horizon_s": horizon, "curve_padding_m": padding}
                kind = MapObstacleRefused if evidence == "map" else MotionBlocked
                return kind(f"execution footprint blocked by {evidence}: {result['reason']}",
                            details=details)
        self._execution_pose = (epoch, start)
        return None

    def motion_refusal(self, vx: float, vy: float, wz: float = 0.):
        src = self._explore_source
        if getattr(src, "observe_only", False):
            return Refused("observation-only mode")
        if src.frame_reset:
            return Refused(f"SuperOdom reset its frame ({src.frame_reset_why})")
        unavailable = self.observation_refusal()
        if unavailable is not None:
            return unavailable
        try:
            return self.execution_refusal(vx, vy, wz)
        except (ObservationUnavailable, TaskScopeRefused) as exc:
            return exc


    def apply_lower_body(self, cmd) -> None:
        """Send CMD unless refused: every command after a stop, a moving one by motion_refusal."""
        src = self._explore_source
        src.require_actuation()
        twist = (0.0, 0.0, 0.0) if cmd is None else (cmd.vx, cmd.vy, cmd.wz)
        vx, vy, wz = (float(v or 0.0) for v in twist)
        refused = Refused("explore stop pressed") if src.stop_flag.is_set() else None
        if refused is None and (vx or vy or wz):
            refused = self.motion_refusal(vx, vy, wz)
        if refused is not None:  # state=None leaves the walk state unchanged
            super().apply_lower_body(LowerBody(vx=0.0, vy=0.0, wz=0.0, state=None))
            src.note_twist(0.0, 0.0, 0.0)
            self._execution_pose = None  # a repair starts from the stopped pose
            refused.args = (f"explore: {refused}",)
            raise refused
        super().apply_lower_body(cmd)
        if not (vx or vy or wz):
            self._execution_pose = None
        if cmd is not None and (cmd.vx, cmd.vy, cmd.wz) != (None, None, None):
            src.note_twist(vx, vy, wz)  # dead reckoning follows the sent twist

    def observation_source(self, name):
        if name == "lidar_obstacles":
            return "registered_lidar"
        return super().observation_source(name)

    def lidar_obstacles(self):
        src = self._explore_source
        if src.frame_reset:
            raise RealHardwareRequired("SuperOdom reset its frame")
        if src.odom_age_s() > 0.5 or time.time() - src.ring_t > 0.5:
            raise ObservationUnavailable("LiDAR near-field scan or registered pose is stale")
        with src.localization.lock:
            points, stamp = src.localization._last_points.copy(), src.ring_t
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                or not np.isfinite(points).all()):
            raise RealHardwareRequired("LiDAR near-field scan is empty or non-finite")
        self._term_stamp["lidar_obstacles"] = stamp
        return points.astype(np.float32)

    def gripper_state(self):
        """RealBackend's Dex3 read; hands that stop answering are served as UNKNOWN from then on."""
        if self._hands_note is None:
            try:
                return super().gripper_state()
            except RealHardwareRequired as absent:
                self._hands_note = f"Dex3 hands not read ({absent}); served as UNKNOWN this run"
        self._term_stamp["gripper_state"] = time.time()
        return {"closure_l": 0.0, "closure_r": 0.0, "held_l": UNKNOWN, "held_r": UNKNOWN}

    def body_state(self) -> str:
        """RealBackend's body line plus the body host's last event."""
        line = super().body_state()
        event = getattr(getattr(self._kernel, "status", None), "last_event", "")
        return f"{line} · host: {event}" if event else line

    def hands_note(self) -> str:
        """Read the hands once and return the note about them, or ''."""
        self.gripper_state()
        return self._hands_note or ""
