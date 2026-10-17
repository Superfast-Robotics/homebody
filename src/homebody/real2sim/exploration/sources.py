"""Where the explorer's sensing and motion come from.

RobotSource: SuperOdom's /ws scan and odometry over DDS, homebody's colour stream, and every move
through the `drive` skill on a RealBackend whose pose, map and lower-body channel it supplies.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from homebody.backends.g1 import profile as g1
from homebody.backends.g1.body_host import BodyClient
from homebody.helpers.geometry import wrap
from homebody.motion.exploration import gait
from homebody.motion.exploration.localizer import PoseGate
from homebody.primitives.localization.superodom import SuperOdomLocalization
from homebody.primitives.navigation import actions, observations
from homebody.primitives.navigation.context import Ctx, Deadline
from homebody.primitives.navigation.contracts import (
    DriveFailure,
    HardwareUnavailable,
    Refused,
    StaleObservation,
)
from homebody.primitives.navigation.envelope import validate_radius_m
from homebody.primitives.navigation.errors import ActionRefused, Paused, Stopped
from homebody.real2sim.exploration.scene import Feeds
from homebody.real2sim.exploration.transport import SshTransport
from homebody.skills.navigate_real import controller as drive
from homebody.skills.navigate_real.navigation import (
    NavigationCapabilities,
    NavigationPort,
    NavigationState,
    Navigator,
)

from . import superodom_ws
from .drive_log import DriveLog
from .explore_backend import (
    ExploreRealBackend,
    MotionBlocked,
    ObservationUnavailable,
    TaskScopeRefused,
)
from .freshness import CAMERA_MAP_MAX_AGE_S, DECISION_FRESH_STABLE_S
from .planner import EXECUTION_RADIUS_M, PADDED_RADII_M, RADIUS_M
from .settings import current

REPO = Path(__file__).resolve().parents[4]
SLAM_RESTART_WAIT_S = 90.0  # the engine's start plus its first registration
STEP_TOL_M = 0.10  # a step back ends this close to its point
GO_COUNTDOWN_S = 3.0  # countdown after the walking policy starts
SLAM_STILL_S = 3.0  # odometry within 5 cm this long before the start frame is taken
SLAM_SCANS_MIN_HZ = 8.0  # the registered scan rate the restart waits for
TURN_GIVE_UP_S = 25.0



class RobotSource:
    """The robot: SuperOdom's /ws scan and odometry, the phone's colour stream, and motion through
    the drive skill only. `pose()` is the torso (the MID-360's xy and heading), start frame."""

    name = "robot"

    camera_description = ("a chest-mounted wide-angle phone camera whose frames arrive late "
                          "and carry no calibrated pose or depth")

    def __init__(self, iface: str, live_map_path: str, *, observe_only: bool = False):
        from homebody.primitives.cameras.context import ContextCamera
        settings = current()
        if not settings.iphone.endpoint.strip():
            raise ValueError("exploration requires iphone.endpoint")
        self.observe_only = observe_only
        self.iface, self.live_map_path = iface, live_map_path
        self.drive_log = DriveLog(Path(live_map_path).with_name("drive_events.jsonl"))
        self.turn_drift_m = settings.exploration.turn_drift_m
        self.pose_gate = PoseGate()
        self._closed = False
        self._close_errors = ()
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.pause_wait = None
        self.motion_event = None
        self.paused_wall_s = 0.0
        self.frame_epoch = None  # motion_epoch's fallback when no explorer is attached
        self._context_distance = None
        self._context_hold_since = None
        self._body = None
        self.backend = None
        self._watchdog_hold = None
        self.localization = SuperOdomLocalization(iface)
        self.view_camera = ContextCamera(settings.iphone.endpoint.strip(), settings.iphone.max_age_s)

    def attach(self, motion_event, pause_wait) -> None:
        """Attach the explorer's status sink for controller events and its resume wait."""
        self.motion_event, self.pause_wait = motion_event, pause_wait

    def health_stable_s(self):
        return self.localization.health_stable_s()

    def note_twist(self, vx: float, vy: float, wz: float):
        return self.localization.note_twist(vx, vy, wz)

    def zeroed(self):
        return self.localization.zeroed()

    def pose(self):
        return self.localization.pose()

    def odom_age_s(self):
        return self.localization.odom_age_s()

    def pose_age_s(self):
        return self.localization.pose_age_s()

    def ring_fresh_m(self):
        return self.localization.ring_fresh_m()

    def ring_ahead_m(self, vx: float, vy: float):
        return self.localization.ring_ahead_m(vx, vy)

    def scans(self):
        return self.localization.scans()

    def live_points(self):
        return self.localization.live_points()

    def scan_hz(self):
        return self.localization.scan_hz()

    def raw_scan_hz(self, window_s: float = 3.0):
        return self.localization.raw_scan_hz(window_s)

    def freshness_status(self):
        return self.localization.freshness_status(stable_s=DECISION_FRESH_STABLE_S)

    def zero(self):
        self.localization.zero()
        if self.backend is not None:
            self.backend.forget_pose_anchor()
        self.pose_gate.forget()

    @property
    def frame_reset(self):
        return self.localization.frame_reset

    @frame_reset.setter
    def frame_reset(self, value):
        self.localization.frame_reset = value

    @property
    def frame_reset_why(self):
        return self.localization.frame_reset_why

    @frame_reset_why.setter
    def frame_reset_why(self, value):
        self.localization.frame_reset_why = value

    @property
    def ring_t(self):
        return self.localization.ring_t

    @ring_t.setter
    def ring_t(self, value):
        self.localization.ring_t = value

    @property
    def ring_m(self):
        return self.localization.ring_m

    @ring_m.setter
    def ring_m(self, value):
        self.localization.ring_m = value

    @property
    def ring_hits(self):
        return self.localization.ring_hits

    @ring_hits.setter
    def ring_hits(self, value):
        self.localization.ring_hits = value

    @property
    def start_frame_metadata(self):
        return self.localization.start_frame_metadata

    def context_image(self):
        return self.view_camera.snapshot()

    def context_ready(self):
        return self.view_camera.fresh()

    def navigation_pose(self) -> tuple[float, float, float]:
        """Gated pelvis pose for planning, using the driver's observation path."""
        if self.backend is None:
            return self.pose()  # unarmed: no body correction
        try:
            return tuple(float(v) for v in gait.pose(self._ctx(5.0)))
        except (DriveFailure, HardwareUnavailable, Refused) as refused:
            raise Stopped(f"the robot's position is unavailable: {refused}") from refused

    def footprint_pose(self) -> tuple[float, float, float]:
        """Read-only pelvis pose for display, outside any motion or recovery gate."""
        if self.backend is None:
            return self.pose()  # unarmed: no body correction
        return self.backend.footprint_pose()

    def base_xy(self) -> tuple[float, float]:
        """The same pelvis centre used for navigation and departure checks."""
        x, y, _ = self.navigation_pose()
        return x, y


    def now(self) -> float:
        return time.time()




    # -- the cockpit's 3-D room ------------------------------------------------------------
    def feeds(self):
        """scene.Feeds over this source: the G1 URDF at the live pose with the host's joints."""
        def pose():
            try:
                return self.footprint_pose()
            except (RuntimeError, ObservationUnavailable):
                return None

        def qpos():
            body = self._body
            if body is None or not body.fresh() or not body.state:
                return None
            return body.state.get("q")

        return Feeds(pose=pose, qpos=qpos, joint_names=tuple(g1.BODY_JOINTS))

    def listen_body(self, addr: str) -> None:
        """Listen to the body host for the live joints of the URDF view."""
        self._body = BodyClient(addr)

    def restart_slam(self):
        """Restart the workstation SuperOdom engine with its origin where the robot stands, then
        wait for live, still odometry. Yields progress lines; raises when it cannot."""
        self.localization.reset_samples()
        # the robot's LiDAR watchdog is frozen during the restart
        watchdog = _WatchdogPause()
        watchdog.freeze(allow_absent=True)
        try:
            yield "SuperOdom engine: stopping"
            superodom_ws.down()
            yield "SuperOdom engine: starting where the robot stands"
            superodom_ws.up()
            yield from self._await_still_odometry()
        finally:
            watchdog.thaw()

    def hold_watchdog(self) -> str:
        """Freeze the robot's LiDAR watchdog until released; returns a status line."""
        if self._watchdog_hold is None:
            hold = _WatchdogPause()
            hold.freeze(allow_absent=True)
            self._watchdog_hold = hold
        return ("the robot's LiDAR watchdog is frozen for this run" if self._watchdog_hold.frozen
                else "no robot LiDAR watchdog; using independent workstation SLAM")

    def release_watchdog(self) -> None:
        hold, self._watchdog_hold = self._watchdog_hold, None
        if hold is not None:
            hold.thaw()

    def _await_still_odometry(self):
        t0, ref, still_since = time.time(), None, None
        while time.time() - t0 < SLAM_RESTART_WAIT_S:
            latest = self.localization._latest()
            scans = self.raw_scan_hz()
            if latest is not None and self.odom_age_s() < 0.3 and scans >= SLAM_SCANS_MIN_HZ:
                xy = latest[1][:2, 3].copy()
                if ref is None or float(np.hypot(*(xy - ref))) > 0.05:
                    ref, still_since = xy, time.time()
                elif time.time() - still_since >= SLAM_STILL_S:
                    yield f"odometry live and still · scans {scans:.0f} Hz"
                    return
                settling = time.time() - still_since
                yield f"odometry live: settling {settling:.0f}/{SLAM_STILL_S:.0f} s"
            else:
                ref, still_since = None, None
                yield (
                    f"waiting for /ws odometry + scans after the restart ({time.time() - t0:.0f} s"
                    f", scans {scans:.0f} Hz)"
                )
            time.sleep(0.5)
        raise RuntimeError(f"no still odometry within {SLAM_RESTART_WAIT_S:.0f} s of the restart")


    def begin_context_retention(self, context, *, sample_context=None):
        if self.view_camera is not None:
            self._context_distance = {"epoch": self.motion_epoch(), "active": True,
                                      "last_pose": None,
                                      "traveled_m": 0.}
            self.view_camera.begin_retention(context, sample_context=sample_context)

    def finish_context_retention(self, ended_wall_s, *, tail_s=3.0):
        if self.view_camera is not None:
            self.view_camera.finish_retention(ended_wall_s, tail_s=tail_s)
            distance = self._context_distance
            if distance is not None:
                distance["active"] = False

    def retained_context_samples(self):
        if self.view_camera is None:
            return None
        result = self.view_camera.retained_samples()
        distance = self.context_traveled_m()
        if distance is not None:
            result["traveled_m"] = distance
        return result

    def context_traveled_m(self):
        distance = self._context_distance
        if distance is not None and distance["epoch"] == self.motion_epoch():
            return distance["traveled_m"]
        return None


    # -- motion: the drive skill --------------------------------------------------------------
    def arm(self, explorer, status_cb) -> None:
        """Arm the kernel once and wait for the remote's Y then A to deploy the walking policy."""
        self.require_actuation()
        if not self.zeroed():
            raise Stopped("press Zero at the start point before starting the exploration")
        if self.backend is None:
            self.backend = ExploreRealBackend(source=self, explorer=explorer, iface=self.iface,
                                              map_artifact=self.live_map_path)
        self.backend.arm_body()
        note = self.backend.hands_note()  # printed once
        if note:
            print(f"[explore] {note}", flush=True)
        while not self.backend.body_deployed():
            status_cb("The body is ready. On the remote, press Y for the initial pose, then A to start walking.")
            if self.stop_flag.is_set():
                raise Stopped("STOP was pressed before the walking policy started")
            time.sleep(0.5)
        t0 = time.monotonic()
        while time.monotonic() - t0 < GO_COUNTDOWN_S:  # a stop cancels the countdown
            left = GO_COUNTDOWN_S - (time.monotonic() - t0)
            status_cb(f"Walking policy started. The run begins in {left:.0f} s; STOP cancels it.")
            if self.stop_flag.is_set():
                raise Stopped("STOP was pressed during the countdown")
            time.sleep(0.25)

    def close(self) -> None:
        if self._closed:
            if self._close_errors:
                raise RuntimeError("source shutdown failed: " + "; ".join(self._close_errors))
            return
        self._closed = True
        self._close_errors = errors = []
        self.stop_flag.set()
        self.localization.stop()
        def finish(name, operation):
            try:
                operation()
            except Exception as exc:  # noqa: BLE001 -- attempt every owner even when the final hold fails
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if self.backend is not None:
            finish("motion hold", self.backend.close)
        if self._body is not None:
            finish("body display listener", lambda: self._body.close(hold=False))
        finish("camera", self.view_camera.close)
        finish("sensor worker", self.localization.close)
        finish("watchdog", self.release_watchdog)
        if errors:
            raise RuntimeError("source shutdown failed: " + "; ".join(errors))

    def require_actuation(self):
        if self.observe_only:
            raise Stopped("preview only: the robot cannot move in this mode")

    def _ctx(self, seconds: float, operation: str = "drive"):
        """The drive skill's Ctx: Observer and Actuator over this backend, the driver clock and
        a Deadline, with every event logged."""
        self.require_actuation()
        log = self.drive_log
        context, tick, last_pose = time.monotonic_ns(), 0, -float("inf")
        explorer = getattr(self.backend, "explorer", None)
        epoch = self.motion_epoch()

        def emit(kind, payload=None):
            if kind == "command":
                payload = {"tick": tick, **payload}
            if (kind == "drive.route_revalidated"
                    and payload.get("reason") == "tracking reference repaired"):
                points = payload["remaining_world_xy"]
                prior = self.navigation_state._last_drive_route
                joins = [i for i, p in enumerate(prior) if tuple(p) == tuple(points[-1])]
                # the repaired prefix plus the accepted route's tail
                tail = prior[joins[-1] + 1:] if joins else []
                payload = {**payload, "route_world_xy": points + tail}
            log.emit(context, kind, payload)
            if kind == "drive.route" and payload.get("phase") == "execute":
                self.navigation_state._last_drive_route = payload.get("route_world_xy", [])
            elif kind == "drive.route_rejoin" and payload["phase"] in ("start", "complete") or (kind == "drive.route_revalidated"
                    and payload.get("reason") == "tracking reference repaired"):
                self.navigation_state._last_drive_route = payload["route_world_xy"]
            self._motion_event(kind, payload)

        def tap(term, value, skill):
            nonlocal last_pose
            now = time.monotonic()
            if term == "base_pose" and now - last_pose >= 0.1:
                last_pose = now
                distance = self._context_distance
                if (distance is not None and distance["active"]
                        and distance["epoch"] == epoch):
                    previous = distance["last_pose"]
                    if previous is not None:
                        distance["traveled_m"] += math.dist(previous[:2], value[:2])
                    distance["last_pose"] = tuple(value)
                emit("base.pose", {"xyyaw": value, "frame": "exploration_start"})

        backend = self.backend
        refresh = getattr(backend, "refresh_execution_geometry", None)
        if callable(refresh):
            refresh()
        needs = drive.NEEDS
        observer = observations.Observer("real", backend, needs.obs, emit=emit, skill="drive",
                                         tap=tap)
        observer.execution_geometry_guarded = (
            callable(getattr(backend, "execution_refusal", None)))
        actuator = actions.Actuator(
            "real", backend, drive.NEEDS.actions, observe=observer, emit=emit, skill="drive"
        )
        def step(channel, payload):
            nonlocal tick
            moving = any(getattr(payload, k, None) for k in ("vx", "vy", "wz"))
            if channel == "lower_body" and moving:
                try:
                    self._motion_guard(epoch)
                except (Stopped, Paused) as exc:
                    emit("motion.preempted", {"request": repr(payload), "reason": str(exc)})
                    raise
            tick += 1
            try:
                actuator.step(channel, payload)
                self._motion_guard(epoch, allow_pause=not moving)
                if channel == "lower_body":
                    self._context_hold_since = (None if moving else
                        self._context_hold_since or time.time())
            except Exception as exc:  # logged, then re-raised
                emit("command.result", {"tick": tick, "channel": channel,
                                        "request": repr(payload), "ok": False,
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "details": getattr(exc, "details", None)})
                raise
            emit("command.result", {"tick": tick, "channel": channel, "ok": True})

        seconds = max(5., seconds)
        if explorer is not None and getattr(explorer, "t0", None) is not None:
            spent = getattr(explorer, "active_elapsed", explorer.elapsed)()
            seconds = min(seconds, max(0., explorer.budget_s - spent))
        emit("motion.context", {"operation": operation, "seconds": seconds,
                                "run_t0_wall_s": getattr(explorer, "t0_wall", None),
                                "leg": len(getattr(explorer, "vlm_picks", [])),
                                "frame_epoch": getattr(getattr(explorer, "visual_history", None),
                                                       "epoch", None)})
        return Ctx(
            observe=observer,
            step=step,
            now=time.monotonic,
            deadline=Deadline(time.monotonic, seconds),
            progress=lambda p: emit("drive.phase", {"phase": p}),
            log=emit,
            gate=self.pose_gate,
        )

    def motion_epoch(self):
        explorer = getattr(self.backend, "explorer", None)
        return getattr(getattr(explorer, "visual_history", None), "epoch", self.frame_epoch)

    def _motion_guard(self, epoch, *, allow_pause=False):
        if self.observe_only:
            raise Stopped("preview only: the robot cannot move in this mode")
        if self.stop_flag.is_set():
            raise Stopped("STOP was pressed")
        if self.frame_reset or self.motion_epoch() != epoch:
            raise Stopped(f"the odometry reset its frame: {self.frame_reset_why}")
        if not allow_pause and self.pause_flag.is_set():
            raise Paused("operator requested pause")

    def stationary_guard(self, generation=None):
        """The motion generation, or a check that GENERATION is current; raises on body loss."""
        if self.observe_only:
            return generation
        backend = self.backend
        if backend is None:
            raise Stopped("body unavailable during stationary recovery")
        try:
            backend.require_body_authority()
            current = backend.motion_generation()
            if not backend.body_deployed():
                raise Stopped("body policy inactive during stationary recovery")
        except (HardwareUnavailable, Refused) as exc:
            if str(exc) == "body authority or robot link lost":
                raise Stopped("body authority or robot link lost during stationary recovery") from exc
            raise Stopped(f"body unavailable during stationary recovery: {exc}") from exc
        if generation is not None and current != generation:
            raise Stopped("body link interrupted during stationary recovery")
        return current

    def _motion_event(self, kind, payload):
        publish = self.motion_event
        if publish is not None:
            publish(kind, payload)

    def _motion_stage(self, stage, reason):
        self._motion_event("status", {"stage": stage, "message": reason})

    def _fresh_for_motion(self, ctx, epoch, *, require_map=False, force_wait=False):
        """Wait for fresh observations before motion; logs and re-raises a stop."""
        try:
            return self._fresh_camera_motion(ctx, epoch, require_map=require_map,
                                             force_wait=force_wait)
        except (Stopped, HardwareUnavailable, Refused, DriveFailure) as exc:
            ctx.log("drive.observations_stopped", {"reason": str(exc),
                                                  "remaining_s": ctx.deadline.remaining()})
            raise

    def _fresh_camera_motion(self, ctx, epoch, *, require_map, force_wait=False):
        """Wait standing still for sustained fresh observations, within the action deadline."""
        explorer = getattr(self.backend, "explorer", None)
        frame = getattr(explorer, "live", None)
        wait_s = getattr(explorer, "decision_recovery_wait_s", 60.)
        readiness = getattr(self.backend, "held_observation_refusal",
                            getattr(self.backend, "observation_refusal", None))
        started = stable_since = generation = last_probe = None
        last_reason, last_log = "observations unavailable", -math.inf

        def guard():
            self._motion_guard(epoch)
            if getattr(explorer, "live", None) is not frame:
                raise Stopped("map frame changed during motion recovery")
            if started is None:
                return
            self.stationary_guard(generation)
            if ctx.deadline.expired():
                raise Stopped(f"the action ran out of time waiting for sensor data: {last_reason}")
            if (explorer is not None and explorer.t0 is not None and
                    getattr(explorer, "active_elapsed", explorer.elapsed)() >= explorer.budget_s):
                raise Stopped(f"the exploration time budget ran out waiting for sensor data: {last_reason}")
            if ctx.now() - started >= wait_s:
                raise Stopped(f"sensor data is still unavailable after {wait_s:g} s: "
                              f"{last_reason}")

        def refusal():
            reason = readiness() if readiness is not None else None
            quality = float(ctx.observe("loc_quality"))
            if not math.isfinite(quality) or quality < drive.LOC_QUALITY_FLOOR:
                reason = f"localization quality {quality} below {drive.LOC_QUALITY_FLOOR}"
            if require_map and explorer is not None:
                age = explorer.map_age_s()
                if age is None or not 0 <= age <= CAMERA_MAP_MAX_AGE_S:
                    reason = ("fresh planning map unavailable "
                              f"(>{CAMERA_MAP_MAX_AGE_S:g} s or empty)")
            return reason

        while True:
            guard()
            before = ctx.now()
            try:
                reason = refusal()
                if reason is None:
                    # one observation, without the localizer's retry loop
                    ctx.observe("base_pose")
                    reason = refusal()
            except (HardwareUnavailable, StaleObservation) as exc:
                reason = str(exc)
            now = ctx.now()
            if reason is not None:
                last_reason = str(reason)
            guard()  # a backend observation may have blocked before returning
            continuous = now - before <= 0.5 and (last_probe is None or before - last_probe <= 0.5)
            last_probe = now
            health_stable = self.health_stable_s()
            if (not force_wait and reason is None and started is None and continuous
                    and health_stable >= DECISION_FRESH_STABLE_S):
                return
            if started is None:
                started = before
                # a zero command through the original Actuator
                gait.hold(ctx, 0.1, 0., 0., 0.)
                generation = self.stationary_guard()
                guard()
            stable_since = (now if stable_since is None else stable_since) if (
                reason is None and continuous) else None
            stable_for = 0. if stable_since is None else min(now - stable_since, health_stable)
            if stable_for >= DECISION_FRESH_STABLE_S:
                ctx.log("drive.observations_ready", {"wait_s": now - started,
                                                    "stable_s": stable_for})
                return
            stage = "verifying fresh data" if reason is None else "waiting for fresh data"
            self._motion_stage(stage, last_reason)
            if now - last_log >= 1.:
                ctx.log("drive.observations_wait", {"reason": last_reason,
                        "wait_s": now - started, "stable_s": stable_for, "limit_s": wait_s,
                        "remaining_s": ctx.deadline.remaining()})
                last_log = now
            gait.hold(ctx, 0.1, 0., 0., 0.)

    def hold_position(self) -> None:
        """Command zero velocity through the current writer and confirm it, still balancing."""
        epoch = self.motion_epoch()
        self._motion_guard(epoch, allow_pause=True)
        generation = self.stationary_guard()
        self._hold_motion(self._ctx(1., "hold"))
        self._motion_guard(epoch, allow_pause=True)
        self.stationary_guard(generation)

    def _hold_motion(self, ctx):
        """Command zero velocity, also after the action deadline."""
        try:
            gait.hold(ctx, 0.2, 0., 0., 0.)
        except (HardwareUnavailable, Refused, DriveFailure) as exc:
            raise Stopped(f"could not hold after motion: {exc}") from exc

    def _wait_for_resume(self, generation, epoch):
        callback = self.pause_wait
        if not callable(callback):
            raise Paused("operator pause has no resume handler")
        started = time.monotonic()
        try:
            callback(generation, epoch)
        finally:
            self.paused_wall_s = self.paused_wall_s + time.monotonic() - started
        self._motion_guard(epoch, allow_pause=True)
        self.stationary_guard(generation)
        if self.pause_flag.is_set():
            raise Stopped("pause handler returned without an approved resume")

    def service_pause(self):
        """Stand still for a requested pause and wait for the resume."""
        if not self.pause_flag.is_set():
            return
        epoch = self.motion_epoch()
        generation = self.stationary_guard()
        self.hold_position()
        self.stationary_guard(generation)
        self._wait_for_resume(generation, epoch)

    def _resumable_motion(self, seconds, operation, action):
        """Run ACTION, resuming it after a pause with the time it had left."""
        if self.backend is None:
            raise Stopped("not armed")
        epoch = self.motion_epoch()
        motion_generation = getattr(self.backend, "motion_generation", None)
        generation = motion_generation() if callable(motion_generation) else None
        explorer = getattr(self.backend, "explorer", None)
        frame = getattr(explorer, "live", None)
        remaining = seconds
        while True:
            ctx = self._ctx(remaining) if operation == "drive" else self._ctx(remaining, operation)
            # the deadline never exceeds the time remaining
            ctx.deadline = Deadline(ctx.now, min(remaining, ctx.deadline.remaining()))
            try:
                self._motion_guard(epoch)
                result = action(ctx)
                self._motion_guard(epoch)
                return result
            except Paused:
                self._hold_motion(ctx)
                self._motion_guard(epoch, allow_pause=True)
                self.stationary_guard(generation)
                remaining = ctx.deadline.remaining()
                ctx.log("motion.paused", {"operation": operation, "remaining_s": remaining})
                self._wait_for_resume(generation, epoch)
                if getattr(explorer, "live", None) is not frame:
                    raise Stopped("map frame changed while paused")
                if remaining <= 0:
                    raise ActionRefused("action deadline reached before pause", code="TIMEOUT")
                ctx.log("motion.resumed", {"operation": operation, "remaining_s": remaining})

    def planning_radius_m(self, requested_radius_m):
        """The validated radius of the idle-arm, unloaded footprint."""
        return validate_radius_m(requested_radius_m)

    def drive_to(self, goal_xy, facing_xy, radius_m: float, seconds: float = 90.0,
                 *, requested_radius_m=None) -> None:
        return self._resumable_motion(seconds, "drive", lambda ctx:
            self.navigation.drive(ctx, goal_xy, facing_xy, radius_m,
                                  **({} if requested_radius_m is None else
                                     {"requested_radius_m": requested_radius_m})))

    @property
    def navigation(self):
        """The checked navigation skill bound to this source's sensors and motion guards."""
        def require_ready():
            if self.backend is None:
                raise Stopped("not armed")

        port = NavigationPort(
            capabilities=NavigationCapabilities(
                execution_radius_m=EXECUTION_RADIUS_M,
                minimum_radius_m=RADIUS_M, padded_radii_m=PADDED_RADII_M,
                turn_translation_limit_m=self.turn_drift_m,
                map_max_age_s=CAMERA_MAP_MAX_AGE_S),
            epoch=lambda: self.motion_epoch(),
            guard=lambda *args, **kwargs: self._motion_guard(*args, **kwargs),
            fresh=lambda *args, **kwargs: self._fresh_for_motion(*args, **kwargs),
            hold=lambda ctx: self._hold_motion(ctx),
            stage=lambda *args: self._motion_stage(*args),
            planning_radius=lambda radius: self.planning_radius_m(radius),
            recovery_map=lambda: self._recovery_map(),
            map_age=lambda: self.backend.explorer.map_age_s(),
            stationary_guard=lambda *args: self.stationary_guard(*args),
            motion_refusal=getattr(self.backend, "motion_refusal", None),
            stopped=lambda: self.stop_flag.is_set(), require_ready=require_ready,
            can_refresh_observations=callable(
                getattr(self.backend, "observation_refusal", None)))
        if not hasattr(self, "_navigation_controller"):
            self._navigation_controller = Navigator(port, self.navigation_state)
        else:
            self._navigation_controller.port = port
            self._navigation_controller.capabilities = port.capabilities
        return self._navigation_controller

    @property
    def navigation_state(self):
        if not hasattr(self, "_navigation_state"):
            self._navigation_state = NavigationState()
        return self._navigation_state

    @property
    def _last_drive_result(self):
        return self.navigation_state._last_drive_result

    def _recovery_map(self):
        explorer = self.backend.explorer
        age = explorer.map_age_s()
        map_limit = CAMERA_MAP_MAX_AGE_S
        if age is None or not 0 <= age <= map_limit:
            raise ActionRefused("recovery requires a fresh map", code="OBSERVATIONS_UNAVAILABLE")
        with explorer.lock:
            live = explorer.live
            occupancy = live.blocked(include_boundary=False)
            art = SimpleNamespace(
                occupancy=occupancy,
                T_map_px=live.T_map_px(), center_allowed=live.center_allowed.copy())
            return art, live.blocking_obstacles(), getattr(live, "version", None)

    def recover_departure(self, radius_m, seconds=30., goal_xy=None, *, requested_radius_m=None):
        """Walk the robot back to a checked departure point."""
        return self._resumable_motion(seconds, "recovery", lambda ctx:
            self.navigation.recover(ctx, radius_m, goal_xy, requested_radius_m=requested_radius_m))

    def turn_hold(self, yaw: float, hold_s: float, *, seconds: float | None = None,
                  recover_blocked: bool = False) -> None:
        """Turn to an absolute heading and settle there within the action deadline."""
        allowed_s = TURN_GIVE_UP_S + 5.0 if seconds is None else min(TURN_GIVE_UP_S + 5.0, seconds)
        if not math.isfinite(allowed_s) or allowed_s <= 0:
            raise ActionRefused("viewing turn deadline expired", code="TIMEOUT")
        anchor = [None, None]  # turn origin and recovery trigger, kept across a pause
        return self._resumable_motion(allowed_s, "turn", lambda ctx:
            self._turn_hold(ctx, yaw, hold_s, anchor, recover_blocked=recover_blocked))

    def _turn_hold(self, ctx, yaw, hold_s, anchor, *, recover_blocked=False):
        started = ctx.now()
        epoch = self.motion_epoch()
        explorer = getattr(self.backend, "explorer", None)
        frame = getattr(explorer, "live", None)
        start_xy = anchor[0]
        departure_check = None
        previous_xy = None
        last_turn_pose = None
        generation = None
        waits = None
        interruptions = []
        self._last_turn_result = None

        def guard(*, allow_pause=False):
            self._motion_guard(epoch, allow_pause=allow_pause)
            if getattr(explorer, "live", None) is not frame:
                raise Stopped("map frame changed during viewing turn")
            if ctx.deadline.expired():
                raise Stopped("action deadline reached during viewing turn")

        def err(pose=None) -> float:
            nonlocal previous_xy, last_turn_pose
            guard()
            pose = gait.pose(ctx) if pose is None else pose
            last_turn_pose = pose
            guard()
            if (self.pose_gate.waits != waits
                    or getattr(ctx.observe, "recovery_generation", 0) != generation):
                raise ActionRefused("viewing turn interrupted by localization recovery",
                                    code="LOOK_INTERRUPTED")
            if (start_xy is not None
                    and math.dist(start_xy, pose[:2]) > self.turn_drift_m):
                ctx.log("turn.drift", {"translation_m": math.dist(start_xy, pose[:2]),
                                      "translation_limit_m": self.turn_drift_m})
                raise ActionRefused("viewing turn exceeded the translation bound",
                                    code="TURN_DRIFT")
            if (departure_check is not None
                    and not drive._line_free(departure_check, previous_xy, pose[:2])):
                raise ActionRefused("viewing turn violated the strict execution footprint",
                                    code="LOOK_BLOCKED")
            previous_xy = pose[:2]
            return (yaw - pose[2] + math.pi) % (2 * math.pi) - math.pi

        def aligned():
            nonlocal reached_heading
            reached_heading = abs(err()) <= drive.LEG_ALIGN_RAD
            return reached_heading

        def restore_departure(trigger, *, resumed=False):
            if not (recover_blocked and getattr(frame, "scope_ready", False)):
                if resumed:
                    raise ActionRefused("view recovery no longer permitted", code="LOOK_BLOCKED")
                return
            guard()
            self._motion_stage("repairing", "view blocked; checking a usable departure")
            radius = self.planning_radius_m(RADIUS_M)
            anchor[1] = str(trigger)
            result = (self.navigation._recovery_departure(ctx, radius, None,
                                                          requested_radius_m=RADIUS_M)
                      if resumed else None)
            if result is None:
                result = self.navigation._checked_escape(ctx, radius, requested_radius_m=RADIUS_M)
            guard()
            # the base moved: the view is decided again
            details = {"completed": False, "departure_recovery": result,
                       "original_reason": str(trigger), "remaining_s": ctx.deadline.remaining()}
            ctx.log("turn.repositioned", details)
            raise ActionRefused(
                "view incomplete; the robot was moved back to a safe starting spot; decide again from here",
                code="LOOK_REPOSITIONED", details=details)

        if anchor[1] is not None:
            restore_departure(anchor[1], resumed=True)

        for attempt in range(2):
            try:
                guard()
                self._fresh_for_motion(ctx, epoch, require_map=bool(attempt))
                guard()
                if attempt:
                    self._motion_stage("looking", "observations recovered; retrying the same view")
                pose = gait.pose(ctx)
                if start_xy is None:
                    start_xy = anchor[0] = pose[:2]
                generation = getattr(ctx.observe, "recovery_generation", 0)
                waits = self.pose_gate.waits
                previous_xy = pose[:2]
                checker, _ = gait.occupancy_grid(ctx.observe("map"), EXECUTION_RADIUS_M,
                                                 exact_footprint=True)
                # the departure check, None when the start is already blocked
                departure_check = (checker if drive._line_free(
                    checker, previous_xy, previous_xy) else None)
                if recover_blocked and departure_check is None:
                    raise ActionRefused("viewing turn starts inside the execution footprint",
                                        code="LOOK_BLOCKED")
                off = err()
                while True:
                    reached_heading = abs(off) <= drive.LEG_ALIGN_RAD
                    if not reached_heading:
                        drive._hold_until(
                            ctx,
                            lambda: (0.0, 0.0, drive._turn_rate(err())),
                            aligned,
                            "explore.turn",
                            "turn in place until the heading is on the nose",
                            min(TURN_GIVE_UP_S, ctx.deadline.remaining()),
                            expect_s=abs(off) / max(gait.turn_speed(), 1e-6),
                        )
                    # aligned if the controller reported it or the heading is within the band
                    reached_heading = abs(err()) <= drive.LEG_ALIGN_RAD or reached_heading
                    settle_s, remaining_s = max(hold_s, 0.2), ctx.deadline.remaining()
                    gait.hold(ctx, min(settle_s, remaining_s), 0.0, 0.0, 0.0)
                    off = err()
                    if remaining_s < settle_s:
                        raise Stopped("action deadline leaves insufficient time to settle the view")
                    if abs(off) <= drive.LEG_ALIGN_RAD or not reached_heading:
                        break
                    ctx.log("turn.settle_correction", {"yaw": yaw, "remaining_rad": off,
                            "remaining_s": ctx.deadline.remaining()})
                if getattr(frame, "scope_ready", False):
                    self._fresh_for_motion(ctx, epoch, require_map=True)
                    settled_pose = self.navigation._settled_recovery_pose(ctx)
                    off = err(settled_pose)
                    radius = self.planning_radius_m(RADIUS_M)
                    checker, _ = gait.occupancy_grid(ctx.observe("map"), radius,
                                                     exact_footprint=True)
                    if not checker.is_safe(*settled_pose[:2]):
                        raise ActionRefused(
                            "viewing turn left the starting footprint blocked at planning clearance",
                            code="LOOK_BLOCKED", details={"pose": list(settled_pose),
                                "planning_radius_m": radius,
                                "execution_radius_m": EXECUTION_RADIUS_M})
                break
            except (Refused, DriveFailure, HardwareUnavailable) as refused:
                self._hold_motion(ctx)
                guard(allow_pause=True)
                cause = refused.__cause__ if isinstance(refused, DriveFailure) else refused
                trigger = {"error": str(refused), "details": getattr(cause, "details", {})}
                ctx.log("turn.failure", {"yaw": yaw, "attempt": attempt + 1, **trigger})
                if isinstance(cause, TaskScopeRefused):
                    raise ActionRefused(str(refused), code=cause.code,
                                        details=trigger) from refused
                if isinstance(cause, MotionBlocked):
                    self._fresh_for_motion(ctx, epoch)
                    restore_departure(refused)
                    raise ActionRefused(f"viewing turn blocked: {refused}",
                                        code="LOOK_BLOCKED", details=trigger) from refused
                if (isinstance(cause, ObservationUnavailable)
                        and callable(getattr(self.backend, "observation_refusal", None))):
                    interruptions.append(trigger)
                    self._fresh_for_motion(ctx, epoch, require_map=True, force_wait=True)
                    guard()
                    if attempt == 0:
                        ctx.log("turn.retry", {"yaw": yaw, "retry_count": 1,
                                "remaining_s": ctx.deadline.remaining(), "trigger": trigger})
                        continue
                    raise ActionRefused(
                        "viewing turn interrupted again after one fresh retry; "
                        f"observations recovered while holding; last trigger: {refused}",
                        code="LOOK_INTERRUPTED", details={"yaw": yaw, "retry_count": 1,
                            "observations_recovered": True, "interruptions": interruptions,
                            "completed": False}) from refused
                raise Stopped(f"the turn was refused: {refused}") from refused
            except ActionRefused as exc:
                self._hold_motion(ctx)
                guard()
                ctx.log("turn.failure", {"yaw": yaw, "attempt": attempt + 1,
                                         "error": str(exc), "code": exc.code})
                if exc.code == "LOOK_BLOCKED":
                    restore_departure(exc)
                raise
            except Exception as exc:
                self._hold_motion(ctx)
                ctx.log("turn.failure", {"yaw": yaw, "attempt": attempt + 1, "error": str(exc)})
                raise
        result = {"yaw": yaw, "remaining_rad": off,
                  "completed": abs(off) <= drive.LEG_ALIGN_RAD,
                  "translation_limit_m": self.turn_drift_m,
                  "translation_m": math.dist(start_xy, last_turn_pose[:2]),
                  "elapsed_s": ctx.now() - started, "retry_count": attempt,
                  "interruptions": interruptions}
        ctx.log("turn.result", result)
        self._last_turn_result = result
        if abs(off) > drive.LEG_ALIGN_RAD:
            why = (f"turn to {math.degrees(yaw):.0f} deg did not settle within "
                   f"{math.degrees(drive.LEG_ALIGN_RAD):g} deg after {ctx.now()-started:.1f} s"
                   f"; {math.degrees(off):.1f} deg still to turn")
            raise ActionRefused(why, code="LOOK_INCOMPLETE")

    def step_to(self, xy, seconds: float = 6.0) -> None:
        """Step onto a start-frame point with the drive's close-in step, keeping the heading,
        until within STEP_TOL_M."""
        return self._resumable_motion(seconds + 5.0, "step_to", lambda ctx:
            self._step_to(ctx, xy, min(seconds, ctx.deadline.remaining())))

    def _step_to(self, ctx, xy, seconds):
        if self.backend is None:
            raise Stopped("not armed")

        def bearing_gap() -> tuple[float, float]:
            x, y, th = gait.pose(ctx)
            dx, dy = xy[0] - x, xy[1] - y
            return wrap(math.atan2(dy, dx) - th), math.hypot(dx, dy)

        try:
            yaw0 = gait.pose(ctx)[2]
            gap0 = bearing_gap()[1]
            if gap0 > STEP_TOL_M:
                drive._hold_until(
                    ctx,
                    lambda: drive._close_twist(
                        *bearing_gap(), STEP_TOL_M, wrap(yaw0 - gait.pose(ctx)[2])
                    ),
                    lambda: bearing_gap()[1] <= STEP_TOL_M,
                    "explore.step_back",
                    "step back onto the spot the turn drifted off, no turn",
                    seconds,
                    expect_s=(gap0 - STEP_TOL_M) / max(current().drive.close_mps, 1e-6),
                )
            gait.hold(ctx, 0.2, 0.0, 0.0, 0.0)
        except (Refused, DriveFailure, HardwareUnavailable) as refused:
            cause = refused.__cause__ if isinstance(refused, DriveFailure) else refused
            if isinstance(cause, TaskScopeRefused):
                raise ActionRefused(str(refused), code=cause.code,
                                    details=cause.details) from refused
            raise Stopped(f"the step back was refused: {refused}") from refused

    def walk(
        self, path, look=None, mps: float = 0.30, radius_m: float = 0.45, seconds: float = 90.0,
        *, requested_radius_m=None
    ) -> None:
        """Drive to the end of PATH, facing where LOOK points."""
        gx, gy = path[-1]
        facing = None
        if look is not None:
            yaw = look(gx, gy)
            facing = (gx + 2.0 * math.cos(yaw), gy + 2.0 * math.sin(yaw))
        self.drive_to((gx, gy), facing, radius_m, seconds,
                      **({} if requested_radius_m is None else
                         {"requested_radius_m": requested_radius_m}))


class _WatchdogPause:
    """SIGSTOP/SIGCONT on the robot's lidar_watchdog.sh, by its pidfile, over homebody's own ssh."""

    def __init__(self):
        self.transport, self.frozen = SshTransport(), False

    def freeze(self, *, allow_absent: bool = False) -> None:
        """Stop the watchdog named by its pidfile; raises unless SIGSTOP reached it."""
        try:
            said = (
                self.transport.run(
                    "p=$(cat /tmp/homebody-lidar-watchdog.pid 2>/dev/null) && "
                    "grep -q lidar_watchdog /proc/$p/cmdline 2>/dev/null && kill -STOP $p && "
                    "echo FROZEN-$p || { pgrep -f '[/]lidar_watchdog.sh' >/dev/null "
                    "&& echo NOT-FROZEN || echo ABSENT; }",
                    "freeze the robot's LiDAR watchdog",
                )
                .strip()
                .splitlines()
            )
        except Exception as exc:
            raise RuntimeError(
                f"the robot's LiDAR watchdog could not be frozen ({exc}); launch without restarting "
                "SLAM, or stop the watchdog on the robot"
            ) from exc
        if said and said[-1] == "ABSENT" and allow_absent:
            return
        if not said or not said[-1].startswith("FROZEN-"):
            raise RuntimeError(
                "the robot's LiDAR watchdog is not running from its pidfile; launch without restarting "
                "SLAM, or stop the watchdog on the robot"
            )
        self.frozen = True
        print(f"[slam] watchdog {said[-1]}", flush=True)

    def thaw(self) -> None:
        if not self.frozen:
            return
        try:
            said = (
                self.transport.run(
                    "p=$(cat /tmp/homebody-lidar-watchdog.pid) && kill -CONT $p && echo THAWED-$p",
                    "thaw the robot's LiDAR watchdog",
                )
                .strip()
                .splitlines()
            )
            if not said or not said[-1].startswith("THAWED-"):
                raise RuntimeError(said[-1] if said else "no answer")
            print(f"[slam] watchdog {said[-1]}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- best-effort thaw, operator told
            print(
                f"[slam] the watchdog was not thawed: {exc} — `kill -CONT` it on the robot",
                flush=True,
            )
