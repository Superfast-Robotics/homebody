"""The inner program's robot API: what a model-written `run(env)` can call.

Everything goes through the skill Context (the Robot interface), so a program gets camera-derived
geometry and measured robot state only. Arm poses are 4x4 numpy arrays in the CURRENT torso frame
(x forward, y left, z up, metres); base goals and scene memory are in the world (map) frame.
"""
import math

import numpy as np

from homebody.helpers.curobo import CuroboPlanner, PlannerUnavailable, thinned
from homebody.helpers.geometry import inverse, points_in, unproject
from homebody.motion.arm import close_or_open, move_path, servo_best_effort
from homebody.skills.contract import Runner, SkillFailure
from homebody.toolkit import calibrated_grasps, clearance, diagnose_ik

MIN_VISIBLE_PIXELS = 12
LOG_LINES = 200


class Env:
    def __init__(self, ctx, memory, labels):
        self._ctx, self._memory, self._labels = ctx, memory, dict(labels)
        self._log = []

    # --- bookkeeping ---
    @property
    def _epoch(self):
        return self._ctx.observations.epoch

    def _frame(self):
        return self._ctx.observe(self._epoch)

    def _label(self, entity):
        if entity not in self._labels:
            raise KeyError(f"no entity {entity!r}; known: {sorted(self._labels)}")
        return self._labels[entity]

    def log(self, *parts):
        """Print to the attempt's report (also what print() does)."""
        if len(self._log) < LOG_LINES:
            self._log.append(" ".join(str(p) for p in parts)[:300])

    def trace(self, what, **data):
        self.log(f"[{what}]", {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()})

    def wait(self, seconds):
        """Let SECONDS of simulated time pass, standing still."""
        for _ in range(max(1, round(seconds / self._ctx.settings.servo.tick))):
            self._ctx.advance(self._epoch)

    # --- perception ---
    def memory(self):
        """The scene memory: [{id, label, location, anchor_xyz (world), portable}] from the scan."""
        return [dict(item) for item in self._memory]

    def visible(self):
        """Entity ids in view now, with their visible pixel counts."""
        frame = self._frame()
        counts = {}
        for entity, label in self._labels.items():
            n = int(np.count_nonzero((frame.labels == label) & np.isfinite(frame.depth)))
            if n >= MIN_VISIBLE_PIXELS:
                counts[entity] = n
        return counts

    def locate(self, entity):
        """Geometry of ENTITY from the head camera's depth, torso frame: {center, top_z, bottom_z,
        radius, extent, n_points}, or None when it is not visible."""
        frame = self._frame()
        mask = (frame.labels == self._label(entity)) & np.isfinite(frame.depth)
        if np.count_nonzero(mask) < MIN_VISIBLE_PIXELS:
            return None
        points = points_in(inverse(frame.world_torso) @ frame.world_camera, unproject(frame.depth, frame.intrinsics, mask))
        center = points.mean(axis=0)
        radial = np.linalg.norm(points[:, :2] - center[:2], axis=1)
        return {"center": center.round(3).tolist(), "top_z": round(float(points[:, 2].max()), 3),
                "bottom_z": round(float(points[:, 2].min()), 3), "radius": round(float(np.percentile(radial, 95)), 3),
                "extent": np.ptp(points, axis=0).round(3).tolist(), "n_points": len(points)}

    def points_near(self, center, radius=0.25, exclude=()):
        """Sensed points (torso frame) within RADIUS of CENTER, the EXCLUDE entities removed."""
        frame = self._frame()
        mask = frame.labels >= 0
        for entity in exclude:
            mask &= frame.labels != self._label(entity)
        points = points_in(inverse(frame.world_torso) @ frame.world_camera, unproject(frame.depth, frame.intrinsics, mask))
        near = points[np.linalg.norm(points - np.asarray(center, float), axis=1) < radius]
        return near[:: max(1, len(near) // 3000)]

    def state(self):
        """Base pose [x, y, yaw] (world), and per hand: palm position (torso), closure, holding."""
        m = self._ctx.measure(self._epoch)
        hands = {side: {"palm": (inverse(m.world_torso) @ m.palm(self._ctx.arms[side], side))[:3, 3].round(3).tolist(),
                        "closure": round(m.hands[side].closure, 2),
                        "holding": bool(m.hands[side].retained(self._ctx.settings.grasp))} for side in ("left", "right")}
        return {"base": np.round(m.base_pose, 3).tolist(), "hands": hands}

    def to_world(self, xyz):
        """A torso-frame point in world coordinates."""
        m = self._ctx.measure(self._epoch)
        return (m.world_torso @ np.array([*np.asarray(xyz, float)[:3], 1.0]))[:3].round(4).tolist()

    def to_torso(self, xyz):
        """A world point in the current torso frame."""
        m = self._ctx.measure(self._epoch)
        return (inverse(m.world_torso) @ np.array([*np.asarray(xyz, float)[:3], 1.0]))[:3].round(4).tolist()

    # --- base ---
    def navigate(self, goal_xy, facing_xy, exact=True, hand="right"):
        """Walk to world GOAL_XY facing world FACING_XY. EXACT stands at the goal; otherwise
        HomeBody snaps to a clear stance square to the faced surface. Returns {ok, code, message, base}."""
        result = Runner(self._ctx).run("navigate", {"goal_xy_m": [float(v) for v in goal_xy[:2]],
                                                    "facing_xy_m": [float(v) for v in facing_xy[:2]],
                                                    "snap": not exact, "hand": hand}, self._frame())
        self.log("navigate", result.code, result.message[:120])
        return {"ok": result.ok, "code": result.code, "message": result.message, "base": self.state()["base"]}

    # --- arm kinematics and collision ---
    def palm_pose(self, side="right"):
        m = self._ctx.measure(self._epoch)
        return inverse(m.world_torso) @ m.palm(self._ctx.arms[side], side)

    def arm_joints(self, side="right"):
        return np.array(self._ctx.measure(self._epoch).arm_joints(side))

    def fk(self, joints, side="right"):
        return self._ctx.arms[side].forward(np.asarray(joints, float))[0]

    def ik(self, pose, side="right", seed=None, position_only=False):
        seed = self.arm_joints(side) if seed is None else np.asarray(seed, float)
        return self._ctx.arms[side].solve(np.asarray(pose, float), seed, position_only=position_only)

    def diagnose_ik(self, pose, side="right", seed=None):
        """Why POSE fails: {ok, reason: solved/beyond_reach/orientation/no_solution, joints, ...}."""
        return diagnose_ik(self._ctx.arms[side], pose, self.arm_joints(side) if seed is None else seed)

    @staticmethod
    def make_pose(position, finger_direction, closing_axis):
        """A palm pose: column 0 the finger direction, column 1 the closing axis (orthonormalised)."""
        x = np.asarray(finger_direction, float)
        x = x / np.linalg.norm(x)
        y = np.asarray(closing_axis, float)
        y = y - x * (x @ y)
        y = y / np.linalg.norm(y)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
        pose[:3, 3] = position
        return pose

    def collides(self, joints, side="right", exempt=(), hand="current"):
        """None if the arm (and hand: "current" shape or "open") at JOINTS clears the sensed scene
        with the EXEMPT entities removed; otherwise the reason."""
        gate = clearance(self._frame(), self._ctx.arms, side, self._ctx.settings,
                         [self._label(e) for e in exempt], hand=hand, carried=self._ctx.carried.get(side))
        return gate.violation(np.asarray(joints, float))

    def path_collides(self, start, end, side="right", exempt=(), hand="current"):
        gate = clearance(self._frame(), self._ctx.arms, side, self._ctx.settings,
                         [self._label(e) for e in exempt], hand=hand, carried=self._ctx.carried.get(side))
        return gate.path_violation(np.asarray(start, float), np.asarray(end, float))

    def grasp_poses(self, entity, side="right", limit=24):
        """Calibrated top grasps of ENTITY, best first: [{pre, grasp, width, tier, jaw_in_palm}]."""
        found = calibrated_grasps(self._frame(), self._label(entity), self._ctx.jaw_samples[side],
                                  self._ctx.settings.grasp, limit)
        return [{"pre": g.pre, "grasp": g.grasp, "width": round(g.width, 3), "tier": g.tier,
                 "jaw_in_palm": g.jaw_in_palm.round(3).tolist()} for g in found]

    def plan_path(self, pre_poses, goal_poses, side="right", exempt=()):
        """cuRobo: the first (pre, goal) torso-frame pair reachable collision-free from here against
        the head depth (EXEMPT removed): {ok, index, approach, descent, status}. Needs the server."""
        frame = self._frame()
        pres = [frame.world_torso @ np.asarray(p, float) for p in pre_poses]
        goals = [frame.world_torso @ np.asarray(p, float) for p in goal_poses]
        if not pres or len(pres) != len(goals):
            return {"ok": False, "status": "give equal, non-empty lists of poses"}
        planner = CuroboPlanner(self._ctx.settings)
        try:
            reply = planner.plan(frame, side, pres, goals, tuple(self._label(e) for e in exempt))
        except PlannerUnavailable as error:
            return {"ok": False, "status": f"planner unavailable: {error}"}
        finally:
            planner.close()
        limit = self._ctx.settings.curobo.max_rows
        return {"ok": bool(reply.get("success")), "index": int(reply.get("index", -1)),
                "approach": [r.tolist() for r in thinned(np.asarray(reply.get("approach", [])), limit)],
                "descent": [r.tolist() for r in thinned(np.asarray(reply.get("descent", [])), limit)],
                "status": str(reply.get("status"))}

    # --- arm motion and hand ---
    def follow(self, joint_waypoints, side="right"):
        """Stream the arm through waypoints from plan_path (no live check). None, or why it stopped."""
        try:
            move_path(self._ctx, self._epoch, side, np.asarray(joint_waypoints, float), planned=True)
            return None
        except SkillFailure as error:
            return str(error)

    def move_through(self, joint_waypoints, side="right"):
        """Stream the arm through waypoints with the live depth check. None, or why it stopped."""
        try:
            move_path(self._ctx, self._epoch, side, np.asarray(joint_waypoints, float))
            return None
        except SkillFailure as error:
            return str(error)

    def servo_to(self, pose, side="right", exempt=()):
        """Cartesian servo of the palm to torso POSE, live-checked: (reached_pose, message)."""
        world = self._ctx.measure(self._epoch).world_torso @ np.asarray(pose, float)
        try:
            servo_best_effort(self._ctx, self._epoch, side, world, exclude_labels=tuple(self._label(e) for e in exempt))
            message = "ok"
        except SkillFailure as error:
            message = str(error)
        reached = self.palm_pose(side)
        short = float(np.linalg.norm(reached[:3, 3] - np.asarray(pose)[:3, 3]))
        return reached, (message if short < 0.015 else f"{message}; {short * 100:.1f} cm short")

    def grip(self, closure, side="right"):
        """Hand closure: 0 open, 1 closed, up to 1.12 squeezing. Returns the closure reached."""
        frame = close_or_open(self._ctx, self._epoch, side, float(np.clip(closure, 0, 1.12)),
                              self._ctx.settings.grasp.close_seconds)
        return round(frame.hands[side].closure, 2)

    def holding(self, side="right"):
        return bool(self._ctx.measure(self._epoch).hands[side].retained(self._ctx.settings.grasp))

    # --- HomeBody's skills ---
    def _pixel(self, frame, entity):
        v, u = np.nonzero((frame.labels == self._label(entity)) & np.isfinite(frame.depth) & (frame.depth > 0))
        if len(u) < MIN_VISIBLE_PIXELS:
            return None
        k = int(np.argmin((u - np.median(u)) ** 2 + (v - np.median(v)) ** 2))
        return [float(min(1000.0, (u[k] + 0.25) * 1000 / (frame.rgb.shape[1] - 1))),
                float(min(1000.0, (v[k] + 0.25) * 1000 / (frame.rgb.shape[0] - 1)))]

    def skill(self, name, **arguments):
        """Run a HomeBody skill (navigate, pick, place) with its own arguments: {ok, code, message}."""
        result = Runner(self._ctx).run(name, arguments, self._frame())
        self.log(name, result.code, result.message[:160])
        return {"ok": result.ok, "code": result.code, "message": result.message}

    def pick(self, entity, side="right"):
        """HomeBody's pick of a visible ENTITY (walks to a grasp stance if needed, lifts into carry)."""
        frame = self._frame()
        pixel = self._pixel(frame, entity)
        if pixel is None:
            return {"ok": False, "code": "NOT_VISIBLE", "message": f"{entity} is not visible"}
        result = Runner(self._ctx).run("pick", {"point_normalized_1000": pixel, "hand": side}, frame)
        self.log("pick", entity, result.code, result.message[:160])
        return {"ok": result.ok, "code": result.code, "message": result.message}

    def place_on(self, entity, side="right"):
        """HomeBody's place of the held object onto a visible support ENTITY (its visible middle)."""
        frame = self._frame()
        pixel = self._pixel(frame, entity)
        if pixel is None:
            return {"ok": False, "code": "NOT_VISIBLE", "message": f"{entity} is not visible"}
        result = Runner(self._ctx).run("place", {"point_normalized_1000": pixel, "hand": side}, frame)
        self.log("place_on", entity, result.code, result.message[:160])
        return {"ok": result.ok, "code": result.code, "message": result.message}

    def place_at(self, x_ahead, y_left, delta_z=0.0, side="right"):
        """HomeBody's place releasing the held object's centre at torso (x, y), hand height + DELTA_Z."""
        result = Runner(self._ctx).run("place", {"x_ahead_m": float(x_ahead), "y_left_m": float(y_left),
                                                 "delta_z_m": float(delta_z), "hand": side}, self._frame())
        self.log("place_at", round(x_ahead, 3), round(y_left, 3), result.code, result.message[:160])
        return {"ok": result.ok, "code": result.code, "message": result.message}

    @staticmethod
    def yaw_to(base, xy):
        """Heading (rad) from BASE [x, y, ...] toward world XY."""
        return math.atan2(xy[1] - base[1], xy[0] - base[0])
