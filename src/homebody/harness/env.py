"""The primitives a customized skill's `run(env, **arguments)` calls.

Everything goes through the skill Context (the Robot interface), so a skill gets camera-derived
geometry and measured robot state only. Objects are chosen as the agent chooses them: by a click
on the head image the decision was made from (`env.target(click)`), which yields a Target that
later calls re-find in fresh frames. Arm poses are 4x4 numpy arrays in the CURRENT torso frame
(x forward, y left, z up, metres); base goals are world (map) coordinates.
"""
from dataclasses import dataclass

import numpy as np

from homebody.helpers.curobo import CuroboPlanner, PlannerUnavailable, thinned
from homebody.helpers.geometry import inverse, points_in, unproject
from homebody.motion.arm import close_or_open, move_path, servo_best_effort
from homebody.skills.contract import Runner, SkillFailure
from homebody.toolkit import calibrated_grasps, clearance, diagnose_ik

MIN_VISIBLE_PIXELS = 12
LOG_LINES = 120


@dataclass(frozen=True)
class Target:
    """One segmented thing in the scene, chosen by a click; stable across frames."""
    label: int


def _click(frame, u, v):
    return [float(min(1000.0, (u + 0.25) * 1000 / (frame.rgb.shape[1] - 1))),
            float(min(1000.0, (v + 0.25) * 1000 / (frame.rgb.shape[0] - 1)))]


class Env:
    def __init__(self, ctx, frame):
        self._ctx, self._decision_frame = ctx, frame
        self._log = []

    # --- bookkeeping ---
    @property
    def _epoch(self):
        return self._ctx.observations.epoch

    def _frame(self):
        return self._ctx.observe(self._epoch)

    def log(self, *parts):
        """Add a line to the skill's result (also what print() does)."""
        if len(self._log) < LOG_LINES:
            self._log.append(" ".join(str(p) for p in parts)[:300])

    def wait(self, seconds):
        """Let SECONDS of simulated time pass, standing still."""
        for _ in range(max(1, round(seconds / self._ctx.settings.servo.tick))):
            self._ctx.advance(self._epoch)

    # --- perception ---
    def target(self, click):
        """The Target under CLICK [x, y] (0-1000) on the head image the decision saw, or None."""
        frame = self._decision_frame
        h, w = frame.labels.shape
        u = int(np.clip(round(float(click[0]) / 1000 * (w - 1)), 0, w - 1))
        v = int(np.clip(round(float(click[1]) / 1000 * (h - 1)), 0, h - 1))
        label = int(frame.labels[v, u])
        return Target(label) if label > 0 else None

    def visible(self):
        """Targets in view now: [{target, click (its middle on the current head image), pixels}]."""
        frame = self._frame()
        found = []
        for label in np.unique(frame.labels):
            if label <= 0:
                continue
            v, u = np.nonzero((frame.labels == label) & np.isfinite(frame.depth))
            if len(u) >= MIN_VISIBLE_PIXELS:
                k = int(np.argmin((u - np.median(u)) ** 2 + (v - np.median(v)) ** 2))
                found.append({"target": Target(int(label)), "click": _click(frame, u[k], v[k]), "pixels": len(u)})
        return found

    def locate(self, target):
        """TARGET's geometry from the current head depth, torso frame: {center, top_z, bottom_z,
        radius, extent, n_points}, or None when it is not visible."""
        frame = self._frame()
        mask = (frame.labels == target.label) & np.isfinite(frame.depth)
        if np.count_nonzero(mask) < MIN_VISIBLE_PIXELS:
            return None
        points = points_in(inverse(frame.world_torso) @ frame.world_camera, unproject(frame.depth, frame.intrinsics, mask))
        center = points.mean(axis=0)
        radial = np.linalg.norm(points[:, :2] - center[:2], axis=1)
        return {"center": center.round(3).tolist(), "top_z": round(float(points[:, 2].max()), 3),
                "bottom_z": round(float(points[:, 2].min()), 3), "radius": round(float(np.percentile(radial, 95)), 3),
                "extent": np.ptp(points, axis=0).round(3).tolist(), "n_points": len(points)}

    def click_of(self, target):
        """TARGET's middle as a click on the CURRENT head image, or None when not visible."""
        frame = self._frame()
        v, u = np.nonzero((frame.labels == target.label) & np.isfinite(frame.depth) & (frame.depth > 0))
        if len(u) < MIN_VISIBLE_PIXELS:
            return None
        k = int(np.argmin((u - np.median(u)) ** 2 + (v - np.median(v)) ** 2))
        return _click(frame, u[k], v[k])

    def points_near(self, center, radius=0.25, exclude=()):
        """Sensed points (torso frame) within RADIUS of CENTER, the EXCLUDE targets removed."""
        frame = self._frame()
        mask = frame.labels >= 0
        for target in exclude:
            mask &= frame.labels != target.label
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

    # --- built-in skills, callable from a customized skill ---
    def skill(self, name, **arguments):
        """Run a built-in skill (navigate, pick, place) with its own arguments: {ok, code, message}."""
        result = Runner(self._ctx).run(name, arguments, self._frame())
        self.log(name, result.code, result.message[:160])
        return {"ok": result.ok, "code": result.code, "message": result.message}

    def navigate(self, goal_xy, facing_xy, exact=True, hand="right"):
        """Built-in navigate to world GOAL_XY facing world FACING_XY. EXACT stands at the goal;
        otherwise a clear stance square to the faced surface is chosen. {ok, code, message, base}."""
        result = self.skill("navigate", goal_xy_m=[float(v) for v in goal_xy[:2]],
                            facing_xy_m=[float(v) for v in facing_xy[:2]], snap=not exact, hand=hand)
        return {**result, "base": self.state()["base"]}

    def pick(self, target, side="right"):
        """Built-in pick of a visible TARGET (walks to a grasp stance when needed, lifts into carry)."""
        click = self.click_of(target)
        if click is None:
            return {"ok": False, "code": "NOT_VISIBLE", "message": "the target is not visible"}
        return self.skill("pick", point_normalized_1000=click, hand=side)

    def place_on(self, target, side="right"):
        """Built-in place of the held object onto a visible support TARGET (its visible middle)."""
        click = self.click_of(target)
        if click is None:
            return {"ok": False, "code": "NOT_VISIBLE", "message": "the target is not visible"}
        return self.skill("place", point_normalized_1000=click, hand=side)

    def place_at(self, x_ahead, y_left, delta_z=0.0, side="right"):
        """Built-in place releasing the held object's centre at torso (x, y), hand height + DELTA_Z."""
        return self.skill("place", x_ahead_m=float(x_ahead), y_left_m=float(y_left), delta_z_m=float(delta_z), hand=side)

    # --- arm kinematics, collision and planning ---
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

    def _gate(self, side, exempt, hand):
        return clearance(self._frame(), self._ctx.arms, side, self._ctx.settings, [t.label for t in exempt],
                         hand=hand, carried=self._ctx.carried.get(side))

    def collides(self, joints, side="right", exempt=(), hand="current"):
        """None if the arm (and hand: "current" shape or "open") at JOINTS clears the sensed scene
        with the EXEMPT targets removed; otherwise the reason."""
        return self._gate(side, exempt, hand).violation(np.asarray(joints, float))

    def path_collides(self, start, end, side="right", exempt=(), hand="current"):
        return self._gate(side, exempt, hand).path_violation(np.asarray(start, float), np.asarray(end, float))

    def grasp_poses(self, target, side="right", limit=24):
        """Calibrated top grasps of TARGET, best first: [{pre, grasp, width, tier, jaw_in_palm}]."""
        found = calibrated_grasps(self._frame(), target.label, self._ctx.jaw_samples[side], self._ctx.settings.grasp, limit)
        return [{"pre": g.pre, "grasp": g.grasp, "width": round(g.width, 3), "tier": g.tier,
                 "jaw_in_palm": g.jaw_in_palm.round(3).tolist()} for g in found]

    def plan_path(self, pre_poses, goal_poses, side="right", exempt=()):
        """cuRobo: the first (pre, goal) torso-frame pair reachable collision-free from here against
        the head depth (EXEMPT targets removed): {ok, index, approach, descent, status}."""
        frame = self._frame()
        pres = [frame.world_torso @ np.asarray(p, float) for p in pre_poses]
        goals = [frame.world_torso @ np.asarray(p, float) for p in goal_poses]
        if not pres or len(pres) != len(goals):
            return {"ok": False, "status": "give equal, non-empty lists of poses"}
        planner = CuroboPlanner(self._ctx.settings)
        try:
            reply = planner.plan(frame, side, pres, goals, tuple(t.label for t in exempt))
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
            servo_best_effort(self._ctx, self._epoch, side, world, exclude_labels=tuple(t.label for t in exempt))
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
