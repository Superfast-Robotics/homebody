"""What the dashboard draws over observations, derived from decisions and skill events."""
import numpy as np

from homebody.skills import navigate, pick, place
from homebody.skills.contract import SkillFailure
from homebody.ui.selection import Pen, Selection, decision_topdown, draw_path
from homebody.ui.theme import FACING, GOAL, HEADING, ROBOT, ROUTE, STANCE
from homebody.vlm.observations import depth_preview, label_preview, map_preview


def palm_path(arm, frame, plan):
    """World palm positions a prepared pick or place follows, from the frame it was planned from."""
    start = frame.palm(arm, plan.side)[:3, 3]
    if isinstance(plan, pick.Plan):
        waypoints = [(frame.world_torso @ arm.forward(q)[0])[:3, 3] for q in plan.approach_joints]
        return np.array([start, *waypoints, plan.pre[:3, 3], plan.palm[:3, 3]])
    return np.array([start, plan.pre[:3, 3], plan.palm[:3, 3]])


class Overlays:
    """What the cockpit draws over observations, derived from decisions and skill events. A prepared plan is drawn against the newest observation at its event."""
    def __init__(self, topdown_settings, arms=None):
        self.topdown_settings, self.arms = topdown_settings, arms or {}
        self.epoch = self.latest = None
        self.clear()

    def clear(self):
        self.decision = self.captured = self.selection = self.palm_path = self.resolved = None
        self.marks = {}

    def observe(self, frame):
        """Track the newest observation; True when a new epoch invalidated every overlay."""
        stale = frame.epoch != self.epoch
        if stale:
            self.clear()
            self.epoch = frame.epoch
        self.latest = frame
        return stale

    def select(self, decision, frame):
        self.clear()
        self.epoch, self.decision, self.captured = frame.epoch, decision, frame
        self.selection = Selection.from_decision(decision, frame)
        if decision.skill == "navigate":
            try:
                goal, facing, _, _ = navigate.goal_arguments(decision.arguments, frame.navigation)
            except SkillFailure:
                return
            self.marks = {"goal": goal} if facing is None else {"goal": goal, "facing": facing}

    def prepared(self, plan):
        """Add a prepared plan's overlays; returns the observation they are drawn against."""
        frame = self.captured if self.latest is None else self.latest
        route = getattr(plan, "drive_plan", plan)
        if isinstance(route, navigate.Plan):
            points = route.waypoints
            if route.departure_goal is not None:
                points = [frame.base_pose[:2], route.departure_goal, *points]
            yaw = float(frame.base_pose[2]) if route.yaw is None else route.yaw
            self.marks = {**self.marks, "route": points, "stance": [*route.goal, yaw]}
        if isinstance(plan, (pick.Plan, place.Plan)):
            arm = self.arms.get(plan.side)
            self.palm_path = None if arm is None else palm_path(arm, frame, plan)
        if isinstance(plan, place.Plan):
            self.resolved = plan.point
        return frame

    def returning(self, event):
        """Clear the approach path once the arm starts its return."""
        self.palm_path = None

    def ego_rgb(self, frame):
        """RGB with the selected segment, its click and the palm path."""
        selection, path = self.selection, self.palm_path  # clear() may run on another thread
        rgb = frame.rgb if selection is None else selection.overlay(frame, frame.rgb)
        if path is not None:
            rgb = draw_path(frame, rgb, path)
        return rgb

    def ego(self, frame):
        """Replay view: RGB, depth and segmentation from the same observation."""
        return self.ego_rgb(frame), depth_preview(frame.depth), label_preview(frame.labels)

    def map(self, frame):
        """The agent's map with the decision's marks, and the robot drawn over them."""
        pen, world_to_pixel = Pen(map_preview(frame)), np.linalg.inv(frame.navigation.T_map_px)

        def px(xy):
            point = world_to_pixel @ np.r_[np.asarray(xy, dtype=float)[:2], 1.]
            return point[:2] / point[2]

        def pose(xy_yaw, colour):
            x, y, yaw = xy_yaw
            pen.pose(px((x, y)), px((x + HEADING * np.cos(yaw), y + HEADING * np.sin(yaw))), colour)

        marks = self.marks
        if len(marks.get("route", ())) > 1:
            pen.path([px(point) for point in marks["route"]], ROUTE)
            for point in marks["route"][1:]:
                pen.dot(px(point), ROUTE)
        for name, colour in (("goal", GOAL), ("facing", FACING)):
            if name in marks:
                pen.ring(px(marks[name]), colour)
        if "stance" in marks:
            pose(marks["stance"], STANCE)
        pose(frame.base_pose, ROBOT)
        return pen.pixels()

    def topdown(self):
        """The decision's captured top-down: the requested release ring, the resolved dot."""
        return decision_topdown(self.decision, self.captured, self.topdown_settings, self.resolved)
