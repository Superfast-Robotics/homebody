"""The walkable floor a map click lands on: the planning footprint, the seen network a click
snaps to, the limits on correcting a click, and the waypoint a walk is planned to.

`planner` is the explorer's planner (`live`, `radius_m`, `seen_network()`) and `live` its live
map (`uv`, `xy`, `known_free()`, `geodesic`, `collision_checker`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

RADIUS_M = 0.30  # planning footprint, measured against cell squares without radius rounding
TOO_CLOSE_M = 0.5  # no snap onto the network this near the robot: a leg is a walk
FACE_MIN_M = 0.2  # a face point nearer the goal than this says nothing about the facing
MAX_GOAL_ADJUST_M = 0.75  # total click-to-goal correction, including clearance padding


@dataclass
class Waypoint:
    kind: str  # "frontier" | "voronoi" | "vlm" | "look" | "home"
    xy: tuple[float, float]
    headings: list[float] = field(default_factory=list)
    look: str | None = None  # key into Planner.look_trees: face the nearest point of it
    through: bool = False  # walked through without stopping

    def stops(self) -> bool:
        return not self.through


def path_checker(live, radius_m: float = RADIUS_M, *, task_scope: bool = True):
    """The driver's exact footprint admission, shared by selection and path planning."""
    return live.collision_checker(radius_m, task_scope=task_scope)


def click_note(live, boundary, xy) -> str:
    """What the click landed on, for the record: outside the limit, black, grey, or floor."""
    x, y = xy
    u, v = (int(c) for c in live.uv(x, y))
    if not boundary.contains(np.array([[x, y]]))[0] or not (0 <= u < live.W and 0 <= v < live.H):
        return "outside the limit"
    if live.avoid()[v, u]:
        return "on black"
    if not (live.known_free()[v, u] or live.assumed[v, u]):
        return "on grey"
    return "on floor"


def reachable_network(planner, x, y):
    """The same selectable network for the model image and destination acceptance."""
    live = planner.live
    checker = path_checker(live, planner.radius_m)
    reach = np.isfinite(live.geodesic(x, y, planner.radius_m, checker=checker))
    network = planner.seen_network() & live.known_free() & reach
    vv, uu = np.nonzero(network)
    xs, ys = live.xy(uu, vv)
    network[vv, uu] &= np.hypot(xs - x, ys - y) >= TOO_CLOSE_M
    return network, reach


def departure_status(live, pose, radius_m, *, reach=None):
    """Explain normal reachability from this snapshot without changing its admission."""
    if pose is None or not np.isfinite(pose[:2]).all():
        code, message = "pose_unavailable", "Current footprint position unavailable."
    elif not live.scope_ready:
        code, message = "region_required", "Waiting for an accepted exploration region."
    else:
        x, y = pose[:2]
        if reach is None:
            reach = np.isfinite(live.geodesic(x, y, radius_m))
        if reach.any():
            code, message = "available", "Normal reachable floor available."
        else:
            u, v = (int(c) for c in live.uv(x, y))
            if 0 <= u < live.W and 0 <= v < live.H and not live.center_allowed[v, u]:
                code, message = "outside_region", "Current position is outside the permitted region."
            elif not live.collision_checker(radius_m).is_safe(x, y):
                code, message = "starting_footprint_blocked", (
                    "Current footprint blocked—normal reachable area unavailable.")
            else:
                code, message = "no_checked_connector", (
                    "No checked departure connector—normal reachable area unavailable.")
    return {"status": code, "message": message, "planning_radius_m": float(radius_m)}


def goal_adjustment(xy, goal):
    """Total correction after padding, and an explicit refusal when it loses the request."""
    distance = math.dist(xy, goal)
    if not math.isfinite(distance):
        return None, "nonfinite requested or planned destination"
    if distance > MAX_GOAL_ADJUST_M:
        return distance, (f"goal adjustment {distance:.2f} m exceeds {MAX_GOAL_ADJUST_M:.2f} m; "
                          "the requested destination cannot be preserved within this limit")
    return distance, ""


def snap_to_network(planner, xy, x, y, face_xy=None, *, network=None):
    """The nearest walkable cell of the seen network to the click: known floor reachable
    under the exact planning footprint and connector checks, at least TOO_CLOSE_M from the
    robot. (spot, snap_m, facing yaw), or None with no such cell.
    The facing is the model's alone: from the spot toward its second point; without one, the
    direction of its click from the robot."""
    live = planner.live
    if network is None:
        network, _ = reachable_network(planner, x, y)
    vv, uu = np.nonzero(network)
    if not len(vv):
        return None
    xs, ys = live.xy(uu, vv)
    j = int(np.argmin(np.hypot(xs - xy[0], ys - xy[1])))
    spot = (float(xs[j]), float(ys[j]))
    snap_m = float(math.hypot(spot[0] - xy[0], spot[1] - xy[1]))
    if face_xy is not None and math.hypot(face_xy[0] - spot[0], face_xy[1] - spot[1]) > FACE_MIN_M:
        return spot, snap_m, math.atan2(face_xy[1] - spot[1], face_xy[0] - spot[0])
    return spot, snap_m, math.atan2(xy[1] - y, xy[0] - x)
