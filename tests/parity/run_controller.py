"""Drive the checked navigation controller on the synthetic room with a kinematic body.

Entry point: RobotSource.walk -> drive_to -> Navigator -> skills.navigate_real.controller.
Run from the source tree's root in a fresh interpreter; writes one JSON report.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario

SCENARIOS = {"clear": (scenario.CLEAR_GOAL, None),
             "blocked": (scenario.BLOCKED_GOAL, scenario.BLOCKED_FACING)}
VISIBLE = {"drive.request", "drive.route", "drive.phase", "drive.retry", "drive.failure",
           "drive.arrival", "drive.arrival_confirmed", "drive.result", "drive.action_refused",
           "drive.stopped", "motion.preempted"}


def bind(root: Path):
    """The modules and names the scenario drives."""
    sys.path[:0] = [str(root / "src"), str(root / "vendor")]
    import homebody
    assert str(root) in homebody.__file__, homebody.__file__
    from stub_body import StubBody

    from homebody.real2sim.exploration import settings
    settings.install(settings.load(root, options={"exploration": {"record": False}}))

    from homebody.helpers.floor import RADIUS_M
    from homebody.motion.exploration.localizer import PoseGate
    from homebody.real2sim.exploration import sources
    from homebody.skills.navigate_real import controller

    def source_for(body):
        source = object.__new__(sources.RobotSource)
        from homebody.primitives.localization.superodom import SuperOdomLocalization
        source.localization = SuperOdomLocalization("fixture", start=False)
        source.backend = body
        source.observe_only = False
        source.pose_gate, source.turn_drift_m = PoseGate(), settings.current().exploration.turn_drift_m
        source.stop_flag, source.pause_flag = threading.Event(), threading.Event()
        source.motion_event = source.pause_wait = None
        source.paused_wall_s = 0.
        source._context_distance = source._context_hold_since = None
        source.frame_reset, source.frame_epoch, source.frame_reset_why = False, 3, ""
        source.health_stable_s = lambda: math.inf
        return source

    return {"StubBody": StubBody, "sources": sources, "controller": controller,
            "radius": RADIUS_M, "source_for": source_for,
            "refusals": (sources.ActionRefused, sources.Stopped)}


def run(root: Path, name: str) -> dict:
    names = bind(root)
    # Deterministic medial-axis tie breaking inside the real planner.
    import skimage.morphology
    medial_axis = skimage.morphology.medial_axis
    skimage.morphology.medial_axis = lambda *a, **kw: medial_axis(*a, **(kw | {"rng": 0}))
    controller, sources = names["controller"], names["sources"]
    clock = scenario.Clock()
    sources.time = controller.time = clock
    grid = scenario.room(blocked=name == "blocked")
    goal, facing = SCENARIOS[name]
    rows, ticks, plans = [], [], []

    class Log:
        def emit(self, context, kind, payload=None):
            if kind != "observe":
                rows.append({"at_s": round(clock.time() - 1000., 6), "event": kind,
                             "data": scenario.plain(payload or {})})

    class Body(names["StubBody"]):
        """Integrates each accepted planar twist for one 4 ms tick; serves the room grid."""

        def __init__(self):
            super().__init__()
            self.position = np.array(scenario.START, float)
            self.accepted, self.last_wall = [], None
            self.art = {"raster": np.zeros((scenario.SIZE, scenario.SIZE, 3), np.uint8),
                        "T_map_px": scenario.transform(), "occupancy": grid.copy(),
                        "scan_ref": "synthetic", "scan_hash": "0" * 64}

        def now(self):
            return clock.time()

        def base_pose(self):
            return self.position.copy()

        def map_artifact(self):
            return self.art

        def lidar_obstacles(self):
            return np.empty((0, 3), np.float32)

        def require_body_authority(self):
            return None

        def motion_generation(self):
            return 1

        def body_deployed(self):
            return True

        def observation_refusal(self):
            return None

        def apply_lower_body(self, cmd):
            wall = time.perf_counter()
            if self.last_wall is not None:
                ticks.append(wall - self.last_wall)
            self.last_wall = wall
            vx, vy, wz = cmd.vx or 0., cmd.vy or 0., cmd.wz or 0.
            self.accepted.append((round(vx, 6), round(vy, 6), round(wz, 6), cmd.state))
            assert len(self.accepted) <= 25000, "controller exceeded the bounded fake clock"
            if cmd.state == "walk":
                yaw = self.position[2]
                self.position += .004 * np.array([math.cos(yaw) * vx - math.sin(yaw) * vy,
                                                  math.sin(yaw) * vx + math.cos(yaw) * vy, wz])
            clock.value = round(clock.value + .004, 9)

    plan_route = controller._plan_route

    def timed_plan(*args, **kwargs):
        started = time.perf_counter()
        try:
            return plan_route(*args, **kwargs)
        finally:
            plans.append(time.perf_counter() - started)

    controller._plan_route = timed_plan
    body = Body()
    source = names["source_for"](body)
    source.drive_log = Log()
    look = None if facing is None else (
        lambda gx, gy: math.atan2(facing[1] - gy, facing[0] - gx))
    terminal = {"status": "returned", "code": None}
    started = time.perf_counter()
    try:
        source.walk([goal], look=look, mps=.30, radius_m=names["radius"], seconds=35.,
                    requested_radius_m=names["radius"])
    except names["refusals"] as exc:
        terminal = {"status": type(exc).__name__, "code": getattr(exc, "code", None),
                    "reason": str(exc)}
    wall_s = time.perf_counter() - started
    stream = []
    for command in body.accepted:
        if stream and stream[-1][:4] == list(command):
            stream[-1][4] = round(stream[-1][4] + .004, 6)
        else:
            stream.append([*command, .004])
    result = source._last_drive_result
    routes = [r["data"].get("route_world_xy") for r in rows if r["event"] == "drive.route"]
    return {
        "scenario": name, "goal": list(goal), "facing": None if facing is None else list(facing),
        "terminal": terminal,
        "refusal_codes": [r["data"].get("code") for r in rows
                          if r["event"] in {"drive.failure", "drive.action_refused"}],
        "commands": {"count": len(body.accepted), "stream": stream,
                     "moving": sum(1 for c in body.accepted if any(c[:3]))},
        "waypoints": routes[-1] if routes else None,
        "routes": routes,
        "final_pose": scenario.plain(body.position),
        "goal_error_m": round(math.dist(body.position[:2], goal), 6),
        "arrived": bool(result and result.get("arrival", {}).get("arrived")),
        "events": [r for r in rows if r["event"] in VISIBLE],
        "event_counts": dict(Counter(r["event"] for r in rows)),
        "event_trace_sha256": scenario.digest(rows),
        "command_trace_sha256": scenario.digest(body.accepted),
        "simulated_s": round(clock.time() - 1000., 6),
        "timings": {"wall_s": round(wall_s, 4), "tick": scenario.percentiles(ticks),
                    "route_plan": scenario.percentiles(plans)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"controller": {name: run(args.source_root.resolve(), name) for name in SCENARIOS}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    for name, row in report["controller"].items():
        print(json.dumps({"scenario": name, "terminal": row["terminal"]["status"],
                          "code": row["terminal"]["code"], "commands": row["commands"]["count"],
                          "final_pose": row["final_pose"], "tick": row["timings"]["tick"]}))
