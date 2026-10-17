"""Run the exploration decision loop headless on the synthetic room with SCRIPT decisions.

Entry point: Explorer(...)._run_safe() -> Session.for_exploration -> session.exploration.run.
Motion is kinematic (the route walked at the commanded speed); sensing comes from the grid.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario

LIVE_MODULES = ("homebody.real2sim.exploration.explorer", "homebody.real2sim.exploration.policy",
                "homebody.real2sim.exploration.vlm_pick", "homebody.session.exploration",
                "homebody.skills.viewing")


def bind(root: Path):
    """The modules and names the scenario drives."""
    import importlib
    sys.path[:0] = [str(root / "src"), str(root / "vendor")]
    import homebody
    assert str(root) in homebody.__file__, homebody.__file__
    from homebody.real2sim.exploration import settings
    resolved = settings.install(settings.load(root, options={"exploration": {"record": False}}))
    from homebody.helpers.raster import px_to_grid, world_to_px
    from homebody.real2sim.exploration import explorer as module
    from homebody.real2sim.exploration import vlm_pick
    from homebody.real2sim.exploration.sources import RobotSource
    from homebody.session import exploration as session
    Source.camera_description = RobotSource.camera_description  # named by the prompt
    from homebody.vlm.provider import Decision
    modules = [importlib.import_module(name) for name in LIVE_MODULES]

    def install_provider(provider):
        session.make = vlm_pick.make = lambda *args: provider

    def explorer_for(source, output):
        explorer = module.Explorer(source, None, output, budget_s=resolved.exploration.budget_s)
        explorer.brief = "Explore this room."
        explorer.pick_hook = vlm_pick.Picker(workspace=output / "provider")
        return explorer

    return {"px_to_grid": px_to_grid, "world_to_px": world_to_px, "modules": modules,
            "Decision": Decision,
            "install_provider": install_provider, "explorer_for": explorer_for,
            "run": lambda explorer: explorer._run_safe()}


IMPORT = "homebody.real2sim.exploration.explorer"


def import_seconds(root: Path, repeats: int = 5) -> float:
    """Median cold import time of the exploration package in fresh interpreters."""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith("HOMEBODY_")}
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "vendor")])
    code = (f"import time; t = time.perf_counter(); import {IMPORT}; "
            "print(time.perf_counter() - t)")
    samples = [float(subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True,
                                    capture_output=True, text=True).stdout.split()[-1])
               for _ in range(repeats)]
    return float(np.median(samples))


class Stop:
    def __init__(self, clock):
        self.clock, self.stopped = clock, False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def clear(self):
        self.stopped = False

    def wait(self, seconds):
        self.clock.sleep(seconds)
        return self.stopped


class Source:
    """Kinematic G1: walks the accepted route at the commanded speed, turns in place,
    serves PIL frames; the LiDAR map is the room grid published into the live map."""
    name, observe_only = "synthetic-parity", False

    def __init__(self, clock):
        self.clock, self.stop_flag = clock, Stop(clock)
        self.pause_flag, self.motion_event, self.pause_wait = threading.Event(), None, None
        self.position, self.sequence, self.calls = scenario.START, 0, []

    def attach(self, motion_event, pause_wait):
        self.motion_event, self.pause_wait = motion_event, pause_wait

    def now(self):
        return self.clock.time()

    def pose(self):
        return self.position

    def base_xy(self):
        return self.position[:2]

    def odom_age_s(self):
        return .1

    def health_stable_s(self):
        return math.inf

    def motion_epoch(self):
        return 1

    def stationary_guard(self, expected=None):
        return 1

    def context_image(self):
        self.sequence += 1
        return scenario.frame(self.position, self.sequence), {
            "status": "fresh", "sequence": self.sequence, "receive_age_s": .01,
            "receiver_decode_time_s": self.now() - .01}

    def image(self, name):
        return None

    def arm(self, explorer, callback):
        self.calls.append({"call": "arm"})

    def hold_position(self):
        self.calls.append({"call": "hold", "pose": list(self.position)})

    def turn_hold(self, heading, hold_s, **kwargs):
        turn = abs((heading - self.position[2] + math.pi) % (2 * math.pi) - math.pi)
        self.calls.append({"call": "turn", "heading": heading, "hold_s": hold_s,
                           "kwargs": kwargs, "from": list(self.position)})
        self.position = (*self.position[:2], heading)
        self.clock.sleep(turn / .5 + hold_s)

    def walk(self, points, mps=.3, **kwargs):
        self.calls.append({"call": "walk", "points": [list(p) for p in points],
                           "mps": mps, "kwargs": {k: v for k, v in kwargs.items() if k != "look"},
                           "look": kwargs.get("look") is not None})
        x, y, yaw = self.position
        for px, py in points:
            step = math.dist((x, y), (px, py))
            if step > 1e-9:
                yaw = math.atan2(py - y, px - x)
                self.clock.sleep(step / mps)
                x, y = float(px), float(py)
        self.position = (x, y, yaw)


def run(root: Path, output: Path) -> dict:
    names = bind(root)
    import skimage.morphology
    medial_axis = skimage.morphology.medial_axis
    skimage.morphology.medial_axis = lambda *a, **kw: medial_axis(*a, **(kw | {"rng": 0}))
    clock = scenario.Clock()
    for module in names["modules"]:
        module.time = clock
    source = Source(clock)
    explorer = names["explorer_for"](source, output)
    grid = scenario.room()
    live = explorer.live
    map_s = []

    def publish():
        """The room as LiDAR evidence: free floor where the grid is free, hits on walls."""
        started = time.perf_counter()
        with explorer.lock:
            yy, xx = np.indices(live.floor.shape)
            x, y = live.xy(xx, yy)
            inside = (x >= scenario.ORIGIN_M) & (x < scenario.ORIGIN_M + scenario.SIZE * scenario.CELL_M) \
                & (y >= scenario.ORIGIN_M) & (y < scenario.ORIGIN_M + scenario.SIZE * scenario.CELL_M)
            free = inside & ~scenario.world(grid, x, y)
            live.floor[free] = np.maximum(live.floor[free], 1)
            v, u = np.nonzero(grid)
            wx, wy = u * scenario.CELL_M + scenario.ORIGIN_M, v * scenario.CELL_M + scenario.ORIGIN_M
            live.mark_obstacle(np.c_[wx, wy, np.full(len(wx), .5)])
            live.version += 1
        map_s.append(time.perf_counter() - started)

    publish()
    explorer.map_age_s = lambda: .1
    provider = scenario.ScriptedProvider(explorer, names["px_to_grid"], names["world_to_px"],
                                         names["Decision"])
    names["install_provider"](provider)
    marks = []
    generate = provider.generate

    def timed(*args, **kwargs):
        marks.append(("start", time.perf_counter()))
        try:
            return generate(*args, **kwargs)
        finally:
            marks.append(("end", time.perf_counter()))
            publish()

    provider.generate = timed
    started = time.perf_counter()
    names["run"](explorer)
    marks.append(("start", time.perf_counter()))
    wall_s = time.perf_counter() - started
    between = [b[1] - a[1] for a, b in itertools.pairwise(marks) if a[0] == "end" and b[0] == "start"]
    decisions = []
    for row in explorer.vlm_picks:
        result = row.get("action_result") or {}
        decisions.append({"n": row["n"], "kind": row["kind"], "used": row.get("used"),
                          "outcome": row.get("outcome"), "rejection": row.get("rejection"),
                          "rejection_code": row.get("rejection_code"), "goal": row.get("goal"),
                          "face_yaw": row.get("face_yaw"),
                          "result_code": result.get("code"),
                          "result_outcome": result.get("outcome"),
                          "pose_after": result.get("pose_xyyaw")})
    history = explorer.visual_history.manifest() if explorer.visual_history is not None else {}
    region = explorer.task_scope_context()
    return scenario.plain({
        "decisions": decisions,
        "requests": provider.requests,
        "calls": source.calls,
        "phase": explorer.phase,
        "previous_action_result": explorer.previous_action_result,
        "coverage": explorer.coverage, "enclosure": explorer.enclosure,
        "region": {k: region.get(k) for k in ("revision", "status", "ownership",
                                               "vertices_start_frame_m")},
        "scope_ready": bool(getattr(explorer.live, "scope_ready", False)),
        "views": {"count": len(history.get("view_ids") or []),
                  "coverage_assessment": history.get("coverage_assessment")},
        "final_pose": list(source.position),
        "artifacts": sorted(str(p.relative_to(output)) for p in output.rglob("*")
                            if p.is_file() and not str(p.relative_to(output)).startswith(
                                ("session/", "provider/"))),
        "timings": {"wall_s": round(wall_s, 4),
                    "decision": scenario.percentiles(between),
                    "provider": scenario.percentiles(provider.provider_s),
                    "map_update": scenario.percentiles(map_s)},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    report = {"explore": run(root, args.workdir.resolve())}
    report["explore"]["timings"]["import_s"] = round(import_seconds(root), 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    row = report["explore"]
    print(json.dumps({"phase": row["phase"],
                      "decisions": [(d["n"], d["kind"], d["outcome"], d["rejection_code"])
                                    for d in row["decisions"]], "timings": row["timings"]}))
