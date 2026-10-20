"""A credit-free load test of a host's Dex1 planning servers: CLIENTS processes, standing in for that many concurrent
sessions, each make CALLS plan requests through the path a skill or the agent's code takes, Env.plan_path ->
CuroboPlanner.plan (admission, configuration, plan), with the Dex1 settings (configs/dex1.toml), replaying REQUEST
(a saved plan request: depth, intrinsics, camera, start, pre and palm poses) from a frame rebuilt around it. With
--mixed, half the clients turn the waist, so servers rebuild their planners as real mixed postures make them. Prints
the time each request waited for a server and the time the whole call took, as percentiles, and every reply status.

    PYTHONPATH=CHECKOUT/src python curobo_load.py REQUEST CLIENTS CALLS [--root CHECKOUT] [--mixed]
"""
import argparse
import json
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")


def client(args):
    index, request_path, calls, root, mixed = args
    from contextlib import contextmanager

    from homebody.harness.env import Env
    from homebody.helpers import curobo
    from homebody.session.settings import Settings
    settings = Settings.load(Path(root) / "configs/simulation.toml", override=Path(root) / "configs/dex1.toml")
    saved = np.load(request_path, allow_pickle=True)
    side = str(saved["side"])
    kinematics = curobo.CuroboPlanner(settings).kinematics
    q = np.zeros(len(kinematics.names))
    q[kinematics.names.index("waist_yaw_joint")] = 0.2 if mixed and index % 2 else 0.0
    arm = {s: [kinematics.names.index(f"{s}_{n}_joint") for n in ARM] for s in ("left", "right")}
    for s in ("left", "right"):  # the other arm at rest, as a session holds it
        q[arm[s]] = np.asarray(saved["start"]) if s == side else settings.placement.rest_arm(s)
    world_torso = kinematics.torso_in_pelvis(q)  # the pelvis at the world origin: the saved poses are pelvis-frame
    gripper = len(json.loads((settings.robot.directory(Path(root)) / "calibration.json").read_text())["hand_joints"]["left"])
    frame = SimpleNamespace(joints=q, world_torso=world_torso, arm_joints=lambda s: q[arm[s]],
                            hands={s: SimpleNamespace(joints=np.zeros(gripper)) for s in ("left", "right")},
                            depth=np.asarray(saved["depth"]), labels=np.zeros(np.asarray(saved["depth"]).shape, int),
                            intrinsics=np.asarray(saved["intrinsics"]), world_camera=np.asarray(saved["camera"]))
    env = Env(SimpleNamespace(settings=settings, observations=SimpleNamespace(epoch=0), observe=lambda epoch: frame), frame)
    torso = np.linalg.inv(world_torso)
    offsets = [np.array([dx, dy, 0.0]) for dx in (0.0, 0.02) for dy in (0.0, 0.02)]

    def shifted(poses):
        return [torso @ np.r_[np.c_[p[:3, :3], p[:3, 3] + o], [[0, 0, 0, 1]]] for p in poses for o in offsets]
    waits, admission = [], curobo.admission

    @contextmanager
    def timed(s):
        asked = time.monotonic()
        with admission(s) as address:
            waits.append(time.monotonic() - asked)
            yield address
    curobo.admission = timed
    rows = []
    for _ in range(calls):
        asked = time.monotonic()
        reply = env.plan_path(shifted(np.asarray(saved["pre"])), shifted(np.asarray(saved["palm"])), side)
        rows.append({"wait": waits[-1] if len(waits) > len(rows) else float("nan"), "total": time.monotonic() - asked,
                     "success": bool(reply.get("ok")), "status": str(reply.get("status"))[:60],
                     "rebuilt": bool(reply.get("reconfigured_for"))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request", type=Path)
    parser.add_argument("clients", type=int)
    parser.add_argument("calls", type=int)
    parser.add_argument("--root", default=".", help="the checkout (its configs and assets/robot/dex1)")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    with multiprocessing.get_context("spawn").Pool(args.clients) as pool:
        rows = [r for part in pool.map(client, [(i, args.request, args.calls, args.root, args.mixed)
                                                for i in range(args.clients)]) for r in part]
    total = time.monotonic() - started

    def pct(key):
        values = np.array([r[key] for r in rows if r[key] == r[key]])
        return {p: round(float(np.percentile(values, p)), 1) for p in (50, 90, 100)} if len(values) else None
    print(json.dumps({"clients": args.clients, "calls": len(rows), "mixed": args.mixed, "seconds": round(total, 1),
                      "wait_s": pct("wait"), "total_s": pct("total"),
                      "rebuilt": sum(r["rebuilt"] for r in rows), "planned": sum(r["success"] for r in rows),
                      "statuses": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}}))


if __name__ == "__main__":
    main()
