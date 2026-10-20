"""A credit-free load test of a host's Dex1 planning servers: CLIENTS processes, standing in for that many concurrent
sessions, each make CALLS plan requests through HomeBody's admission (helpers/curobo.admission), replaying REQUEST
(a saved plan request: depth, intrinsics, camera, start, pre and palm poses) with the Dex1 posture locks. With
--mixed, half the clients lock the waist elsewhere, so servers rebuild their planners as real mixed postures make
them. Prints the time each request waited for a server and the time the server took, as percentiles.

    PYTHONPATH=CHECKOUT/src python curobo_load.py REQUEST CLIENTS CALLS [--hand assets/robot/dex1] [--mixed]
"""
import argparse
import json
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
REST = {"left": (0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0), "right": (0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0)}


def locks(hand, waist_yaw=0.0):
    """The posture a plan locks: waist, both arms at rest, both hands open (the profile's calibration)."""
    calibration = json.loads((Path(hand) / "calibration.json").read_text())
    out = {"waist_yaw_joint": waist_yaw, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0}
    for side in ("left", "right"):
        out.update({f"{side}_{name}_joint": q for name, q in zip(ARM, REST[side])})
        names = calibration["hand_joints"][side]
        out.update(zip(names, calibration.get("hand_open", {}).get(side, [0.0] * len(names))))
    return out


def client(args):
    index, request_path, calls, hand, mixed = args
    from homebody.helpers.curobo import admission
    from homebody.remote.wire import PolicyClient
    saved = np.load(request_path, allow_pickle=True)
    pre, palm = np.asarray(saved["pre"]), np.asarray(saved["palm"])
    offsets = [np.array([dx, dy, 0.0]) for dx in (0.0, 0.02) for dy in (0.0, 0.02)]
    def shifted(poses):
        return np.stack([np.r_[np.c_[p[:3, :3], p[:3, 3] + o], [[0, 0, 0, 1]]] for p in poses for o in offsets])
    posture = locks(hand, 0.2 if mixed and index % 2 else 0.0)
    request = {"side": str(saved["side"]), "start": np.asarray(saved["start"]), "pre": shifted(pre),
               "palm": shifted(palm), "depth": np.asarray(saved["depth"]), "intrinsics": np.asarray(saved["intrinsics"]),
               "camera": np.asarray(saved["camera"]), "locks": posture}
    settings = SimpleNamespace(endpoint="tcp://127.0.0.1:5561", servers=3, timeout=900.0)
    rows = []
    for _ in range(calls):
        asked = time.monotonic()
        with admission(settings) as address:
            admitted = time.monotonic()
            planner = PolicyClient(address, settings.timeout)
            try:
                if not planner.call("ping").get("configured"):
                    planner.call("configure", {"locks": posture})
                reply = planner.call("plan_grasp", request)
            finally:
                planner.close()
        rows.append({"wait": admitted - asked, "service": time.monotonic() - admitted, "server": address,
                     "success": bool(reply.get("success")), "rebuilt": bool(reply.get("reconfigured_for"))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request", type=Path)
    parser.add_argument("clients", type=int)
    parser.add_argument("calls", type=int)
    parser.add_argument("--hand", default="assets/robot/dex1")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    with multiprocessing.get_context("spawn").Pool(args.clients) as pool:
        rows = [r for part in pool.map(client, [(i, args.request, args.calls, args.hand, args.mixed)
                                                for i in range(args.clients)]) for r in part]
    total = time.monotonic() - started

    def pct(key):
        values = np.array([r[key] for r in rows])
        return {p: round(float(np.percentile(values, p)), 1) for p in (50, 90, 100)}
    print(json.dumps({"clients": args.clients, "calls": len(rows), "mixed": args.mixed, "seconds": round(total, 1),
                      "wait_s": pct("wait"), "service_s": pct("service"),
                      "rebuilt": sum(r["rebuilt"] for r in rows), "planned": sum(r["success"] for r in rows),
                      "per_server": {s: sum(r["server"] == s for r in rows) for s in sorted({r["server"] for r in rows})}}))


if __name__ == "__main__":
    main()
