"""score_episode.py SESSION TARGETS EVALUATE_PY: score one recorded HomeBody session (or the latest
session under an --output root) from physics.

Each targeted object gets a stage: 0 untouched, 1 lifted by the robot, 2 ended inside its target
region, 3 physics-verified placement (the frozen evaluator's verdict). Writes nothing; prints the
result JSON, with the session's step and coder-call counts.
"""
import json
import runpy
import sys
from pathlib import Path

import numpy as np

LIFT_M = 0.05


def main(session, targets_path, evaluate_py):
    session = Path(session)
    if not (session / "events.jsonl").exists():  # an --output root: its latest session
        session = max(session.glob("session-*"))
    spec = json.loads(Path(targets_path).read_text())
    goals = {s["fixture_id"]: s["evaluator_target"] for s in spec["steps"]} if "steps" in spec else spec["objects"]
    physics = max(session.glob("tasks/task-*/physics.jsonl"))
    rows = [json.loads(line) for line in physics.read_text().splitlines() if line.strip()]
    verdict = runpy.run_path(evaluate_py)["score_attempt"](rows, {"schema_version": 1, "frame": "ws_map", "objects": goals},
                                                           require_settled=True)
    objects = {}
    for name, goal in goals.items():
        reason = verdict["objects"].get(name, {}).get("terminal_reason", "missing")
        start = rows[0]["objects"][name]["position"]
        lifted = any(r["objects"][name]["position"][2] > start[2] + LIFT_M and r["objects"][name]["hand_contacts"] for r in rows)
        end = np.asarray(rows[-1]["objects"][name]["position"])
        low, high = (np.asarray(b) for b in goal["bounds"])
        inside = bool(np.all(end >= low) and np.all(end <= high))
        objects[name] = {"stage": 3 if reason is None else 2 if inside else 1 if lifted else 0, "reason": reason}
    outcome = json.loads((physics.parent / "outcome.json").read_text()) if (physics.parent / "outcome.json").exists() else {}
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines() if line.strip()]
    skills = [e["decision"]["skill"] for e in events if e.get("kind") == "decision"]
    print(json.dumps({"success": bool(verdict["success"]) and outcome.get("status") == "model_done",
                      "stage": round(sum(o["stage"] for o in objects.values()) / max(1, len(objects)), 3),
                      "objects": objects, "untargeted": verdict.get("untargeted"), "status": outcome.get("status"),
                      "steps": len(skills), "customizations": skills.count("call_coder"),
                      "wall_seconds": outcome.get("wall_seconds", 0.0)}, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
