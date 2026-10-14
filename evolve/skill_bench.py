"""skill_bench.py ROOT EPISODE OUT: one skill-benchmark episode, no model involved.

The pick episode stands the robot at the episode's stance (the scene's scripted stance plus the
episode's offset), calls the candidate's pick once on the object (a click on its visible pixels, as
the agent would give), carries it along the task's carry path and stands still for a moment. Scored
from physics: 0 not lifted, 1 lifted but not held at the end (dropped in the pick or the carry),
2 still held (success). An episode whose stance or view fails before the pick is a setup failure and
is not scored. Writes OUT/result.json; the session evidence (events, physics) stays under OUT.
"""
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

LIFT_M = 0.05  # rise of the object's lowest corner that counts as lifted
HELD_M = 0.03  # still at least this far above its start at the end, in hand and unsupported
SETTLE_SECONDS = 2.0


def load_check_manipulation(root):
    spec = importlib.util.spec_from_file_location("check_manipulation", root / "tools/check_manipulation.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"no tools/check_manipulation.py under {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_manipulation"] = module
    spec.loader.exec_module(module)
    return module


def physics_verdict(rows, name):
    """(lifted, held, max rise) of object NAME over the recorded physics rows."""
    def lowest(row):
        return min(point[2] for point in row["objects"][name]["corners"])
    start = lowest(rows[0])
    lifted = any(lowest(r) - start >= LIFT_M and r["objects"][name]["hand_contacts"] for r in rows)
    end = rows[-1]["objects"][name]
    held = bool(lowest(rows[-1]) - start >= HELD_M and end["hand_contacts"] and not end["support_contacts"])
    return lifted, held, max(lowest(r) - start for r in rows)


def run_pick(cm, session, episode, step):
    port, execute = session.port, session.execute
    timing = {}
    if not execute("navigate", cm.navigate_arguments(episode["stance"], step.facing, step.hand), port.snapshot(),
                   evidence_name="stance"):
        return {"setup_failed": "stance navigation refused"}
    frame = port.snapshot()
    click = cm.target_click(frame, step.label)
    if click is None:
        return {"setup_failed": "object not visible from the stance"}
    u, v = click
    side = step.hand if step.hand != "auto" else ("left" if u < frame.rgb.shape[1] / 2 else "right")
    started = time.monotonic()
    result = session.runner.run("pick", {"point_normalized_1000": cm.normalized(frame, u, v), "hand": side}, frame)
    timing["pick_wall"] = round(time.monotonic() - started, 1)
    pick = {key: value for key, value in asdict(result).items() if key != "details"}
    pick["details"] = {k: v for k, v in (result.details or {}).items() if k != "traceback"}
    carry = None
    if result.ok and step.carry_goal is not None:
        carried = execute("navigate", cm.navigate_arguments(step.carry_goal, step.carry_facing, side),
                          port.snapshot(), evidence_name="carry")
        carry = session.outcome["steps"][-1] if session.outcome["steps"] else {"code": "OK" if carried else "?"}
    if result.ok:
        for _ in range(int(SETTLE_SECONDS / 0.02)):
            port.advance(port.epoch, 0.02)
    return {"pick": pick, "carry": carry and {k: carry.get(k) for k in ("code", "message")}, "side": side,
            "timing": timing}


def write_trace(out):
    """OUT/trace.json: the skill's own events (start, stages, traces, finish) and the scripted requests,
    without the per-tick commands, for diagnosis."""
    events = []
    for path in (out / "session").rglob("events.jsonl"):
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == "skill" and event.get("event", {}).get("type") != "observation":
                body = event["event"]
                if body.get("type") == "skill_prepared":  # the plan holds whole joint paths; keep its outline
                    body = {**body, "plan": shorten(body.get("plan"))}
                events.append({"time": event.get("time"), **body})
            elif event.get("kind") in ("scripted_request", "grip_command"):
                events.append({k: v for k, v in event.items() if k not in ("wall_time", "frame")})
    (out / "trace.json").write_text(json.dumps(events, default=str))


def shorten(value, depth=0):
    """VALUE with long lists and nested structure cut down to an outline."""
    if isinstance(value, dict):
        return {k: shorten(v, depth + 1) for k, v in list(value.items())[:30]} if depth < 3 else "{...}"
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return f"[{len(value)} items]"
        return [shorten(v, depth + 1) for v in value] if depth < 3 else "[...]"
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "..."
    return value


def main(root, episode_path, out):
    root, out = Path(root).resolve(), Path(out)
    episode = json.loads(Path(episode_path).read_text())
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root / "src"), str(root / "vendor")]
    cm = load_check_manipulation(root)
    from homebody.backends.mujoco import MujocoBackend
    from homebody.session.recorder import Recorder
    from homebody.session.settings import Settings

    scene = Path(episode["scene"])
    settings = Settings.load(root / "configs/simulation.toml", scene=scene)
    recorder = Recorder(root, settings, out / "session")
    task = recorder.start_task(f"Skill benchmark: {episode['skill']} {episode['id']}", None)
    outcome = {"status": "error", "steps": [], "vlm_used": False}
    args = SimpleNamespace(no_visuals=True, place_candidates=256, wall_timeout=episode.get("wall_timeout", 900))
    start, session, record = time.monotonic(), None, {"episode": episode["id"], "skill": episode["skill"],
                                                    "scene": scene.name, "family": episode["family"]}
    try:
        backend = MujocoBackend(root, settings, scene=scene)
        step = cm.resolved(cm.Step(**episode["step"]), backend.label_ids)
        session = cm.ScriptedSession(backend, recorder, task, outcome, root, settings, args, start)
        session.begin_cycle(0, step)
        record.update(run_pick(cm, session, episode, step))
    except Exception as error:  # noqa: BLE001 -- an episode that crashes is recorded, not fatal
        record.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-2000:])
    finally:
        if session is not None:
            session.end_cycle({"index": 0})
            session.close()
        recorder.end_task(outcome)
        recorder.close()
    rows = [json.loads(line) for line in max((out / "session").rglob("physics.jsonl")).read_text().splitlines()
            if line.strip()]
    write_trace(out)
    if rows and "setup_failed" not in record:
        lifted, held, rise = physics_verdict(rows, episode["step"]["fixture_id"])
        record.update(lifted=lifted, held=held, max_rise=round(rise, 3), stage=2 if held else 1 if lifted else 0,
                      success=held)
    record.update(wall_seconds=round(time.monotonic() - start, 1), simulation_seconds=round(rows[-1]["time"], 1) if rows else 0)
    (out / "result.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps({k: record.get(k) for k in ("episode", "stage", "success", "setup_failed", "error")}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
