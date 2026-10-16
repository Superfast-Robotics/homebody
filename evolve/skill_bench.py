"""skill_bench.py ROOT EPISODE OUT: one skill-benchmark episode.

A skill takes part through a driver in DRIVERS (episode["skill"] picks it): run() sets the episode up,
calls the skill once (with one coder rescue when enabled) and finishes the episode; score() reads the
recorded physics into stage 0, 1 or 2 (2: success). Adding a skill to the benchmark means adding a
driver here and an episode generator in skill_episodes.py.

The place episode stands the robot at the stance, picks the object with the built-in pick and carries
it to the step's place stance (a failure in that setup is not scored), then calls the candidate's place
once with the release the scripted sequence would give, over the destination (with one coder rescue
when enabled). Scored by the evaluator's own placement test against the step's target: 0 still in
hand, 1 released but not placed in the target, 2 placed (inside, released, supported and still).

The pick episode stands the robot at the episode's stance (the scene's scripted stance plus the
episode's offset), calls the candidate's pick once on the object (a click on its visible pixels, as
the agent would give); if the pick fails and the episode allows it (coder: true), the coding
assistant gets one call_coder to change pick and run it on the robot, as in a task. A held object is
then carried along the task's carry path and the robot stands still for a moment. Scored from
physics: 0 not lifted, 1 lifted but not held at the end (dropped in the pick or the carry), 2 still
held (success); the record also keeps the built-in pick's own outcome, the coder's result, its
credits and the seconds spent. An episode whose stance or view fails before the pick is a setup failure and
is not scored. Writes OUT/result.json; the session evidence (events, physics) stays under OUT.
"""
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

SOL_RATES = (50.0, 2.5, 250.0)  # coder credits per 1M tokens: input, cached input, output
UNMETERED_CREDITS = 8.2  # a full coder call (~120k uncached, ~480k cached, ~4k output tokens)
LIFT_M = 0.05  # rise of the object's lowest corner that counts as lifted
HELD_M = 0.03  # still at least this far above its start at the end, in hand and unsupported
SETTLE_SECONDS = 2.0
CARRY_RECHECKS = 2  # pick rechecks of an uncertain grasp during a place episode's carry


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
    arguments = {"point_normalized_1000": cm.normalized(frame, u, v), "hand": side}
    name = str(episode["step"]["label"]).replace("_", " ")
    record = call_skill(session, episode, "pick", arguments, frame, request={
        "goal": f"Pick up the {name} (the object under the click {arguments['point_normalized_1000']} on image 1) "
                f"with the {side} hand and hold it.",
        "success": "pick returns OK and the object stays in the hand, lifted off its support.",
        "constraints": "Change only the pick skill; do not walk away from the object.",
        "task": f"Pick up the {name} and hold it."})
    held = record["ok"] and bool(session.context.carried)
    side = next(iter(session.context.carried), side)
    timing = record["timing"]
    carry = None
    if held and step.carry_goal is not None:
        carried = execute("navigate", cm.navigate_arguments(step.carry_goal, step.carry_facing, side),
                          port.snapshot(), evidence_name="carry")
        carry = session.outcome["steps"][-1] if session.outcome["steps"] else {"code": "OK" if carried else "?"}
    if held:
        for _ in range(int(SETTLE_SECONDS / 0.02)):
            port.advance(port.epoch, 0.02)
    return {**{k: v for k, v in record.items() if k != "ok"}, "carry": carry and {k: carry.get(k) for k in ("code", "message")},
            "side": side, "timing": timing}


def call_skill(session, episode, skill, arguments, frame, request):
    """Call SKILL once from FRAME; if it refuses and the run enables the coder, one call_coder limited to
    that skill (REQUEST: goal, success, constraints, task). Returns {ok, call, coder?, timing}: ok is the
    skill's own OK, or the coder's when it rescued the call."""
    started = time.monotonic()
    carried_before = {side: dict(carry) for side, carry in session.context.carried.items()}
    result = session.runner.run(skill, arguments, frame)
    timing = {"call_wall": round(time.monotonic() - started, 1)}
    call = {key: value for key, value in asdict(result).items() if key != "details"}
    call["details"] = {k: v for k, v in (result.details or {}).items() if k != "traceback"}
    changed = harness_changes(frame, session.port.snapshot(), carried_before, session.context.carried)
    record = {"ok": result.ok, "call": call, "what_changed": changed, "timing": timing}
    if not result.ok and episode.get("coder", os.environ.get("HOMEBODY_BENCH_CODER") == "1"):
        coder = rescue(session, skill, arguments, call, request, changed)
        record.update(coder=coder, ok=coder["code"] == "OK")
        timing["coder_wall"] = coder["seconds"]
    timing["skill_wall"] = round(timing["call_wall"] + timing.get("coder_wall", 0), 1)
    return record


def harness_changes(before, after, carried_before, carried_after):
    """What the skill call did to the robot, as the coder's request shows it (none when the candidate's
    harness has no change_report)."""
    try:
        from homebody.harness import change_report
    except ImportError:
        return None
    return change_report(before, after, carried_before, carried_after)


def rescue(session, skill, arguments, call, request, changed=None):
    """One call_coder after a refused SKILL call, limited to that skill, as the agent would make it; its
    result, seconds and credits."""
    from dataclasses import replace

    from homebody.session.coder import Coder

    coder_settings = replace(session.context.settings.coder, calls=1)
    if os.environ.get("HOMEBODY_CODER_MODEL"):  # the run's coder model, whatever the candidate's settings say
        coder_settings = replace(coder_settings, model=os.environ["HOMEBODY_CODER_MODEL"])
    if os.environ.get("HOMEBODY_CODER_SECONDS"):  # a slower (local) model may get a longer limit
        coder_settings = replace(coder_settings, seconds=float(os.environ["HOMEBODY_CODER_SECONDS"]))
    coder = Coder(coder_settings)
    coder.reset(session.task)
    history = [{"operator_task": request["task"]},
               {"decision": {"text": f"Call {skill}.", "skill": skill, "arguments": arguments}, "result": call,
                **({"what_changed": changed} if changed else {})}]
    agent = SimpleNamespace(history=history, observed=session.port, recorder=session.recorder,
                            context=session.context, runner=session.runner)
    ask = {"skill": skill, "goal": request["goal"], "success": request["success"],
           "constraints": request["constraints"],
           "tried": f"{skill} {json.dumps(arguments)} returned {call['code']}: {call['message']}"}
    started = time.monotonic()
    result = coder.call(ask, agent)
    seconds = round(time.monotonic() - started, 1)
    return {"code": result["code"], "message": result["message"][:400], "seconds": seconds,
            "credits": coder_credits(session.task),
            "executes": [{k: e.get(k) for k in ("code", "message")} for e in result.get("details", {}).get("executes", [])]}


def coder_credits(task_dir):
    """Credits of the task's coder calls from Codex's usage records, at Sol's rates or HOMEBODY_CODER_RATES
    ("input,cached,output" per 1M tokens; 0,0,0 for a local model). A call cut off before Codex
    reported its usage is charged a typical full call."""
    rates = tuple(float(x) for x in os.environ["HOMEBODY_CODER_RATES"].split(",")) if os.environ.get(
        "HOMEBODY_CODER_RATES") else SOL_RATES
    total = 0.0
    for events in Path(task_dir).glob("coder/call-*/events.jsonl"):
        turns = [json.loads(line)["usage"] for line in events.read_text(errors="replace").splitlines()
                 if '"turn.completed"' in line]
        if not turns:
            total += UNMETERED_CREDITS * (rates[0] / SOL_RATES[0])
        for usage in turns:
            cached = usage.get("cached_input_tokens", 0)
            total += ((usage["input_tokens"] - cached) * rates[0] + cached * rates[1]
                      + usage["output_tokens"] * rates[2]) / 1e6
    return round(total, 3)


def score_pick(rows, episode, record):
    """Pick's stage from physics: 0 not lifted, 1 lifted but not held at the end, 2 held."""
    lifted, held, rise = physics_verdict(rows, episode["step"]["fixture_id"])
    return {"lifted": lifted, "held": held, "max_rise": round(rise, 3), "stage": 2 if held else 1 if lifted else 0}


def run_place(cm, session, episode, step):
    """Stand at the stance, pick the object with the built-in pick and carry it to the step's place stance
    (setup: a failure there is not scored), then call the candidate's place once with the arguments the
    scripted sequence would give (a release over the destination), with one coder rescue if it refuses."""
    port, execute = session.port, session.execute
    if not execute("navigate", cm.navigate_arguments(episode["stance"], step.facing, step.hand), port.snapshot(),
                   evidence_name="stance"):
        return {"setup_failed": "stance navigation refused"}
    frame = port.snapshot()
    click = cm.target_click(frame, step.label)
    if click is None:
        return {"setup_failed": "object not visible from the stance"}
    u, v = click
    side = step.hand if step.hand != "auto" else ("left" if u < frame.rgb.shape[1] / 2 else "right")
    if not execute("pick", {"point_normalized_1000": cm.normalized(frame, u, v), "hand": side}, frame) \
            or not session.context.carried:
        return {"setup_failed": "the built-in pick did not hold the object"}
    side = next(iter(session.context.carried))
    goals = [(via, None) for via in step.carry_via_goals]
    if step.carry_goal is not None:
        goals.append((step.carry_goal, step.carry_facing))
    for index, (goal, facing) in enumerate(goals):
        if not carry_to(cm, session, step, side, goal, facing, f"carry_{index}"):
            return {"setup_failed": "carry navigation refused"}
    if not session.context.carried:
        return {"setup_failed": "the object was dropped during the carry"}
    frame = port.snapshot()
    palm = frame.palm(session.context.arms[side], side)
    arguments = cm.drop_arguments(frame, palm, step.release_map_xy_m, side, step.release_delta_z_m)
    session.record("target.json", {"schema_version": 1, "frame": "ws_map", "objects": {step.fixture_id: step.evaluator_target}})
    name, destination = (str(value).replace("_", " ") for value in (episode["step"]["label"], step.destination_label))
    record = call_skill(session, episode, "place", arguments, frame, request={
        "goal": f"Place the {name} held in the {side} hand into the {destination} "
                f"(release {json.dumps({k: round(v, 3) for k, v in arguments.items() if k != 'hand'})}).",
        "success": f"place returns OK and the {name} rests inside the {destination}, released and still.",
        "constraints": "Change only the place skill; keep holding the object until the release.",
        "task": f"Put the {name} in the {destination}."})
    for _ in range(int(SETTLE_SECONDS / 0.02)):
        port.advance(port.epoch, 0.02)
    return {**{k: v for k, v in record.items() if k != "ok"}, "side": side}


def carry_to(cm, session, step, side, goal, facing, name, rechecks=CARRY_RECHECKS):
    """Walk the held object to GOAL; when navigate stops to recheck the grasp (HOLD_UNCERTAIN), repeat pick on
    the object to re-verify it and walk on, as the agent would. Whether it arrived still holding."""
    port, execute = session.port, session.execute
    for attempt in range(rechecks + 1):
        if execute("navigate", cm.navigate_arguments(goal, facing, side), port.snapshot(),
                   evidence_name=f"{name}_{attempt}" if attempt else name):
            return True
        last = session.outcome["steps"][-1] if session.outcome["steps"] else {}
        frame = port.snapshot()
        click = cm.target_click(frame, step.label)
        if last.get("code") != "HOLD_UNCERTAIN" or click is None or attempt == rechecks:
            return False
        if not execute("pick", {"point_normalized_1000": cm.normalized(frame, *click), "hand": side}, frame,
                       evidence_name=f"{name}_recheck_{attempt}") or side not in session.context.carried:
            return False
    return False


def score_place(rows, episode, record):
    """Place's stage from physics, by the evaluator's own placement test against the step's target: 2 placed
    (inside the target, released, supported and still), 1 released but not placed there, 0 still in hand."""
    evaluate = load_evaluator()
    name, goal = episode["step"]["fixture_id"], episode["step"]["evaluator_target"]
    end = rows[-1]
    earlier = next((r for r in reversed(rows) if end["time"] - r["time"] >= 0.5), rows[0])  # speeds over 0.5 s
    state = end["objects"][name]
    seconds = max(1e-6, end["time"] - earlier["time"])
    reason = evaluate.object_reason(state, goal, speeds=evaluate.pose_speeds(state, earlier["objects"][name], seconds),
                                    require_still=True)
    released = not state["hand_contacts"]
    return {"placed": reason is None, "released": released, "place_reason": reason,
            "stage": 2 if reason is None else 1 if released else 0}


def load_evaluator():
    """evaluation/evaluate.py of the episode's code root (the scorer is frozen, so every candidate has the same)."""
    path = Path(os.environ["HOMEBODY_BENCH_ROOT"]) / "evaluation/evaluate.py"
    spec = importlib.util.spec_from_file_location("homebody_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# skill name -> (run(cm, session, episode, step) -> record, score(rows, episode, record) -> {stage, ...})
DRIVERS = {"pick": (run_pick, score_pick), "place": (run_place, score_place)}


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
    os.environ["HOMEBODY_BENCH_ROOT"] = str(root)
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
    args = SimpleNamespace(no_visuals=True, place_candidates=256, wall_timeout=episode.get("wall_timeout", 3600))
    start, session, record = time.monotonic(), None, {"episode": episode["id"], "skill": episode["skill"],
                                                    "scene": scene.name, "family": episode["family"]}
    try:
        backend = MujocoBackend(root, settings, scene=scene)
        step = cm.resolved(cm.Step(**episode["step"]), backend.label_ids)
        session = cm.ScriptedSession(backend, recorder, task, outcome, root, settings, args, start)
        session.begin_cycle(0, step)
        record.update(DRIVERS[episode["skill"]][0](cm, session, episode, step))
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
    if rows and "setup_failed" not in record and "error" not in record:
        record.update(DRIVERS[episode["skill"]][1](rows, episode, record))
        record.update(success=record["stage"] == 2,
                      builtin_success=bool(record["stage"] == 2 and (record.get("call") or {}).get("code") == "OK"))
    record.update(wall_seconds=round(time.monotonic() - start, 1), simulation_seconds=round(rows[-1]["time"], 1) if rows else 0)
    (out / "result.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps({k: record.get(k) for k in ("episode", "stage", "success", "setup_failed", "error")}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
