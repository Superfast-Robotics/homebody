"""score_episode.py SESSION TARGETS EVALUATE_PY [MODEL]: score one recorded HomeBody session (or the latest
session under an --output root) from physics.
score_episode.py --usage SESSION [MODEL]: only the session's cost (written whether or not it can be scored).

Each targeted object gets a stage: 0 untouched, 1 lifted by the robot, 2 ended inside its target
region, 3 physics-verified placement: the frozen evaluator finds it placed at the end AND still in place for
its whole settling window, in a valid recording (no invalid physics, robot fall, warning, gap or reset; those
are listed under "invalid" and no object gets stage 3). Non-target products the evaluator finds disturbed at the
end are listed under "disturbed". An object the robot lifted
that ends on the floor is flagged "dropped" and keeps stage 1 (lifted): the robot cannot bend to pick
it up again, so a drop is not scored below an object never touched, and every run stays in the score.
Writes nothing; prints the result JSON, with the session's step and coder-call counts and its cost: the
tokens of every agent decision and coder call, in credits at each model's rates (the agent's MODEL key,
Sol when not given; the coder is Sol). A coder call cut off
at its time limit leaves no usage record; it is counted in coder_unmetered and charged the mean of the
metered calls (or CODER_FALLBACK when none are).
"""
import json
import runpy
import sys
from pathlib import Path

import numpy as np

LIFT_M = 0.05
RATES = {"input": 50.0, "cached": 2.5, "output": 250.0}  # credits per 1M tokens (GPT-6.1 Sol)
AGENT_RATES = {"vllm": (0.0, 0.0, 0.0), "astra": (250.0, 25.0, 1250.0), "sol": (50.0, 2.5, 250.0), "luna": (2.5, 0.25, 12.5)}  # by MODEL key
CODER_FALLBACK = (120_000, 480_000, 4_000)  # (uncached, cached, output) tokens of a typical coder call
FALL_M = 0.5  # ending this far below its start means it fell to the floor (scenes differ in floor height)


def credits(uncached, cached, output, rates=None):
    rate_in, rate_cached, rate_out = rates or (RATES["input"], RATES["cached"], RATES["output"])
    return (uncached * rate_in + cached * rate_cached + output * rate_out) / 1e6


def agent_rates(model):
    """The agent's credit rates from its MODEL key (astra-litellm, sol61-litellm, ...); Sol when unknown."""
    return next((rates for family, rates in AGENT_RATES.items() if family in (model or "")), AGENT_RATES["sol"])


def decision_usage(reply):
    """(input, cached, output) tokens of one agent decision's reply: a Responses API event stream or a
    JSON body with usage; None when it reports none (a CLI provider, a cut-off stream): its cost is unknown."""
    try:
        text = reply.read_text(errors="replace")
    except OSError:  # no reply recorded
        return None
    usage = None
    if text.startswith("data:"):
        for line in text.splitlines():
            if line.startswith("data:") and '"response.completed"' in line:
                try:
                    usage = json.loads(line[5:])["response"].get("usage")
                except (ValueError, KeyError, AttributeError, TypeError):
                    usage = None
    else:
        try:
            usage = json.loads(text).get("usage")
        except (ValueError, AttributeError):
            usage = None
    if not isinstance(usage, dict):
        return None
    total = usage.get("input_tokens", usage.get("prompt_tokens"))
    output = usage.get("output_tokens", usage.get("completion_tokens"))
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else None
    if not all(tokens(value) for value in (total, output, cached)) or cached > total:
        return None  # incomplete or malformed usage: its cost is unknown
    return total, cached, output


def tokens(value):
    """VALUE is a token count: an int >= 0 below 10**9, not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 10**9


FREE = (0.0, 0.0, 0.0)  # a model on our own GPUs


def decision_calls(decision, model):
    """[(tokens or None, rates, model)] of one decision folder: its one reply at MODEL's rates, or a multi-model
    decision's calls (vlm/composite.py routing.json), each at its own model's rates (free on our own GPUs). A local
    call without usage costs nothing and is not unmetered."""
    routing = decision / "routing.json"
    if not routing.is_file():
        return [(decision_usage(decision / "reply.json"), agent_rates(model), model)]
    try:
        calls = json.loads(routing.read_text())["calls"]
    except (ValueError, KeyError, TypeError):
        return [(None, agent_rates(model), model)]
    return [(decision_usage(decision / str(call.get("dir", "")) / "reply.json")
             or ((0, 0, 0) if call.get("local") else None),
             FREE if call.get("local") else agent_rates(call.get("model")), call.get("model"))
            for call in calls if isinstance(call, dict)]


def cost(task_dir, model=None):
    """The task's token use and credits: agent decisions (every attempt, retries included) and coder calls. A decision
    or coder call that reported no usage is counted (main_unmetered, coder_unmetered) and makes the record
    "unmetered": its credits are then a lower estimate, and the loop charges the session its cap instead."""
    calls = [call for decision in sorted(task_dir.glob("decisions/*")) if decision.is_dir()
             for call in decision_calls(decision, model)]
    main, main_unmetered = [used for used, _, _ in calls if used is not None], sum(used is None for used, _, _ in calls)
    by_model = {}
    for used, rates, name in calls:
        entry = by_model.setdefault(str(name), {"calls": 0, "input": 0, "cached": 0, "output": 0, "credits": 0.0})
        entry["calls"] += 1
        if used is not None:
            entry.update(input=entry["input"] + used[0], cached=entry["cached"] + used[1],
                         output=entry["output"] + used[2],
                         credits=round(entry["credits"] + credits(used[0] - used[1], used[1], used[2], rates), 4))
    coder, unmetered = [], 0
    for events in task_dir.glob("coder/call-*/events.jsonl"):
        turns = []
        for line in events.read_text(errors="replace").splitlines():
            if '"turn.completed"' in line:
                try:
                    turns.append(json.loads(line)["usage"])
                except (ValueError, KeyError, TypeError):
                    continue
        last = turns[-1] if turns and isinstance(turns[-1], dict) else {}
        used = (last.get("input_tokens"), last.get("cached_input_tokens", 0), last.get("output_tokens"))
        if all(tokens(value) for value in used) and used[1] <= used[0]:
            coder.append(used)
        else:
            unmetered += 1
    main_in, main_cached, main_out = (sum(column) for column in zip(*main)) if main else (0, 0, 0)
    coder_in, coder_cached, coder_out = (sum(column) for column in zip(*coder)) if coder else (0, 0, 0)
    metered = (sum(credits(used[0] - used[1], used[1], used[2], rates) for used, rates, _ in calls if used is not None)
               + credits(coder_in - coder_cached, coder_cached, coder_out))
    per_call = (credits(coder_in - coder_cached, coder_cached, coder_out) / len(coder) if coder
                else credits(*CODER_FALLBACK))
    return {"main_input": main_in, "main_cached": main_cached, "main_output": main_out,
            "coder_input": coder_in, "coder_cached": coder_cached, "coder_output": coder_out,
            "coder_metered": len(coder), "coder_unmetered": unmetered, "main_unmetered": main_unmetered,
            "unmetered": bool(main_unmetered or unmetered), "credits": round(metered + unmetered * per_call, 3),
            "models": by_model}


def latest(session):
    session = Path(session)
    if not (session / "events.jsonl").exists():  # an --output root: its latest session
        session = max(session.glob("session-*"))
    return session


def usage(session, model=None):
    """The session's cost alone (zero when it recorded no task)."""
    tasks = sorted(latest(session).glob("tasks/task-*"))
    print(json.dumps(cost(tasks[-1], model) if tasks else {"credits": 0.0, "no_task": True}))


def main(session, targets_path, evaluate_py, model=None):
    session = latest(session)
    spec = json.loads(Path(targets_path).read_text())
    goals = {s["fixture_id"]: s["evaluator_target"] for s in spec["steps"]} if "steps" in spec else spec["objects"]
    physics = max(session.glob("tasks/task-*/physics.jsonl"))
    rows = [json.loads(line) for line in physics.read_text().splitlines() if line.strip()]
    verdict = runpy.run_path(evaluate_py)["score_attempt"](rows, {"schema_version": 1, "frame": "ws_map", "objects": goals},
                                                           require_settled=True)
    settle = verdict.get("criteria", {}).get("settle_seconds", 0.0)
    invalid = verdict.get("invalid_reasons") or []
    objects = {}
    for name, goal in goals.items():
        judged = verdict["objects"].get(name, {})
        reason = judged.get("terminal_reason", "missing")
        if reason is None and judged.get("settled_seconds", 0.0) < settle - 1e-9:
            reason = "not_settled"
        if reason is None and invalid:
            reason = "invalid_recording"
        start = rows[0]["objects"][name]["position"]
        lifted = any(r["objects"][name]["position"][2] > start[2] + LIFT_M and r["objects"][name]["hand_contacts"] for r in rows)
        end = np.asarray(rows[-1]["objects"][name]["position"])
        low, high = (np.asarray(b) for b in goal["bounds"])
        inside = bool(np.all(end >= low) and np.all(end <= high))
        dropped = lifted and not inside and end[2] < start[2] - FALL_M
        objects[name] = {"stage": 3 if reason is None else 2 if inside else 1 if lifted else 0, "reason": reason,
                         "dropped": bool(dropped)}
    outcome = json.loads((physics.parent / "outcome.json").read_text()) if (physics.parent / "outcome.json").exists() else {}
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines() if line.strip()]
    skills = [e["decision"]["skill"] for e in events if e.get("kind") == "decision"]
    print(json.dumps({"success": bool(verdict["success"]) and outcome.get("status") == "model_done",
                      "stage": round(sum(o["stage"] for o in objects.values()) / max(1, len(objects)), 3),
                      "dropped": sum(o["dropped"] for o in objects.values()),
                      "objects": objects, "untargeted": verdict.get("untargeted"), "status": outcome.get("status"),
                      "invalid": invalid, "targets": sorted(goals),
                      "disturbed": sorted(n for n, v in (verdict.get("untargeted") or {}).items()
                                          if (v or {}).get("terminal_reason") == "disturbed"),
                      "steps": len(skills), "customizations": skills.count("call_coder"),
                      "wall_seconds": outcome.get("wall_seconds", 0.0), "cost": cost(physics.parent, model)},
                     indent=1, default=str))


if __name__ == "__main__":
    if sys.argv[1] == "--usage":
        usage(*sys.argv[2:4])
    else:
        main(*sys.argv[1:5])
