"""Extract dataset: the sample types a monitoring session can give, ordered by value, and a rule-based extraction of
them from what this dashboard holds (its store, the robot state trace and the stored camera frames).

Value: failures with what recovered them, the monitor's incidents, and the operator's corrections are rare, hard to
simulate and hard to label, so they come first; state and action traces, task episodes and RGB clips are common.
Each extraction writes one folder: manifest.json (types, counts, source, what each record holds and what is not
captured yet), one JSON-lines file per sample type, and rgb/ with the camera frames the samples point to. Every record
carries dashboard (wall) times on one clock, and simulation times where they are known, so video, robot state,
actions and monitor events line up.
"""
import json
import time
from pathlib import Path

CONTEXT_S = 5.0  # seconds of context kept before and after an event
FOLLOW_S = 30.0  # seconds after an operator's correction in which what the agent did next is kept
CLIP_S = 3.0  # half-length of the RGB clip around an event
TYPES = (  # (id, title, description, badge, tone), highest value first
    ("failure_recovery", "Failure + recovery samples", "Failed attempts, recovery actions, and their outcomes.",
     "Highest value", "highest"),
    ("monitor_incidents", "Monitor incident samples",
     "Monitor observations, anomaly signals, interventions, and evidence.", "High value", "high"),
    ("human_interventions", "Human intervention samples", "Operator corrections, overrides, and intervention context.",
     "High value", "high"),
    ("robot_state_actions", "Robot state + actions", "Robot state, end-effector pose, gripper state, and action traces.",
     "Core", "core"),
    ("task_language", "Task + language episodes", "Task instruction, episode context, and outcome labels.",
     "Core", "core"),
    ("rgb", "RGB camera trajectories", "Synchronized robot camera clips for behavior review and learning.",
     "Common", "common"),
)
PRESELECTED = ("failure_recovery", "monitor_incidents", "human_interventions")
EMPTY = {"failure_recovery": "No failed attempts in this session", "monitor_incidents": "No monitor incidents recorded",
         "human_interventions": "No human interventions recorded", "robot_state_actions": "No robot state recorded",
         "task_language": "No task episodes in this session", "rgb": "No camera frames stored"}
NOT_CAPTURED = [("joint states and end-effector poses are not streamed to this dashboard: robot state is the hands, "
                 "base pose and target it is told"), "depth is not stored by the dashboard (Depth / 3D is not offered)"]


def within(t, start, end):
    return t is not None and start <= t <= end


def frames_in(frames, start, end):
    """The ids of the stored frames FRAMES ([[id, t, sim]]) between START and END."""
    return [frame_id for frame_id, t, _ in frames if within(t, start, end)]


def readings_in(state, start, end):
    return [{k: r.get(k) for k in ("t", "sim_time", "level", "risk", "signal", "activity", "activity_p", "reason",
                                   "scope", "call")} for r in state.get("readings", []) if within(r["t"], start, end)]


def decisions_in(state, start, end):
    return [{k: d[k] for k in ("t", "skill", "text", "arguments", "goal")} for d in state.get("decisions", [])
            if within(d["t"], start, end)]


def tool_row(tool):
    return {k: tool.get(k) for k in ("id", "skill", "arguments", "goal", "start", "end", "state", "code", "message",
                                     "stages", "attempt", "recovery", "call")}


def failure_recovery(state, frames):
    """Each failed attempt with the decisions before it, the monitor's readings around it, its incident, and what
    the agent tried next in the same goal (the recovery) and how that ended."""
    tools = state.get("tools", [])
    samples = []
    for index, tool in enumerate(tools):
        if tool["state"] != "failed":
            continue
        end = tool["end"] if tool["end"] is not None else tool["start"]
        recovery = next((t for t in tools[index + 1:] if t["goal"] == tool["goal"]), None)
        samples.append({
            "failure": tool_row(tool), "label": tool["code"] or "FAILED",
            "context": {"decisions": decisions_in(state, tool["start"] - CONTEXT_S, tool["start"]),
                        "readings": readings_in(state, tool["start"] - CONTEXT_S, end + CONTEXT_S)},
            "incidents": [i["id"] for i in state.get("incidents", []) if i.get("tool") == tool["id"]],
            "recovery": tool_row(recovery) if recovery else None,
            "recovered": bool(recovery and recovery["state"] == "succeeded"),
            "frames": frames_in(frames, tool["start"] - CONTEXT_S, end + CONTEXT_S)})
    return samples


def monitor_incidents(state, frames):
    """Each incident the monitor raised, and each run of elevated or anomalous readings, with its evidence."""
    samples = []
    for incident in state.get("incidents", []):
        if incident.get("kind") != "monitor":
            continue
        t = incident["t"]
        samples.append({
            "incident": {k: incident.get(k) for k in ("id", "t", "severity", "title", "reason", "node", "tool", "state")},
            "label": incident.get("title", ""), "readings": readings_in(state, t - CONTEXT_S, t + CONTEXT_S),
            "interventions": [{k: i.get(k) for k in ("id", "t", "action", "initiator", "state")}
                              for i in state.get("interventions", []) if within(i.get("t"), t - CONTEXT_S, t + 60)],
            "frames": frames_in(frames, t - CONTEXT_S, t + CONTEXT_S)})
    flagged = [r for r in state.get("readings", []) if r.get("level") in ("elevated", "anomaly")]
    runs = []
    for reading in flagged:  # readings within CONTEXT_S of each other make one episode of concern
        if runs and reading["t"] - runs[-1][-1]["t"] <= CONTEXT_S:
            runs[-1].append(reading)
        else:
            runs.append([reading])
    for run in runs:
        start, end = run[0]["t"], run[-1]["t"]
        signals = sorted({r.get("signal") or "" for r in run} - {""})
        samples.append({"incident": None, "label": f"{run[-1]['level']}: {', '.join(signals) or 'risk'}",
                        "readings": readings_in(state, start - CONTEXT_S, end + CONTEXT_S),
                        "peak_risk": max((r.get("risk") or 0.0) for r in run), "interventions": [],
                        "frames": frames_in(frames, start - CONTEXT_S, end + CONTEXT_S)})
    return samples


def human_interventions(state, frames):
    """Each operator instruction and operator-initiated stop or pause, with what the agent did in the next FOLLOW_S."""
    events = [{"kind": "instruction", "t": i["t"], "text": i["text"], "state": i.get("state", "")}
              for i in state.get("instructions", [])]
    events += [{"kind": i.get("action", "intervention"), "t": i["t"], "text": i.get("reason", ""),
                "state": i.get("state", "")} for i in state.get("interventions", [])
               if "operator" in str(i.get("initiator", "")).lower()]
    return [{**event, "label": event["kind"],
             "followed_by": {"decisions": decisions_in(state, event["t"], event["t"] + FOLLOW_S),
                             "attempts": [tool_row(t) for t in state.get("tools", [])
                                          if within(t["start"], event["t"], event["t"] + FOLLOW_S)]},
             "frames": frames_in(frames, event["t"] - CONTEXT_S, event["t"] + FOLLOW_S)}
            for event in sorted(events, key=lambda e: e["t"])]


def episodes(state):
    return [task for task in state.get("tasks", []) if task.get("start") is not None]


def robot_state_actions(state, frames, trace):
    """Per task: the robot state samples and the actions (decisions and executed attempts) on one time axis."""
    samples = []
    for task in episodes(state):
        start, end = task["start"], task["end"] if task["end"] is not None else time.time()
        samples.append({"task": task.get("text", ""), "start": start, "end": task["end"],
                        "states": [{"t": t, "status": status, **robot} for t, status, robot in trace
                                   if within(t, start, end)],
                        "actions": decisions_in(state, start, end),
                        "attempts": [tool_row(t) for t in state.get("tools", []) if within(t["start"], start, end)],
                        "frames": frames_in(frames, start, end)})
    return [s for s in samples if s["states"] or s["actions"]]


def task_language(state, frames):
    """Per task: the instruction, its outcome label and the agent's own account, step by step."""
    return [{"instruction": task.get("text", ""), "start": task["start"], "end": task["end"],
             "outcome": task.get("outcome"), "result": task.get("result"),
             "steps": [d["text"] for d in decisions_in(state, task["start"], task["end"] or time.time())],
             "frames": frames_in(frames, task["start"], task["end"] or time.time())} for task in episodes(state)]


def rgb(state, frames):
    """Per task, its stored camera frames; and a short clip around each failure and monitor incident."""
    clips = [{"kind": "episode", "label": task.get("text", ""),
              "frames": frames_in(frames, task["start"], task["end"] or time.time())} for task in episodes(state)]
    events = [(t["end"] or t["start"], f"{t['skill']} {t['code']}") for t in state.get("tools", [])
              if t["state"] == "failed"]
    events += [(i["t"], i.get("title", "")) for i in state.get("incidents", []) if i.get("kind") == "monitor"]
    clips += [{"kind": "event", "label": label, "t": t, "frames": frames_in(frames, t - CLIP_S, t + CLIP_S)}
              for t, label in events]
    return [clip for clip in clips if clip["frames"]]


def samples(kind, state, frames, trace):
    if kind == "robot_state_actions":
        return robot_state_actions(state, frames, trace)
    return {"failure_recovery": failure_recovery, "monitor_incidents": monitor_incidents,
            "human_interventions": human_interventions, "task_language": task_language, "rgb": rgb}[kind](state, frames)


def available(state, frames, trace, *, human_supported=True) -> list[dict]:
    """The sample types to offer, highest value first: {id, title, description, badge, tone, count, available,
    reason, preselected, preview (dataset_preview.preview)}. Human interventions are left out where the controls cannot take them."""
    from homebody.ui.dashboard.dataset_preview import preview
    rows = []
    for kind, title, description, badge, tone in TYPES:
        if kind == "human_interventions" and not human_supported:
            continue
        found = samples(kind, state, frames, trace)
        rows.append({"id": kind, "title": title, "description": description, "badge": badge, "tone": tone,
                     "count": len(found), "available": bool(found), "reason": "" if found else EMPTY[kind],
                     "preselected": bool(found) and kind in PRESELECTED,
                     "preview": preview(kind, state, found, frames)})
    return rows


def extract(state, frames, trace, kinds, folder: Path, jpeg, source: str) -> dict:
    """Write the KINDS' samples under FOLDER (a new folder per extraction); JPEG(frame_id) gives a stored frame's
    bytes. Returns {path, counts, frames}."""
    folder.mkdir(parents=True, exist_ok=False)
    counts, referenced = {}, set()
    for kind in kinds:
        rows = samples(kind, state, frames, trace)
        counts[kind] = len(rows)
        with (folder / f"{kind}.jsonl").open("w") as out:
            for number, row in enumerate(rows):
                out.write(json.dumps({"sample": f"{kind}-{number:04d}", "type": kind, **row}, default=str) + "\n")
                referenced.update(row.get("frames", []))
    times = {frame_id: (t, sim) for frame_id, t, sim in frames}
    (folder / "rgb").mkdir()
    written = []
    for frame_id in sorted(referenced):
        data = jpeg(frame_id)
        if data:
            (folder / "rgb" / f"{frame_id:06d}.jpg").write_bytes(data)
            written.append({"frame": frame_id, "t": times[frame_id][0], "sim_time": times[frame_id][1],
                            "file": f"rgb/{frame_id:06d}.jpg"})
    (folder / "rgb" / "index.json").write_text(json.dumps(written, indent=1))
    manifest = {"schema_version": 1, "created": time.time(), "source": source, "types": list(kinds),
                "counts": counts, "frames": len(written), "extraction": "rule-based, from the dashboard's own record",
                "time_axis": "t: dashboard wall seconds (one clock for every record); sim_time where the source gave it",
                "not_captured": NOT_CAPTURED}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {"path": str(folder), "counts": counts, "frames": len(written)}


def attach(dashboard, folder: Path):
    """DASHBOARD's Extract dataset: `dataset_types` (what this session offers) and `extract_dataset` (write the chosen
    types under FOLDER/<scene>-<time>/)."""
    from homebody.ui.controls import capabilities

    def gathered():
        return dashboard.store.snapshot({}), dashboard.history.index(), dashboard.store.robot_samples()

    def source():
        return "recorded rollout (replay)" if dashboard.read_only else "current monitoring session"

    def types(request):
        state, frames, trace = gathered()
        return {"ok": True, "source": source(),
                "types": available(state, frames, trace, human_supported=capabilities(dashboard.controls)["tell"])}

    def run(request):
        kinds = request.get("types")
        known = {kind for kind, *_ in TYPES}
        if not isinstance(kinds, list) or not kinds or not all(kind in known for kind in kinds):
            return {"ok": False, "message": "Choose at least one sample type."}
        state, frames, trace = gathered()
        name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dashboard.store.scene_name)[:40]
        target = folder / f"{name}-{time.strftime('%Y%m%dT%H%M%S')}"
        result = extract(state, frames, trace, [k for k, *_ in TYPES if k in kinds], target, dashboard.history.get,
                         source())
        total = sum(result["counts"].values())
        return {"ok": True, **result,
                "message": f"Extracted {total} sample{'s' if total != 1 else ''} and {result['frames']} frames to "
                           f"{result['path']}"}

    dashboard.handlers.update(dataset_types=types, extract_dataset=run)
