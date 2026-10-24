"""What one sample of each dataset type looks like, for the Extract dataset window: a thumbnail, three key camera
frames with captions, a compact record of the sample, and what it includes. Built from the same samples the
extraction writes (dataset.py), from the most telling one of each type."""
import time

INCLUDES = ("camera frames", "robot state", "actions", "monitor events", "outcome label", "language")


def nearest(frames, t, ids=None):
    """The stored frame (of IDS, when given) closest to T: its id, or None."""
    pool = [(frame_id, ft) for frame_id, ft, _ in frames if ids is None or frame_id in ids]
    return min(pool, key=lambda item: abs(item[1] - t))[0] if pool and t is not None else None


def keyframes(frames, ids, points):
    """[{id, caption}] for each (t, caption) of POINTS: the stored frame of IDS nearest t (repeats dropped)."""
    chosen, seen = [], set()
    for number, (t, caption) in enumerate(points, 1):
        frame_id = nearest(frames, t, set(ids)) if ids else None
        if frame_id is not None and frame_id not in seen:
            seen.add(frame_id)
            chosen.append({"id": frame_id, "caption": f"{number}. {caption}"})
    return chosen


def spread(frames, ids):
    """Three points across the frames IDS (first, middle, last) as (t, caption)."""
    times = sorted(t for frame_id, t, _ in frames if frame_id in set(ids))
    if not times:
        return []
    return [(times[0], "Start"), (times[len(times) // 2], "Middle"), (times[-1], "End")]


def task_of(state, t):
    return next((task.get("text", "") for task in reversed(state.get("tasks", []))
                 if task["start"] <= t and (task["end"] is None or task["end"] >= t)), "")


def failure_recovery(state, rows, frames):
    row = next((r for r in rows if r["recovered"]), rows[0])
    failure, recovery = row["failure"], row["recovery"]
    end = failure["end"] or failure["start"]
    points = [(failure["start"], f"{failure['skill'].capitalize()} attempt {failure['attempt']}"),
              (end, f"{failure['skill'].capitalize()} failed: {failure['code'] or 'failed'}")]
    if recovery:
        points.append((recovery["end"] or recovery["start"],
                       f"Recovery: {recovery['skill']} {(recovery['code'] or recovery['state']).lower()}"))
    record = {"task": task_of(state, failure["start"]), "skill": failure["skill"],
              "failure_type": failure["code"] or "failed",
              "recovery": f"{recovery['skill']} attempt {recovery['attempt']}" if recovery else None,
              "outcome": "recovered" if row["recovered"] else "not recovered", "frames": len(row["frames"]),
              "labels": ["failure", "recovery"] if recovery else ["failure"]}
    return rows.index(row), points, record, row


def monitor_incidents(state, rows, frames):
    row = max(rows, key=lambda r: r.get("peak_risk") or (1.0 if r["incident"] else 0.0))
    readings = row["readings"]
    peak = max(readings, key=lambda r: r.get("risk") or 0.0) if readings else None
    points = [(readings[0]["t"], "Before")] if readings else []
    if peak:
        points.append((peak["t"], f"Peak: {peak.get('signal') or peak['level']}"))
    if readings:
        points.append((readings[-1]["t"], "After"))
    activities = sorted({r.get("activity") for r in readings if r.get("activity")})
    record = {"task": task_of(state, readings[0]["t"]) if readings else "", "label": row["label"],
              "peak_risk": round(row.get("peak_risk") or (peak.get("risk") or 0.0 if peak else 0.0), 2),
              "readings": len(readings), "activities": activities[:4], "interventions": len(row["interventions"]),
              "frames": len(row["frames"])}
    return rows.index(row), points, record, row


def human_interventions(state, rows, frames):
    row = rows[-1]
    attempts = row["followed_by"]["attempts"]
    points = [(row["t"] - 2.0, "Before"), (row["t"], f"Operator: {row['kind']}")]
    if attempts:
        points.append((attempts[0]["end"] or attempts[0]["start"], f"Then: {attempts[0]['skill']}"))
    record = {"task": task_of(state, row["t"]), "kind": row["kind"], "operator": row.get("text", ""),
              "agent_next": [d["skill"] for d in row["followed_by"]["decisions"]][:3],
              "outcome": attempts[0]["code"] if attempts else None, "frames": len(row["frames"])}
    return rows.index(row), points, record, row


def robot_state_actions(state, rows, frames):
    row = max(rows, key=lambda r: len(r["states"]) + len(r["actions"]))
    last = row["states"][-1] if row["states"] else {}
    record = {"task": row["task"], "states": len(row["states"]), "actions": len(row["actions"]),
              "attempts": [f"{a['skill']} {a['code'] or a['state']}" for a in row["attempts"]][:4],
              "hands": last.get("hands"), "frames": len(row["frames"])}
    return rows.index(row), spread(frames, row["frames"]), record, row


def task_language(state, rows, frames):
    row = next((r for r in reversed(rows) if r["end"] is not None), rows[-1])
    end = row["end"] or time.time()
    record = {"instruction": row["instruction"], "outcome": row["outcome"], "steps": len(row["steps"]),
              "first_steps": [step[:60] for step in row["steps"][:2]],
              "duration_s": round(end - row["start"], 1)}
    return rows.index(row), spread(frames, row["frames"]), record, row


def rgb(state, rows, frames):
    row = next((r for r in rows if r["kind"] == "event"), rows[0])
    record = {"clip": row["kind"], "label": row["label"], "frames": len(row["frames"]),
              "camera": "robot head", "rate_hz": 1.0}
    return rows.index(row), spread(frames, row["frames"]), record, row


BUILDERS = {"failure_recovery": failure_recovery, "monitor_incidents": monitor_incidents,
            "human_interventions": human_interventions, "robot_state_actions": robot_state_actions,
            "task_language": task_language, "rgb": rgb}
NAMES = {"failure_recovery": "failure_recovery_episode", "monitor_incidents": "monitor_incident",
         "human_interventions": "human_intervention", "robot_state_actions": "state_action_trajectory",
         "task_language": "task_episode", "rgb": "rgb_clip"}


def includes(row) -> list[str]:
    """What a sample carries, in INCLUDES order."""
    found = {"camera frames": bool(row.get("frames")),
             "robot state": bool(row.get("states")),
             "actions": bool(row.get("actions") or row.get("context", {}).get("decisions") or row.get("followed_by")
                             or row.get("failure") or row.get("steps")),
             "monitor events": bool(row.get("readings") or row.get("context", {}).get("readings")),
             "outcome label": any(key in row for key in ("recovered", "outcome", "label")),
             "language": bool(row.get("instruction") or row.get("text") or row.get("steps"))}
    return [name for name in INCLUDES if found[name]]


def preview(kind, state, rows, frames) -> dict | None:
    """The preview of KIND from its sample ROWS: {name, frames: [{id, caption}], record, includes, thumbnail,
    text}; None without a sample."""
    if not rows:
        return None
    index, points, record, row = BUILDERS[kind](state, rows, frames)
    shown = keyframes(frames, row.get("frames", []), points)
    return {"name": f"{NAMES[kind]}_{index:04d}", "frames": shown, "record": record, "includes": includes(row),
            "thumbnail": shown[len(shown) // 2]["id"] if shown else None,
            "text": record.get("instruction") or record.get("task") or ""}
