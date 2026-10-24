// Interventions and the monitor's current condition, in words: pure, so node --test checks it (tests/js). Every
// step shown comes from a recorded stage (store interventions.py); a step no component reported is "not reported",
// never assumed, and the robot physically stopping is never claimed.

export const OPEN = ["requested", "executing", "timed_out"];
const STATE = {
  requested: ["warning", "Stop requested"], executing: ["running", "Executor stopping the skill"],
  timed_out: ["danger", "No response from the executor"], confirmed: ["success", "Skill stopped (executor confirmed)"],
  rejected: ["danger", "Request rejected"], failed: ["danger", "The skill did not stop"],
  cancelled: ["neutral", "Request withdrawn"],
};

/** The intervention state's tone and words. */
export function stateText(state) {
  return STATE[state] || ["neutral", state];
}

/** The monitor's signal names (monitor/decide.signals) as an operator reads them. */
export function signalLabel(signal) {
  if (!signal) return "Monitor risk";
  if (signal === "stall") return "Stall probability";
  if (signal === "unsafe") return "Unsafe-situation probability";
  if (signal === "human_needed") return "Needs-a-person probability";
  if (signal.startsWith("failure:")) return `Failure probability: ${signal.slice(8).replace(/_/g, " ")}`;
  if (signal.startsWith("undone:")) return `Goal ${signal.slice(7)} undone probability`;
  return `${signal.replace(/_/g, " ")} probability`;
}

const stage = (row, name) => row.stages.find(([s]) => s === name) || null;

/**
 * An intervention's steps, in order, each {key, label, state, t, source, note}: state "done", "pending", "failed",
 * "unknown" or "skipped". Only recorded stages are "done"; the physical stop is "unknown" (no motion telemetry).
 */
export function steps(row) {
  const detected = stage(row, "detected"), requested = stage(row, "requested"), accepted = stage(row, "accepted");
  const final = ["confirmed", "failed", "rejected", "cancelled"].map((s) => stage(row, s)).find(Boolean) || null;
  const step = (key, label, state, rec, note) => ({ key, label, state, t: rec ? rec[1] : null, source: rec ? rec[2] : "",
    note: note ?? (rec ? rec[3] : "") });
  const out = [];
  out.push(detected ? step("detected", "Detection", "done", detected)
    : step("detected", "Detection", row.initiator === "Human operator" ? "skipped" : "unknown", null,
      row.initiator === "Human operator" ? "The operator acted without a monitor detection" : "Not reported"));
  out.push(requested && requested[1] !== null ? step("requested", `Stop request (${row.initiator})`, "done", requested)
    : step("requested", `Stop request (${row.initiator})`, "unknown", requested, "The request itself did not reach the dashboard"));
  const rejected = final && final[0] === "rejected";
  const ackState = accepted ? "done" : rejected ? "failed" : final ? "skipped" : row.state === "timed_out" ? "failed" : "pending";
  const ackNote = accepted ? null : rejected ? final[3] : final ? "Never taken: the request ended first"
    : row.state === "timed_out" ? `No acknowledgement within ${row.ack_timeout} s (still waiting)` : "Awaiting the skill's next cancellation check";
  out.push(step("accepted", "Executor acknowledgement", ackState, accepted || (rejected ? final : null), ackNote));
  const stopState = !final ? "pending" : final[0] === "confirmed" ? "done" : final[0] === "cancelled" ? "skipped" : "failed";
  out.push(step("stopped", "Skill stop", stopState, final && final[0] !== "rejected" ? final : null,
    final ? undefined : accepted ? "The skill is ending (returning the arm)" : "Not yet"));
  out.push(step("physical", "Physical stop", "unknown", null, "Unverified: no motion or safety telemetry reaches the dashboard"));
  return out;
}

/** "Stall detected → Stop requested → Executor accepted → Skill stopped": the recorded stages only, in order. */
export function flowText(row, detectedLabel = "Anomaly detected") {
  const words = { detected: detectedLabel, requested: "Stop requested", accepted: "Executor accepted",
    confirmed: "Skill stopped", rejected: "Rejected", failed: "Did not stop", cancelled: "Withdrawn" };
  const parts = row.stages.filter(([, t], i) => t !== null || i > 0).map(([s]) => words[s] || s);
  if (row.state === "timed_out") parts.push("Timed out");
  return parts.join(" → ");
}

/** The intervention to put on the card: the newest one of the current task still open, or one that went wrong
 * (failed, rejected, timed out) whose detection the operator has not resolved. */
export function activeIntervention(state) {
  const task = state.task?.id ?? null;
  const incidents = new Map(state.incidents.map((i) => [i.id, i]));
  return (state.interventions || []).findLast((row) => row.task === task && (OPEN.includes(row.state) ||
    (["failed", "rejected"].includes(row.state) && incidents.get(row.incident)?.state !== "resolved"))) || null;
}

/**
 * The monitor's current condition, apart from its connectivity, by priority: an open intervention, an active
 * anomaly (the newest fresh observation), elevated risk, unknown or stale observations, then normal. Returns
 * {level: "intervention" | "anomaly" | "elevated" | "unknown" | "normal" | "unavailable", text}. HISTORY counts the
 * anomalies in the window, said beside a normal current condition rather than hidden.
 */
export function monitorCondition(state, status, now, windowStart) {
  const open = activeIntervention(state);
  if (open && OPEN.includes(open.state)) {
    const head = open.stages.find(([s]) => s === "detected")?.[3] || "Stop requested";
    const tail = { requested: "Stop requested · Awaiting executor acknowledgement",
      executing: "Executor accepted · Skill stopping", timed_out: "Stop requested · No executor acknowledgement" }[open.state];
    return { level: "intervention", text: `${head} · ${tail}` };
  }
  const observations = state.readings.filter((r) => r.kind === "observation");
  const latest = observations.at(-1) || null;
  const anomalies = state.incidents.filter((i) => i.kind === "monitor" && i.t >= windowStart).length;
  if (!state.readings.length) return { level: "unavailable", text: "No monitoring service connected: session, agent and skill events only" };
  if (!latest) return { level: "unknown", text: "Monitoring connected · No behaviour observation yet" };
  const fresh = status.state === "connected" && status.observationAge !== null && status.observationAge <= Math.max(3, status.staleAfter || 3);
  if (!fresh) {
    return { level: "unknown", text: `Monitoring ${status.state === "connected" ? "connected" : status.state} · Current behaviour assessment unavailable (last observation ${Math.round(status.observationAge ?? 0)} s ago)` };
  }
  const what = signalLabel(latest.signal).toLowerCase();
  const during = latest.skill ? ` during ${latest.skill}` : "";
  if (latest.level === "anomaly") return { level: "anomaly", text: `Anomaly: ${latest.reason || what}${during}` };
  if (latest.level === "elevated" || (latest.risk !== null && latest.threshold !== null && latest.risk >= latest.threshold)) {
    const sustained = sustainedFor(observations, now);
    return { level: "elevated", text: `Elevated ${what}${during}${sustained >= 10 ? ` · sustained ${Math.round(sustained)} s` : ""}` };
  }
  if (latest.level === "unknown") return { level: "unknown", text: "Monitoring connected · Current behaviour assessment unavailable" };
  return { level: "normal", text: anomalies ? `Current condition normal · ${anomalies} anomal${anomalies > 1 ? "ies" : "y"} in the last minute`
    : "Monitoring active · No anomalies detected in the last minute" };
}

/** How long the newest run of elevated or anomalous observations has lasted, in seconds. */
export function sustainedFor(observations, now) {
  let start = null;
  for (let i = observations.length - 1; i >= 0; i -= 1) {
    const r = observations[i];
    const high = r.level === "anomaly" || r.level === "elevated" || (r.risk !== null && r.threshold !== null && r.risk >= r.threshold);
    if (!high) break;
    start = r.t;
  }
  return start === null ? 0 : now - start;
}

/** Open alerts grouped by incident: each {incident, interventions}, newest first; an alert without a shared
 * incident stands alone. */
export function alertGroups(state) {
  const rows = state.interventions || [];
  return state.incidents.filter((i) => i.alert && i.state !== "resolved").reverse()
    .map((incident) => ({ incident, interventions: rows.filter((row) => row.incident === incident.id) }));
}
