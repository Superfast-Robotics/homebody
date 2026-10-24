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

/** The monitor's signal names (monitor/decide.signals) as an operator reads them. They are scores (the decision
 * model's uncalibrated answer probabilities), so they are never called calibrated probabilities. */
export function signalLabel(signal) {
  if (!signal) return "Monitor score";
  if (signal === "stall") return "Stall score";
  if (signal === "unsafe") return "Unsafe-situation score";
  if (signal === "human_needed") return "Needs-a-person score";
  if (signal.startsWith("failure:")) return `Failure score: ${signal.slice(8).replace(/_/g, " ")}`;
  if (signal.startsWith("undone:")) return `Goal ${signal.slice(7)} undone score`;
  return `${signal.replace(/_/g, " ")} score`;
}

const ORDER = ["detected", "decided", "requested", "accepted", "rejected", "confirmed", "failed", "cancelled", "retry_started", "retry_finished"];

/** ROW's stages in lifecycle order, whatever order their events arrived in. */
export function ordered(row) {
  return [...row.stages].sort((a, b) => ORDER.indexOf(a[0]) - ORDER.indexOf(b[0]));
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
  const decided = stage(row, "decided");
  out.push(detected ? step("detected", "Detection", "done", detected)
    : step("detected", "Detection", row.initiator === "Human operator" ? "skipped" : "unknown", null,
      row.initiator === "Human operator" ? "The operator acted without a monitor detection" : "Not reported"));
  out.push(decided ? step("decided", "Intervention decision", "done", decided)
    : step("decided", "Intervention decision", row.initiator === "Human operator" ? "done" : "unknown", null,
      row.initiator === "Human operator" ? "The operator's own decision" : "Not reported"));
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
  const retry = stage(row, "retry_started"), retried = stage(row, "retry_finished");
  if (retry) {
    out.push(step("retry", "Retry", retried ? (/ended with OK/.test(retried[3]) ? "done" : "failed") : "pending", retried || retry,
      retried ? retried[3].replace(/ \[[^\]]+\]$/, "") : retry[3].replace(/ \[[^\]]+\]$/, " running")));
  } else if (final && final[0] === "confirmed") {
    out.push(step("retry", "Retry or recovery", "pending", null, "None started yet"));
  }
  return out;
}

/** "Stall detected → Stop requested → Executor accepted → Skill stopped": the recorded stages only, in order. */
export function flowText(row, detectedLabel = "Anomaly detected") {
  const words = { detected: detectedLabel, requested: "Abort requested", accepted: "Executor accepted",
    confirmed: "Skill stopped", rejected: "Rejected", failed: "Did not stop", cancelled: "Withdrawn",
    retry_started: "Retry started", retry_finished: "Retry ended" };
  const parts = ordered(row).filter(([s, t], i) => s !== "decided" && (t !== null || i > 0)).map(([s]) => words[s] || s);
  if (row.state === "requested") parts.push("Awaiting executor");
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

/** Open alerts grouped by what they are about: the alerts of one attempt (a detection and what followed on it) are
 * one group with a count, newest first; an alert without an attempt stands alone, and unrelated ones never merge.
 * Each {incident: the newest, incidents, interventions}. */
export function alertGroups(state) {
  const rows = state.interventions || [];
  const groups = new Map();
  for (const incident of state.incidents.filter((i) => i.alert && i.state !== "resolved").reverse()) {
    const key = incident.tool ? `tool:${incident.tool}` : `incident:${incident.id}`;
    if (!groups.has(key)) groups.set(key, { incident, incidents: [], interventions: [] });
    const group = groups.get(key);
    group.incidents.push(incident);
    for (const row of rows) if (row.incident === incident.id && !group.interventions.includes(row)) group.interventions.push(row);
  }
  return [...groups.values()];
}
