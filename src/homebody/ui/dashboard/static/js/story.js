// How the dashboard words what happened: the monitoring summary's milestones, readable error text, operator labels
// for tree nodes, and the Auto camera's choice. Pure, so node --test checks it (tests/js); every sentence comes from
// a recorded event, and "→" joins two milestones only where one event names the other as its cause.

// Readable text for the failure codes skills declare (skills/contract.py and each skill's FAILURES) and the plan
// runner's own; any other code is shown as it is rather than guessed at.
const ERRORS = {
  CANCELLED: "Cancelled", STALE_EPOCH: "The scene changed under it", STALE_OBSERVATION: "Observation too old",
  INVALID_ARGUMENT: "Invalid action argument", TARGET_LOST: "Target lost from view",
  ACTION_REJECTED: "The robot rejected the action", PHYSICS_INVALID: "Physics became invalid", COLLISION: "Collision",
  NO_ROUTE: "No route to the goal", STALL: "The robot stalled", TIMEOUT: "Timed out",
  ARRIVAL_NOT_VERIFIED: "Arrival not verified", CARRY_CLEARED: "Carried object no longer held",
  HOLD_UNCERTAIN: "Grasp uncertain", HAND_OCCUPIED: "The hand is already holding something",
  CALIBRATION_MISSING: "Calibration missing", NO_GRASP: "No grasp found", UNREACHABLE: "Target out of reach",
  TARGET_MOVED: "The target moved", ARM_NOT_REACHED: "The arm did not reach its pose",
  LIFT_NOT_VERIFIED: "Lift not verified", NOT_HOLDING: "The hand is not holding the object",
  SURFACE_CHANGED: "The surface changed", RELEASE_NOT_VERIFIED: "Release not verified",
  OUTCOME_UNKNOWN: "Outcome unknown", PLACEMENT_NOT_VERIFIED: "Placement not verified",
  CHECK_FAILED: "The goal's check does not hold", RECOVERY_LIMIT: "Recovery limit reached", ERROR: "Skill error",
  MONITOR_STOPPED: "Stopped by monitor", OPERATOR_STOPPED: "Stopped by operator",
};
const DETERMINERS = new Set(["a", "an", "the", "one", "some", "any"]);
// where a goal's sentence leaves its action for the details (where to, how, with what kept in place)
const CLAUSE = [",", ";", " from ", " to the ", " into ", " inside ", " onto ", " on the ", " in the ", " with ", " while ",
  " keeping ", " without ", " so that "];

export function errorText(code) {
  return ERRORS[code] || code;
}

const capital = (text) => (text ? text[0].toUpperCase() + text.slice(1) : text);

/** A concise label: the first words of TEXT up to LIMIT characters, the full text kept for details. */
export function shortLabel(text, limit = 42) {
  const clean = capital(String(text || "").trim());
  if (clean.length <= limit) return clean;
  const cut = clean.slice(0, limit);
  return `${cut.slice(0, cut.lastIndexOf(" ") > 20 ? cut.lastIndexOf(" ") : limit).trimEnd()}…`;
}

/** A goal's concise label: its sentence up to the first clause after the action ("Pick one jam jar from the shelf
 * with the right hand" → "Pick one jam jar"), when that leaves at least two words; the full text stays in details. */
export function goalLabel(text, limit = 42) {
  const clean = String(text || "").trim().replace(/\.$/, "");
  const cuts = CLAUSE.map((c) => clean.toLowerCase().indexOf(c)).filter((i) => i > 0 && clean.slice(0, i).trim().split(/\s+/).length >= 2);
  const head = cuts.length ? clean.slice(0, Math.min(...cuts)) : clean;
  // "Pick one accessible jam jar" -> "Pick accessible jam jar": determiners and counts after the verb carry no action
  const words = head.split(/\s+/);
  const kept = [words[0], ...words.slice(1).filter((w, i) => i > 0 || !DETERMINERS.has(w.toLowerCase()))];
  return shortLabel(kept.length >= 2 ? kept.join(" ") : head, limit);
}

const STATUS = { running: "Running", success: "Completed", failure: "Failed", pending: "Pending", skipped: "Ended" };
const REQUEST = { requested: "stop requested by the {who}, awaiting the skill", executing: "stop accepted, the skill is ending",
  timed_out: "stop requested by the {who}, no acknowledgement" };

/** A tree node as an operator reads it: {label, meta}. The technical view keeps the raw fields. */
export function operatorText(node, elapsed) {
  const d = node.detail;
  const time = elapsed ? ` · ${elapsed}` : "";
  if (node.type === "root") return { label: shortLabel(node.label, 60), meta: d.note || `Task · ${STATUS[node.status] || node.status}` };
  if (node.type === "sequence") return { label: "Sequence", meta: STATUS[node.status] || node.status };
  if (node.type === "fallback") {
    const meta = d.selected === null ? "All options failed" : d.selected > 0 ? `Recovering with option ${d.selected + 1}`
      : `Option 1 · ${STATUS[node.status] || node.status}`;
    return { label: "Fallback", meta };
  }
  if (node.type === "goal") {
    const label = `${d.recovers ? "Recovery: " : ""}${goalLabel(node.label, d.recovers ? 34 : 42)}`;
    const attempts = d.used > 1 ? ` · ${d.used} attempts` : "";
    if (node.status === "failure") return { label, meta: `Failed${attempts}${time}` };
    if (node.status === "success") return { label, meta: `Completed${time}` };
    if (node.status === "running") return { label, meta: `Running · attempt ${d.used} of ${d.budget}${time}` };
    return { label, meta: STATUS[node.status] || node.status };
  }
  if (node.type === "attempts") {
    const reasons = [...new Set(d.codes.filter((code) => code && code !== "OK").map(errorText))].join(", ");
    return { label: `${d.count} earlier attempt${d.count > 1 ? "s" : ""}`,
      meta: d.failed === d.count ? `All failed · ${reasons}` : `${d.failed} failed · ${reasons}` };
  }
  const name = capital(node.label.replace(/_/g, " "));
  const label = d.number > 1 || d.attempt > 1 ? `${name} · attempt ${d.number || d.attempt}` : name;
  if (node.status === "failure") return { label, meta: `Failed · ${errorText(d.code)}` };
  if (node.status === "success") return { label, meta: `Succeeded · ${elapsed || "done"}` };
  if (node.status === "running") {
    const open = (d.interventions || []).findLast((r) => REQUEST[r.state]);
    const request = open ? ` · ${REQUEST[open.state].replace("{who}", open.initiator.toLowerCase())}` : "";
    return { label, meta: `Running${time}${request}` };
  }
  return { label, meta: STATUS[node.status] || node.status };
}

const KIND = {
  skill: "failure", goal: "failure", task: "failure", monitor: "detection", intervention: "operator",
};

function milestone(incident, state) {
  const goal = (id) => state.tree?.nodes[`goal:${id}`]?.label || id;
  if (incident.kind === "recovery") {
    const who = incident.initiator ? ` (${incident.initiator.replace("Behavior Tree ", "BT ")})` : "";
    const label = { recovery_start: `Recovery initiated${who}`, recovery_ok: "Recovery succeeded", recovery_fail: "Recovery failed",
      replan: "Replanned by the agent" }[incident.tag] || incident.title;
    const kind = incident.tag === "recovery_ok" ? "success" : incident.tag === "recovery_fail" ? "failure" : "recovery";
    return { kind, label };
  }
  if (incident.kind === "skill") return { kind: "failure", label: `${capital(incident.skill || "Skill")} failed` };
  if (incident.kind === "goal") return { kind: "failure", label: `Goal failed: ${shortLabel(goal(incident.node), 28)}` };
  if (incident.kind === "monitor") return { kind: "detection", label: `Monitor: ${shortLabel(incident.title, 30)}` };
  return { kind: KIND[incident.kind] || "neutral", label: shortLabel(incident.title, 34) };
}

const TASK_KIND = { success: "success", success_with_recoveries: "success", failed: "failure", partial: "failure" };

/**
 * The monitoring summary for [start, end]: major milestones, oldest first, each {id, t, kind, label, count,
 * incident, node, linked}. Repeated failures of one skill in one goal are one milestone; LINKED says the event was
 * caused by the milestone before it (a recorded caused_by), else the two are only consecutive.
 */
export function summarize(state, start, end, limit = 6) {
  const events = state.incidents.filter((i) => i.t >= start && i.t <= end && i.kind !== "task").sort((a, b) => a.t - b.t);
  const blocks = [];
  for (const incident of events) {
    const last = blocks.at(-1);
    const sameFailure = incident.kind === "skill" && last?.kind === "failure" && last.incident.kind === "skill" &&
      last.incident.skill === incident.skill && last.incident.node === incident.node;
    const sameDetection = incident.kind === "monitor" && last?.incident?.kind === "monitor" && last.incident.title === incident.title;
    if (last && (sameFailure || sameDetection)) {
      last.count += 1;
      last.ids.push(incident.id);
      last.end = incident.t;
      continue;
    }
    const { kind, label } = milestone(incident, state);
    blocks.push({ id: incident.id, ids: [incident.id], t: incident.t, end: incident.t, kind, label, count: 1, incident,
      node: incident.node, linked: Boolean(last && incident.caused_by && last.ids.includes(incident.caused_by)) });
  }
  for (const task of state.tasks || []) {
    if (task.end === null || task.end < start || task.end > end || !task.result) continue;
    blocks.push({ id: `task:${task.id}`, ids: [], t: task.end, end: task.end, kind: TASK_KIND[task.result.state] || "neutral",
      label: `Task finished · ${task.result.label}`, count: 1, incident: null, node: null, linked: false, task });
  }
  blocks.sort((a, b) => a.t - b.t);
  for (const block of blocks) if (block.count > 1) block.label = `${block.label} ×${block.count}`;
  return { blocks: blocks.slice(-limit), earlier: Math.max(0, blocks.length - limit) };
}

/**
 * The Auto camera: the first view the activity prefers that this robot serves. AUTO maps an activity (a skill name,
 * "finished" or "idle") to view names, where "{hand}" is the running skill's hand.
 */
export function cameraFor(state, auto, available) {
  const tool = state.tools.findLast((t) => t.state === "running");
  const finished = state.task && state.task.end !== null && !state.controls.busy;
  const activity = tool ? tool.skill : finished ? "finished" : "idle";
  const hand = tool?.arguments?.hand === "left" ? "left" : "right";
  const wanted = (auto[activity] || auto.idle || ["robot"]).map((name) => name.replace("{hand}", hand));
  return wanted.find((name) => available.includes(name)) || available[0] || "robot";
}
