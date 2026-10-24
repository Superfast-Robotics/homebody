// The inspection selection every panel follows: one incident (or intervention, attempt or moment) resolved by the
// identities the records carry (intervention -> incident -> attempt -> plan node, the detecting observation, the
// recorded camera frame near the moment), never by matching labels. Inspecting is read-only: nothing here reaches the
// robot. Pure, so node --test checks it.
import { cameraEvidence } from "./evidence.js";
import { attemptLabel, toolById } from "./execution.js";

export const WINDOW = 60; // s of the main monitoring view
const STATES = { succeeded: "Completed", failed: "Failed", cancelled: "Cancelled", ended: "Ended" };

/** The stored robot-camera frame nearest T within TOLERANCE seconds: {id, t, delta}, or null (never a far one). */
export function recordedFrame(history, t, tolerance) {
  let best = null;
  for (const [id, at] of history || []) {
    const delta = Math.abs(at - t);
    if (delta <= tolerance && (best === null || delta < best.delta)) best = { id, t: at, delta };
  }
  return best;
}

/** The timeline window that shows T: the live window when T is in it, else SECONDS around T (never past NOW). */
export function windowFor(t, now, seconds = WINDOW) {
  if (t >= now - seconds && t <= now) return { start: now - seconds, end: now, historical: false };
  const end = Math.min(now, t + seconds / 2);
  return { start: end - seconds, end, historical: true };
}

/** What attempt TOOL was doing at moment T, from its recorded start and end. */
export function stateAt(tool, t) {
  if (!tool) return "Unknown";
  if (t < tool.start) return "Not started";
  if (tool.end === null || tool.end === undefined || t < tool.end) return "Running";
  const word = STATES[tool.state] || tool.state;
  return tool.code ? `${word} (${tool.code})` : word;
}

/** The moment an intervention began: its detection, else its request, else when it was recorded. */
function startOf(row) {
  const first = row.stages.find(([s, t]) => (s === "detected" || s === "requested") && t !== null);
  return first ? first[1] : row.t ?? null;
}

/**
 * The selection for REF ({intervention, incident, tool, t}): {t, intervention, incident, tool, node, goal, observation,
 * detecting, frame, window, historical, label, missing}. NODE is the plan node of the attempt (tool:<id>), GOAL its goal
 * node; OBSERVATION the reading that detected the incident (its first, not the episode's newest; DETECTING is false
 * when only the newest was recorded). MISSING lists, in words,
 * what the record does not establish.
 */
export function resolveSelection(state, ref, now) {
  const row = ref.intervention ? (state.interventions || []).find((r) => r.id === ref.intervention) || null : null;
  const incidentId = ref.incident ?? row?.incident ?? null;
  const incident = incidentId ? state.incidents.find((i) => i.id === incidentId) || null : null;
  const tool = toolById(state, ref.tool ?? row?.tool ?? incident?.tool ?? null);
  const t = ref.t ?? (row ? startOf(row) : null) ?? incident?.t ?? tool?.start ?? now;
  const nodes = state.tree?.nodes || {};
  const node = tool && nodes[`tool:${tool.id}`] ? `tool:${tool.id}` : null;
  const goalId = tool?.goal ?? incident?.node ?? row?.node ?? null;
  const goal = goalId && nodes[`goal:${goalId}`] ? `goal:${goalId}` : null;
  // the detecting observation; an incident recorded before it was kept has only the episode's newest one, said so
  const detecting = Boolean(incident?.evidence?.first_observation);
  const observation = incident?.evidence?.first_observation || incident?.observation || null;
  // the camera evidence, attributed (evidence.js): the observation's own (synchronized by simulation time) only at the
  // detection; any other moment of the incident gets the recorded frame near that moment, as context
  const detectedAt = (row ? startOf(row) : null) ?? incident?.t ?? null;
  const atDetection = ref.t === undefined || ref.t === null || detectedAt === null || Math.abs(ref.t - detectedAt) < 0.5;
  const evidence = cameraEvidence(state, atDetection ? { t, observation, incident, intervention: row } : { t, observation: null });
  const frame = evidence.frame ? { id: evidence.frame.id, t: evidence.frame.t, delta: Math.abs(evidence.frame.t - t) } : null;
  const window = windowFor(t, now);
  const missing = [];
  if (!tool) missing.push("Attempt unknown: the record names no skill call");
  else if (!node) missing.push("The attempt is no longer in the current behavior tree");
  if (!observation) missing.push(row?.initiator === "Human operator" ? "No monitor observation: an operator's request" : "No monitor observation linked");
  if (!frame) missing.push("Camera evidence unavailable for this timestamp");
  if (state.keep_s && t < now - state.keep_s) missing.push(`Timeline data older than ${Math.round(state.keep_s / 60)} minutes has expired`);
  const label = `Inspecting ${tool ? attemptLabel(tool) : "an unknown attempt"}`;
  return { t, intervention: row, incident, tool, node, goal, observation, detecting, frame, evidence, window, historical: window.historical || t < now - 1,
    label, missing };
}
