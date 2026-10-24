// Where each piece of incident evidence comes from and what it is to the detection: detector input (the observation
// the monitor itself consumed, only when a record establishes it), a synchronized robot observation (the dashboard's
// own recorded robot-camera frame matched to the observation's simulation time) or contextual evidence (a recorded
// frame near the moment by the dashboard's clock). Quality checks are attributed to the item they checked. Pure, so
// node --test checks it.

const MATCH = (state) => 2 * (state.history_period || 1); // s within which a recorded frame is context for a moment
const SYNC = (state) => Math.max(0.25, (state.history_period || 1) / 2); // s of simulated time for a synchronized frame

function signed(delta) {
  const s = Math.abs(delta) < 0.05 ? "0.0 s" : `${Math.abs(delta).toFixed(1)} s`;
  return delta >= 0.05 ? `${s} after` : delta <= -0.05 ? `${s} before` : "at";
}

/**
 * The camera evidence for SELECTION (selection.js): {frame: {id, t, sim}, relation, label, short, offset, basis,
 * metadata}, or {frame: null, relation: "unavailable", label}. RELATION is "detector_input" only when the observation
 * names the recorded frame the detector consumed (OBSERVATION.detector_frame), "synchronized" when a recorded frame's
 * simulation time matches the observation's, else "contextual" (nearest by the dashboard clock). OFFSET is the frame's
 * time minus the moment's on BASIS ("simulation" or "dashboard clock").
 */
export function cameraEvidence(state, selection) {
  const history = (state.history || []).map(([id, t, sim]) => ({ id, t, sim: sim ?? null }));
  const o = selection.observation;
  const tolerance = MATCH(state);
  const meta = (frame, relation, offset, basis) => ({ evidence_id: `frame:${frame.id}`, incident_id: selection.incident?.id ?? null,
    observation_id: o?.id ?? null, source_component: "Dashboard history (robot camera)", camera: "robot (head)",
    capture_sim_time: frame.sim, receipt_time: frame.t, relation, offset_s: offset, offset_basis: basis,
    quality_check: o && relation === "synchronized" && o.quality_relation === "synchronized"
      ? `${o.frame_quality} (the dashboard's check of the robot camera frames at the observation's time, not of this stored frame alone)`
      : "not checked for this frame",
    monitor_evaluation_id: o?.id ?? null });
  const own = o?.detector_frame !== undefined && o?.detector_frame !== null ? history.find((f) => f.id === o.detector_frame) : null;
  if (own) {
    return { frame: own, relation: "detector_input", label: `Detector input · sim ${own.sim?.toFixed(2) ?? "?"} s`, short: "Recorded detector input",
      offset: 0, basis: "simulation", metadata: meta(own, "detector_input", 0, "simulation") };
  }
  if (o && typeof o.sim_time === "number") {
    let best = null;
    for (const f of history) {
      if (f.sim === null) continue;
      const delta = f.sim - o.sim_time;
      if (Math.abs(delta) <= SYNC(state) && (best === null || Math.abs(delta) < Math.abs(best.delta))) best = { f, delta };
    }
    if (best) {
      return { frame: best.f, relation: "synchronized", label: `Robot camera · Synchronized observation (${signed(best.delta)} the monitor's newest frame, simulation time)`,
        short: `Recorded robot camera · synchronized (${best.delta >= 0 ? "+" : "−"}${Math.abs(best.delta).toFixed(2)} s sim)`, offset: best.delta, basis: "simulation", metadata: meta(best.f, "synchronized", best.delta, "simulation") };
    }
  }
  let near = null;
  for (const f of history) {
    const delta = f.t - selection.t;
    if (Math.abs(delta) <= tolerance && (near === null || Math.abs(delta) < Math.abs(near.delta))) near = { f, delta };
  }
  if (!near) return { frame: null, relation: "unavailable", label: "Camera evidence unavailable for this timestamp", short: "No recording" };
  const what = selection.incident || selection.intervention ? "detection" : "the selected moment";
  return { frame: near.f, relation: "contextual", label: `Contextual camera frame · ${signed(near.delta)} ${what} (dashboard clock)`,
    short: `Contextual footage · ${near.delta >= 0 ? "+" : "−"}${Math.abs(near.delta).toFixed(1)} s`, offset: near.delta,
    basis: "dashboard clock", metadata: meta(near.f, "contextual", near.delta, "dashboard clock") };
}

const STATE_WORD = { valid: "valid", suspected_invalid: "suspected invalid", invalid: "invalid", unknown: "not checked" };

/**
 * The quality checks of an observation, each attributed to what it checked: DETECTOR (the monitor's own input: known
 * only when the check ran on it) and CAMERA (the dashboard's robot-camera frame at the observation's time, or when it
 * arrived). Each {state, text, tone, note}.
 */
export function qualityAttribution(o) {
  const checked = o?.frame_quality || "unknown";
  const bad = checked === "invalid" || checked === "suspected_invalid";
  const tone = checked === "invalid" ? "danger" : checked === "suspected_invalid" ? "warning" : checked === "valid" ? "success" : "";
  if (o?.quality_relation === "detector_input") {
    return { detector: { state: checked, tone, text: bad ? `Detector image ${STATE_WORD[checked]}` : `Detector input ${STATE_WORD[checked]}`,
      note: o.quality_source || "" }, camera: null };
  }
  const when = o?.quality_relation === "synchronized" ? "at the observation's time" : "when the reading arrived";
  return {
    detector: { state: "unknown", tone: "", text: "Detector input quality unknown",
      note: "The monitor's own copy of its input is not validated by any component that reports to the dashboard" },
    camera: { state: checked, tone, text: bad ? `Robot camera quality warning: ${STATE_WORD[checked]} frames` : `Robot camera frames ${STATE_WORD[checked]}`,
      note: [o?.quality_reason, o?.quality_source || `Dashboard check ${when}`].filter(Boolean).join(" · ") },
  };
}
