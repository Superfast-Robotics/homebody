// A monitor result and how far its evidence holds, as four separate properties: the monitor's output (its score and
// what that score is), the model's confidence (with its calibration), the quality of the input it read (checked at
// the observation's own time), and the assessment's reliability (an independent check, Unknown when none reports).
// They are never combined into one number, and nothing here changes a score: pure, so node --test checks it.
import { attemptLabel, toolById } from "./execution.js";
import { signalLabel } from "./interventions.js";

const QUALITY = { valid: ["success", "Valid"], suspected_invalid: ["warning", "Suspected invalid"], invalid: ["danger", "Invalid"],
  unknown: ["", "Not checked"] };
const CALIBRATION = { calibrated: "Calibrated", uncalibrated: "Uncalibrated" };
const RELIABILITY = { reliable: ["success", "Reliable"], unreliable: ["danger", "Unreliable"], unknown: ["", "Unknown"] };
const SCOPES = { task: "The task", robot: "The robot", environment: "The environment" };

const fixed = (v) => (v === null || v === undefined ? null : Number(v).toFixed(2));

/** The assessment's rows for OBSERVATION (a recorded reading): each {key, label, value, tone, note}. The input quality
 * is the one recorded with the observation; CURRENT (the camera's quality now) is never used for it. */
export function assessment(observation, { current = null } = {}) {
  void current; // a historical incident keeps its own observation-time quality
  const o = observation || {};
  const [qTone, qWord] = QUALITY[o.frame_quality] || QUALITY.unknown;
  const [rTone, rWord] = RELIABILITY[o.reliability] || RELIABILITY.unknown;
  const calibration = CALIBRATION[o.calibration] || "Calibration unknown";
  const level = o.level ? ` · ${o.level === "anomaly" ? "detection" : o.level}` : "";
  return [
    { key: "output", label: signalLabel(o.signal), value: fixed(o.risk) ?? "Not reported", tone: o.level === "anomaly" ? "danger" : "",
      note: `${o.semantics || "The monitor did not say what this score is"}${level}` },
    { key: "threshold", label: "Detection threshold", value: fixed(o.threshold) ?? "None reported", tone: "",
      note: o.threshold !== null && o.threshold !== undefined ? "The rules compare their running average of the score with it" : "" },
    { key: "confidence", label: "Model confidence", value: o.confidence === null || o.confidence === undefined ? "Not reported"
      : `${fixed(o.confidence)} · ${calibration}`, tone: "", note: o.confidence_semantics || "" },
    { key: "input", label: "Camera input", value: qWord, tone: qTone,
      note: [o.quality_reason, o.quality_source || "Checked when the reading arrived (no observation-time record)"].filter(Boolean).join(" · ") },
    { key: "reliability", label: "Assessment reliability", value: rWord, tone: rTone,
      note: o.reliability === "reliable" || o.reliability === "unreliable" ? "" : "No independent check of the monitor's assessment reports to the dashboard" },
  ];
}

/** A compact warning beside a monitor result whose input frames were not valid at its observation, or null. */
export function inputWarning(observation) {
  const q = observation?.frame_quality;
  if (q !== "invalid" && q !== "suspected_invalid") return null;
  return { text: `Input warning: ${QUALITY[q][1].toLowerCase()} frames`, reason: observation.quality_reason || "",
    tone: QUALITY[q][0], source: observation.quality_source || "" };
}

/** What a reading applies to, by the scope the monitor gave it; an unmatched or unscoped reading is said to be so. A
 * reading without a scope (an older record) is its attempt's when it names one, as execution.sameContext reads it. */
export function scopeText(reading, state) {
  const scope = reading?.scope || (reading?.tool ? "skill_attempt" : "unknown");
  if (scope === "skill_attempt") {
    const tool = toolById(state, reading.tool);
    return tool ? attemptLabel(tool) : "A skill attempt the dashboard could not match";
  }
  if (scope === "goal") return reading.scope_goal ? `Goal ${reading.scope_goal}` : "A goal (not named)";
  return SCOPES[scope] || "Unknown";
}
