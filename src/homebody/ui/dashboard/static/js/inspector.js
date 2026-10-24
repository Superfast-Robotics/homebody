// The incident inspector, in five parts: (1) an overview that says what happened, to which attempt, and where the
// intervention stands now; (2) the lifecycle, one row per event; (3) a compact evidence card (the attributed camera
// frame, the score against its threshold, the input checks, each labelled by source) with the full evidence one click
// away; (4) recovery and outcome, as verified; (5) technical details, collapsed. A stage no component reported says
// Unknown or Unverified, never a guess; times link to that moment on the dashboard. Nothing here reaches the robot.
import { assessment, scopeText } from "./assessment.js";
import { ago, clock, duration, h, json } from "./dom.js";
import { qualityAttribution } from "./evidence.js";
import { attemptLabel, incidentCard } from "./execution.js";
import { ACK_TEXT, COMMAND_TEXT, PHYSICAL_TEXT, ordered, requestStatus, signalLabel, steps } from "./interventions.js";
import { resolveSelection, stateAt } from "./selection.js";

const TONE = { done: "success", pending: "warning", overdue: "danger", failed: "danger", unknown: "neutral", skipped: "neutral" };
const WORD = { done: "Done", pending: "Pending", overdue: "Overdue", failed: "Failed", unknown: "Unknown", skipped: "Not applicable" };
const SEVERITY = { critical: "danger", warning: "warning", info: "neutral" };
const INCIDENT_STATE = { active: "Active", acknowledged: "Acknowledged", resolved: "Resolved" };
const COMMAND_TONE = { pending: "warning", active: "running", completed: "success", failed: "danger", cancelled: "neutral", expired: "danger", unknown: "neutral" };
const RECOVERY_TONE = { "Not started": "neutral", Running: "running", Completed: "success", Failed: "danger", Unknown: "neutral" };

function facts(pairs) {
  return h("dl", { class: "kv small" }, ...pairs.filter(([, v]) => v !== undefined && v !== null && v !== "")
    .flatMap(([k, v]) => [h("dt", { text: k }), h("dd", {}, v)]));
}

const note = (text) => h("p", { class: "note", text });
const badge = (tone, text) => h("span", { class: `badge ${tone}`, text });
const lower = (text) => text.charAt(0).toLowerCase() + text.slice(1);

/** What the detection means for the operator, from the monitor's signal or the initiator. */
export function concern(signal, initiator) {
  if (initiator === "Human operator") return "The operator asked the executor to stop the running skill.";
  if (signal === "stall") return "The monitor judged that the skill was not making progress.";
  if (signal === "unsafe") return "The monitor judged the situation unsafe.";
  if (signal === "human_needed") return "The monitor judged that a person is needed.";
  if (signal?.startsWith("failure:")) return `The monitor judged that the skill was failing: ${signal.slice(8).replace(/_/g, " ")}.`;
  if (signal?.startsWith("undone:")) return `The monitor judged that goal ${signal.slice(7)}, already done, no longer holds.`;
  return "The monitor reported an anomaly.";
}

/** The overview's status lines: "Attempt 4 · Abort requested · Controller acknowledgement overdue" and the physical
 * outcome, from the request's three separate states (interventions.requestStatus). */
export function overviewLines(tool, status) {
  const ack = { awaiting: "Awaiting controller acknowledgement", deadline_exceeded: status.command === "pending" ? "Controller acknowledgement overdue"
    : "No controller acknowledgement", acknowledged: "Controller acknowledged", acknowledged_late: "Controller acknowledged after deadline",
  rejected: "Request rejected", not_acknowledged: "Not acknowledged", not_requested: "No request recorded", unknown: "Acknowledgement unknown" }[status.ack];
  const command = { pending: "Abort requested", active: "Abort accepted", completed: "Skill stopped (executor result)", failed: "Abort failed",
    cancelled: "Request ended", expired: "Abort request expired", unknown: "Request state unknown" }[status.command];
  return [`${tool ? `Attempt ${tool.number}` : "Attempt unknown"} · ${command} · ${ack}`, `Physical stop ${lower(PHYSICAL_TEXT[status.physical])}`];
}

/** The recovery after a stop, as recorded: Not started, Running, Completed, Failed or Unknown (no request). */
export function recoveryState(row) {
  if (!row) return { state: "Unknown", note: "No intervention recorded" };
  const started = row.stages.find(([s]) => s === "retry_started"), finished = row.stages.find(([s]) => s === "retry_finished");
  if (finished) return { state: /ended with OK/.test(finished[3]) ? "Completed" : "Failed", note: finished[3].replace(/ \[[^\]]+\]$/, ""), t: finished[1], source: finished[2] };
  if (started) return { state: "Running", note: started[3].replace(/ \[[^\]]+\]$/, ""), t: started[1], source: started[2] };
  return { state: "Not started", note: "No retry or recovery has been recorded" };
}

/** The lifecycle rows; each shows its event id on hover. */
function lifecycle(app, row, list, observation, ref) {
  return h("ol", { class: "stepper" }, ...list.map((step) => {
    const tone = TONE[step.state] || "neutral";
    const event = step.key === "detected" && observation?.id ? `observation ${observation.id}` : step.t !== null ? `${row.id}/${step.key}` : "none recorded";
    return h("li", { class: `step ${step.state}`, title: `Event: ${event}` }, h("span", { class: `step-dot ${tone}`, "aria-hidden": "true" }),
      h("div", { class: "step-body" },
        h("div", { class: "step-head" }, h("strong", { text: step.label }), badge(tone, step.word || WORD[step.state] || step.state),
          step.t !== null ? h("button", { class: "link when", type: "button", title: "Show this moment on the timeline, tree and camera",
            onclick: () => app.selectMoment(step.t, ref) }, clock(step.t)) : h("span", { class: "when muted", text: "No time" })),
        h("p", { class: "step-meta", text: [step.source, step.note].filter(Boolean).join(" · ") || "Not reported" })));
  }));
}

function lifecycleSteps(row, now, detectedLabel) {
  const labels = { detected: detectedLabel, decided: "Abort decided", requested: "Abort requested", accepted: "Controller acknowledgement",
    stopped: "Skill stopped", physical: "Physical stop" };
  return steps({ ...row, stages: ordered(row) }, now).filter((s) => s.key !== "retry").map((s) => ({ ...s, label: labels[s.key] || s.label,
    word: s.key === "physical" ? "Unverified" : s.key === "accepted" && s.state === "done" && /after the deadline/.test(s.note) ? "After deadline"
      : s.key === "detected" && s.state === "done" ? "Confirmed by monitor" : s.key === "requested" && s.state === "done" ? "Sent to executor" : null }));
}

/** The request's times (requested, deadline, acknowledged, ended, elapsed) and its three states. */
function requestTimes(app, status, ref) {
  const at = (t) => (t === null ? null : h("button", { class: "link", type: "button", onclick: () => app.selectMoment(t, ref) }, clock(t)));
  const t = status.times;
  return facts([["Requested", at(t.requested)], ["Ack deadline", t.deadline === null ? "Unknown" : at(t.deadline)],
    ["Acknowledged", t.acknowledged === null ? "Not received" : at(t.acknowledged)], ["Ended", t.ended === null ? "Still open" : at(t.ended)],
    ["Elapsed", t.elapsed === null ? null : duration(t.elapsed)],
    ["States", `Acknowledgement: ${ACK_TEXT[status.ack]} · Command: ${COMMAND_TEXT[status.command]} · Physical: ${PHYSICAL_TEXT[status.physical]}`]]);
}

/** The compact evidence card: the attributed camera frame, the score against its threshold, the input checks with their
 * sources; the full evidence (assessment, score meaning, metadata) is behind "Inspect full evidence". */
function evidenceCard(app, selection, ref) {
  const state = app.state;
  const o = selection.observation;
  const e = selection.evidence;
  const q = o ? qualityAttribution(o) : null;
  const thumb = e?.frame ? h("button", { class: "evidence-thumb", type: "button", title: "Show this frame's moment",
    onclick: () => app.selectMoment(e.frame.t, ref) }, h("img", { src: `history/${e.frame.id}.jpg`, alt: e.label,
    onerror: (event) => event.target.replaceWith(note("No longer kept")) })) : h("div", { class: "evidence-thumb empty", text: "No recording" });
  const score = o ? `${signalLabel(o.signal)} ${o.risk === null || o.risk === undefined ? "not reported" : Number(o.risk).toFixed(2)}`
    + (o.threshold !== null && o.threshold !== undefined ? ` · threshold ${Number(o.threshold).toFixed(2)}` : "") : "No monitor observation linked";
  return h("section", { class: "evidence-card" },
    thumb,
    h("div", { class: "evidence-facts" },
      h("span", { class: `evidence-source ${e?.relation || "unavailable"}`, text: e?.label || "Camera evidence unavailable for this timestamp" }),
      h("strong", { text: score }),
      o ? h("span", { class: "agent-meta", text: `${selection.detecting ? "Detecting observation" : "Newest observation of the episode"} ${o.id || ""} · ${clock(o.t)} · applies to ${scopeText(o, state)}` }) : null,
      q ? h("span", { class: "quality-line" }, badge(q.detector.tone || "neutral", q.detector.text)) : null,
      q?.camera ? h("span", { class: "quality-line" }, badge(q.camera.tone || "neutral", q.camera.text)) : null),
    h("details", { class: "evidence-more", dataset: { key: `evidence:${ref.intervention || ref.incident}` } },
      h("summary", { text: "Inspect full evidence" }),
      o ? h("dl", { class: "assessment" }, ...assessment(o).flatMap((row) => [h("dt", { text: row.label }),
        h("dd", {}, row.tone ? badge(row.tone, row.value) : h("strong", { text: row.value }), row.note ? h("span", { class: "agent-meta", text: row.note }) : null)])) : null,
      e?.frame ? h("img", { src: `history/${e.frame.id}.jpg`, alt: e.label }) : null,
      e?.frame ? note(`${e.label}.${e.relation === "detector_input" ? "" : " Not the detector's input: the frames the monitor read are not available to the dashboard."}`) : null,
      facts([["Input", o?.input_source || "Not reported"], ["Evidence", e?.metadata?.evidence_id],
        ["Offset", e?.frame ? `${e.offset >= 0 ? "+" : "−"}${Math.abs(e.offset).toFixed(2)} s (${e.basis})` : null]])));
}

/** The affected attempt at the selected moment and its current result, from its recorded start and end; the tree's
 * composite nodes keep no history, so only the attempt has one. */
function attemptCard(app, selection) {
  const tool = selection.tool;
  if (!tool) return note("Affected attempt: unknown (the record names no skill call).");
  return h("section", { class: "evidence-block" },
    h("div", { class: "evidence-head" }, h("strong", { text: attemptLabel(tool) }),
      h("button", { class: "link", type: "button", onclick: () => app.drawer.tool(tool.id) }, "Open tool call")),
    facts([[`At ${clock(selection.t)}`, stateAt(tool, selection.t)], ["Current result", tool.end === null ? "Running" : stateAt(tool, Infinity)],
      ["Controller", controllerText(app.state.controller)]]));
}

function controllerText(c) {
  if (!c) return "No report reaches this dashboard";
  if (c.state === "in_process") return "In process (the session's executor: its events are the lifecycle)";
  return `${c.state}${c.checked ? ` · checked ${clock(c.checked)}` : ""} (current report; no history is kept)`;
}

function technical(key, data) {
  return h("details", { class: "technical", dataset: { key } }, h("summary", { text: "Technical details" }),
    h("pre", { class: "json", text: json(data) }));
}

function alertActions(drawer, incident) {
  if (!incident?.alert) return null;
  return h("div", { class: "controls wrap" },
    incident.state === "active" ? h("button", { class: "btn small", type: "button", onclick: () => drawer.alertAction("acknowledge", incident.id) }, "Acknowledge") : null,
    incident.state !== "resolved" ? h("button", { class: "btn small", type: "button", onclick: () => drawer.alertAction("resolve", incident.id) }, "Resolve") : null);
}

/** The drawer's view of intervention ID. */
export function interventionView(drawer, id) {
  const app = drawer.app;
  const state = app.state;
  const row = (state.interventions || []).find((r) => r.id === id);
  if (!row) return note("This intervention is no longer kept.");
  const now = app.link.now();
  const ref = { intervention: id };
  const card = incidentCard(state, row);
  const selection = resolveSelection(state, ref, now);
  const { incident, tool } = selection;
  const status = requestStatus(row, now);
  const [statusLine, physicalLine] = overviewLines(tool, status);
  const detected = `${card.title.replace(/ detected$/, "").replace(/ stop requested by the operator$/, " stop")} detected`;
  const recovery = recoveryState(row);
  const node = selection.goal && state.tree?.nodes[selection.goal];
  return [
    h("header", { class: "inspector-head" }, h("strong", { text: card.title }), badge(COMMAND_TONE[status.command], COMMAND_TEXT[status.command]),
      badge(SEVERITY[card.severity] || "neutral", card.severity)),
    h("section", { class: "inspector-summary" },
      h("p", { class: "summary-attempt", text: `${tool ? attemptLabel(tool) : "Attempt unknown"} · detected ${clock(selection.t)} (${ago(now - selection.t)})` }),
      h("p", { text: concern(selection.observation?.signal, row.initiator) }),
      h("p", { class: "summary-status", text: statusLine }),
      h("p", { class: "summary-physical", text: physicalLine }),
      selection.missing.length ? h("p", { class: "agent-meta", text: `Not established: ${selection.missing.join(" · ")}` }) : null),
    h("h3", { text: "Intervention lifecycle" }),
    lifecycle(app, row, lifecycleSteps(row, now, row.initiator === "Human operator" ? "Detection" : detected), selection.observation, ref),
    requestTimes(app, status, ref),
    h("h3", { text: "Evidence" }),
    evidenceCard(app, selection, ref), attemptCard(app, selection),
    h("h3", { text: "Recovery and outcome" }),
    facts([["Recovery", h("span", {}, badge(RECOVERY_TONE[recovery.state], recovery.state), ` ${recovery.note}`)],
      ["Initiator", recovery.source || null],
      ["Outcome", `${COMMAND_TEXT[status.command]}: ${status.headline}`],
      ["Plan goal now", node ? `${node.label} · ${node.status}` : selection.goal ? "No longer in the tree" : "Unknown"],
      ["Alert", incident?.alert ? INCIDENT_STATE[incident.state] || incident.state : "No alert linked"]]),
    alertActions(drawer, incident),
    technical(`technical:${id}`, { intervention: { ...row, stages: ordered(row) }, request_status: status, incident,
      [selection.detecting ? "detecting_observation" : "newest_observation_of_episode"]: selection.observation,
      camera_evidence: selection.evidence?.metadata || "unavailable",
      tool: tool && { id: tool.id, execution: tool.execution, call: tool.call, skill: tool.skill, arguments: tool.arguments, stages: tool.stages,
        code: tool.code, message: tool.message },
      node: node ? { id: node.id, ...node.detail } : null,
      clocks: "Times are the dashboard's clock when each event arrived; sim times are the simulator's. No trace ids are reported." }),
  ];
}

/** The drawer's view of incident ID (a detection or event without an intervention), in the same order. */
export function incidentView(drawer, id) {
  const app = drawer.app;
  const state = app.state;
  const incident = state.incidents.find((i) => i.id === id);
  if (!incident) return note("This incident is no longer kept.");
  const now = app.link.now();
  const ref = { incident: id };
  const selection = resolveSelection(state, ref, now);
  const tool = selection.tool;
  const cause = incident.caused_by && state.incidents.find((i) => i.id === incident.caused_by);
  return [
    h("header", { class: "inspector-head" }, h("strong", { text: incident.title }),
      badge(SEVERITY[incident.severity] || "neutral", incident.severity),
      badge("neutral", incident.alert ? INCIDENT_STATE[incident.state] || incident.state : "History")),
    h("section", { class: "inspector-summary" },
      h("p", { class: "summary-attempt", text: `${tool ? attemptLabel(tool) : "Attempt unknown"} · ${clock(incident.t)} (${ago(now - incident.t)})` }),
      h("p", { text: incident.reason || (selection.observation ? concern(selection.observation.signal) : "No description recorded.") }),
      h("p", { class: "summary-status", text: `Detected by ${incident.detection || "an unrecorded source"} · ${incident.action || "no action reported"}` }),
      incident.evidence?.synthetic ? h("p", { class: "badge synthetic", text: "Synthetic data: not a real safety or execution outcome" }) : null),
    h("h3", { text: "Evidence" }),
    evidenceCard(app, selection, ref), attemptCard(app, selection),
    h("h3", { text: "Outcome" }),
    facts([["Outcome", incident.outcome || "None recorded"], ["Recovery initiator", incident.initiator || "None recorded"],
      ["Recovery executor", incident.executor || "None recorded"],
      ["Caused by", cause ? h("button", { class: "link", type: "button", onclick: () => app.openIncident(cause.id) }, `${cause.title} (${cause.id})`) : null],
      ["Acknowledged", incident.acknowledged_at ? clock(incident.acknowledged_at) : null]]),
    alertActions(drawer, incident),
    technical(`technical:${id}`, { incident, camera_evidence: selection.evidence?.metadata || "unavailable",
      tool: tool && { id: tool.id, execution: tool.execution, arguments: tool.arguments, code: tool.code } }),
  ];
}
