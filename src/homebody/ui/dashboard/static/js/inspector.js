// The incident inspector, evidence first: (A) a summary that answers what happened, to which attempt, what the harness
// asked, whether the executor answered and whether the outcome is verified; (B) the intervention's lifecycle as steps;
// (C) the evidence, strongest first (the detecting observation's assessment, the recorded camera frame labelled as
// context, the attempt, the controller's report); (D) recovery and outcome; (E) technical details, collapsed. A stage
// no component reported says Unknown or Unverified, never a guess. Every time is a link that moves the dashboard's
// inspection to that moment; nothing here reaches the robot.
import { assessment, inputWarning, scopeText } from "./assessment.js";
import { ago, clock, duration, h, json } from "./dom.js";
import { attemptLabel, incidentCard } from "./execution.js";
import { ordered, stateText, steps } from "./interventions.js";
import { resolveSelection, stateAt } from "./selection.js";

const TONE = { done: "success", pending: "warning", failed: "danger", unknown: "neutral", skipped: "neutral" };
const WORD = { done: "Done", pending: "Pending", failed: "Failed", unknown: "Unknown", skipped: "Not applicable" };
const SEVERITY = { critical: "danger", warning: "warning", info: "neutral" };
const INCIDENT_STATE = { active: "Active", acknowledged: "Acknowledged", resolved: "Resolved" };

function facts(pairs) {
  return h("dl", { class: "kv small" }, ...pairs.filter(([, v]) => v !== undefined && v !== null && v !== "")
    .flatMap(([k, v]) => [h("dt", { text: k }), h("dd", {}, v)]));
}

const note = (text) => h("p", { class: "note", text });
const badge = (tone, text) => h("span", { class: `badge ${tone}`, text });

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

/** The two lines that answer "what did the harness ask, did the executor answer, is the outcome verified". */
export function answers(row, byKey) {
  const ack = byKey.accepted;
  const ackText = ack.state === "done" ? "Executor acknowledged" : row.state === "timed_out" ? "Executor acknowledgement timed out"
    : ack.state === "failed" ? "Request refused" : ack.state === "skipped" ? "Never acknowledged: the request ended first"
      : "Awaiting executor acknowledgement";
  const asked = byKey.requested.state === "done" ? "Abort requested" : "Abort request not seen by the dashboard";
  const stop = byKey.stopped.state;
  const stopText = stop === "done" ? "Skill stopped (executor result)" : stop === "failed" ? "The skill did not stop as asked"
    : stop === "skipped" ? "No stop: the request was withdrawn" : "Skill stop not yet reported";
  return [`${asked} · ${ackText}`, `${stopText} · Physical stop unverified`];
}

/** The lifecycle steps, labelled for this intervention, with an explicit Recovery step even when none started. */
function lifecycleSteps(row, detectedLabel) {
  const list = steps({ ...row, stages: ordered(row) });
  const labels = { detected: detectedLabel, decided: "Abort decided", requested: "Abort requested", accepted: "Controller acknowledgement",
    stopped: "Skill stopped", physical: "Physical stop", retry: "Recovery" };
  const out = list.map((step) => ({ ...step, label: labels[step.key] || step.label }));
  if (!out.some((s) => s.key === "retry")) {
    const final = ["confirmed", "failed", "rejected", "cancelled"].includes(row.state);
    out.push({ key: "retry", label: "Recovery", state: "unknown", t: null, source: "",
      note: final ? "Not started: no retry or recovery was recorded" : "Not yet: the intervention has not finished" });
  }
  return out;
}

function stepWord(row, step) {
  if (step.key === "accepted" && step.state === "failed") return row.state === "timed_out" ? "Timed out" : "Refused";
  if (step.key === "physical") return "Unverified";
  if (step.key === "retry" && step.state === "unknown") return "Not started";
  if (step.key === "detected" && step.state === "done") return "Confirmed by monitor";
  if (step.key === "requested" && step.state === "done") return "Sent to executor";
  return WORD[step.state] || step.state;
}

function stepper(app, row, list, observation, ref) {
  return h("ol", { class: "stepper" }, ...list.map((step) => {
    const tone = TONE[step.state] || "neutral";
    const event = step.key === "detected" && observation?.id ? `observation ${observation.id}` : step.t !== null ? `${row.id}/${step.key}` : "none recorded";
    return h("li", { class: `step ${step.state}` }, h("span", { class: `step-dot ${tone}`, "aria-hidden": "true" }),
      h("div", { class: "step-body" },
        h("div", { class: "step-head" }, h("strong", { text: step.label }), badge(tone, stepWord(row, step)),
          step.t !== null ? h("button", { class: "link when", type: "button", title: "Show this moment on the timeline, tree and camera",
            onclick: () => app.selectMoment(step.t, ref) }, clock(step.t)) : h("span", { class: "when muted", text: "No time" })),
        h("p", { class: "step-meta", text: [step.source, step.note].filter(Boolean).join(" · ") || "Not reported" }),
        h("p", { class: "step-id", text: `Event: ${event}` })));
  }));
}

/** The detecting observation's assessment: score, threshold, confidence, input quality and reliability, apart. */
function assessmentBlock(app, observation, incident, ref, detecting = true) {
  const state = app.state;
  const warning = inputWarning(observation);
  const peak = incident?.evidence?.readings > 1 ? `This episode: ${incident.evidence.readings} anomalous readings, peak score ${Number(incident.evidence.risk).toFixed(2)}` : "";
  return h("section", { class: "evidence-block" },
    h("div", { class: "evidence-head" }, h("strong", { text: detecting ? "Monitor assessment" : "Monitor assessment (newest reading)" }),
      h("button", { class: "link when", type: "button", title: "Show this observation's moment", onclick: () => app.selectMoment(observation.t, ref) }, clock(observation.t))),
    detecting ? null : note("This incident was recorded before its detecting observation was kept: this is the newest reading of the episode, not necessarily the one that detected it."),
    note([`Observation ${observation.id || "(no id: an older record)"}`, observation.source,
      observation.sim_time !== null && observation.sim_time !== undefined ? `newest frame at sim ${observation.sim_time.toFixed(2)} s` : "no simulation time reported",
      `applies to ${scopeText(observation, state)}`].join(" · ")),
    warning ? h("p", { class: `input-warning ${warning.tone}`, text: `${warning.text}${warning.reason ? `: ${warning.reason}` : ""}. The score is shown as the model gave it.` }) : null,
    h("dl", { class: "assessment" }, ...assessment(observation).flatMap((row) => [
      h("dt", { text: row.label }),
      h("dd", {}, row.tone ? badge(row.tone, row.value) : h("strong", { text: row.value }), row.note ? h("span", { class: "agent-meta", text: row.note }) : null)])),
    facts([["Input", observation.input_source || "Not reported (older record)"], ["Episode", peak]]));
}

/** The recorded robot-camera frame for the moment, labelled as context: the dashboard stores its own frames, not the
 * frames the monitor read. */
function cameraBlock(app, selection, incident, ref) {
  const history = app.state.history || [];
  const stored = incident?.frame ? history.find(([id]) => id === incident.frame) : null;
  const frame = stored ? { id: stored[0], t: stored[1], delta: Math.abs(stored[1] - selection.t) } : selection.frame;
  if (!frame) return h("section", { class: "evidence-block" }, h("strong", { text: "Camera" }), note("Camera evidence unavailable for this timestamp."));
  return h("section", { class: "evidence-block" },
    h("div", { class: "evidence-head" }, h("strong", { text: "Camera" }), badge("neutral", "Recorded"),
      h("button", { class: "link when", type: "button", onclick: () => app.selectMoment(frame.t, ref) }, `captured ${clock(frame.t)}`)),
    note(`Context, not detection input: the robot camera as the dashboard stored it, ${frame.delta.toFixed(1)} s from the selected moment (both on the dashboard's clock). The frames the monitor read are not available to the dashboard.`),
    h("img", { src: `history/${frame.id}.jpg`, alt: `Recorded robot camera at ${clock(frame.t)}`,
      onerror: (event) => event.target.replaceWith(note("That camera frame is no longer kept.")) }));
}

function attemptBlock(app, selection) {
  const tool = selection.tool;
  if (!tool) return h("section", { class: "evidence-block" }, h("strong", { text: "Skill attempt" }), note("Unknown: the record names no skill call."));
  return h("section", { class: "evidence-block" },
    h("div", { class: "evidence-head" }, h("strong", { text: attemptLabel(tool) }),
      h("button", { class: "link", type: "button", onclick: () => app.drawer.tool(tool.id) }, "Open tool call")),
    facts([["Then", `${stateAt(tool, selection.t)} at ${clock(selection.t)}`], ["Now", tool.end === null ? "Running" : stateAt(tool, Infinity)],
      ["Execution id", tool.execution || "Unknown"], ["Started", clock(tool.start)],
      ["Duration", duration((tool.end ?? app.link.now()) - tool.start)], ["Result message", tool.message]]));
}

function controllerBlock(app) {
  const c = app.state.controller;
  const text = !c ? "No controller report reaches this dashboard."
    : c.state === "in_process" ? "In process: the session's skill executor is the controller; its reports are the lifecycle events above."
      : `${c.state}${c.checked ? ` · checked ${clock(c.checked)}` : ""}${c.detail ? ` · ${c.detail}` : ""}. This is the current report: no history of controller reports is kept.`;
  return h("section", { class: "evidence-block" }, h("div", { class: "evidence-head" }, h("strong", { text: "Controller" }),
    h("button", { class: "link", type: "button", onclick: () => app.drawer.controller() }, "Controller status")), note(text));
}

function technical(key, data) {
  return h("details", { class: "technical", dataset: { key } }, h("summary", { text: "Technical details" }),
    h("pre", { class: "json", text: json(data) }));
}

function incidentActions(drawer, incident) {
  if (!incident?.alert) return null;
  return h("div", { class: "controls wrap" },
    incident.state === "active" ? h("button", { class: "btn small", type: "button", onclick: () => drawer.alertAction("acknowledge", incident.id) }, "Acknowledge") : null,
    incident.state !== "resolved" ? h("button", { class: "btn small", type: "button", onclick: () => drawer.alertAction("resolve", incident.id) }, "Resolve") : null);
}

function inspectButtons(app, ref) {
  return h("div", { class: "controls wrap" },
    h("button", { class: "btn small", type: "button", onclick: () => app.select(ref, { open: false }) }, "Show on dashboard"),
    h("button", { class: "btn small", type: "button", onclick: () => app.returnToLive() }, "Return to live"));
}

/** The drawer's view of intervention ID. */
export function interventionView(drawer, id) {
  const app = drawer.app;
  const state = app.state;
  const row = (state.interventions || []).find((r) => r.id === id);
  if (!row) return note("This intervention is no longer kept.");
  const card = incidentCard(state, row);
  const selection = resolveSelection(state, { intervention: id }, app.link.now());
  const { incident, tool } = selection;
  const observation = selection.observation;
  const detected = `${card.title.replace(/ detected$/, "").replace(/ stop requested by the operator$/, " stop")} detected`;
  const list = lifecycleSteps(row, row.initiator === "Human operator" ? "Detection" : detected);
  const byKey = Object.fromEntries(list.map((s) => [s.key, s]));
  const [tone, word] = stateText(row.state);
  const retry = byKey.retry;
  const node = selection.goal && state.tree?.nodes[selection.goal];
  return [
    h("header", { class: "inspector-head" }, h("strong", { text: card.title }), badge(tone, word),
      badge(SEVERITY[card.severity] || "neutral", card.severity)),
    h("section", { class: "inspector-summary" },
      h("p", { class: "summary-attempt", text: `${tool ? attemptLabel(tool) : "Attempt unknown"} · detected ${clock(selection.t)} (${ago(app.link.now() - selection.t)})` }),
      h("p", { text: concern(observation?.signal, row.initiator) }),
      h("ul", { class: "summary-answers" }, ...answers(row, byKey).map((line) => h("li", { text: line }))),
      card.warning ? h("p", { class: "input-warning warning", text: card.warning }) : null,
      selection.missing.length ? h("p", { class: "agent-meta", text: `Not established: ${selection.missing.join(" · ")}` }) : null,
      inspectButtons(app, { intervention: id })),
    h("h3", { text: "Intervention lifecycle" }),
    stepper(app, row, list, observation, { intervention: id }),
    h("h3", { text: "Evidence" }),
    observation ? assessmentBlock(app, observation, incident, { intervention: id }, selection.detecting)
      : note(row.initiator === "Human operator" ? "The operator acted; no monitor observation is linked." : "No monitor observation reached the dashboard for this incident."),
    cameraBlock(app, selection, incident, { intervention: id }), attemptBlock(app, selection), controllerBlock(app),
    h("h3", { text: "Recovery and outcome" }),
    facts([["Recovery", retry.state === "unknown" ? retry.note : `${WORD[retry.state] || retry.state}: ${retry.note}`],
      ["Initiator", retry.source || (retry.state === "unknown" ? "None recorded" : "Unknown")],
      ["Verification", retry.state === "done" ? "The retry attempt ended OK (its skill result; the goal's check decides the goal)" : "Not verified"],
      ["Final outcome", card.outcome], ["Physical stop", "Unverified: no motion or safety telemetry reaches the dashboard"],
      ["Plan goal now", node ? `${node.label} · ${node.status}` : selection.goal ? "No longer in the tree" : "Unknown"],
      ["Alert", incident?.alert ? INCIDENT_STATE[incident.state] || incident.state : "No alert linked"]]),
    incidentActions(drawer, incident),
    technical(`technical:${id}`, { intervention: { ...row, stages: ordered(row) }, incident, [selection.detecting ? "detecting_observation" : "newest_observation_of_episode"]: observation,
      newest_observation: incident?.observation || null, tool: tool && { id: tool.id, execution: tool.execution, call: tool.call, skill: tool.skill,
        arguments: tool.arguments, stages: tool.stages, code: tool.code, message: tool.message },
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
  const selection = resolveSelection(state, { incident: id }, app.link.now());
  const observation = selection.observation;
  const tool = selection.tool;
  const cause = incident.caused_by && state.incidents.find((i) => i.id === incident.caused_by);
  return [
    h("header", { class: "inspector-head" }, h("strong", { text: incident.title }),
      badge(SEVERITY[incident.severity] || "neutral", incident.severity),
      badge("neutral", incident.alert ? INCIDENT_STATE[incident.state] || incident.state : "History")),
    h("section", { class: "inspector-summary" },
      h("p", { class: "summary-attempt", text: `${tool ? attemptLabel(tool) : "Attempt unknown"} · ${clock(incident.t)} (${ago(app.link.now() - incident.t)})` }),
      h("p", { text: incident.reason || (observation ? concern(observation.signal) : "No description recorded.") }),
      h("ul", { class: "summary-answers" }, h("li", { text: `Detected by ${incident.detection || "an unrecorded source"}` }),
        h("li", { text: incident.kind === "recovery" ? `Recovery by ${incident.executor || "an unrecorded executor"}` : `Action: ${incident.action || "none reported"}` }),
        h("li", { text: `Outcome: ${incident.outcome || "none recorded"}` })),
      incident.evidence?.synthetic ? h("p", { class: "badge synthetic", text: "Synthetic data: not a real safety or execution outcome" }) : null,
      inspectButtons(app, { incident: id })),
    h("h3", { text: "Evidence" }),
    observation ? assessmentBlock(app, observation, incident, { incident: id }, selection.detecting) : null,
    cameraBlock(app, selection, incident, { incident: id }), attemptBlock(app, selection),
    h("h3", { text: "Outcome" }),
    facts([["Outcome", incident.outcome || "None recorded"], ["Recovery initiator", incident.initiator || "None recorded"],
      ["Recovery executor", incident.executor || "None recorded"],
      ["Caused by", cause ? h("button", { class: "link", type: "button", onclick: () => drawer.incident(cause.id) }, `${cause.title} (${cause.id})`) : null],
      ["Acknowledged", incident.acknowledged_at ? clock(incident.acknowledged_at) : null]]),
    incidentActions(drawer, incident),
    technical(`technical:${id}`, { incident, tool: tool && { id: tool.id, execution: tool.execution, arguments: tool.arguments, code: tool.code } }),
  ];
}
