// The incident inspector: the full lifecycle of a request to stop a skill attempt, in seven sections (detection,
// intervention decision, control request, controller acknowledgement, execution confirmation, recovery, final
// outcome), each with its status, time, source, attempt and evidence. A stage no component reported reads "Unknown"
// or "Unverified", never a guess. Clicking a section pins the timeline, the tree and the camera history to its moment;
// opening the inspector changes nothing on the robot.
import { ago, clock, duration, h } from "./dom.js";
import { attemptLabel, incidentCard } from "./execution.js";
import { flowText, signalLabel, stateText, steps } from "./interventions.js";

const STATUS = { done: ["success", "Done"], pending: ["warning", "Pending"], failed: ["danger", "Failed"],
  unknown: ["neutral", "Unknown"], skipped: ["neutral", "Not applicable"] };

function facts(pairs) {
  return h("dl", { class: "kv small" }, ...pairs.filter(([, v]) => v !== undefined && v !== null && v !== "")
    .flatMap(([k, v]) => [h("dt", { text: k }), h("dd", {}, v)]));
}

const number = (v) => (v === undefined || v === null ? "Not reported" : Number(v).toFixed(2));

/** The drawer's view of intervention ID. */
export function interventionView(drawer, id) {
  const app = drawer.app;
  const state = app.state;
  const row = (state.interventions || []).find((r) => r.id === id);
  if (!row) return h("p", { class: "note", text: "This intervention is no longer kept." });
  const card = incidentCard(state, row);
  const { incident, tool } = card;
  const attempt = tool ? `${attemptLabel(tool)} (${tool.execution})` : "Unknown (the reading named no skill call)";
  const node = row.node && state.tree?.nodes[`goal:${row.node}`];
  const goto = (t) => {
    if (t === null || t === undefined) return;
    app.focusAttempt(row.tool, t);
  };
  const byKey = Object.fromEntries(steps(row).map((step) => [step.key, step]));
  const final = row.stages.find(([s]) => ["confirmed", "failed", "rejected", "cancelled"].includes(s)) || null;
  const next = final && state.decisions.find((d) => d.t > final[1]);
  const observation = incident?.observation;
  const evidence = {
    detected: incident ? [facts([["Detector", incident.detection || "Unknown"], ["Observation", observation ? `${observation.level} · ${observation.reason || ""}` : "Not recorded"],
      ["Signal", observation ? signalLabel(observation.signal) : null], ["Risk", number(incident.evidence?.risk)],
      ["Threshold", number(incident.evidence?.threshold)],
      ["Confidence", incident.evidence?.confidence !== undefined ? `${number(incident.evidence.confidence)} (the top action's probability, not calibrated)` : null],
      ["Frames", observation?.frame_quality ? `${observation.frame_quality.replace("_", " ")} at detection` : null]]),
      incident.frame ? h("img", { src: `history/${incident.frame}.jpg`, alt: "Robot camera at detection",
        onerror: (event) => event.target.replaceWith(h("p", { class: "note", text: "That camera frame is no longer kept." })) }) : null]
      : h("p", { class: "note", text: row.initiator === "Human operator" ? "The operator acted; no monitor detection is linked." : "No detection reached the dashboard." }),
    decided: facts([["Deciding component", row.decided_by || "Unknown"], ["Policy", row.initiator === "Human operator" ? "The operator's own decision" : "Abort on a failure label, stall or unsafe reading"], ["Reason", row.reason || "Unknown"]]),
    requested: facts([["Requested action", "Stop the running skill"], ["Initiator", row.initiator], ["Target", row.target]]),
    accepted: facts([["Controller", row.target], ["Answer", byKey.accepted.note || "None"]]),
    stopped: facts([["Skill result", tool?.end ? `${tool.code || tool.state}` : "Still running"],
      ["Physical stop", "Unverified: no motion or safety telemetry reaches the dashboard"],
      ["Duration", final && byKey.requested.t !== null ? duration(final[1] - byKey.requested.t) : null]]),
    retry: byKey.retry ? facts([["Initiator", "Plan goal (its next skill call, chosen by the agent)"], ["Result", byKey.retry.note]])
      : next ? facts([["Next decision", `${next.skill} at ${clock(next.t)}`], ["Link", "Consecutive only: no recorded causal link"]])
        : h("p", { class: "note", text: final ? "No retry or recovery recorded yet." : "The intervention has not finished." }),
  };
  const sections = [["detected", "Detection"], ["decided", "Intervention decision"], ["requested", "Control request"],
    ["accepted", "Controller acknowledgement"], ["stopped", "Execution confirmation"], ["retry", "Recovery"]];
  const section = ([key, title]) => {
    const step = byKey[key] || { state: "unknown", t: null, source: "", note: "" };
    const [tone, word] = STATUS[step.state] || STATUS.unknown;
    return h("section", { class: `lifecycle ${step.state}` },
      h("button", { class: "lifecycle-head", type: "button", disabled: step.t === null,
        title: step.t === null ? "No recorded time" : "Show this moment on the timeline, tree and camera", onclick: () => goto(step.t) },
      h("strong", { text: title }), h("span", { class: `badge ${tone}`, text: word }),
      h("span", { class: "when", text: step.t !== null ? clock(step.t) : "—" })),
      facts([["Source", step.t !== null ? step.source : step.note], ["Attempt", attempt], ["Note", step.t !== null ? step.note : null]]),
      evidence[key]);
  };
  const [tone, word] = stateText(row.state);
  return [
    h("h3", { text: card.title }),
    h("p", {}, h("span", { class: `badge ${tone}`, text: word }), ` ${card.attempt} · ${ago(app.link.now() - row.t)}`),
    h("p", { class: "note", text: flowText(row, `${card.title.replace(" detected", "")} detected`) }),
    ...sections.map(section),
    h("section", { class: "lifecycle final" }, h("div", { class: "lifecycle-head" }, h("strong", { text: "Final outcome" }),
      h("span", { class: `badge ${tone}`, text: word })),
    facts([["Outcome", card.outcome], ["Incident", incident ? `${incident.id} · ${incident.severity} · ${incident.state}${incident.outcome ? ` · ${incident.outcome}` : ""}` : "None linked"],
      ["Plan node", node ? `${node.label} (${row.node})` : row.node || "Unknown"], ["Intervention", row.id]])),
    h("div", { class: "controls wrap" },
      incident ? h("button", { class: "btn", type: "button", onclick: () => drawer.incident(incident.id) }, "Open incident") : null,
      node ? h("button", { class: "btn", type: "button", onclick: () => drawer.node(`goal:${row.node}`) }, "Open plan node") : null,
      tool ? h("button", { class: "btn", type: "button", onclick: () => drawer.tool(tool.id) }, "Open tool call") : null),
  ];
}
