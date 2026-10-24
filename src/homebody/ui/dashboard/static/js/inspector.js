// The intervention provenance inspector: how a request to stop the running skill went, stage by stage, each with
// its recorded time and source. Detection, request, the executor's acknowledgement, the skill's stop and what came
// next are separate stages; one no component reported reads "Unknown", never a guess. Clicking a stage pins the
// timeline, the tree and the camera history to its moment; opening the inspector changes nothing on the robot.
import { ago, clock, duration, h } from "./dom.js";
import { flowText, signalLabel, stateText, steps } from "./interventions.js";

function facts(pairs) {
  return h("dl", { class: "kv" }, ...pairs.filter(([, v]) => v !== undefined && v !== null && v !== "")
    .flatMap(([k, v]) => [h("dt", { text: k }), h("dd", {}, v)]));
}

const known = (value) => value || "Unknown";
const final = (row) => row.stages.find(([s]) => ["confirmed", "failed", "rejected", "cancelled"].includes(s)) || null;

/** The drawer's view of intervention ID. */
export function interventionView(drawer, id) {
  const app = drawer.app;
  const state = app.state;
  const row = (state.interventions || []).find((r) => r.id === id);
  if (!row) return h("p", { class: "note", text: "This intervention is no longer kept." });
  const incident = state.incidents.find((i) => i.id === row.incident) || null;
  const tool = state.tools.find((t) => t.id === row.tool) || null;
  const node = row.node && state.tree?.nodes[`goal:${row.node}`];
  const [tone, label] = stateText(row.state);
  const stage = (name) => row.stages.find(([s]) => s === name) || null;
  const requested = stage("requested"), accepted = stage("accepted"), done = final(row);
  const start = (stage("detected") || requested)?.[1] ?? row.t;
  const next = done && state.decisions.find((d) => d.t > done[1]);
  const goto = (t) => {
    if (t === null || t === undefined) return;
    app.inspect(t);
    if (row.tool) app.tree.highlightAt(t);
  };
  const flow = steps(row).map((step) => h("li", { class: `flow-step ${step.state}` }, h("button", {
    type: "button", disabled: step.t === null, title: step.t === null ? "No recorded time" : "Show this moment on the timeline, tree and camera",
    onclick: () => goto(step.t),
  }, h("strong", { text: step.label }), h("span", { text: step.t !== null ? ` · ${clock(step.t)}` : "" }),
  h("br"), h("span", { class: "agent-meta", text: [step.t !== null ? step.source : "", step.note].filter(Boolean).join(" · ") || "Unknown" }))));
  return [
    h("h3", { text: incident ? incident.title : `${row.initiator}: stop ${row.skill || "the running skill"}` }),
    h("p", { class: `badge ${tone}`, text: label }),
    h("p", { class: "note", text: flowText(row, incident ? `${signalLabel(incident.observation?.signal)} high` : "Detected") }),
    h("ol", { class: "flow" }, ...flow),
    h("h3", { text: "Incident" }),
    facts([["Intervention", `${row.id} · ${clock(row.t)} (${ago(app.link.now() - row.t)})`],
      ["Incident", incident ? `${incident.id} · ${incident.kind} · ${incident.severity} · ${incident.state}` : "None: no detection recorded"],
      ["Task", row.task], ["Plan node", node ? `${node.label} (${row.node})` : known(row.node)], ["Current status", label]]),
    h("h3", { text: "Detection" }),
    incident ? facts([["Detecting component", known(incident.detection)],
      ["Observation", incident.observation ? `${incident.observation.level} · ${incident.observation.reason || ""}` : "Not recorded"],
      ["Signal", incident.observation ? signalLabel(incident.observation.signal) : null],
      ["Risk", incident.evidence?.risk !== undefined ? Number(incident.evidence.risk).toFixed(2) : "Not reported"],
      ["Threshold", incident.evidence?.threshold !== undefined ? Number(incident.evidence.threshold).toFixed(2) : "Not reported"],
      ["Confidence", incident.evidence?.confidence !== undefined ? `${Number(incident.evidence.confidence).toFixed(2)} (the top action's probability, not calibrated)` : "Not reported"]])
      : h("p", { class: "note", text: row.initiator === "Human operator" ? "The operator acted; no monitor detection is linked." : "Unknown: no detection reached the dashboard." }),
    h("h3", { text: "Decision and request" }),
    facts([["Deciding component", known(row.decided_by)], ["Reason", known(row.reason)], ["Requested action", "Stop the running skill"],
      ["Request initiator", row.initiator], ["Request time", requested && requested[1] !== null ? clock(requested[1]) : "Unknown"]]),
    h("h3", { text: "Execution" }),
    facts([["Target", row.target], ["Acknowledgement", accepted ? `${clock(accepted[1])} · ${accepted[2]}` : row.state === "timed_out" ? `None within ${row.ack_timeout} s` : "None reported"],
      ["Action taken", tool ? `${tool.skill}() ended ${tool.end ? `with ${tool.code}` : "— still running"}` : "Unknown: no tool call linked"],
      ["Verification", done ? `${done[0]} · ${done[2]}` : "Pending"],
      ["Physical stop", "Unverified: no motion or safety telemetry reaches the dashboard"],
      ["Duration", done && start !== null ? duration(done[1] - start) : `${duration(app.link.now() - start)} so far`]]),
    h("h3", { text: "What came next" }),
    next ? facts([["Next decision", `${next.skill} at ${clock(next.t)}`], ["Decided by", "Agent planner"],
      ["Link", "Consecutive only: no recorded causal link to this intervention"]])
      : h("p", { class: "note", text: done ? "No decision recorded after it yet." : "The intervention has not finished." }),
    h("div", { class: "controls wrap" },
      h("button", { class: "btn", type: "button", onclick: () => goto(start) }, "Inspect on timeline"),
      incident ? h("button", { class: "btn", type: "button", onclick: () => drawer.incident(incident.id) }, "Open incident") : null,
      node ? h("button", { class: "btn", type: "button", onclick: () => drawer.node(`goal:${row.node}`) }, "Open plan node") : null,
      tool ? h("button", { class: "btn", type: "button", onclick: () => drawer.tool(tool.id) }, "Open tool call") : null),
  ];
}
