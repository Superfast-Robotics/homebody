// The right panel, ordered by what the operator needs now: while a task runs, robot status, alerts and the agent's
// account; once it has finished, its outcome, the unresolved alerts and a compact status. A critical alert and a
// pause each get a fixed slot above; minor warnings never rearrange the panel. Nothing here is invented.
import { $, ago, clock, duration, h, icon, setText, toggleClass } from "./dom.js";
import { attemptLabel, attemptRelation, consistency, runningAttempt, toolById } from "./execution.js";
import { activeIntervention, alertGroups, flowText, signalLabel, stateText, steps } from "./interventions.js";
import { monitorStatus } from "./model.js";

const HANDS = { free: "Free", verified: "Holding (verified)", held: "Holding (unverified)" };
const SEVERITY = { critical: ["alert", "Critical"], warning: ["alert", "Warning"], info: ["info", "Info"] };
const STATE = { active: "Active", acknowledged: "Acknowledged", resolved: "Resolved" };
const OUTCOME = { success: "success", success_with_recoveries: "success", partial: "partial", unverified: "unverified", failed: "failed" };
const SKILLS = { navigate: "Navigate", pick: "Pick", place: "Place", call_coder: "Coding assistant" };
const MARKS = { done: "✓", pending: "…", failed: "✗", unknown: "?", skipped: "–" };
// a step's right edge when it has no time to show: one word; the full explanation is the step's tooltip
const SHORT = { "accepted:pending": "Waiting", "accepted:failed": "No answer", "stopped:pending": "Not yet",
  "physical:unknown": "Unverified", "retry:pending": "None yet", unknown: "Not reported", skipped: "—", failed: "Failed", pending: "Waiting" };
const CAMERA = { live: "success", stale: "warning", reconnecting: "warning", offline: "danger", unknown: "" };
const CONTROLLER_QUIET_S = 15; // s without a report from the skill executor, while a skill runs, before it is said

/** When the skill executor last reported (a stage, a result, a stop it took), from the recorded events only. */
export function lastControllerReport(state) {
  const times = [];
  for (const tool of state.tools) {
    times.push(tool.start);
    for (const [t] of tool.stages) times.push(t);
    if (tool.end !== null) times.push(tool.end);
  }
  for (const row of state.interventions || []) {
    for (const [, t, source] of row.stages) if (t !== null && source.startsWith("Skill executor")) times.push(t);
  }
  return times.length ? Math.max(...times) : null;
}

/** A target in words, from the decision's own arguments: a click on the camera image or a map point, never a name. */
export function targetText(target) {
  if (!target) return null;
  const skill = SKILLS[target.skill] || target.skill;
  const point = target.point_normalized_1000 || target.release_point_1000;
  const hand = target.hand ? ` with the ${target.hand} hand` : "";
  if (point) return `${skill} at the camera point (${point.join(", ")})${hand}`;
  const goal = target.goal_xy_m || target.goal_map_1000;
  if (goal) return `${skill} to ${target.goal_xy_m ? `x ${goal[0]} m, y ${goal[1]} m` : `map point (${goal.join(", ")})`}`;
  return `${skill}${hand}`;
}

export function coordinates(pose, frame) {
  const heading = Math.round((pose[2] * 180) / Math.PI);
  return `x ${pose[0].toFixed(2)} m, y ${pose[1].toFixed(2)} m, heading ${heading}°${frame ? ` (${frame} frame)` : ""}`;
}

export class Info {
  constructor(app) {
    this.app = app;
    this.panel = $("info-panel");
    this.finished = null;
    $("info-toggle").addEventListener("click", () => this.toggle());
    $("alerts-all").addEventListener("click", () => app.drawer.alerts());
    $("agent-history").addEventListener("click", () => app.drawer.decisions());
    $("critical-inspect").addEventListener("click", () => this.critical && app.openIncident(this.critical.id));
    $("intervention-inspect").addEventListener("click", () => this.intervention && app.drawer.intervention(this.intervention.id));
    $("intervention-trace").addEventListener("click", () => {
      if (this.intervention?.tool) app.focusAttempt(this.intervention.tool, null, () => app.drawer.tool(this.intervention.tool));
    });
    $("mismatch-inspect").addEventListener("click", () => {
      const tool = this.check?.attempt;
      if (tool) app.focusAttempt(tool.id, null, () => app.drawer.tool(tool.id));
    });
    $("mismatch-controller").addEventListener("click", () => app.drawer.controller());
    $("outcome-details").addEventListener("click", () => app.drawer.outcome());
  }

  toggle(open = !this.panel.classList.contains("open")) {
    toggleClass(this.panel, "open", open);
    $("info-toggle").setAttribute("aria-expanded", String(open));
  }

  reveal() {
    this.toggle(true);
    $("alerts").querySelector("button")?.focus();
  }

  render(state) {
    const finished = Boolean(state.task && state.task.end !== null && !state.controls.busy);
    if (finished !== this.finished) { // collapse the agent's account once there is an outcome to read instead
      this.finished = finished;
      toggleClass(this.panel, "finished", finished);
      $("agent-section").open = !finished;
    }
    this.renderOutcome(state);
    this.renderIntervention(state);
    this.renderMismatch(state);
    this.renderRecovery(state);
    this.renderCritical(state);
    this.renderPause(state);
    this.renderStatus(state);
    this.renderAlerts(state);
    this.renderAgent(state);
  }

  renderOutcome(state) {
    const outcome = state.outcome;
    const shown = this.finished && outcome && outcome.state !== "not_evaluated";
    $("outcome-section").hidden = !shown;
    if (!shown) return;
    const o = outcome.objectives;
    const r = outcome.recoveries;
    const facts = [
      o ? `${o.verified}/${o.total} objectives verified${o.agent_closed ? ` · ${o.agent_closed} closed by the agent without a check` : ""}` : "No behavior tree: no objectives to verify",
      r.started ? `${r.started} recover${r.started > 1 ? "ies" : "y"} started · ${r.succeeded} succeeded${r.failed ? ` · ${r.failed} failed` : ""}` : "No recovery",
      r.replans ? `${r.replans} replan${r.replans > 1 ? "s" : ""}` : null,
      outcome.failed_objectives.length ? `Failed: ${outcome.failed_objectives.join(", ")}` : null,
      `${outcome.unresolved_alerts} unresolved alert${outcome.unresolved_alerts === 1 ? "" : "s"}`,
    ].filter(Boolean);
    const box = $("outcome-summary");
    box.className = `outcome-summary ${OUTCOME[outcome.state] || ""}`;
    box.replaceChildren(h("strong", { text: outcome.label }), h("ul", {}, ...facts.map((f) => h("li", { text: f }))),
      h("p", { text: outcome.reason }));
  }

  /** The critical intervention card: the request, and each stage a component reported, while it is open or went
   * wrong; a step nobody reported says so. */
  renderIntervention(state) {
    const row = activeIntervention(state);
    const shown = row !== null;
    if (shown && this.intervention?.id !== row.id) $("agent-section").open = false; // the card leads; the agent's account waits
    this.intervention = row;
    const card = $("intervention-card");
    card.hidden = !shown;
    if (!shown) return;
    const now = this.app.link.now();
    const incident = state.incidents.find((i) => i.id === row.incident) || null;
    const [, label] = stateText(row.state);
    card.className = `intervention-card ${row.state}`;
    setText($("intervention-title"), incident ? incident.title : `${row.initiator}: stop ${row.skill || "the running skill"}`);
    const risk = incident?.evidence?.risk;
    const signal = incident?.observation?.signal;
    setText($("intervention-source"), [incident ? `Detected by ${incident.detection || "an unreported source"}` : `Requested by ${row.initiator}`,
      risk !== undefined && risk !== null ? `Reported ${signalLabel(signal).toLowerCase()}: ${Number(risk).toFixed(2)}` : ""].filter(Boolean).join(" · "));
    const since = row.stages.find(([, t]) => t !== null)?.[1] ?? row.t;
    setText($("intervention-age"), ago(now - since));
    const tool = toolById(state, row.tool);
    const relation = { current: "running now", previous: tool?.end ? `ended ${tool.code ? `with ${tool.code}` : ""}` : "", unknown: "" }[attemptRelation(state, row.tool)];
    setText($("intervention-attempt"), tool ? `Affected attempt: ${attemptLabel(tool)}${relation ? ` (${relation})` : ""} · ${tool.execution}`
      : "Affected attempt: Unknown (the reading named no skill call)");
    setText($("intervention-state"), `Requested action: stop the skill · Status: ${label}`);
    // one line per step; a completed step's source and note are its tooltip, an open or failed one says why
    $("intervention-steps").replaceChildren(...steps(row).map((step) => {
      const detail = [step.source && step.t !== null ? step.source : "", step.note].filter(Boolean).join(" · ");
      const short = step.t !== null && step.state !== "failed" ? clock(step.t) : SHORT[`${step.key}:${step.state}`] || SHORT[step.state] || "";
      return h("li", { class: step.state, title: detail },
        h("span", { class: "mark", text: MARKS[step.state] || "?" }), h("span", { text: step.label }),
        h("span", { class: "when", text: short }));
    }));
    $("intervention-trace").disabled = !row.tool;
  }

  /** A lasting disagreement between the tree, the executor and the controller, past the configured tolerance. */
  renderMismatch(state) {
    const now = this.app.link.now();
    const check = consistency(state, now);
    this.check = check;
    const card = $("mismatch-card");
    card.hidden = !(check.state === "suspected" || check.state === "confirmed");
    if (card.hidden) return;
    card.className = `mismatch-card ${check.severity}`;
    const rows = [["Behavior Tree", check.bt], ["Executor", check.executor], ["Controller", check.controller],
      ["Last controller update", check.controllerAge === null ? "Not reported" : `${duration(check.controllerAge)} ago`],
      ["Classification", `${check.state === "confirmed" ? "Confirmed" : "Suspected"} mismatch · ${check.severity}`],
      ["Attempt", check.attempt ? `${attemptLabel(check.attempt)} · ${check.attempt.execution}` : "None running"], ["Why", check.rule]];
    $("mismatch-facts").replaceChildren(...rows.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]));
    $("mismatch-inspect").disabled = !check.attempt;
  }

  /** A recovery the plan is running: its goal, and who started it, from the recorded recovery event. */
  renderRecovery(state) {
    const span = state.intervals.findLast((i) => i.track === "goal" && i.end === null && i.detail.recovery) || null;
    const card = $("recovery-card");
    card.hidden = !span || !state.controls.busy;
    if (card.hidden) return;
    const started = state.incidents.findLast((i) => i.tag === "recovery_start" && i.node === span.detail.goal);
    setText($("recovery-title"), `Recovery in progress: ${span.label}`);
    setText($("recovery-text"), [started ? `Started by ${started.initiator || "an unrecorded source"}` : "Start not recorded",
      started?.reason || "", `running ${duration(this.app.link.now() - span.start)}`].filter(Boolean).join(" · "));
  }

  renderCritical(state) {
    // only a detection on the attempt running now (or on the task itself) is current: a previous attempt's stays in
    // the alerts list, named by its attempt; the intervention card already shows its own detection
    const running = runningAttempt(state)?.id ?? null;
    this.critical = state.incidents.findLast((i) => i.alert && i.severity === "critical" && i.state === "active" &&
      i.id !== this.intervention?.incident && (!i.tool || i.tool === running) &&
      (i.task === null || i.task === state.task?.id)) || null;
    $("critical-banner").hidden = !this.critical;
    if (!this.critical) return;
    setText($("critical-title"), this.critical.title);
    const why = [this.critical.detection ? `${this.critical.detection}:` : "", this.critical.reason || "No reason given."];
    setText($("critical-reason"), `${why.join(" ").trim()} · ${ago(this.app.link.now() - this.critical.t)}`);
  }

  renderPause(state) {
    const paused = Boolean(state.controls.paused);
    $("pause-card").hidden = !paused;
    if (!paused) return;
    const request = state.incidents.findLast((i) => i.tag === "pause");
    const node = state.tree?.nodes[`goal:${request?.node}`];
    const rows = [
      ["Paused", request ? `${clock(request.t)} (${ago(this.app.link.now() - request.t)})` : "Reported by the controller"],
      ["Source", request ? request.initiator || request.source : "Controller (no operator request recorded)"],
      ["Reason", request?.reason || "None given"],
      ["Node", node ? node.label : request?.node || "None active"],
      ["Awaiting", "Resume from the header"],
    ];
    $("pause-facts").replaceChildren(...rows.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]));
  }

  /** Robot, camera, monitor and controller, each from its own signal: a connected page is not a live camera. */
  freshness(state, now) {
    const link = state.robot_link;
    const robot = !this.app.link.connected ? ["danger", "Disconnected"]
      : link ? [{ connected: "success", lost: "danger" }[link.state] || "", { connected: "Connected", lost: "Lost", unknown: "Unknown" }[link.state] || link.state, link.detail]
        : ["success", "Connected", "The session runs beside this dashboard"];
    const cam = this.app.camera.freshness();
    const camera = [CAMERA[cam.kind], { live: "Live", stale: "Stale", reconnecting: "Reconnecting", offline: "Offline", unknown: "Unknown" }[cam.kind]
      + (cam.age !== null && cam.kind !== "unknown" ? ` · ${duration(cam.age)} old` : cam.kind === "unknown" ? " · no frames received" : "")];
    const m = monitorStatus(state.readings, state.incidents, now, this.app.link.connected);
    const observed = m.observationAge === null ? "no observation yet" : `last observation ${duration(m.observationAge)} ago`;
    const monitor = { connected: ["success", `Connected · ${observed}`], stale: ["warning", `Stale · ${observed}`],
      offline: ["danger", "Offline"], unknown: ["", "Not connected"] }[m.state];
    const report = lastControllerReport(state);
    const quiet = report === null ? null : Math.max(0, now - report);
    const running = runningAttempt(state);
    const executor = report === null ? ["", "Unknown · no report yet"]
      : running && quiet > CONTROLLER_QUIET_S ? ["warning", `${attemptLabel(running)} · no report for ${duration(quiet)}`]
        : ["success", `${running ? `${attemptLabel(running)} running` : "No skill running"} · last report ${duration(quiet)} ago`];
    const c = state.controller;
    const age = c?.checked ? Math.max(0, now - c.checked) : null;
    const [tone, word] = !c ? ["", "Unknown · not reported"] : { in_process: ["success", "In process"], running: ["success", "Running"],
      stopped: ["danger", "Stopped"], error: ["danger", "Failed"], unknown: ["", "Unknown"] }[c.state] || ["", c.state];
    const controller = [tone, age !== null ? `${word} · checked ${duration(age)} ago` : word, c?.detail || ""];
    const check = consistency(state, now);
    const execution = [{ consistent: "success", transitioning: "", unknown: "", suspected: "warning", confirmed: "danger" }[check.state],
      { consistent: "Consistent", transitioning: "Transitioning", unknown: "Unknown", suspected: "Suspected mismatch", confirmed: "Confirmed mismatch" }[check.state],
      check.rule || "The tree, the executor and the controller agree"];
    return { robot, camera, monitor, executor, controller, execution, cam, m, report };
  }

  renderStatus(state) {
    const now = this.app.link.now();
    const telemetry = this.app.telemetry;
    const f = this.freshness(state, now);
    const tool = state.tools.findLast((t) => t.state === "running");
    const pose = telemetry?.pose;
    const hands = state.robot.hands;
    const row = (label, value, muted = false, title = "") => ({ label, value: [value].flat(), muted, title });
    const dot = ([tone, text, title]) => row("", [h("span", { class: `status-dot ${tone}` }), text], false, title || "");
    const rows = [
      { ...dot(f.robot), label: "Robot" }, { ...dot(f.camera), label: "Camera" },
      { ...dot(f.monitor), label: "Monitor" }, { ...dot(f.executor), label: "Executor" },
      { ...dot(f.controller), label: "Controller" }, { ...dot(f.execution), label: "Execution" },
      row("Current skill", tool ? `${SKILLS[tool.skill] || tool.skill}${tool.recovery ? " (recovery)" : ""} · ${duration(now - tool.start)}` : "None", !tool),
      row("Location", telemetry?.place || (pose ? "Coordinates only (see Details)" : "Unavailable"), !telemetry?.place,
        pose ? coordinates(pose, telemetry.frame) : ""),
      row("Target", targetText(state.robot.target) || "None", !state.robot.target),
      row("Hands", hands ? `Left ${HANDS[hands.left] || hands.left} · Right ${HANDS[hands.right] || hands.right}` : "Unavailable", !hands),
    ];
    if (this.finished) rows.splice(1, 1); // a finished task has no current skill
    $("robot-status").replaceChildren(...rows.flatMap((r) => [
      h("dt", { text: r.label }), h("dd", { class: r.muted ? "muted" : "", title: r.title || null }, ...r.value)]));
    const details = [
      ["Position", pose ? coordinates(pose, telemetry.frame) : "Unavailable"],
      ["Target arguments", state.robot.target ? JSON.stringify(state.robot.target) : "None"],
      ["Simulation time", typeof telemetry?.sim === "number" ? `${telemetry.sim.toFixed(1)} s` : "Unavailable"],
      ["Frame", telemetry?.seq !== undefined ? `#${telemetry.seq} · epoch ${telemetry.epoch}` : "Unavailable"],
      ["Last state from the session", state.now ? clock(state.now) : "Unavailable"],
      ["Last camera frame", this.app.camera.frameAt ? clock(this.app.camera.frameAt) : "None received"],
      ["Last monitor observation", f.m.observationAge === null ? "None" : clock(now - f.m.observationAge)],
      ["Last executor report", f.report === null ? "None" : clock(f.report)],
      ["Clock offset to the session", `${this.app.link.offset.toFixed(3)} s (times shown are the session's)`],
      ["Transport", this.app.link.connected ? "WebSocket connected" : "WebSocket disconnected"],
    ];
    $("robot-details").replaceChildren(...details.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]));
  }

  renderAlerts(state) {
    const open = state.incidents.filter((i) => i.alert && i.state !== "resolved");
    const groups = alertGroups(state);
    const shown = groups.slice(0, 2).map((g) => g.incident); // the newest two; the rest are one click away in View all
    const active = open.filter((i) => i.state === "active").length;
    const count = $("alert-count");
    count.hidden = open.length === 0;
    setText(count, String(open.length));
    count.title = `${active} active, ${open.length - active} acknowledged`;
    const now = this.app.link.now();
    const task = state.task?.id ?? null;
    $("alerts").replaceChildren(...shown.map((incident) => {
      const [name, label] = SEVERITY[incident.severity] || SEVERITY.info;
      const historical = incident.task !== null && incident.task !== task;
      const requests = groups.find((g) => g.incident.id === incident.id).interventions;
      const toolId = incident.tool || requests.at(-1)?.tool || null;
      const relation = attemptRelation(state, toolId);
      const attempt = `${attemptLabel(toolById(state, toolId))}${{ current: " · current", previous: " · previous attempt", unknown: "" }[relation]}`;
      const status = historical ? "Earlier task" : incident.state === "acknowledged" ? "Acknowledged" : relation === "previous" ? "Historical" : "Inspect";
      return h("li", {}, h("button", {
        class: `alert ${incident.severity} ${incident.state}${historical ? " historical" : ""}`, type: "button",
        "aria-label": `${label}, ${historical ? "from an earlier task" : STATE[incident.state]}: ${incident.title}. Inspect the incident.`,
        onclick: () => this.app.focusAttempt(toolId, incident.t,
          () => (requests.length ? this.app.drawer.intervention(requests.at(-1).id) : this.app.drawer.incident(incident.id))),
      }, icon(name), h("span", {}, h("span", { class: "alert-attempt", text: attempt }), h("span", { class: "alert-title", text: incident.title }),
        h("span", { class: "alert-reason", text: incident.reason || incident.action })),
      h("span", { class: "alert-when" }, ago(now - incident.t), h("br"), status),
      requests.length ? h("ul", { class: "alert-steps" }, ...requests.map((r) => h("li", {
        text: `${flowText(r, "Detected")} · ${stateText(r.state)[1]}` }))) : null));
    }));
    setText($("alerts-note"), shown.length ? (open.length > 2 ? `${open.length - 2} more in View all.` : "")
      : state.readings.length ? "No unresolved alerts from the session or the monitor."
        : "No unresolved alerts. No monitoring service is connected, so this is not a health verdict.");
  }

  renderAgent(state) {
    const thinking = state.status === "Thinking";
    const text = state.agent_text || (thinking ? "Thinking about the next step…" : state.task ? "No explanation yet." : "Waiting for a task.");
    const meta = [state.model, thinking ? "thinking" : null, "agent-generated, not verified"].filter(Boolean).join(" · ");
    const box = $("agent-context");
    if (!box.firstChild) {
      box.append(h("span", { class: "agent-avatar" }, icon("robot")),
        h("div", {}, h("span", { class: "agent-text" }), h("span", { class: "agent-meta" })));
    }
    setText(box.querySelector(".agent-text"), text);
    setText(box.querySelector(".agent-meta"), meta);
  }
}
