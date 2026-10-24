// The right panel, ordered by what the operator needs now: while a task runs, robot status, alerts and the agent's
// account; once it has finished, its outcome, the unresolved alerts and a compact status. A critical alert and a
// pause each get a fixed slot above; minor warnings never rearrange the panel. Nothing here is invented.
import { $, ago, clock, duration, h, icon, setText, toggleClass } from "./dom.js";

const HANDS = { free: "Free", verified: "Holding (verified)", held: "Holding (unverified)" };
const SEVERITY = { critical: ["alert", "Critical"], warning: ["alert", "Warning"], info: ["info", "Info"] };
const STATE = { active: "Active", acknowledged: "Acknowledged", resolved: "Resolved" };
const OUTCOME = { success: "success", success_with_recoveries: "success", partial: "partial", unverified: "unverified", failed: "failed" };
const SKILLS = { navigate: "Navigate", pick: "Pick", place: "Place", call_coder: "Coding assistant" };

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

  renderCritical(state) {
    this.critical = state.incidents.findLast((i) => i.alert && i.severity === "critical" && i.state === "active") || null;
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

  renderStatus(state) {
    const now = this.app.link.now();
    const telemetry = this.app.telemetry;
    const age = telemetry ? now - telemetry.t : null;
    const connection = !this.app.link.connected ? ["danger", "Disconnected"]
      : age === null ? ["", "Connected · no frames yet"]
        : age > 2.5 ? ["warning", `Connected · frames stale (${duration(age)})`] : ["success", "Connected"];
    const tool = state.tools.findLast((t) => t.state === "running");
    const pose = telemetry?.pose;
    const hands = state.robot.hands;
    const row = (label, value, muted = false, title = "") => ({ label, value: [value].flat(), muted, title });
    const rows = [
      row("Connection", [h("span", { class: `status-dot ${connection[0]}` }), connection[1]]),
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
    ];
    $("robot-details").replaceChildren(...details.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]));
  }

  renderAlerts(state) {
    const open = state.incidents.filter((i) => i.alert && i.state !== "resolved");
    const shown = open.slice(-2).reverse(); // the newest two; the rest are one click away in View all
    const active = open.filter((i) => i.state === "active").length;
    const count = $("alert-count");
    count.hidden = open.length === 0;
    setText(count, String(open.length));
    count.title = `${active} active, ${open.length - active} acknowledged`;
    const now = this.app.link.now();
    $("alerts").replaceChildren(...shown.map((incident) => {
      const [name, label] = SEVERITY[incident.severity] || SEVERITY.info;
      return h("li", {}, h("button", {
        class: `alert ${incident.severity} ${incident.state}`, type: "button",
        "aria-label": `${label}, ${STATE[incident.state]}: ${incident.title}. Inspect the incident.`,
        onclick: () => this.app.openIncident(incident.id),
      }, icon(name), h("span", {}, h("span", { class: "alert-title", text: incident.title }),
        h("span", { class: "alert-reason", text: incident.reason || incident.action })),
      h("span", { class: "alert-when" }, ago(now - incident.t), h("br"), incident.state === "acknowledged" ? "Acknowledged" : "Inspect")));
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
