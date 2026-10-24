// The details drawer: incidents, tree nodes, decisions, tool calls, lists, diagnostics and the expanded timeline.
import { $, ago, clock, duration, h, json } from "./dom.js";
import { incidentView, interventionView } from "./inspector.js";
import { attemptLabel, consistency } from "./execution.js";
import { stateText } from "./interventions.js";
import { errorText, summarize } from "./story.js";
import { Timeline } from "./timeline.js";

const LABELS = ["Goal", "Skill", "Agent decisions", "Monitor observations", "Risk", "Interventions", ""];

function facts(pairs) {
  return h("dl", { class: "kv" }, ...pairs.filter(([, v]) => v !== undefined && v !== null && v !== "")
    .flatMap(([k, v]) => [h("dt", { text: k }), h("dd", {}, v)]));
}

export class Drawer {
  constructor(app) {
    this.app = app;
    this.element = $("drawer");
    this.body = $("drawer-body");
    this.current = null; // {kind, id} of what is shown, refreshed on new state
    this.timeline = null;
    $("drawer-close").addEventListener("click", () => this.close());
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !this.element.hidden && !$("confirm").open) this.close();
    });
  }

  /** SIGNATURE() names what the view shows; the view is rebuilt only when it changes. */
  open(title, kind, id, build, { wide = false, signature = () => this.app.state.version } = {}) {
    if (this.element.hidden) this.returnFocus = document.activeElement;
    this.current = { kind, id, build, title, signature, shown: signature() };
    this.element.classList.toggle("wide", wide);
    this.timeline?.stop();
    this.timeline = null;
    $("drawer-title").textContent = title;
    this.body.replaceChildren(...this.content(build));
    this.body.scrollTop = 0; // a new view starts at its summary
    this.element.hidden = false;
    $("drawer-close").focus();
  }

  /** A view's nodes: builders return a node or a list with optional (null) sections, which are left out. */
  content(build) {
    return [build()].flat(Infinity).filter((part) => part !== null && part !== undefined && part !== false);
  }

  close() {
    this.element.hidden = true;
    this.app.tree?.deselect();
    this.timeline?.stop();
    this.timeline = null;
    this.current = null;
    this.returnFocus?.focus?.();
  }

  /** Re-render list and detail views when the state changes; the expanded timeline redraws itself. */
  refresh() {
    const current = this.current;
    if (!current || current.kind === "monitor") return;
    const shown = current.signature();
    if (shown === current.shown) return;
    current.shown = shown;
    const open = [...this.body.querySelectorAll("details[open]")].map((d) => d.dataset.key);
    const scroll = this.body.scrollTop;
    this.body.replaceChildren(...this.content(current.build));
    open.forEach((key) => { const d = this.body.querySelector(`details[data-key="${key}"]`); if (d) d.open = true; });
    this.body.scrollTop = scroll;
  }

  /** An incident without an intervention, in the inspector's order (inspector.js). */
  incident(id) {
    this.open("Incident", "incident", id, () => incidentView(this, id),
      { signature: () => this.signatureOf({ incident: id }) });
  }

  /** What an incident or intervention view shows: the record, its incident and its attempt, so it rebuilds when
   * any of them changes and not on every state. */
  signatureOf({ intervention = null, incident = null }) {
    const s = this.app.state;
    const row = intervention ? (s.interventions || []).find((r) => r.id === intervention) || null : null;
    const found = s.incidents.find((i) => i.id === (incident ?? row?.incident)) || null;
    const tool = s.tools.find((t) => t.id === (row?.tool ?? found?.tool)) || null;
    return JSON.stringify([row, found, tool && [tool.state, tool.end, tool.code], s.controller?.state, Boolean(s.tree)]);
  }

  async alertAction(action, id) {
    const reply = await this.app.link.request(action, { incident: id });
    this.app.notify(reply.ok ? "ok" : "bad", reply.message);
  }

  /** The finished task's outcome, with what verified each objective and the final camera frame. */
  outcome() {
    this.open("Task outcome", "outcome", null, () => {
      const state = this.app.state;
      const outcome = state.outcome;
      if (!outcome) return h("p", { class: "note", text: "No task yet." });
      const goals = Object.values(state.tree?.nodes || {}).filter((n) => n.type === "goal");
      const how = (g) => (g.status === "success" ? (g.detail.check === "the agent closes it" ? "closed by the agent (no check)" : `verified: ${g.detail.check}`)
        : g.status === "failure" ? `failed${g.detail.reason ? `: ${g.detail.reason}` : ""}` : g.status);
      return [
        h("h3", { text: outcome.label }),
        h("p", { class: "note", text: outcome.reason }),
        facts([["Executor", state.lifecycle], ["Task", state.task?.text], ["Ended", state.task?.end ? clock(state.task.end) : "Running"],
          ["Objectives", outcome.objectives ? `${outcome.objectives.verified} verified of ${outcome.objectives.total}` : "No behavior tree"],
          ["Recoveries", `${outcome.recoveries.started} started · ${outcome.recoveries.succeeded} succeeded`],
          ["Unresolved alerts", String(outcome.unresolved_alerts)]]),
        goals.length ? [h("h3", { text: "Goals" }), h("ol", { class: "drawer-list" }, ...goals.map((g) => h("li", {},
          h("strong", { text: g.label }), h("span", { class: "agent-meta", text: how(g) }))))] : null,
        h("h3", { text: "Robot camera at task end" }),
        outcome.frame ? h("img", { src: `history/${outcome.frame}.jpg`, alt: "Robot camera at the end of the task",
          onerror: (event) => event.target.replaceWith(h("p", { class: "note", text: "No recorded camera evidence: that frame is no longer kept." })) })
          : h("p", { class: "note", text: "No recorded camera evidence." }),
        h("p", { class: "note", text: "The outcome comes from the plan's goals and their checks, never from the agent's own report." }),
      ].flat();
    }, { signature: () => JSON.stringify(this.app.state.outcome) });
  }

  /** One second of the risk band: the readings it summarizes, as the monitor reported them. */
  risk(block) {
    this.open("Risk", "risk", null, () => [
      facts([["From", clock(block.start)], ["To", clock(block.end)], ["Class", block.state], ["Source", block.source],
        ["Maximum risk", block.max === null ? "No sample" : block.max.toFixed(2)], ["Readings", String(block.count)],
        ["Threshold", block.threshold === null ? "None reported" : block.threshold.toFixed(2)],
        ["Aggregation", "The block shows the second's maximum; every reading is listed below"]]),
      block.count ? h("ol", { class: "drawer-list" }, ...block.readings.map((r) => h("li", {
        text: `${clock(r.t)} · risk ${r.risk.toFixed(2)} · ${r.level}${r.synthetic ? " · synthetic" : ""}` })))
        : h("p", { class: "note", text: "No risk sample in this second: missing data, not low risk." }),
    ], { signature: () => "fixed" });
  }

  /** A monitor observation segment (or a single reading): its readings, as the monitor reported them. */
  observation(segment) {
    const readings = segment.readings || [segment];
    this.open(segment.level === "missing" ? "Missing monitor readings" : "Monitor observation", "observation", null, () => {
      if (segment.level === "missing") {
        return [facts([["From", clock(segment.start)], ["To", clock(segment.end)], ["Duration", duration(segment.end - segment.start)]]),
          h("p", { class: "note", text: "No monitor reading stands for this stretch of the task. That is missing data, not a normal reading." })];
      }
      const rows = readings.slice(-50).reverse().map((r) => h("li", {}, h("strong", { text: `${clock(r.t)} · ${r.level}` }),
        ` · ${r.source}${r.risk !== null ? ` · risk ${r.risk.toFixed(2)}` : ""}${r.confidence !== null ? ` · confidence ${r.confidence.toFixed(2)}` : ""}${r.synthetic ? " · synthetic" : ""}`,
        r.reason ? h("span", { class: "agent-meta", text: r.reason }) : null,
        r.node || r.skill ? h("span", { class: "agent-meta", text: `During ${r.skill ? `${r.skill}()` : "a task"}${r.node ? ` in goal ${r.node}` : ""}` }) : null));
      return [facts([["Level", segment.level], ["Readings", String(readings.length)], ["From", clock(readings[0].t)],
        ["To", clock(readings.at(-1).t)], ["Source", readings.at(-1).source]]),
      readings.some((r) => r.synthetic) ? h("p", { class: "badge synthetic", text: "Synthetic data: not a real safety or execution outcome" }) : null,
      h("ol", { class: "drawer-list" }, ...rows)].flat().filter(Boolean);
    }, { signature: () => "fixed" });
  }

  superseded(index) {
    this.open("Superseded plan", "superseded", index, () => {
      const plan = this.app.state.superseded?.[index];
      if (!plan) return h("p", { class: "note", text: "This plan is no longer kept." });
      const goals = Object.values(plan.tree.nodes).filter((n) => n.type === "goal");
      return [facts([["Replaced", clock(plan.t)], ["Rewrite", String(plan.replan)]]),
        h("p", { class: "note", text: "The agent replaced this tree. Its goals as they stood:" }),
        h("ol", { class: "drawer-list" }, ...goals.map((g) => h("li", {}, h("strong", { text: `${g.detail.goal}: ${g.label}` }),
          ` · ${g.status}${g.detail.reason ? ` · ${g.detail.reason}` : ""}`)))];
    }, { signature: () => String(this.app.state.superseded?.length) });
  }

  node(id) {
    this.open("Behavior tree node", "node", id, () => {
      const node = this.app.state.tree?.nodes[id];
      if (!node) return h("p", { class: "note", text: "This node is no longer in the tree." });
      const d = node.detail;
      if (node.type === "action") return this.toolView(d.tool);
      if (node.type === "attempts") {
        const tools = node.children.map((child) => this.app.state.tree.nodes[child]?.detail.tool).filter(Boolean)
          .map((tool) => this.app.state.tools.find((t) => t.id === tool)).filter(Boolean);
        return [h("p", { class: "note", text: "The goal's earlier attempts, grouped for reading; the goal is the tree's retry boundary." }),
          this.toolList(tools), this.technical(node)];
      }
      const recovery = d.recovers ? `Recovers from goal ${d.recovers} (${d.recovers_label || d.recovers})` : null;
      const tools = this.app.state.tools.filter((t) => d.goal && t.goal === d.goal);
      const span = this.app.state.intervals.findLast((i) => i.track === "goal" && i.detail.goal === d.goal);
      const failures = this.app.state.incidents.filter((i) => d.goal && i.node === d.goal && i.severity !== "info");
      return [
        h("h3", { text: node.label }),
        facts([
          ["Type", node.type], ["Status", node.status], ["Goal id", d.goal], ["Check", d.check],
          ["Attempts", d.budget ? `${d.used} of ${d.budget} skill calls` : null], ["Recovery", recovery],
          ["Started", span ? clock(span.start) : null], ["Ended", span?.end ? clock(span.end) : null],
          ["Duration", span ? duration((span.end ?? this.app.link.now()) - span.start) : null],
          ["Selected child", d.selected !== undefined && d.selected !== null ? `child ${d.selected + 1}` : null],
          ["Reason", d.reason], ["Note", d.note],
        ]),
        tools.length ? [h("h3", { text: "Tool calls" }), this.toolList(tools)] : null,
        failures.length ? [h("h3", { text: "Failures" }), this.incidentList(failures)] : null,
        this.technical(node, { tools: tools.map((t) => ({ id: t.id, skill: t.skill, state: t.state, code: t.code, attempt: t.attempt })) }),
      ].flat();
    });
  }

  /** The raw node, for debugging: names, ids, codes and recovery metadata as the runtime has them. */
  technical(node, extra = {}) {
    return h("details", { dataset: { key: `technical:${node.id}` } }, h("summary", { text: "Technical details" }),
      h("pre", { class: "json", text: json({ id: node.id, type: node.type, name: node.label, status: node.status, ...node.detail, ...extra }) }));
  }

  toolView(id) {
    const tool = this.app.state.tools.find((t) => t.id === id);
    if (!tool) return h("p", { class: "note", text: "This tool call is no longer kept." });
    return [
      h("h3", { text: `${attemptLabel(tool)} · ${tool.skill}()` }),
      h("div", { class: "controls wrap" }, h("button", { class: "btn small", type: "button", onclick: () => this.app.select({ tool: tool.id }, { open: false }) },
        "Show on dashboard")),
      facts([["State", tool.state], ["Goal", tool.goal], ["Attempt", `${tool.number} in its goal (${tool.attempt} of this skill)`],
        ["Execution id", tool.execution || "Unknown"], ["Skill call", tool.call ?? "Not reported"],
        ["Outcome", tool.code ? `${errorText(tool.code)} (${tool.code})` : tool.state], ["Recovery", tool.recovery ? "Yes" : null],
        ["Started", clock(tool.start)], ["Ended", tool.end ? clock(tool.end) : null],
        ["Duration", duration((tool.end ?? this.app.link.now()) - tool.start)], ["Result", tool.code], ["Message", tool.message]]),
      h("h3", { text: "Arguments" }), h("pre", { class: "json", text: json(tool.arguments) }),
      tool.stages.length ? [h("h3", { text: "Stages" }), h("pre", { class: "json", text: tool.stages.map(([t, s]) => `${clock(t)}  ${s}`).join("\n") })] : null,
      this.toolInterventions(tool),
      this.toolReadings(tool),
    ].flat();
  }

  /** The requests to stop this call, each opening its inspector. */
  toolInterventions(tool) {
    const rows = (this.app.state.interventions || []).filter((r) => r.tool === tool.id);
    if (!rows.length) return null;
    return [h("h3", { text: "Interventions" }), h("ol", { class: "drawer-list" }, ...rows.map((r) => h("li", {},
      h("button", { type: "button", onclick: () => this.intervention(r.id) }, h("strong", { text: `${r.initiator}: stop` }),
        ` · ${stateText(r.state)[1]} · ${r.id}${r.incident ? ` · incident ${r.incident}` : ""}`))))];
  }

  /** The monitor's observations made while this call ran. */
  toolReadings(tool) {
    const end = tool.end ?? this.app.link.now();
    const readings = this.app.state.readings.filter((r) => r.kind === "observation" && r.t >= tool.start && r.t <= end);
    if (!readings.length) return null;
    const counts = {};
    for (const r of readings) counts[r.level] = (counts[r.level] || 0) + 1;
    return [h("h3", { text: "Monitor observations during the call" }),
      h("p", { class: "note", text: Object.entries(counts).map(([level, n]) => `${n} ${level}`).join(" · ") })];
  }

  tool(id) {
    this.open("Tool call", "tool", id, () => this.toolView(id));
  }

  /** Events that share a moment on the timeline, each opening its own record. */
  events(items) {
    this.open("Events at this moment", "events", null, () => h("ol", { class: "drawer-list" }, ...items.map((m) => h("li", {},
      h("button", { type: "button", onclick: () => this.app.open("event", m) }, h("strong", { text: `${clock(m.t)} ${m.label}` }),
        h("br"), m.detail || "")))), { signature: () => items.map((m) => m.id).join("|") });
  }

  /** The incident inspector of an intervention (inspector.js). */
  intervention(id) {
    this.open("Incident", "intervention", id, () => interventionView(this, id),
      { signature: () => this.signatureOf({ intervention: id }) });
  }

  toolList(tools) {
    return h("ol", { class: "drawer-list" }, ...tools.map((t) => h("li", {}, h("button", { type: "button", onclick: () => this.tool(t.id) },
      h("strong", { text: `${t.skill}() ` }), `${t.code || t.state} · ${clock(t.start)}`))));
  }

  incidentList(incidents) {
    return h("ol", { class: "drawer-list" }, ...incidents.map((i) => h("li", {}, h("button", { type: "button", onclick: () => this.incident(i.id) },
      h("strong", { text: i.title }), h("br"), `${i.severity} · ${i.alert ? i.state : "history"} · ${clock(i.t)}`))));
  }

  decision(id) {
    this.open("Agent decision", "decision", id, () => {
      const d = this.app.state.decisions.find((x) => x.id === id);
      if (!d) return h("p", { class: "note", text: "This decision is no longer kept." });
      const tool = this.app.state.tools.find((t) => t.id === d.tool);
      return [
        facts([["Time", clock(d.t)], ["Skill", d.skill], ["Goal", d.goal], ["Outcome", tool ? `${tool.state}${tool.code ? ` · ${tool.code}` : ""}` : "Answered by the session"]]),
        h("h3", { text: "Agent's explanation (agent-generated, not verified)" }), h("p", { class: "agent-text", text: d.text }),
        h("h3", { text: "Arguments" }), h("pre", { class: "json", text: json(d.arguments) }),
        tool ? [h("h3", { text: "Tool call" }), this.toolList([tool])] : null,
      ].flat();
    });
  }

  decisions() {
    this.open("Recent decisions", "decisions", null, () => {
      const list = [...this.app.state.decisions].reverse();
      if (!list.length) return h("p", { class: "note", text: "No decisions yet." });
      return h("ol", { class: "drawer-list" }, ...list.map((d) => h("li", {}, h("button", { type: "button", onclick: () => this.decision(d.id) },
        h("strong", { text: `${clock(d.t)} ${d.skill}` }), h("br"), d.text))));
    });
  }

  alerts() {
    this.open("Alerts", "alerts", null, () => {
      const list = this.app.state.incidents.filter((i) => i.alert).reverse();
      const history = this.app.state.incidents.filter((i) => !i.alert && i.severity !== "info").reverse();
      return [list.length ? this.incidentList(list) : h("p", { class: "note", text: "No alerts recorded." }),
        history.length ? [h("h3", { text: "Failures recorded as history" }), this.incidentList(history)] : null].flat();
    });
  }

  instructions() {
    this.open("Instruction history", "instructions", null, () => {
      const list = [...this.app.state.instructions].reverse();
      if (!list.length) return h("p", { class: "note", text: "No instructions sent from this dashboard yet." });
      return h("ol", { class: "drawer-list" }, ...list.map((i) => h("li", {}, h("strong", { text: `${clock(i.t)} · ${i.state}` }),
        h("br"), i.text, i.reason ? h("span", { class: "agent-meta", text: i.reason }) : null)));
    });
  }

  /** The controller's own report and how it compares with the tree and the executor. */
  controller() {
    this.open("Controller state", "controller", null, () => {
      const s = this.app.state;
      const check = consistency(s, this.app.link.now());
      return [
        facts([["Classification", check.state], ["Severity", check.severity], ["Behavior Tree", check.bt], ["Executor", check.executor],
          ["Controller", check.controller], ["Rule", check.rule || "None: the sources agree"],
          ["Tolerances", `transition ${s.timing?.transition_s} s · stale ${s.timing?.stale_s} s · stop ${s.timing?.stop_s} s ([dashboard] settings)`]]),
        h("h3", { text: "Controller report" }), h("pre", { class: "json", text: json(s.controller ?? "Not reported to this dashboard") }),
      ];
    });
  }

  diagnostics() {
    this.open("Diagnostics", "diagnostics", null, () => {
      const s = this.app.state;
      return [
        facts([["Connection", this.app.link.connected ? "Connected" : "Disconnected"], ["Server clock offset", `${this.app.link.offset.toFixed(3)} s`],
          ["Status", s.status], ["Model", s.model], ["State version", String(s.version)],
          ["Stored camera frames", String(s.history.length)], ["3D scene port", s.scene_port ? String(s.scene_port) : "not running"]]),
        h("h3", { text: "Controller capabilities" }), h("pre", { class: "json", text: json(s.controls) }),
        h("h3", { text: "Robot" }), h("pre", { class: "json", text: json(s.robot) }),
        h("h3", { text: "Last telemetry" }), h("pre", { class: "json", text: json(this.app.telemetry) }),
      ];
    });
  }

  monitor() {
    this.open("Monitoring: last 10 minutes", "monitor", null, () => {
      const canvas = h("canvas", { tabindex: "0", role: "img", "aria-label": "Monitoring timeline for the last 10 minutes" });
      const view = h("div", { class: "timeline" },
        h("ul", { class: "track-labels", "aria-hidden": "true" }, ...LABELS.map((l) => h("li", { text: l }))),
        h("div", { class: "timeline-canvas" }, canvas));
      requestAnimationFrame(() => { this.timeline = new Timeline(this.app, canvas, { seconds: 600 }); });
      const now = this.app.link.now();
      const { blocks } = summarize(this.app.state, now - 600, now, 30);
      const readings = this.app.state.readings.filter((r) => r.kind === "observation");
      const last = readings.at(-1);
      return [h("p", { class: "note", text: "Click a moment to inspect it; the main view keeps its 60-second window." }), view,
        h("h3", { text: "What happened" }),
        blocks.length ? h("ol", { class: "drawer-list" }, ...blocks.map((b) => h("li", {}, h("button", { type: "button",
          onclick: () => (b.incident ? this.app.openIncident(b.incident.id) : this.app.inspect(b.t)) },
        h("strong", { text: clock(b.t) }), ` ${b.label}${b.linked ? " (caused by the step before)" : ""}`))))
          : h("p", { class: "note", text: "No significant events in the last 10 minutes." }),
        h("h3", { text: "Monitor" }),
        facts([["Last observation", last ? `${clock(last.t)} (${ago(now - last.t)})` : "None"],
          ["Observations in the window", String(readings.filter((r) => r.t >= now - 600).length)],
          ["Source", last?.source], ["Synthetic data", readings.some((r) => r.synthetic) ? "Yes: not a real outcome" : "No"]])];
    }, { wide: true });
  }
}
