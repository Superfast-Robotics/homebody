// The dashboard: one state from the session drives every panel; inspection pins every panel to one moment.
import { Logs, Tools } from "./activity.js";
import { Camera } from "./camera.js";
import { Controls } from "./controls.js";
import { $, clock, duration, h, setText, toggleClass } from "./dom.js";
import { Drawer } from "./drawer.js";
import { Header } from "./header.js";
import { Info } from "./info.js";
import { consistency, contextSummary, systemSummary } from "./execution.js";
import { signalLabel } from "./interventions.js";
import { gapFor, liveEnd, momentAt, monitorStatus, riskSources } from "./model.js";
import { Rollouts } from "./rollouts.js";
import { Scenarios } from "./scenarios.js";
import { Link } from "./socket.js";
import { Timeline } from "./timeline.js";
import { TreeView, setupTabs } from "./tree.js";

const TICK_MS = 500; // clocks, ages and freshness between state updates
const MONITOR = {
  connected: ["tone-success", "Connected"], stale: ["tone-warning", "Stale"], offline: ["tone-danger", "Offline"], unknown: ["", "Unknown"],
};

class App {
  constructor() {
    this.state = null;
    this.telemetry = null;
    this.inspectAt = null; // the inspected moment, or null while live
    this.inspectEnd = null; // the frozen end of the timeline window while inspecting
    this.riskSource = null; // the risk series shown when several monitors report one
    this.thawed = null; // the task end after which the operator un-froze the timeline
    this.lastSeen = null;
    this.link = new Link({
      onState: (state) => this.update(state),
      onFrame: (meta, blob) => this.camera.frame(meta, blob),
      onTelemetry: (telemetry) => { this.telemetry = telemetry; },
      onConnection: () => this.connection(),
    });
    this.drawer = new Drawer(this);
    this.header = new Header(this);
    this.scenarios = new Scenarios(this);
    this.rollouts = new Rollouts(this);
    this.controls = new Controls(this);
    this.camera = new Camera(this);
    this.tree = new TreeView(this);
    this.logs = new Logs(this);
    this.tools = new Tools(this);
    this.info = new Info(this);
    this.timeline = new Timeline(this, $("timeline-canvas"));
    setupTabs(document.querySelector(".tabs"));
    $("monitor-expand").addEventListener("click", () => this.drawer.monitor());
    const legend = $("legend"), toggle = $("legend-toggle");
    toggle.addEventListener("click", () => {
      legend.hidden = !legend.hidden;
      toggle.setAttribute("aria-expanded", String(!legend.hidden));
    });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") { legend.hidden = true; toggle.setAttribute("aria-expanded", "false"); } });
    $("risk-source").addEventListener("change", (event) => { this.riskSource = event.target.value; });
    document.querySelectorAll("[data-action='live']").forEach((button) => button.addEventListener("click", () => this.returnToLive()));
    setInterval(() => this.tick(), TICK_MS);
    this.banner("loading", "Connecting to the session…");
    this.link.connect();
  }

  update(state) {
    this.state = state;
    this.lastSeen = state.now;
    for (const part of [this.header, this.scenarios, this.controls, this.camera, this.tree, this.logs, this.tools, this.info]) {
      try {
        part.render(state);
      } catch (error) {
        console.error("dashboard: a panel failed to render", error);
      }
    }
    $("logs-empty").hidden = state.logs.length > 0;
    $("menu-rollouts").hidden = !state.rollouts;
    this.renderMonitor(state);
    this.drawer.refresh();
  }

  banner(kind, text) {
    const banner = $("connection-banner");
    banner.hidden = !text;
    banner.className = `connection-banner ${kind}`;
    setText(banner, text);
  }

  connection() {
    const connected = this.link.connected;
    toggleClass($("app"), "offline", !connected && this.state !== null);
    this.banner(this.state ? "" : "loading", connected ? "" : this.state
      ? `Disconnected from the session; reconnecting. The data shown is from ${clock(this.lastSeen)}.`
      : "Connecting to the session…");
    if (this.state) this.update(this.state);
    this.camera.renderFreshness();
    this.camera.renderEmpty();
  }

  tick() {
    if (!this.state) return;
    this.header.tick();
    this.camera.renderFreshness();
    this.info.renderStatus(this.state);
    this.renderMonitor(this.state);
  }

  /** The monitor's connectivity, the timeline's mode and the task's lifecycle are three things, shown apart. */
  renderMonitor(state) {
    const now = this.link.now();
    const status = monitorStatus(state.readings, state.incidents, now, this.link.connected);
    const [tone, label] = MONITOR[status.state];
    const pill = $("monitor-status");
    pill.className = `pill ${tone}`;
    setText(pill.querySelector(".pill-text"), label);
    const age = status.observationAge;
    pill.title = status.state === "unknown" ? "No monitoring service has reported to this session"
      : [`Monitor ${label.toLowerCase()}`, age === null ? "no behavioural observation yet" : `last observation ${age < 1 ? `${age.toFixed(1)}s` : duration(age)} ago`,
        `${status.observations} observations in the last minute`].join(" · ");
    $("monitor-synthetic").hidden = !status.synthetic;
    const held = this.timelineEnd();
    this.renderMode(state, held);
    this.renderRiskSources(state);
    this.renderRiskLabel(state, status);
    this.renderSummary(state, status);
  }

  /** Live timeline, Task end · Frozen, or Inspecting history; the last two offer Return to live. */
  renderMode(state, held) {
    const inspecting = this.inspectAt !== null;
    const mode = inspecting ? ["history", `Inspecting history · ${clock(this.inspectAt)}`]
      : held.held ? ["frozen", held.reason === "ended" ? `Task end · Frozen at ${clock(state.task.end)}` : `Session lost · Frozen at ${clock(this.lastSeen)}`]
        : ["live", "Live timeline"];
    const pill = $("timeline-mode");
    pill.className = `pill timeline-mode ${mode[0]}`; // shown only when not live: the header stays minimal
    setText(pill.querySelector(".pill-text"), mode[1]);
    $("monitor-inspect").hidden = !inspecting && !(held.held && held.reason === "ended");
    toggleClass($("monitor"), "inspecting", inspecting);
    if (inspecting) {
      const m = momentAt(state, this.inspectAt, this.link.now());
      const parts = [m.goal ? `goal ${m.goal.label}` : "no goal", m.skill ? `skill ${m.skill.label}` : "no skill",
        m.reading ? `observation ${m.reading.level}${m.reading.synthetic ? " (synthetic)" : ""}` : "no reading",
        m.risk ? `risk ${m.risk.risk.toFixed(2)}` : ""];
      setText($("monitor-inspect-text"), parts.filter(Boolean).join(" · "));
    } else {
      setText($("monitor-inspect-text"), ""); // the mode pill already says the timeline is frozen
    }
  }

  /** One primary conclusion (execution.contextSummary: an open intervention, an anomaly, a mismatch, elevated risk,
   * a required service, normal), with a stale monitor or unusable camera frames said beside it. While inspecting
   * history it is not current, so the strip says what window it shows; the milestones are in View details. */
  renderSummary(state, status) {
    const end = this.inspectEnd ?? this.timelineEnd().end;
    const live = this.inspectAt === null && !this.timelineEnd().held;
    const now = this.link.now();
    const condition = live ? contextSummary(state, status, now, end - 60, {
      gap: gapFor(state.readings.filter((r) => r.kind === "observation")), check: consistency(state, now),
      system: state.servers ? systemSummary(state.servers) : null })
      : { level: "history", text: `Showing ${clock(end - 60)}–${clock(end)}: View details lists what happened` };
    if (condition.text === this.summaryShown) return;
    this.summaryShown = condition.text;
    $("summary-strip").replaceChildren(h("li", { class: `condition ${condition.level}`, text: condition.text,
      title: "The monitoring panel's primary conclusion, from recorded events" }));
  }

  openMilestone(block) {
    this.inspect(block.t);
    if (block.incident) this.drawer.incident(block.incident.id);
    else if (block.task) this.drawer.outcome();
  }

  /** The risk track's label: the signal the newest scored observation measured, and its current value. */
  renderRiskLabel(state, status) {
    const scored = state.readings.filter((r) => r.kind === "observation" && r.risk !== null);
    const latest = scored.at(-1) || null;
    const label = $("risk-label");
    const fresh = latest && status.state === "connected" && status.observationAge !== null && status.observationAge <= Math.max(3, status.staleAfter);
    setText(label.querySelector(".risk-name"), latest ? signalLabel(latest.signal) : "Risk");
    const cls = !latest ? "" : latest.level === "anomaly" || (latest.threshold !== null && latest.risk >= latest.threshold) ? "High"
      : latest.level === "elevated" ? "Elevated" : "Low";
    setText(label.querySelector(".risk-now"), !latest ? "" : fresh ? `${latest.risk.toFixed(2)} · ${cls}` : `last ${latest.risk.toFixed(2)} (historical)`);
    label.className = `tall risk-label ${fresh ? cls.toLowerCase() : "historical"}`;
    label.title = latest ? `${signalLabel(latest.signal)} from ${latest.source} at ${clock(latest.t)}; threshold ${latest.threshold ?? "none"}.\nThe strongest of the monitor's signals at each reading.` : "";
  }

  renderRiskSources(state) {
    const sources = riskSources(state.readings);
    $("risk-source-wrap").hidden = sources.length < 2;
    const select = $("risk-source");
    if (select.options.length !== sources.length || [...select.options].some((o, i) => o.value !== sources[i])) {
      select.replaceChildren(...sources.map((s) => h("option", { value: s, text: `Risk: ${s}` })));
    }
    select.value = sources.includes(this.riskSource) ? this.riskSource : sources.at(-1) || "";
  }

  /** Where the live timeline ends: the clock while a task runs, held once the run stops. */
  timelineEnd() {
    return liveEnd(this.state, this.link.now(), this.link.connected, this.lastSeen, this.thawed);
  }

  /** Leave inspection, and un-freeze a task-end timeline so it follows the clock again. */
  returnToLive() {
    if (this.state?.task?.end) this.thawed = this.state.task.end;
    this.inspect(null);
  }

  /** Inspect moment T on every panel (null returns to live). The robot is never paused or moved by this. */
  inspect(t) {
    if (t === null) {
      this.inspectAt = this.inspectEnd = null;
    } else {
      const now = this.timelineEnd().end;
      this.inspectAt = Math.min(t, now);
      this.inspectEnd ??= now; // freeze the window where it was
      if (this.inspectAt > this.inspectEnd || this.inspectAt < this.inspectEnd - 60) {
        this.inspectEnd = Math.min(now, this.inspectAt + 30); // bring an outside moment to the middle
      }
    }
    this.camera.inspect(this.inspectAt, this.state?.history || []);
    this.logs.highlight(this.inspectAt);
    this.tools.highlight(this.inspectAt);
    this.tree.highlightAt(this.inspectAt);
    if (this.state) this.renderMonitor(this.state);
  }

  open(kind, item) {
    if (kind === "incident") this.drawer.incident(item.id);
    else if (kind === "decision") this.drawer.decision(item.id);
    else if (kind === "goal") this.drawer.node(`goal:${item.detail.goal}`);
    else if (kind === "skill" && item.detail.tool) this.drawer.tool(item.detail.tool);
    else if (kind === "observation") this.drawer.observation(item);
    else if (kind === "risk") {
      const toolId = item.readings.find((r) => r.tool)?.tool || null;
      const request = (this.state.interventions || []).findLast((r) => toolId && r.tool === toolId);
      this.focusAttempt(toolId, item.start, () => (request ? this.drawer.intervention(request.id) : this.drawer.risk(item)));
    }
    else if (kind === "event") {
      this.focusAttempt(item.tool, item.t, () => (item.intervention ? this.drawer.intervention(item.intervention)
        : item.incident ? this.drawer.incident(item.incident) : null));
    } else if (kind === "cluster") this.drawer.events(item);
  }

  /** Select moment T, highlight attempt TOOL_ID in the tree (expanding its group), then OPEN its details. The
   * robot is never paused or moved by this; Return to live leaves the selection. */
  focusAttempt(toolId, t, open = null) {
    if (t !== null && t !== undefined) this.inspect(t);
    if (toolId) this.tree.highlightTool(toolId);
    open?.();
  }

  openIncident(id) {
    const incident = this.state.incidents.find((i) => i.id === id);
    this.drawer.incident(id);
    if (incident) this.inspect(incident.t);
  }

  /** A short confirmation or rejection under the instruction input. */
  notify(kind, text) {
    this.header.show(kind, text);
  }
}

new App();
