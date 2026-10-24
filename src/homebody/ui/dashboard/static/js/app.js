// The dashboard: one state from the session drives every panel; inspection pins every panel to one moment.
import { Logs, Tools } from "./activity.js";
import { Camera } from "./camera.js";
import { Controls } from "./controls.js";
import { $, clock, duration, h, setText, toggleClass } from "./dom.js";
import { Drawer } from "./drawer.js";
import { Header } from "./header.js";
import { Info } from "./info.js";
import { keepHeaderClear, setupSplitter } from "./layout.js";
import { inputWarning } from "./assessment.js";
import { consistency, contextLabel, contextSummary, systemSummary } from "./execution.js";
import { signalLabel } from "./interventions.js";
import { gapFor, liveEnd, momentAt, monitorStatus, riskSources } from "./model.js";
import { Rollouts } from "./rollouts.js";
import { Scenarios } from "./scenarios.js";
import { resolveSelection } from "./selection.js";
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
    this.selection = null; // the inspected incident, intervention or attempt (selection.js), or null
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
    setupSplitter();
    keepHeaderClear();
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
    if (this.selection) { // the same records by identity, with what they say now; the inspected moment stays
      this.selection = { ref: this.selection.ref, ...resolveSelection(state, this.selection.ref, this.timelineEnd().end) };
    }
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
    const mode = inspecting ? ["history", this.selection ? "Inspecting history" : `Inspecting history · ${clock(this.inspectAt)}`]
      : held.held ? ["frozen", held.reason === "ended" ? `Task end · Frozen at ${clock(state.task.end)}` : `Session lost · Frozen at ${clock(this.lastSeen)}`]
        : ["live", "Live timeline"];
    const pill = $("timeline-mode");
    pill.className = `pill timeline-mode ${mode[0]}`; // shown only when not live: the header stays minimal
    setText(pill.querySelector(".pill-text"), mode[1]);
    $("monitor-inspect").hidden = !inspecting && !(held.held && held.reason === "ended");
    toggleClass($("monitor"), "inspecting", inspecting);
    const end = this.inspectEnd ?? held.end;
    setText($("window-text"), inspecting ? `${clock(end - 60)}–${clock(end)} · Recorded` : "Last 1 min");
    if (inspecting && this.selection) {
      const sel = this.selection;
      setText($("monitor-inspect-text"), [`${sel.label} · ${clock(this.inspectAt)}`,
        sel.observation ? `${sel.detecting ? "observation" : "newest observation"} ${sel.observation.id || "recorded"}` : "no monitor observation linked"].join(" · "));
    } else if (inspecting) {
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
    if (block.incident) return this.openIncident(block.incident.id);
    this.inspect(block.t);
    if (block.task) this.drawer.outcome();
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
    const warning = latest ? inputWarning(latest) : null;
    setText(label.querySelector(".risk-now"), !latest ? "" : `${fresh ? `${latest.risk.toFixed(2)} · ${cls}` : `last ${latest.risk.toFixed(2)} (historical)`}${warning ? " · ⚠ input" : ""}`);
    label.className = `tall risk-label ${fresh ? cls.toLowerCase() : "historical"}`;
    label.title = latest ? [`${signalLabel(latest.signal)} from ${latest.source} at ${clock(latest.t)}; threshold ${latest.threshold ?? "none"}.`,
      `Applies to: ${contextLabel(state, latest)}.`, latest.semantics || "", warning ? `${warning.text}: ${warning.reason}` : "",
      "The strongest of the monitor's signals at each reading."].filter(Boolean).join("\n") : "";
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

  /** Leave inspection on every panel: the live camera, the tree following execution, the rolling 60-second window,
   * no highlights; a task-end timeline follows the clock again. The inspector stays open (it describes a recorded
   * incident, which stays true); its Show on dashboard selects it again. */
  returnToLive() {
    if (this.state?.task?.end) this.thawed = this.state.task.end;
    this.selection = null;
    this.inspect(null);
    this.tree.followLive();
  }

  /**
   * Select REF ({intervention, incident, tool, t}) on every panel: its moment on the timeline (in a recorded window
   * around it when it is older than the live one), its attempt in the tree, the recorded camera frame, and (OPEN) its
   * details in the drawer. Identities come from the records (selection.js), never from labels. Read-only: the robot is
   * never paused, resumed or moved by this.
   */
  select(ref, { open = true } = {}) {
    if (!this.state) return;
    const sel = resolveSelection(this.state, ref, this.timelineEnd().end);
    this.inspect(sel.t, { window: sel.window });
    this.selection = { ref, ...sel };
    this.tree.showSelection(this.selection, sel.t);
    this.renderMonitor(this.state);
    if (!open) return;
    if (ref.intervention) this.drawer.intervention(ref.intervention);
    else if (ref.incident) this.drawer.incident(ref.incident);
    else if (ref.tool) this.drawer.tool(ref.tool);
  }

  /** Move the inspected moment to T, keeping the selected incident (arrow keys); with REF (an inspector's own record),
   * that record is selected at T, so the highlights always belong to what the inspector shows. */
  selectMoment(t, ref = null) {
    if (ref) return this.select({ ...ref, t }, { open: false });
    this.inspect(t, { keep: true });
  }

  /** Inspect moment T on every panel (null returns to live). A new moment clears the selection unless KEEP; WINDOW
   * (selection.windowFor) places a recorded window around a moment outside the live one. The robot is never paused
   * or moved by this. */
  inspect(t, { keep = false, window = null } = {}) {
    if (!keep) this.selection = null;
    if (t === null) {
      this.inspectAt = this.inspectEnd = null;
    } else {
      const now = this.timelineEnd().end;
      this.inspectAt = Math.min(t, now);
      if (window) {
        this.inspectEnd = window.end;
      } else {
        this.inspectEnd ??= now; // freeze the window where it was
        if (this.inspectAt > this.inspectEnd || this.inspectAt < this.inspectEnd - 60) {
          this.inspectEnd = Math.min(now, this.inspectAt + 30); // bring an outside moment to the middle
        }
      }
    }
    this.camera.inspect(this.inspectAt, this.state);
    this.logs.highlight(this.inspectAt);
    this.tools.highlight(this.inspectAt);
    if (this.selection) this.tree.showSelection(this.selection, this.inspectAt);
    else this.tree.highlightAt(this.inspectAt);
    if (this.state) this.renderMonitor(this.state);
  }

  /** Open what the timeline hit names; returns whether it selected (and so inspected) something itself. */
  open(kind, item) {
    if (kind === "incident") return this.openIncident(item.id), true;
    if (kind === "decision") this.drawer.decision(item.id);
    else if (kind === "goal") this.drawer.node(`goal:${item.detail.goal}`);
    else if (kind === "skill" && item.detail.tool) return this.select({ tool: item.detail.tool, t: item.start }), true;
    else if (kind === "observation") this.drawer.observation(item);
    else if (kind === "risk") {
      const toolId = item.readings.find((r) => r.tool)?.tool || null;
      const request = (this.state.interventions || []).findLast((r) => toolId && r.tool === toolId);
      if (request) return this.select({ intervention: request.id, t: item.start }), true;
      if (toolId) this.select({ tool: toolId, t: item.start }, { open: false });
      this.drawer.risk(item);
      return Boolean(toolId);
    } else if (kind === "event") {
      const ref = item.intervention ? { intervention: item.intervention, t: item.t } : item.incident ? { incident: item.incident, t: item.t }
        : { tool: item.tool, t: item.t };
      return this.select(ref), true;
    } else if (kind === "cluster") this.drawer.events(item);
    return false;
  }

  /** An incident, through the intervention that acted on it when there is one (the fuller lifecycle). */
  openIncident(id) {
    const row = (this.state.interventions || []).findLast((r) => r.incident === id);
    this.select(row ? { intervention: row.id } : { incident: id });
  }

  /** A short confirmation or rejection under the instruction input. */
  notify(kind, text) {
    this.header.show(kind, text);
  }
}

new App();
