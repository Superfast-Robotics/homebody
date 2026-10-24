// Session Configuration: which harness server and which simulation server this monitoring session uses. The two
// choices are a draft until Apply; the active pair is only ever what the backend confirms (the harness the monitor app
// drives, and the simulation server that harness reports being attached to). Health, selection and activation are
// shown apart. Servers are configured outside the dashboard: the only action on the configuration is Refresh.
import { $, ago, clock, h, setText, toggleClass } from "./dom.js";
import { Select } from "./select.js";

const HEALTH = { healthy: ["success", "Healthy"], degraded: ["warning", "Degraded"], unavailable: ["danger", "Unavailable"],
  unknown: ["neutral", "Unknown"] };
const BUSY = ["Thinking", "Executing", "Paused", "Stopping"];
const FAULTED = ["Error", "Reset required"]; // the harness's session needs a new attach: the same pair can be applied again
const CONTROLLER = { running: "Controller running", stopped: "Controller stopped", none: "No controller", unknown: "Controller unknown" };
const VIDEO = { on: "Video on", ready: "Video ready", none: "No video", unknown: "Video unknown" };

/** A server row's health as the dropdown shows it: Checking before its first answer. */
export function healthStatus(row) {
  if (row.checking) return { tone: "neutral", text: "Checking" };
  const [tone, text] = HEALTH[row.health] || HEALTH.unknown;
  return { tone, text };
}

function freshness(row, now) {
  const answered = row.last_ok ? (now - row.last_ok < 5 ? "Last response just now" : `Last response ${ago(now - row.last_ok)}`) : "Never answered";
  return [answered, row.latency_ms !== null && row.latency_ms !== undefined ? `${Math.round(row.latency_ms)} ms` : ""].filter(Boolean).join(" · ");
}

/** The dropdown options for KIND ("harness" or "simulation") from the server rows; ACTIVE marks the confirmed one. */
export function serverOptions(rows, kind, active, now) {
  return (rows || []).filter((r) => r.kind === kind).map((r) => {
    const description = kind === "simulation"
      ? (r.connected ? `${CONTROLLER[r.controller] || CONTROLLER.unknown} · ${VIDEO[r.video] || VIDEO.unknown}` : r.error || "No answer")
      : (r.connected ? r.detail : r.error || "No answer");
    return { id: r.name, label: r.name, icon: kind === "harness" ? "⌁" : "▣", status: healthStatus(r),
      description: `${r.name === active ? "Active · " : ""}${description || ""}`, meta: freshness(r, now),
      hint: r.error || r.detail || "" };
  });
}

/**
 * Whether the DRAFT pair ({harness, sim}) can be applied, from the rows and the confirmed SESSION: {enabled, reason}.
 * Nothing changed, a choice missing or no longer configured, a server unavailable or without a controller, a task
 * running on the active harness, or an Apply already in progress each disable it with the reason.
 */
export function applyState(rows, session, draft) {
  const find = (kind, name) => (rows || []).find((r) => r.kind === kind && r.name === name);
  if (session?.applying) return { enabled: false, reason: "Applying…", applying: true };
  if (!draft.harness || !draft.sim) return { enabled: false, reason: "Choose a harness and a simulation server" };
  const same = draft.harness === session?.harness && draft.sim === session?.sim;
  if (same && !FAULTED.includes(session?.lifecycle)) return { enabled: false, reason: "No changes" };
  const harness = find("harness", draft.harness), sim = find("simulation", draft.sim);
  if (!harness) return { enabled: false, reason: `${draft.harness} is no longer configured` };
  if (!sim) return { enabled: false, reason: `${draft.sim} is no longer configured` };
  if (harness.connected === false) return { enabled: false, reason: `${harness.name} is unavailable: ${harness.error || "no answer"}` };
  if (sim.connected === false) return { enabled: false, reason: `${sim.name} is unavailable: ${sim.error || "no answer"}` };
  if (sim.connected && sim.controller !== "running") return { enabled: false, reason: `${sim.name}: ${CONTROLLER[sim.controller] || "controller unknown"}` };
  if (session?.harness && BUSY.includes(session.lifecycle)) {
    return { enabled: false, reason: `A task is running on ${session.harness}: stop it or let it finish before switching` };
  }
  return { enabled: true, reason: "", reconnect: same };
}

/** The header's session label: "harness → sim", or what is missing. */
export function sessionLabel(session) {
  if (!session?.harness) return "Session · not set";
  return `${session.harness} → ${session.sim || "no simulation server"}`;
}

export class SessionPanel {
  constructor(app) {
    this.app = app;
    this.chip = $("servers-chip");
    this.panel = $("session-panel");
    this.draft = { harness: null, sim: null };
    this.edited = false; // the operator changed a choice since the panel opened or the last Apply
    this.harness = new Select($("session-harness"), { label: "Harness server", placeholder: "Choose a harness",
      onChange: (id) => { this.draft = { ...this.draft, harness: id }; this.edited = true; this.render(app.state); } });
    this.sim = new Select($("session-sim"), { label: "Simulation server", placeholder: "Choose a simulation server",
      onChange: (id) => { this.draft = { ...this.draft, sim: id }; this.edited = true; this.render(app.state); } });
    $("session-harness").dataset.menuWidth = $("session-sim").dataset.menuWidth = "420";
    this.chip.addEventListener("click", () => this.toggle());
    $("session-close").addEventListener("click", () => this.toggle(false));
    $("session-apply").addEventListener("click", () => this.apply());
    $("session-refresh").addEventListener("click", () => this.refresh());
    this.panel.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !event.defaultPrevented) { event.preventDefault(); this.toggle(false); }
    });
    document.addEventListener("pointerdown", (event) => {
      if (!this.panel.hidden && !this.panel.contains(event.target) && !this.chip.contains(event.target)
          && !event.target.closest(".dropdown-menu")) this.toggle(false);
    });
  }

  toggle(open = this.panel.hidden) {
    this.panel.hidden = !open;
    this.chip.setAttribute("aria-expanded", String(open));
    if (open) {
      const session = this.app.state?.session;
      this.draft = { harness: session?.harness ?? null, sim: session?.sim ?? null };
      this.edited = false;
      this.render(this.app.state);
      $("session-panel").querySelector(".dropdown-trigger")?.focus();
    } else this.chip.focus();
  }

  render(state) {
    const rows = state?.servers;
    this.chip.hidden = !Array.isArray(rows);
    if (!Array.isArray(rows)) return;
    const session = state.session || {};
    if (this.awaiting && session.applying) this.awaiting = { ...this.awaiting, started: true };
    if (this.awaiting?.started && !session.applying) { // the Apply ended: its "Applying…" note gives way to the outcome
      const { harness, sim } = this.awaiting;
      this.awaiting = null;
      this.app.notify(session.error ? "bad" : "", session.error || `Session: ${harness} → ${sim}`);
    }
    const now = this.app.link.now();
    const active = (kind, name) => rows.find((r) => r.kind === kind && r.name === name);
    const pair = [active("harness", session.harness), active("simulation", session.sim)].filter(Boolean);
    const worst = pair.find((r) => r.health === "unavailable") || pair.find((r) => r.health !== "healthy");
    setText(this.chip, sessionLabel(session));
    for (const tone of ["success", "warning", "danger"]) toggleClass(this.chip, tone, false);
    toggleClass(this.chip, worst ? (worst.health === "unavailable" ? "danger" : "warning") : session.harness ? "success" : "warning", true);
    this.chip.title = `${sessionLabel(session)}${worst ? ` · ${worst.name} ${worst.health}` : ""}\nOpen Session Configuration`;
    if (this.panel.hidden) return;
    if (this.edited && !session.applying && session.harness === this.draft.harness && session.sim === this.draft.sim) this.edited = false;
    if (!this.edited) this.draft = { harness: session.harness ?? this.draft.harness, sim: session.sim ?? this.draft.sim };
    const harnesses = serverOptions(rows, "harness", session.harness, now), sims = serverOptions(rows, "simulation", session.sim, now);
    this.harness.setOptions(harnesses);
    this.sim.setOptions(sims);
    this.harness.value = this.draft.harness;
    this.sim.value = this.draft.sim;
    setText($("session-harness-empty"), harnesses.length ? "" : "No harness servers configured (configuration is managed outside the dashboard)");
    setText($("session-sim-empty"), sims.length ? "" : "No simulation servers configured");
    const gone = (kind, name, list) => (name && !list.some((o) => o.id === name) ? `${name} is no longer configured` : "");
    setText($("session-draft-note"), [gone("harness", this.draft.harness, harnesses), gone("simulation", this.draft.sim, sims)].filter(Boolean).join(" · "));
    const current = $("session-current");
    current.className = `session-current ${session.harness ? (session.sim_confirmed ? "active" : "pending") : "none"}`;
    setText(current.querySelector(".pair"), sessionLabel(session));
    setText(current.querySelector(".session-state"), !session.harness ? "No active session" : session.sim_confirmed
      ? `Active${session.lifecycle ? ` · ${session.lifecycle}` : ""}` : "Harness connected · simulation link not confirmed");
    const can = applyState(rows, session, this.draft);
    const apply = $("session-apply");
    apply.disabled = !can.enabled;
    setText(apply, can.applying ? "Applying…" : can.reconnect ? "Reconnect" : "Apply selection");
    apply.title = can.reason;
    setText($("session-apply-reason"), can.enabled ? "" : can.reason === "No changes" ? "" : can.reason);
    setText($("session-error"), session.error || "");
    setText($("session-warnings"), (session.warnings || []).join(" · "));
    setText($("session-refreshed"), session.refreshed ? `Config refreshed ${clock(session.refreshed)} · ${session.refresh_message}` : "");
  }

  /** The draft stays the operator's until the backend confirms it (render), so a failed Apply keeps the choice. */
  async apply() {
    let reply;
    try {
      reply = await this.app.link.request("apply_session", { harness: this.draft.harness, sim: this.draft.sim });
    } catch (error) {
      reply = { ok: false, message: `Unable to activate the selected session: ${error?.message || error}` };
    }
    this.awaiting = reply.ok ? { ...this.draft, started: false } : null;
    this.app.notify(reply.ok ? "pending" : "bad", reply.message || "Unable to activate the selected session");
    if (!reply.ok) setText($("session-error"), reply.message || "Unable to activate the selected session");
  }

  async refresh() {
    const button = $("session-refresh");
    button.disabled = true;
    setText(button, "Refreshing…");
    let reply;
    try {
      reply = await this.app.link.request("refresh_config");
    } catch (error) {
      reply = { ok: false, message: `Config not reloaded: ${error?.message || error}` };
    } finally {
      button.disabled = false;
      setText(button, "Refresh config");
    }
    this.app.notify(reply.ok ? "pending" : "bad", reply.message || "The request failed.");
  }
}
