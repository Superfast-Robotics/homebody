// Scenarios and servers, when this dashboard runs beside harnesses elsewhere (homebody connect, the monitor app):
// a scenario fills the instruction and runs on its harness and simulation server; the Servers chip opens the list
// of harnesses and simulation servers, each with whether it answers.
import { $, h, setText, toggleClass } from "./dom.js";

const CUSTOM = ""; // the list's first entry: the instruction as typed
const HEALTH = { up: "Answering", down: "Not answering", unknown: "Unknown" };

/** The scenario named NAME in STATE, or null. */
export function scenarioNamed(state, name) {
  return (state?.scenarios || []).find((s) => s.name === name) || null;
}

/** Where a scenario runs, in words: its harness and simulation server, or its scene. */
export function scenarioWhere(scenario) {
  if (!scenario) return "";
  if (scenario.harness) return `Runs on ${scenario.harness}${scenario.sim ? ` with ${scenario.sim}` : ""}.`;
  return scenario.scene ? `Runs in ${scenario.scene}.` : "";
}

/**
 * The System chip: quiet ("System · 2/5 answering") unless a server in use (the harness and simulation server the
 * dashboard runs on) is down or unknown, which is a visible degraded state naming it. Servers not in use answering
 * or not is diagnostics, in the menu.
 */
export function serversSummary(rows) {
  const up = rows.filter((r) => r.health === "up").length;
  const used = rows.filter((r) => r.active);
  const down = used.filter((r) => r.health === "down");
  const unknown = used.filter((r) => r.health !== "up" && r.health !== "down");
  if (down.length) return { text: `System degraded · ${down.map((r) => r.name).join(", ")} not answering`, tone: "danger", degraded: true };
  if (unknown.length) return { text: `System · ${unknown.map((r) => r.name).join(", ")} unknown`, tone: "warning", degraded: true };
  return { text: `System · ${up}/${rows.length} answering`, tone: "", degraded: false };
}

export class Scenarios {
  constructor(app) {
    this.app = app;
    this.select = $("scenario");
    this.chip = $("servers-chip");
    this.signature = "";
    this.select.addEventListener("change", () => this.choose());
    this.chip.addEventListener("click", () => this.open());
  }

  /** The chosen scenario's name, or null for a custom instruction. */
  chosen() {
    return this.select.hidden || this.select.value === CUSTOM ? null : this.select.value;
  }

  choose() {
    const scenario = scenarioNamed(this.app.state, this.select.value);
    if (scenario) {
      this.app.header.input.value = scenario.task;
      this.app.header.resize();
      this.app.header.show("", scenarioWhere(scenario));
    }
  }

  render(state) {
    const scenarios = state.scenarios || [];
    this.select.hidden = scenarios.length === 0 || Boolean(state.controls.busy); // task setup folds while a task runs
    const signature = scenarios.map((s) => s.name).join("|");
    if (signature !== this.signature) {
      const keep = this.select.value;
      this.signature = signature;
      this.select.replaceChildren(h("option", { value: CUSTOM, text: "Custom instruction" }),
        ...scenarios.map((s) => h("option", { value: s.name, text: s.name })));
      this.select.value = scenarios.some((s) => s.name === keep) ? keep : CUSTOM;
    }
    const rows = state.servers;
    this.chip.hidden = !Array.isArray(rows);
    if (Array.isArray(rows)) {
      const summary = serversSummary(rows);
      setText(this.chip, summary.text);
      for (const tone of ["success", "warning", "danger"]) toggleClass(this.chip, tone, summary.tone === tone);
      toggleClass(this.chip, "quiet", !summary.degraded);
      this.chip.title = `${rows.map((r) => `${r.name} (${r.kind}${r.active ? ", in use" : ""}): ${r.health}${r.error ? ` · ${r.error}` : ""}`).join("\n")}\nOpen for details and Refresh config`;
    }
  }

  open() {
    this.app.drawer.open("System status", "servers", null, () => {
      const rows = this.app.state.servers || [];
      const refresh = h("button", { class: "btn", type: "button", onclick: () => this.refresh() }, "Refresh config");
      return [
        h("p", { class: "note", text: "The harnesses and simulation servers the config names, checked every 2 s." }),
        h("ul", { class: "server-list" }, ...rows.map((r) => h("li", { class: `server ${r.health}` },
          h("span", { class: "dot", title: HEALTH[r.health] || r.health }),
          h("strong", { text: r.name }), h("span", { class: "agent-meta", text: ` ${r.kind}${r.active ? " · in use" : ""}` }),
          h("span", { class: "latency", text: r.latency_ms === null ? "—" : `${Math.round(r.latency_ms)} ms` }),
          h("div", { class: "server-detail", text: r.detail }),
          r.error ? h("div", { class: "server-error", text: r.error }) : null))),
        refresh,
      ];
    }, { signature: () => JSON.stringify(this.app.state.servers || []) });
  }

  async refresh() {
    const reply = await this.app.link.request("refresh_config");
    this.app.notify(reply.ok ? "pending" : "bad", reply.message || "The request failed.");
  }
}
