// Rollouts: this machine's recorded and imported rollouts. Export one as a bundle, import a bundle, run the offline
// monitor on one (its cost shown first) and, on a dashboard without a live session (homebody rollout view), open one:
// the dashboard then shows that task as it ran, and the timeline inspects its moments.
import { h } from "./dom.js";

/** The drawer's choice: a rollout's name and task, kept while the list refreshes. */
export function keepChoice(list, name, task) {
  const row = list.find((r) => r.name === name) || list[0];
  if (!row) return { name: null, task: null };
  return { name: row.name, task: row.tasks.includes(task) ? task : row.tasks.at(-1) ?? null };
}

export class Rollouts {
  constructor(app) {
    this.app = app;
    this.name = null;
    this.task = null;
    this.about = "";
    this.estimate = null; // {calls, message}: an evaluation's cost, shown before it runs
  }

  open() {
    this.app.drawer.open("Rollouts", "rollouts", null, () => this.build(),
      { signature: () => JSON.stringify(this.app.state.rollouts || null) + this.about + JSON.stringify(this.estimate) });
    this.describe();
  }

  async request(type, extra = {}) {
    const reply = await this.app.link.request(type, { name: this.name, task: this.task, ...extra });
    this.app.notify(reply.ok ? "pending" : "bad", reply.message || "The request failed.");
    return reply;
  }

  async describe() {
    if (!this.name) return;
    const reply = await this.app.link.request("rollout_describe", { name: this.name, task: this.task });
    this.about = reply.ok ? reply.message : "";
    this.app.drawer.refresh();
  }

  choose(name, task) {
    Object.assign(this, keepChoice(this.app.state.rollouts?.list || [], name, task));
    this.estimate = null;
    this.describe();
  }

  build() {
    const r = this.app.state.rollouts;
    if (!r) return h("p", { class: "note", text: "This dashboard has no rollouts." });
    Object.assign(this, keepChoice(r.list, this.name, this.task));
    if (!this.name) {
      return [h("p", { class: "note", text: "Recorded and imported rollouts appear here." }), this.importer(r)];
    }
    const row = r.list.find((x) => x.name === this.name);
    const rollout = h("select", { class: "scenario-select", "aria-label": "Rollout", onchange: (e) => this.choose(e.target.value, null) },
      ...r.list.map((x) => h("option", { value: x.name, text: x.name, selected: x.name === this.name })));
    const task = h("select", { class: "scenario-select", "aria-label": "Task", onchange: (e) => this.choose(this.name, e.target.value) },
      ...row.tasks.map((t) => h("option", { value: t, text: t, selected: t === this.task })));
    const button = (label, onclick, disabled = false, title = "") =>
      h("button", { class: "btn", type: "button", onclick, disabled: disabled || r.busy, title }, label);
    const estimate = this.estimate;
    return [
      rollout, task,
      this.about ? h("p", { class: "agent-meta", text: this.about }) : null,
      r.opened ? h("p", { class: "note", text: `Showing ${r.opened.name} ${r.opened.task} (${r.opened.source} monitor).` }) : null,
      h("div", { class: "drawer-actions" },
        r.replays ? button("Open", () => this.request("rollout_open")) : null,
        r.replays ? button("Open with the offline monitor", () => this.request("rollout_open", { source: "offline" })) : null,
        button("Export bundle", () => this.request("rollout_export"), false, "Writes a .tar.gz under runs/exports"),
        button("Evaluate with the monitor", () => this.evaluate(false), !r.evaluates,
          r.evaluates ? "Runs the monitor over the recorded frames; shows the cost first" : "No monitor models on this machine")),
      estimate ? h("div", { class: "drawer-actions" }, h("span", { class: "agent-meta", text: estimate.message }),
        estimate.calls ? button("Run evaluation", () => this.evaluate(true)) : null) : null,
      r.replays ? null : h("p", { class: "note", text: "Replays open in their own dashboard: homebody rollout view." }),
      r.message ? h("p", { class: "instruction-state pending", text: r.message }) : null,
      this.importer(r),
    ];
  }

  importer(r) {
    const path = h("input", { type: "text", class: "path-input", placeholder: "Path to a bundle (.tar.gz) on this machine", "aria-label": "Bundle path" });
    return h("form", { class: "drawer-actions", onsubmit: (e) => { e.preventDefault(); this.request("rollout_import", { path: path.value }); } },
      path, h("button", { class: "btn", type: "submit", disabled: r.busy }, "Import"));
  }

  async evaluate(confirm) {
    const reply = await this.request("rollout_evaluate", { confirm });
    this.estimate = confirm || !reply.ok ? null : { calls: reply.calls || 0, message: reply.message };
    this.app.drawer.refresh();
  }
}
