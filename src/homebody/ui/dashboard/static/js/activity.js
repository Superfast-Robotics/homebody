// The Logs and Tools tabs: rows keyed by id, appended or updated in place so scroll and open rows survive.
import { $, clock, duration, h, icon, json, setIcon, setText, toggleClass } from "./dom.js";
import { Select } from "./select.js";

const FILTERS = [
  ["log-severity", "Severity", [["", "All severities"], ["info", "Info"], ["warning", "Warning"], ["error", "Error"]]],
  ["log-source", "Source", [["", "All sources"], ["session", "Session"], ["agent", "Agent"], ["skill", "Skill"], ["operator", "Operator"], ["monitor", "Monitor"]]],
  ["log-node", "Behavior tree goal", [["", "All goals"]]],
  ["log-time", "Time", [["", "All time"], ["60", "Last 1 min"], ["300", "Last 5 min"]]],
];

const TOOL_STATE = { running: "running", succeeded: "success", failed: "failure" };

export class Logs {
  constructor(app) {
    this.app = app;
    this.list = $("logs");
    this.rows = new Map();
    this.filters = FILTERS.map(([id, label, options]) => {
      const select = new Select($(id), { label, onChange: () => this.filter() });
      select.setOptions(options.map(([value, text]) => ({ id: value, label: text })));
      select.value = "";
      return select;
    });
    this.goals = "";
  }

  render(state) {
    const nearBottom = this.list.scrollHeight - this.list.scrollTop - this.list.clientHeight < 24;
    const ids = new Set(state.logs.map((entry) => entry.id));
    for (const [id, row] of this.rows) if (!ids.has(id)) { row.remove(); this.rows.delete(id); }
    for (const entry of state.logs) {
      if (this.rows.has(entry.id)) continue;
      const row = h("li", { dataset: { severity: entry.severity, source: entry.source, node: entry.node || "", t: entry.t } },
        h("span", { class: "log-time", text: clock(entry.t) }),
        h("span", { class: `sev ${entry.severity}`, title: entry.severity }),
        h("span", { class: "log-msg" }, h("span", { class: "log-source", text: entry.source }), entry.message));
      this.rows.set(entry.id, row);
      this.list.append(row);
      this.apply(row);
    }
    this.renderGoals(state);
    if (nearBottom) this.list.scrollTop = this.list.scrollHeight;
  }

  renderGoals(state) {
    const goals = Object.values(state.tree?.nodes || {}).filter((n) => n.type === "goal");
    const signature = goals.map((g) => g.detail.goal).join("|");
    if (signature === this.goals) return;
    this.goals = signature;
    const select = this.filters[2];
    const value = select.value;
    select.setOptions([{ id: "", label: "All goals" }, ...goals.map((g) => ({ id: g.detail.goal, label: `${g.detail.goal}: ${g.label}` }))]);
    select.value = goals.some((g) => g.detail.goal === value) ? value : "";
  }

  apply(row) {
    const [severity, source, node, time] = this.filters.map((select) => select.value || "");
    const recent = !time || this.app.link.now() - Number(row.dataset.t) <= Number(time);
    row.hidden = (severity && row.dataset.severity !== severity) || (source && row.dataset.source !== source) ||
      (node && row.dataset.node !== node) || !recent;
  }

  filter() {
    this.rows.forEach((row) => this.apply(row));
  }

  /** Mark the entries nearest T and scroll the first into view; null clears the mark. */
  highlight(t) {
    let first = null;
    for (const row of this.rows.values()) {
      const near = t !== null && Math.abs(Number(row.dataset.t) - t) <= 1.5;
      toggleClass(row, "highlight", near);
      if (near && !first) first = row;
    }
    first?.scrollIntoView({ block: "center" });
  }
}

export class Tools {
  constructor(app) {
    this.app = app;
    this.list = $("tools");
    this.rows = new Map();
  }

  render(state) {
    $("tools-empty").hidden = state.tools.length > 0;
    const ids = new Set(state.tools.map((tool) => tool.id));
    for (const [id, view] of this.rows) if (!ids.has(id)) { view.row.remove(); this.rows.delete(id); }
    for (const tool of state.tools) {
      const view = this.rows.get(tool.id) || this.create(tool);
      this.update(view, tool, state);
    }
  }

  create(tool) {
    const state = h("span", { class: "state" }, icon("check"));
    const name = h("span", { class: "tool-name" });
    const sub = h("span", { class: "tool-sub" });
    const time = h("span", { class: "duration" });
    const body = h("div");
    const details = h("details", {}, h("summary", {}, state, h("span", {}, name, sub), time), body);
    details.addEventListener("toggle", () => details.open && this.fill(view, this.app.state));
    const row = h("li", { dataset: { t: tool.start } }, details);
    const view = { row, details, state, name, sub, time, body, id: tool.id };
    this.rows.set(tool.id, view);
    this.list.prepend(row);
    return view;
  }

  update(view, tool, state) {
    const status = TOOL_STATE[tool.state] || "skipped";
    view.state.className = `state ${status}`;
    const svg = view.state.querySelector("svg");
    svg.style.display = status === "running" ? "none" : "";
    setIcon(svg, status === "failure" ? "x" : status === "skipped" ? "skip" : "check");
    setText(view.name, `${tool.skill}()`);
    const goal = tool.goal ? `goal ${tool.goal} · ` : "";
    const outcome = tool.state === "running" ? (tool.stages.at(-1)?.[1] || "running") : tool.code || tool.message;
    setText(view.sub, `${goal}${clock(tool.start)} · ${outcome}`);
    setText(view.time, duration((tool.end ?? this.app.link.now()) - tool.start));
    view.tool = tool;
    if (view.details.open) this.fill(view, state);
  }

  fill(view, state) {
    const tool = view.tool;
    if (!tool) return;
    const key = JSON.stringify([tool.state, tool.stages.length, tool.code]);
    if (view.filled === key) return;
    view.filled = key;
    const decision = state?.decisions.find((d) => d.tool === tool.id);
    view.body.replaceChildren(
      h("pre", { class: "json", text: json({
        tool: tool.skill, goal: tool.goal, state: tool.state, started: clock(tool.start),
        ended: tool.end ? clock(tool.end) : null, arguments: tool.arguments,
        stages: tool.stages.map(([t, stage]) => `${clock(t)} ${stage}`),
        result: tool.code ? { code: tool.code, message: tool.message } : tool.message || null,
        agent_reason: decision?.text,
      }) }));
  }

  highlight(t) {
    let first = null;
    for (const view of this.rows.values()) {
      const tool = view.tool;
      const near = t !== null && tool && tool.start <= t && (tool.end ?? Infinity) >= t;
      toggleClass(view.row, "highlight", near);
      if (near && !first) first = view.row;
    }
    first?.scrollIntoView({ block: "nearest" });
  }
}
