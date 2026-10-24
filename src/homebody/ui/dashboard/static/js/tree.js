// The behavior tree, updated in place by node id, through its lifecycle: no task, planning, running, recovery,
// replanning. Collapsing is the operator's; following the running node pauses while the operator looks around.
import { $, clock, duration, h, icon, setIcon, setText, storage, toggleClass } from "./dom.js";
import { retrySummary } from "./execution.js";
import { goalLabel, operatorText } from "./story.js";

const STATE_ICON = { success: "check", failure: "x", skipped: "skip", paused: "pause" };
const KIND_ICON = { root: "tree", sequence: "sequence", fallback: "fallback", attempts: "list" };
const STATUS_TEXT = { running: "Running", success: "Completed", failure: "Failed", pending: "Pending", skipped: "Ended", paused: "Paused" };
const MARK = { success: "✓", failure: "✗", running: "●", pending: "○", skipped: "–" };

export function setupTabs(tablist) {
  const tabs = [...tablist.querySelectorAll("[role='tab']")];
  const select = (tab) => {
    for (const other of tabs) {
      const on = other === tab;
      other.setAttribute("aria-selected", String(on));
      other.tabIndex = on ? 0 : -1;
      $(other.getAttribute("aria-controls")).hidden = !on;
    }
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      const next = tabs[(index + step + tabs.length) % tabs.length];
      select(next);
      next.focus();
    });
  });
}

/** The deepest running node, and the path to it. */
function activePath(tree) {
  const path = [];
  let id = tree.root;
  while (id) {
    path.push(id);
    id = tree.nodes[id].children.find((child) => tree.nodes[child]?.status === "running");
  }
  return path;
}

export class TreeView {
  constructor(app) {
    this.app = app;
    this.container = $("tree");
    this.pane = $("pane-plan");
    this.rows = new Map(); // node id -> {wrap, row, state, label, meta, path, twisty, children}
    this.collapsed = new Set();
    this.active = null;
    this.follow = true;
    this.inspected = new Set();
    this.selected = null; // the node open in the details drawer: a UI choice, styled apart from running
    this.technical = storage.get("tree-technical", false); // raw names, codes and counts instead of operator text
    const technical = $("tree-technical");
    technical.setAttribute("aria-pressed", String(this.technical));
    technical.addEventListener("click", () => {
      this.technical = !this.technical;
      storage.set("tree-technical", this.technical);
      technical.setAttribute("aria-pressed", String(this.technical));
      this.render(this.app.state);
    });
    const pause = () => this.setFollow(false);
    this.pane.addEventListener("wheel", pause, { passive: true });
    this.pane.addEventListener("touchmove", pause, { passive: true });
    this.pane.addEventListener("keydown", (event) => {
      if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End"].includes(event.key)) pause();
    });
    $("tree-follow").addEventListener("click", () => {
      this.setFollow(true);
      this.reveal(this.active, true);
    });
  }

  setFollow(on) {
    if (this.follow === on) return;
    this.follow = on;
    $("tree-follow").hidden = on || !this.app.state?.task || this.app.state.task.end !== null;
  }

  render(state) {
    const tree = state.tree;
    this.renderState(state);
    this.renderSuperseded(state);
    if (!tree) {
      this.container.replaceChildren();
      this.rows.clear();
      return;
    }
    const path = activePath(tree);
    const active = path.length > 1 ? path.at(-1) : null;
    const moved = active !== this.active;
    if (moved && this.follow) path.forEach((id) => this.collapsed.delete(id));
    const seen = new Set();
    const showRoot = tree.mode !== "planning" || tree.nodes.root.children.length > 0;
    const root = showRoot ? this.node(tree, tree.root, 0, state, seen) : null;
    if (root && this.container.firstChild !== root) this.container.replaceChildren(root);
    if (!root) this.container.replaceChildren();
    for (const id of [...this.rows.keys()]) if (!seen.has(id)) this.rows.delete(id);
    $("tree-follow").hidden = this.follow || !active;
    if (moved) {
      this.active = active;
      if (this.follow) this.reveal(active);
    }
  }

  reveal(id, smooth = false) {
    this.rows.get(id)?.row.scrollIntoView({ block: "nearest", behavior: smooth ? "smooth" : "auto" });
  }

  /** The compact state card: what the panel is waiting for, never a made-up tree. */
  renderState(state) {
    const tree = state.tree;
    const c = state.controls;
    let title = "", text = "", spinner = false, banner = "";
    if (!tree) {
      [title, text] = state.initial_task && c.accepting
        ? ["Task ready", "Start it from the header. The behavior tree appears when the agent begins planning."]
        : ["No task", "Send an instruction to start. The behavior tree appears when the agent begins planning."];
      if (state.status === "Loading") [title, text] = ["Loading the simulation", "The panel fills once a task starts."];
    } else if (tree.mode === "planning") {
      [title, text, spinner] = ["Generating execution plan", "Waiting for the agent's behavior tree to be accepted.", c.busy];
    } else if (tree.mode === "replanning") {
      const plan = state.superseded.length;
      banner = `Replanning: the tree failed and the agent is rewriting it${plan ? ` (${plan} earlier plan${plan > 1 ? "s" : ""} below)` : ""}. Goals already done stay done.`;
    }
    $("tree-state").hidden = !title;
    $("tree-spinner").hidden = !spinner;
    setText($("tree-state-title"), title);
    setText($("tree-state-text"), text);
    $("tree-banner").hidden = !banner;
    setText($("tree-banner"), banner);
  }

  renderSuperseded(state) {
    const plans = state.superseded || [];
    $("superseded").hidden = plans.length === 0;
    setText($("superseded-title"), `Superseded plans (${plans.length})`);
    const signature = plans.map((p) => p.t).join("|");
    if (signature === this.supersededShown) return;
    this.supersededShown = signature;
    $("superseded-list").replaceChildren(...plans.map((plan, index) => h("li", {}, h("button", {
      type: "button", onclick: () => this.app.drawer.superseded(index),
    }, `Plan replaced at ${clock(plan.t)} (rewrite ${plan.replan})`))));
  }

  node(tree, id, depth, state, seen) {
    const node = tree.nodes[id];
    seen.add(id);
    if (node.type === "attempts" && !this.rows.has(id)) this.collapsed.add(id); // earlier attempts start folded
    // a recovery branch unfolds only while it runs or when the operator opens it
    if (node.type === "goal" && node.detail.recovers && !this.rows.has(id) && node.status !== "running") this.collapsed.add(id);
    const view = this.rows.get(id) || this.create(id);
    const { wrap, row, children } = view;
    wrap.dataset.depth = depth;
    row.style.setProperty("--depth", depth);
    const leaf = node.children.length === 0;
    toggleClass(view.twisty, "leaf", leaf);
    view.twisty.disabled = leaf;
    view.twisty.setAttribute("aria-label", this.collapsed.has(id) ? "Expand" : "Collapse");
    toggleClass(wrap, "collapsed", this.collapsed.has(id) && !leaf);
    if (leaf) row.removeAttribute("aria-expanded");
    else row.setAttribute("aria-expanded", String(!this.collapsed.has(id)));
    row.className = `row ${node.status}${leaf ? " leaf" : ""}${this.inspected.has(id) ? " inspected" : ""}${this.selected === id ? " selected" : ""}`;
    row.setAttribute("aria-selected", String(this.selected === id));
    this.paintState(view.state, node);
    const text = this.technical ? { label: node.label, meta: this.meta(node, state) } : this.operator(node, state);
    setText(view.label, text.label);
    view.label.title = node.label;
    setText(view.meta, text.meta);
    this.paintPath(view.path, this.technical ? node : null);
    view.tag.hidden = !this.recovering(node);
    row.setAttribute("aria-label", `${node.label}: ${STATUS_TEXT[node.status] || node.status}${this.recovering(node) ? ", recovery" : ""}`);
    const elements = node.children.map((child) => this.node(tree, child, depth + 1, state, seen));
    const current = [...children.children];
    if (current.length !== elements.length || current.some((el, i) => el !== elements[i])) children.replaceChildren(...elements);
    return wrap;
  }

  recovering(node) {
    return Boolean(node.detail.recovers || node.detail.recovery);
  }

  create(id) {
    const twisty = h("button", { class: "twisty", type: "button", tabindex: "-1" }, icon("chevron"));
    const state = h("span", { class: "state" }, icon("check"));
    const label = h("span", { class: "node-label" });
    const meta = h("span", { class: "node-meta" });
    const path = h("span", { class: "node-path" });
    const tag = h("span", { class: "tag recovery", text: "Recovery", hidden: true });
    const row = h("div", { class: "row", role: "treeitem", tabindex: "0" },
      twisty, state, h("span", { class: "node-text" }, label, meta, path), tag);
    const children = h("div", { class: "children", role: "group" });
    const wrap = h("div", { class: "tree-node" }, row, children);
    twisty.addEventListener("click", (event) => {
      event.stopPropagation();
      this.toggle(id);
    });
    row.addEventListener("click", () => this.select(id));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") this.select(id);
      else if (event.key === "ArrowLeft") this.toggle(id, true);
      else if (event.key === "ArrowRight") this.toggle(id, false);
      else return;
      event.preventDefault();
    });
    const view = { wrap, row, twisty, state, label, meta, path, tag, children };
    this.rows.set(id, view);
    return view;
  }

  select(id) {
    this.setFollow(false);
    this.selected = id;
    this.app.drawer.node(id);
    this.render(this.app.state);
  }

  deselect() {
    if (this.selected === null) return;
    this.selected = null;
    if (this.app.state) this.render(this.app.state);
  }

  toggle(id, collapse = !this.collapsed.has(id)) {
    if (collapse) this.collapsed.add(id);
    else this.collapsed.delete(id);
    this.render(this.app.state);
  }

  /** Highlight attempt TOOL_ID and its goal, unfolding the group that holds it, and bring it into view. */
  highlightTool(toolId) {
    const tree = this.app.state?.tree;
    if (!tree) return;
    const key = `tool:${toolId}`;
    const holder = Object.values(tree.nodes).find((n) => n.children.includes(key));
    if (!holder) return;
    this.inspected = new Set([key]);
    for (const n of Object.values(tree.nodes)) if (n.children.includes(holder.id)) this.inspected.add(n.id === "root" ? holder.id : n.id);
    this.collapsed.delete(holder.id);
    for (const n of Object.values(tree.nodes)) if (n.children.includes(holder.id)) this.collapsed.delete(n.id);
    this.setFollow(false);
    this.render(this.app.state);
    this.rows.get(key)?.row.scrollIntoView({ block: "nearest" });
  }

  /** Mark the goal and the tool call that ran at moment T (null clears the marks). */
  highlightAt(t) {
    const state = this.app.state;
    this.inspected = new Set();
    if (t !== null && state) {
      const now = this.app.link.now();
      for (const span of state.intervals) {
        if (span.start > t || (span.end ?? now) < t) continue;
        if (span.track === "goal" && span.detail.goal) this.inspected.add(`goal:${span.detail.goal}`);
        if (span.track === "skill" && span.detail.tool) this.inspected.add(`tool:${span.detail.tool}`);
      }
    }
    this.render(state);
    const first = [...this.inspected].map((id) => this.rows.get(id)).find(Boolean);
    if (first) {
      this.setFollow(false);
      first.row.scrollIntoView({ block: "nearest" });
    }
  }

  paintState(element, node) {
    const composite = node.type in KIND_ICON;
    element.className = `state ${node.status}${composite ? " composite" : ""}`;
    const name = composite ? KIND_ICON[node.type] : node.type === "goal" && node.status === "pending" ? null : STATE_ICON[node.status];
    const svg = element.querySelector("svg");
    svg.style.display = name ? "" : "none";
    if (name) setIcon(svg, name);
    element.title = STATUS_TEXT[node.status] || node.status;
  }

  /** The operator's wording, with the recorded initiator of a recovery goal. */
  operator(node, state) {
    const d = node.detail;
    const now = this.app.link.now();
    let start = d.start, end = d.end;
    if (node.type === "goal") {
      const span = state.intervals.findLast((i) => i.track === "goal" && i.detail.goal === d.goal);
      [start, end] = span ? [span.start, span.end] : [null, null];
    }
    const elapsed = start ? duration((end ?? now) - start) : "";
    const text = operatorText(node, elapsed);
    if (node.type === "goal" && d.used > 0) { // the goal's attempts, from its recorded calls
      const tools = state.tools.filter((t) => t.goal === d.goal && state.task && t.start >= state.task.start);
      const [count, ...previous] = retrySummary(tools, d.budget).split(" · ");
      if (d.used > 1 || node.status === "running") text.label = `${goalLabel(node.label, 24)} · ${count}`;
      text.meta = [STATUS_TEXT[node.status] || node.status, ...previous, elapsed].filter(Boolean).join(" · ");
    }
    if (node.type === "goal" && d.recovers) {
      const started = state.incidents.findLast((i) => i.tag === "recovery_start" && i.node === d.goal);
      text.meta += ` · triggered by ${started?.initiator || "an unrecorded source"}`;
    }
    return text;
  }

  /** A fallback's route through its children so far, e.g. "grasp ✗ → reposition ●" (technical view only). */
  paintPath(element, node) {
    const path = node?.type === "fallback" ? node.detail.path || [] : [];
    const signature = path.map((p) => `${p.label}${p.status}`).join("|");
    if (element.dataset.signature === signature) return;
    element.dataset.signature = signature;
    element.hidden = path.length < 2;
    element.replaceChildren(...path.flatMap((p, i) => [i ? " → " : "", h("span", { class: p.status, text: `${p.label} ${MARK[p.status] || ""}` })]));
  }

  meta(node, state) {
    const status = STATUS_TEXT[node.status] || node.status;
    const d = node.detail;
    if (node.type === "root") return d.note || `Task · ${status}`;
    if (node.type === "sequence") return `Sequence · ${status}`;
    if (node.type === "attempts") return `Earlier attempts (visual grouping) · ${d.codes.join(", ")}`;
    if (node.type === "fallback") {
      return d.selected === null ? "Fallback · every child failed"
        : d.selected > 0 ? `Fallback · recovering with child ${d.selected + 1} · ${status}` : `Fallback · child 1 · ${status}`;
    }
    if (node.type === "goal") {
      const span = state.intervals.findLast((i) => i.track === "goal" && i.detail.goal === d.goal);
      const time = span ? ` · ${duration((span.end ?? this.app.link.now()) - span.start)}` : "";
      const recovers = d.recovers ? ` · after “${d.recovers_label || d.recovers}” failed` : "";
      return `${status}${time} · ${d.used}/${d.budget} skill calls${recovers} · check: ${d.check}`;
    }
    const time = d.start ? ` · ${duration((d.end ?? this.app.link.now()) - d.start)}` : "";
    const attempt = d.attempt > 1 ? ` · attempt ${d.attempt}` : "";
    return `${status}${time}${attempt}${d.code ? ` · ${d.code}` : ""}`;
  }
}
