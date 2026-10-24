// The monitoring timeline: every track on one time axis, drawn on a canvas from the newest state. Readings are
// drawn where they were made and stand only as long as their cadence allows; gaps are drawn as gaps.
import { clock, duration, tokens } from "./dom.js";
import { signalLabel, stateText } from "./interventions.js";
import { interventionMarker, observationSegments, riskBuckets, riskSources, spans } from "./model.js";

const ROWS = [["goal", 26], ["skill", 26], ["decisions", 26], ["readings", 26], ["risk", 30], ["interventions", 34], ["axis", 20]];
const LEFT = 8, RIGHT = 8; // px margins of the plot
const BLOCK = 22; // px height of a risk block
const FRAME_MS = 100;
const STATUS_TEXT = { running: "Running", success: "Completed", failure: "Failed", ended: "Ended", cancelled: "Cancelled" };
const RISK_TEXT = { low: "Low risk", elevated: "Elevated risk", high: "High risk (at or above threshold)", missing: "Missing" };
const LEVEL_TEXT = { expected: "Expected", unknown: "Unknown or unclassified", elevated: "Elevated concern", anomaly: "Anomaly", missing: "No reading" };
// an intervention's recorded stages on the interventions track: shape and tone
const STAGE_MARK = { requested: ["triangle", "warning"], accepted: ["diamond", "running"], confirmed: ["square", "success"],
  rejected: ["cross", "danger"], failed: ["cross", "danger"], cancelled: ["circle", "neutral"], timed_out: ["clock", "danger"] };
const STAGE_TEXT = { requested: "Stop requested", accepted: "Executor acknowledged", confirmed: "Skill stopped (confirmed)",
  rejected: "Request rejected", failed: "Did not stop", cancelled: "Request withdrawn", timed_out: "Acknowledgement timed out" };

export class Timeline {
  /**
   * @param {object} app the dashboard
   * @param {HTMLCanvasElement} canvas
   * @param {{seconds?: number}} options the window length, 60 s on the main view
   */
  constructor(app, canvas, { seconds = 60 } = {}) {
    this.app = app;
    this.canvas = canvas;
    this.seconds = seconds;
    this.hits = [];
    this.hover = null;
    this.hatch = null;
    this.tooltip = document.getElementById("tooltip");
    this.running = true;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener("mousemove", (event) => this.move(event));
    canvas.addEventListener("mouseleave", () => { this.hover = null; this.tooltip.hidden = true; });
    canvas.addEventListener("click", (event) => this.click(event));
    canvas.addEventListener("keydown", (event) => this.keys(event));
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  stop() {
    this.running = false;
  }

  resize() {
    const ratio = devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    this.canvas.height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    this.ratio = ratio;
    this.draw();
  }

  loop(at) {
    if (!this.running) return;
    if (!this.last || at - this.last >= FRAME_MS) {
      this.last = at;
      try {
        this.draw();
      } catch (error) {
        console.error("dashboard: the timeline failed to draw", error); // the next frame tries again
      }
    }
    requestAnimationFrame(this.loop);
  }

  end() {
    return this.app.inspectEnd ?? this.app.timelineEnd().end;
  }

  layout() {
    const fixed = ROWS.reduce((sum, [, size]) => sum + size, 0);
    let y = 0;
    const rows = {};
    for (const [name, size] of ROWS) {
      const h = size || Math.max(20, this.canvas.clientHeight - fixed);
      rows[name] = { y, h };
      y += h;
    }
    return rows;
  }

  x(t, end, width) {
    return LEFT + ((t - (end - this.seconds)) / this.seconds) * (width - LEFT - RIGHT);
  }

  time(x, end, width) {
    return end - this.seconds + ((x - LEFT) / (width - LEFT - RIGHT)) * this.seconds;
  }

  draw() {
    const state = this.app.state;
    const ctx = this.canvas.getContext("2d");
    const width = this.canvas.clientWidth;
    if (!width || !state) return;
    const c = tokens();
    ctx.setTransform(this.ratio || 1, 0, 0, this.ratio || 1, 0, 0);
    ctx.clearRect(0, 0, width, this.canvas.clientHeight);
    ctx.font = `500 11px ${c.font}`;
    ctx.textBaseline = "middle";
    const end = this.end();
    const start = end - this.seconds;
    this.hatch ||= this.pattern(ctx, c.neutral);
    const view = { ctx, c, end, start, width, rows: this.layout(), tasks: spans(state.tasks || [], start, end), hatch: this.hatch };
    this.hits = [];
    this.grid(view);
    this.intervals(view, "goal");
    this.intervals(view, "skill");
    this.decisions(view, state);
    this.observations(view, state);
    this.risk(view, state);
    this.interventions(view, state);
    this.axis(view);
    this.cursors(view);
  }

  pattern(ctx, colour) {
    const tile = document.createElement("canvas");
    tile.width = tile.height = 6;
    const t = tile.getContext("2d");
    t.strokeStyle = colour;
    t.globalAlpha = 0.55;
    t.beginPath();
    t.moveTo(0, 6);
    t.lineTo(6, 0);
    t.stroke();
    return ctx.createPattern(tile, "repeat");
  }

  rect(ctx, x, y, w, h, r) {
    const width = Math.max(0, w);
    ctx.beginPath();
    ctx.roundRect(x, y, width, h, Math.max(0, Math.min(r, width / 2, h / 2)));
  }

  clampX(t, view) {
    return Math.min(view.width - RIGHT, Math.max(LEFT, this.x(t, view.end, view.width)));
  }

  grid({ ctx, c, end, width, rows }) {
    ctx.strokeStyle = c["border-subtle"];
    ctx.lineWidth = 1;
    for (let s = 0; s <= this.seconds; s += this.step()) {
      const x = Math.round(this.x(end - s, end, width)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rows.axis.y);
      ctx.stroke();
    }
    for (const name of ["goal", "skill", "decisions", "readings"]) {
      ctx.fillStyle = c["surface-2"];
      this.rect(ctx, LEFT, rows[name].y + 3, width - LEFT - RIGHT, rows[name].h - 6, 5);
      ctx.fill();
    }
  }

  step() {
    return this.seconds <= 60 ? 15 : this.seconds <= 300 ? 60 : 120;
  }

  intervals(view, track) {
    const { ctx, c, end, start } = view;
    const row = view.rows[track];
    const colours = {
      running: [c["running-light"], c.running], failure: [c["danger-light"], c["state-error-edge"]],
      // a finished goal is muted green; a finished skill recedes into charcoal so the running one stands out
      success: track === "goal" ? [c["success-light"], c["state-success-edge"]] : [c["surface-3"], c.border],
    };
    const shown = this.app.state.intervals.filter((s) => s.track === track && (s.end ?? end) >= start && s.start <= end);
    if (!shown.length && track === "goal") this.empty(view, row, view.tasks.length ? "No goal in this window" : "No task in this window");
    for (const span of shown) {
      const x0 = this.clampX(span.start, view), x1 = this.clampX(span.end ?? end, view);
      const recovery = span.detail.recovery;
      const [fill, stroke] = recovery && span.status !== "failure" && span.status !== "running"
        ? [c["violet-bg"], c.violet] : colours[span.status] || [c["surface-3"], c.border];
      ctx.fillStyle = fill;
      ctx.strokeStyle = recovery ? c.violet : stroke;
      ctx.lineWidth = recovery ? 1.5 : 1;
      ctx.setLineDash(recovery ? [4, 3] : []);
      this.rect(ctx, x0 + 1, row.y + 4, x1 - x0 - 2, row.h - 8, 4);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      const label = `${recovery ? "↻ " : ""}${span.label}${span.detail.attempt > 1 ? ` #${span.detail.attempt}` : ""}`;
      const ink = span.status === "running" ? c.running : recovery ? c.violet : c["text-secondary"];
      this.label(ctx, label, x0 + 6, x1 - 6, row.y + row.h / 2, ink);
      this.hits.push({ x0, x1, y0: row.y, y1: row.y + row.h, kind: track, item: span, t: span.start });
    }
  }

  label(ctx, text, x0, x1, y, colour) {
    if (x1 - x0 < 28) return;
    ctx.fillStyle = colour;
    let shown = text;
    while (shown.length > 1 && ctx.measureText(shown).width > x1 - x0) shown = `${shown.slice(0, -2)}…`;
    ctx.textAlign = "center";
    ctx.fillText(shown, (x0 + x1) / 2, y);
    ctx.textAlign = "left";
  }

  empty({ ctx, c, width }, row, text) {
    ctx.fillStyle = c["text-muted"];
    ctx.textAlign = "center";
    ctx.fillText(text, (LEFT + width - RIGHT) / 2, row.y + row.h / 2);
    ctx.textAlign = "left";
  }

  decisions(view, state) {
    const { ctx, c, end, start } = view;
    const row = view.rows.decisions;
    const mid = row.y + row.h / 2;
    for (const span of state.intervals) {
      if (span.track !== "thinking" || (span.end ?? end) < start) continue;
      const x0 = this.clampX(span.start, view), x1 = this.clampX(span.end ?? end, view);
      ctx.globalAlpha = 0.22;
      ctx.fillStyle = c.primary;
      ctx.fillRect(x0, row.y + 7, x1 - x0, row.h - 14);
      ctx.globalAlpha = 1;
      this.hits.push({ x0, x1, y0: row.y, y1: row.y + row.h, kind: "thinking", item: span, t: span.start, weak: true });
    }
    const tools = new Map(state.tools.map((t) => [t.id, t]));
    for (const d of state.decisions) {
      if (d.t < start || d.t > end) continue;
      const tool = tools.get(d.tool);
      // a decision is cyan; one the session answered itself (plan, goal closes) violet; red only for a failed call
      ctx.fillStyle = !tool ? c.violet : tool.state === "failed" ? c.danger : c.cyan;
      const x = this.x(d.t, end, view.width);
      ctx.beginPath();
      ctx.arc(x, mid, 4, 0, Math.PI * 2);
      ctx.fill();
      this.hits.push({ x0: x - 6, x1: x + 6, y0: row.y, y1: row.y + row.h, kind: "decision", item: d, t: d.t });
    }
  }

  observations(view, state) {
    const { ctx, c, end, start } = view;
    const row = view.rows.readings;
    if (!state.readings.length) return this.empty(view, row, "No monitoring service connected");
    if (!state.readings.some((r) => r.kind === "observation")) {
      return this.empty(view, row, "Monitor connected (heartbeats only): no behavioural observations");
    }
    const segments = observationSegments(state.readings, state.tasks || [], start, end);
    if (!segments.length) this.empty(view, row, view.tasks.length ? "No observations in this window" : "No task in this window");
    const colours = { expected: c.success, unknown: c.neutral, elevated: c.warning, anomaly: c.danger };
    for (const segment of segments) {
      const x0 = this.clampX(segment.start, view), x1 = Math.max(x0 + 2, this.clampX(segment.end, view));
      const missing = segment.level === "missing";
      ctx.fillStyle = missing ? view.hatch : colours[segment.level];
      this.rect(ctx, x0, row.y + 7, x1 - x0 - 1, row.h - 14, 3);
      ctx.fill();
      if (missing) {
        ctx.strokeStyle = c.neutral;
        ctx.setLineDash([2, 2]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      this.hits.push({ x0, x1, y0: row.y, y1: row.y + row.h, kind: "observation", item: segment, t: segment.start });
    }
    for (const incident of state.incidents) {
      if (incident.kind !== "monitor" || incident.t < start || incident.t > end) continue;
      const x = this.x(incident.t, end, view.width);
      ctx.fillStyle = c.danger;
      ctx.beginPath();
      ctx.moveTo(x - 5, row.y);
      ctx.lineTo(x + 5, row.y);
      ctx.lineTo(x, row.y + 7);
      ctx.closePath();
      ctx.fill();
      this.hits.push({ x0: x - 6, x1: x + 6, y0: row.y - 2, y1: row.y + 9, kind: "incident", item: incident, t: incident.t });
    }
  }

  /** Risk as one block per second, the second's maximum, classed by the monitor's own thresholds and levels. */
  risk(view, state) {
    const { ctx, c, end, start } = view;
    const row = view.rows.risk;
    const sources = riskSources(state.readings);
    if (!sources.length) {
      return this.empty(view, row, !state.readings.length ? "No risk score: no monitoring service"
        : view.tasks.length ? "The monitor reports no risk score" : "No task in this window");
    }
    const source = sources.includes(this.app.riskSource) ? this.app.riskSource : sources.at(-1);
    const colours = { low: c.success, elevated: c.warning, high: c.danger, missing: c.neutral };
    const top = row.y + (row.h - BLOCK) / 2;
    const blocks = riskBuckets(state.readings, source, state.tasks || [], start, end);
    blocks.forEach((block, index) => { // a tick where a known reading first reaches the threshold after a lower one
      const before = blocks[index - 1];
      if (block.state !== "high" || !before || before.state === "high" || before.state === "missing") return;
      const x = this.clampX(block.start, view);
      ctx.fillStyle = c.danger;
      ctx.beginPath();
      ctx.moveTo(x - 4, row.y);
      ctx.lineTo(x + 4, row.y);
      ctx.lineTo(x, row.y + 5);
      ctx.closePath();
      ctx.fill();
      this.hits.push({ x0: x - 5, x1: x + 5, y0: row.y - 1, y1: row.y + 6, kind: "crossing", item: { block, before }, t: block.start });
    });
    for (const block of blocks) {
      const x0 = this.clampX(block.start, view), x1 = this.clampX(block.end, view);
      if (x1 - x0 < 1) continue;
      ctx.fillStyle = colours[block.state];
      ctx.globalAlpha = block.state === "missing" ? 0.45 : 1;
      this.rect(ctx, x0 + 0.75, top, x1 - x0 - 1.5, BLOCK, 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      this.hits.push({ x0, x1, y0: row.y, y1: row.y + row.h, kind: "risk", item: block, t: block.start });
    }
  }

  interventions(view, state) {
    const { ctx, c, end, start, width } = view;
    const row = view.rows.interventions;
    const mid = row.y + row.h / 2;
    this.requests(view, state, row, mid);
    const shown = state.incidents.filter((i) => i.t >= start && i.t <= end && interventionMarker(i)).sort((a, b) => a.t - b.t);
    const xs = shown.map((i) => this.x(i.t, end, width));
    shown.forEach((incident, index) => {
      const x = xs[index];
      const { shape, tone } = interventionMarker(incident);
      this.glyph(ctx, shape, x, mid, c[tone] || c.running, c["app-bg"]);
      const limit = Math.min(width - RIGHT, (xs[index + 1] ?? Infinity) - 12); // a label stops before the next marker
      let text = incident.title;
      while (text.length > 4 && x + 11 + ctx.measureText(text).width > limit) text = `${text.slice(0, -2)}…`;
      if (x + 11 + ctx.measureText(text).width <= limit) {
        ctx.fillStyle = c["text-primary"];
        ctx.fillText(text, x + 11, mid);
      }
      this.hits.push({ x0: x - 8, x1: x + 8, y0: row.y, y1: row.y + row.h, kind: "incident", item: incident, t: incident.t });
    });
  }

  /** Each intervention's recorded stages, joined by a line from its request to its last stage. */
  requests(view, state, row, mid) {
    const { ctx, c, end, start, width } = view;
    for (const r of state.interventions || []) {
      const marks = r.stages.filter(([s, t]) => t !== null && STAGE_MARK[s]).map(([s, t]) => [s, t]);
      const requested = marks.find(([s]) => s === "requested");
      if (requested && (r.state === "timed_out" || marks.some(([s, t]) => s === "accepted" && t - requested[1] > r.ack_timeout))) {
        marks.push(["timed_out", requested[1] + r.ack_timeout]);
      }
      const shown = marks.filter(([, t]) => t >= start && t <= end).sort((a, b) => a[1] - b[1]);
      if (!shown.length) continue;
      const xs = shown.map(([, t]) => this.x(t, end, width));
      ctx.strokeStyle = c.border;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(r.state === "timed_out" || r.state === "requested" ? [3, 3] : []);
      ctx.beginPath();
      ctx.moveTo(xs[0], mid);
      ctx.lineTo(r.state === "requested" || r.state === "timed_out" || r.state === "executing" ? this.x(end, end, width) : xs.at(-1), mid);
      ctx.stroke();
      ctx.setLineDash([]);
      shown.forEach(([stage, t], i) => {
        const [shape, tone] = STAGE_MARK[stage];
        this.mark(ctx, shape, xs[i], mid, c[tone] || c.neutral, c["app-bg"]);
        this.hits.push({ x0: xs[i] - 7, x1: xs[i] + 7, y0: row.y, y1: row.y + row.h, kind: "intervention",
          item: { ...r, stage, at: t }, t });
      });
    }
  }

  mark(ctx, shape, x, y, colour, ink) {
    ctx.fillStyle = colour;
    ctx.strokeStyle = colour;
    ctx.beginPath();
    if (shape === "triangle") { ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y + 5); ctx.lineTo(x - 6, y + 5); ctx.closePath(); ctx.fill(); }
    else if (shape === "cross") { ctx.lineWidth = 2.5; ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke(); }
    else if (shape === "clock") {
      ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x, y); ctx.lineTo(x + 3, y + 2); ctx.stroke();
    } else this.glyph(ctx, shape, x, y, colour, ink);
  }

  glyph(ctx, shape, x, y, colour, ink) {
    ctx.fillStyle = colour;
    ctx.beginPath();
    if (shape === "square") ctx.rect(x - 6, y - 6, 12, 12);
    else if (shape === "diamond") { ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.closePath(); }
    else ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = ink;
    ctx.strokeStyle = ink;
    if (shape === "pause") { ctx.fillRect(x - 3, y - 4, 2, 8); ctx.fillRect(x + 1, y - 4, 2, 8); }
    if (shape === "play") { ctx.beginPath(); ctx.moveTo(x - 2, y - 4); ctx.lineTo(x + 4, y); ctx.lineTo(x - 2, y + 4); ctx.fill(); }
    if (shape === "reset" || shape === "replan") {
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0.3, Math.PI * 1.7);
      ctx.stroke();
    }
  }

  axis({ ctx, c, end, width, rows }) {
    const live = this.app.inspectEnd === null || this.app.inspectEnd === undefined;
    const held = this.app.timelineEnd();
    const last = !held.held ? "Now" : held.reason === "ended" ? "Task end" : "Last data";
    const step = this.step();
    ctx.fillStyle = c["text-secondary"];
    for (let s = 0; s <= this.seconds; s += step) {
      const x = this.x(end - s, end, width);
      const text = live ? (s === 0 ? last : `-${step >= 60 ? `${s / 60}m` : `${s}s`}`) : clock(end - s);
      ctx.textAlign = s === 0 ? "right" : s === this.seconds ? "left" : "center";
      ctx.fillText(text, s === 0 ? x : s === this.seconds ? Math.max(0, x - 4) : x, rows.axis.y + rows.axis.h / 2 + 2);
    }
    ctx.textAlign = "left";
  }

  cursors({ ctx, c, end, width, rows }) {
    const line = (t, colour, dash) => {
      const x = this.x(t, end, width);
      if (x < LEFT || x > width - RIGHT) return null;
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rows.axis.y);
      ctx.stroke();
      ctx.setLineDash([]);
      return x;
    };
    if (this.hover !== null) line(this.hover, c["violet-edge"], [3, 3]);
    const t = this.app.inspectAt;
    if (t === null || t === undefined) return;
    const x = line(t, c.violet, []);
    if (x === null) return;
    const text = clock(t);
    const w = ctx.measureText(text).width + 12;
    const left = Math.min(Math.max(LEFT, x - w / 2), width - RIGHT - w);
    ctx.fillStyle = c.primary;
    this.rect(ctx, left, rows.axis.y + 1, w, rows.axis.h - 2, 4);
    ctx.fill();
    ctx.fillStyle = c["on-primary"];
    ctx.textAlign = "center";
    ctx.fillText(text, left + w / 2, rows.axis.y + rows.axis.h / 2 + 1);
    ctx.textAlign = "left";
  }

  at(event) {
    const box = this.canvas.getBoundingClientRect();
    const x = event.clientX - box.left, y = event.clientY - box.top;
    const hits = this.hits.filter((hit) => x >= hit.x0 && x <= hit.x1 && y >= hit.y0 && y <= hit.y1);
    const hit = hits.findLast((h) => !h.weak) || hits[0] || null;
    return { x, hit, t: this.time(x, this.end(), this.canvas.clientWidth) };
  }

  move(event) {
    const { x, hit, t } = this.at(event);
    this.hover = x >= LEFT ? t : null;
    this.tooltip.hidden = false;
    this.tooltip.textContent = hit ? this.describe(hit) : `${clock(t)}\nClick to inspect this moment`;
    const left = Math.min(event.clientX + 14, innerWidth - this.tooltip.offsetWidth - 8);
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${Math.max(8, event.clientY - this.tooltip.offsetHeight - 12)}px`;
  }

  describe({ kind, item }) {
    const now = this.app.link.now();
    const state = this.app.state;
    const span = (s) => duration((s.end ?? now) - s.start);
    if (kind === "goal") {
      return `Goal: ${item.label}\n${STATUS_TEXT[item.status] || item.status} · started ${clock(item.start)} · ${span(item)}${item.detail.recovery ? "\nRecovery goal" : ""}`;
    }
    if (kind === "skill") {
      const goal = item.detail.goal && state.tree?.nodes[`goal:${item.detail.goal}`]?.label;
      const flags = [item.detail.attempt > 1 ? `Attempt ${item.detail.attempt}` : "", item.detail.recovery ? "Recovery" : ""].filter(Boolean);
      return [`Skill: ${item.label}`, `${STATUS_TEXT[item.status] || item.status} · started ${clock(item.start)} · ${span(item)}`,
        goal ? `Node: ${goal}` : "", flags.join(" · ")].filter(Boolean).join("\n");
    }
    if (kind === "thinking") return `Agent thinking · ${span(item)}`;
    if (kind === "decision") {
      const tool = state.tools.find((t) => t.id === item.tool);
      const outcome = tool ? `${tool.skill}() ${tool.code || tool.state}` : "Answered by the session";
      return `Decision ${clock(item.t)}: ${item.skill}\n${item.text}\n${item.goal ? `Node: ${item.goal} · ` : ""}${outcome}\n(agent-generated, not verified)`;
    }
    if (kind === "observation") {
      if (item.level === "missing") {
        return `No reading for ${duration(item.end - item.start)}\n${clock(item.start)}–${clock(item.end)}\nMissing samples are not a normal reading`;
      }
      const last = item.readings.at(-1);
      const count = `${item.readings.length} reading${item.readings.length > 1 ? "s" : ""}`;
      return `${LEVEL_TEXT[item.level]} · ${count}\n${clock(item.start)}–${clock(item.end)} · ${last.source}${last.synthetic ? " · synthetic" : ""}${last.reason ? `\n${last.reason}` : ""}`;
    }
    if (kind === "risk") {
      const span = `${clock(item.start)}–${clock(item.end)}`;
      if (item.state === "missing") return `${span} · no risk sample in this second (missing, not low)`;
      const incident = state.incidents.find((i) => i.kind === "monitor" && i.t >= item.start && i.t < item.end);
      const requests = (state.interventions || []).filter((r) => r.stages.some(([, t]) => t !== null && t >= item.start && t < item.end));
      const peak = item.readings.reduce((top, r) => (r.risk > top.risk ? r : top));
      const signals = [...new Set(item.readings.map((r) => signalLabel(r.signal)))];
      return [`${span} · ${RISK_TEXT[item.state]}`,
        `${signalLabel(peak.signal)}: ${item.max.toFixed(2)} (the second's maximum of ${item.count} reading${item.count > 1 ? "s" : ""})`,
        signals.length > 1 ? `Signals in this second: ${signals.join(", ")}` : "",
        `Threshold ${item.threshold === null ? "none" : item.threshold.toFixed(2)} · ${item.source}${item.readings.some((r) => r.synthetic) ? " · synthetic" : ""}`,
        incident ? `Incident: ${incident.title}` : "",
        ...requests.map((r) => `Intervention ${r.id}: ${stateText(r.state)[1]}`)].filter(Boolean).join("\n");
    }
    if (kind === "crossing") {
      const r = item.block.readings.reduce((top, x) => (x.risk > top.risk ? x : top));
      return `Threshold crossed at ${clock(item.block.start)}\n${signalLabel(r.signal)} ${r.risk.toFixed(2)} ≥ ${r.threshold?.toFixed(2) ?? "?"} after ${item.before.max?.toFixed(2) ?? "?"}`;
    }
    if (kind === "intervention") {
      const stage = item.stages.find(([s]) => s === item.stage);
      return [`${STAGE_TEXT[item.stage]} · ${clock(item.at)}`, `${item.initiator} → ${item.target}`,
        stage ? `${stage[2]}${stage[3] ? ` · ${stage[3]}` : ""}` : `No acknowledgement within ${item.ack_timeout} s of the request`,
        `Now: ${stateText(item.state)[1]} · click to inspect`].join("\n");
    }
    return `${item.title}\n${item.severity} · ${item.state} · ${clock(item.t)}${item.outcome ? `\nThen: ${item.outcome}` : ""}`;
  }

  click(event) {
    const { x, hit, t } = this.at(event);
    if (x < LEFT) return;
    if (hit && !hit.weak) this.app.open(hit.kind, hit.item);
    this.app.inspect(hit && !hit.weak ? hit.t : t);
  }

  keys(event) {
    const step = event.shiftKey ? 5 : 1;
    const at = this.app.inspectAt ?? this.app.timelineEnd().end;
    if (event.key === "ArrowLeft") this.app.inspect(at - step);
    else if (event.key === "ArrowRight") this.app.inspect(Math.min(this.app.timelineEnd().end, at + step));
    else if (event.key === "Escape") this.app.inspect(null);
    else return;
    event.preventDefault();
  }
}
