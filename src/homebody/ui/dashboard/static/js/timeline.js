// The monitoring timeline: every track on one time axis, drawn on a canvas from the newest state. Readings are
// drawn where they were made and stand only as long as their cadence allows; gaps are drawn as gaps.
import { inputWarning, scopeText } from "./assessment.js";
import { clock, duration, tokens } from "./dom.js";
import { MARKERS, attemptLabel, clusterMarkers, eventMarkers, incidentCard } from "./execution.js";
import { signalLabel, stateText } from "./interventions.js";
import { observationSegments, riskBuckets, riskSources, spans } from "./model.js";
import { TIMELINE, segmentLabel, segmentStyle } from "./timeline_style.js";

const ROWS = [["goal", 26], ["skill", 26], ["decisions", 26], ["readings", 26], ["risk", 30], ["interventions", 34], ["axis", 20]];
const LEFT = 8, RIGHT = 8; // px margins of the plot
const BLOCK = 22; // px height of a risk block
const FRAME_MS = 100;
const DIM = 0.55; // opacity of event markers unrelated to the selection (markers only: no text, never hidden)
const STATUS_TEXT = { running: "Running", success: "Completed", failure: "Failed", ended: "Ended", cancelled: "Cancelled" };
const RISK_TEXT = { low: "Low risk", elevated: "Elevated risk", high: "High risk (at or above threshold)", missing: "Missing" };
const LEVEL_TEXT = { expected: "Expected", unknown: "Unknown or unclassified", elevated: "Elevated concern", anomaly: "Anomaly", missing: "No reading" };

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
    ctx.font = `500 11.5px ${c.font}`;
    ctx.textBaseline = "middle";
    const end = this.end();
    const start = end - this.seconds;
    this.hatch ||= this.pattern(ctx, c.neutral);
    const view = { ctx, c, end, start, width, rows: this.layout(), tasks: spans(state.tasks || [], start, end), hatch: this.hatch,
      sel: this.app.selection };
    this.hits = [];
    this.grid(view);
    this.attemptBand(view, state);
    this.intervals(view, "goal");
    this.intervals(view, "skill");
    this.decisions(view, state);
    this.observations(view, state);
    this.risk(view, state);
    this.transitions(view, state);
    this.interventions(view, state);
    this.axis(view);
    this.cursors(view);
  }

  /** The selected attempt's interval (else the running one's), faintly across the tracks, so its readings and markers
   * read as its own. */
  attemptBand({ ctx, c, end, start, width, rows, sel }, state) {
    const tool = sel?.tool || state.tools.findLast((t) => t.state === "running");
    if (!tool || tool.start > end || (tool.end !== null && tool.end < start)) return;
    const x0 = this.clampX(Math.max(tool.start, start), { end, width }), x1 = this.clampX(tool.end ?? end, { end, width });
    ctx.fillStyle = sel?.tool ? TIMELINE.related : c["running-light"];
    ctx.globalAlpha = 0.5;
    ctx.fillRect(x0, 0, x1 - x0, rows.axis.y);
    ctx.globalAlpha = 1;
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

  grid({ ctx, end, width, rows }) {
    ctx.strokeStyle = TIMELINE.grid;
    ctx.lineWidth = 1;
    for (let s = 0; s <= this.seconds; s += this.step()) {
      const x = Math.round(this.x(end - s, end, width)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rows.axis.y);
      ctx.stroke();
    }
    for (const name of ["goal", "skill", "decisions", "readings"]) {
      ctx.fillStyle = TIMELINE.track;
      this.rect(ctx, LEFT, rows[name].y + 3, width - LEFT - RIGHT, rows[name].h - 6, 5);
      ctx.fill();
    }
  }

  step() {
    return this.seconds <= 60 ? 15 : this.seconds <= 300 ? 60 : 120;
  }

  /** Goal and skill intervals (timeline_style.segmentStyle): the execution state as a left accent and an icon, the
   * selected incident's attempt and goal as a quiet tint, the operator's exact selection as a thin border; the label
   * is always drawn at full strength over its own background. */
  intervals(view, track) {
    const { ctx, end, start } = view;
    const row = view.rows[track];
    const shown = this.app.state.intervals.filter((s) => s.track === track && (s.end ?? end) >= start && s.start <= end);
    if (!shown.length && track === "goal") this.empty(view, row, view.tasks.length ? "No goal in this window" : "No task in this window");
    const sel = view.sel;
    const tool = sel?.tool;
    const exact = sel?.ref?.tool && !sel.ref.intervention && !sel.ref.incident; // the operator picked this attempt itself
    for (const span of shown) {
      const x0 = this.clampX(span.start, view), x1 = this.clampX(span.end ?? end, view);
      const mine = tool && (track === "skill" ? span.detail.tool === tool.id : span.detail.goal === tool.goal);
      const paint = segmentStyle(span, track, !mine ? "none" : track === "skill" && exact ? "selected" : "related");
      const y = row.y + 4, h = row.h - 8, w = x1 - x0 - 2;
      ctx.fillStyle = paint.fill;
      this.rect(ctx, x0 + 1, y, w, h, 4);
      ctx.fill();
      if (paint.border) {
        ctx.strokeStyle = paint.border;
        ctx.lineWidth = 1;
        this.rect(ctx, x0 + 1.5, y + 0.5, w - 1, h - 1, 4);
        ctx.stroke();
      }
      if (paint.accent && w > 4) { // the execution state, whatever the selection
        ctx.fillStyle = paint.accent;
        this.rect(ctx, x0 + 1, y, 3, h, 1.5);
        ctx.fill();
      }
      const text = `${paint.icon ? `${paint.icon} ` : ""}${segmentLabel(span, track)}`;
      this.label(ctx, text, x0 + 8, x1 - 6, row.y + row.h / 2, paint.text);
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
    const colours = { expected: c["risk-low"], unknown: c["risk-missing"], elevated: c["risk-elevated"], anomaly: c["risk-high"] };
    const tool = view.sel?.tool;
    for (const segment of segments) {
      const x0 = this.clampX(segment.start, view), x1 = Math.max(x0 + 2, this.clampX(segment.end, view));
      const missing = segment.level === "missing";
      const mine = tool && segment.readings.some((r) => r.tool === tool.id && r.scope !== "goal"); // the attempt's own observations
      ctx.fillStyle = missing ? view.hatch : colours[segment.level];  // semantic colours stay as they are under a selection
      this.rect(ctx, x0, row.y + 7, x1 - x0 - 1, row.h - 14, 3);
      ctx.fill();
      if (mine) { // the selected attempt's own observations: a thin border, the level's colour kept
        ctx.strokeStyle = TIMELINE.selectedBorder;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
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
      ctx.globalAlpha = view.sel && view.sel.incident?.id !== incident.id ? DIM : 1;
      ctx.beginPath();
      ctx.moveTo(x - 5, row.y);
      ctx.lineTo(x + 5, row.y);
      ctx.lineTo(x, row.y + 7);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
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
    const colours = { low: c["risk-low"], elevated: c["risk-elevated"], high: c["risk-high"], missing: c["risk-missing"] };
    const top = row.y + (row.h - BLOCK) / 2;
    const blocks = riskBuckets(state.readings, source, state.tasks || [], start, end);
    const rank = { low: 0, elevated: 1, high: 2 };
    blocks.forEach((block, index) => { // a bright tick where a known reading first rises into elevated or high
      const before = blocks[index - 1];
      if (!before || before.state === "missing" || !(block.state in rank) || rank[block.state] <= rank[before.state] || block.state === "low") return;
      const x = this.clampX(block.start, view);
      ctx.fillStyle = c["event-warning"]; // the standard "risk threshold crossed" symbol: an amber triangle
      ctx.beginPath();
      ctx.moveTo(x - 4, row.y);
      ctx.lineTo(x + 4, row.y);
      ctx.lineTo(x, row.y + 5);
      ctx.closePath();
      ctx.fill();
      this.hits.push({ x0: x - 5, x1: x + 5, y0: row.y - 1, y1: row.y + 6, kind: "crossing", item: { block, before }, t: block.start });
    });
    const selected = this.app.inspectAt ?? this.hover;
    const tool = view.sel?.tool;
    const related = (block) => block.readings.some((r) => r.tool === tool.id && r.scope !== "goal"); // the attempt's own readings
    let extent = null; // the selected attempt's risk interval, underlined
    for (const block of blocks) {
      const x0 = this.clampX(block.start, view), x1 = this.clampX(block.end, view);
      if (x1 - x0 < 1) continue;
      const mine = tool && block.state !== "missing" && related(block);
      if (mine) extent = [Math.min(extent?.[0] ?? x0, x0), Math.max(extent?.[1] ?? x1, x1)];
      ctx.fillStyle = colours[block.state]; // every bucket keeps its semantic colour
      this.rect(ctx, x0 + 0.75, top, x1 - x0 - 1.5, BLOCK, 2);
      ctx.fill();
      if (selected !== null && selected !== undefined && selected >= block.start && selected < block.end) {
        ctx.strokeStyle = TIMELINE.selected; // the selected second: a thin violet border
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      this.hits.push({ x0, x1, y0: row.y, y1: row.y + row.h, kind: "risk", item: block, t: block.start });
    }
    if (extent) {
      ctx.fillStyle = TIMELINE.selectedBorder;
      ctx.fillRect(extent[0], top + BLOCK + 2, extent[1] - extent[0], 2);
    }
  }

  /** Where the skill attempt changes: a subtle dashed boundary across the observation and risk tracks, at the Skill
   * track's own times, so a risk reading reads as its attempt's. */
  transitions(view, state) {
    const { ctx, c, end, start, rows } = view;
    const skills = state.intervals.filter((s) => s.track === "skill").sort((a, b) => a.start - b.start);
    const y0 = rows.readings.y + 2, y1 = rows.risk.y + rows.risk.h - 2;
    for (let i = 1; i < skills.length; i += 1) {
      const t = skills[i].start;
      if (t < start || t > end) continue;
      const x = Math.round(this.x(t, end, view.width)) + 0.5;
      ctx.strokeStyle = c["text-muted"];
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      this.hits.push({ x0: x - 3, x1: x + 3, y0, y1, kind: "transition", item: { from: skills[i - 1], to: skills[i] }, t });
    }
  }

  /** The interventions track: each recorded event with its standard symbol (execution.MARKERS); events that would
   * overlap are one numbered cluster, listed on click, so none is hidden. */
  interventions(view, state) {
    const { ctx, c, end, start, width } = view;
    const row = view.rows.interventions;
    const mid = row.y + row.h / 2;
    const markers = eventMarkers(state, start, end);
    if (!markers.length) return;
    const sel = view.sel;
    const related = (m) => !sel || (sel.intervention ? m.intervention === sel.intervention.id : false) ||
      (sel.incident && m.incident === sel.incident.id) || (!sel.intervention && sel.tool && m.tool === sel.tool.id);
    const clusters = clusterMarkers(markers, (t) => this.x(t, end, width));
    const group = clusters.filter((cl) => sel && cl.items.some(related)).map((cl) => cl.x);
    if (group.length > 1) { // the selected incident's events, joined as one correlated group (by their ids)
      ctx.strokeStyle = TIMELINE.selectedBorder;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(Math.min(...group), mid);
      ctx.lineTo(Math.max(...group), mid);
      ctx.stroke();
    }
    const at = this.app.inspectAt;
    for (const cluster of clusters) {
      const x = cluster.x;
      const mine = cluster.items.some(related);
      const own = (m) => (sel?.intervention && m.intervention === sel.intervention.id) || (sel?.incident && m.incident === sel.incident.id);
      const exact = sel && mine && at !== null && cluster.items.some((m) => own(m) && Math.abs(m.t - at) < 0.05);
      const critical = cluster.items.some((m) => MARKERS[m.kind][1] === "danger");
      if (sel && mine) { // the exact selected event: the strongest ring; its related events: a light one
        ctx.strokeStyle = exact ? TIMELINE.selected : TIMELINE.selectedBorder;
        ctx.lineWidth = exact ? 2 : 1;
        ctx.beginPath();
        ctx.arc(x, mid, (cluster.items.length > 1 ? 12 : 10.5) + (exact ? 0.5 : 0), 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = mine ? 1 : critical ? 0.85 : DIM; // unrelated events stay visible, critical ones most of all
      if (cluster.items.length === 1) {
        const [m] = cluster.items;
        const [shape, tone] = MARKERS[m.kind];
        this.symbol(ctx, shape, x, mid, c[tone] || c.neutral, c["app-bg"]);
        ctx.globalAlpha = 1;
        this.hits.push({ x0: x - 8, x1: x + 8, y0: row.y, y1: row.y + row.h, kind: "event", item: m, t: m.t });
        continue;
      }
      ctx.fillStyle = c["surface-3"];
      ctx.strokeStyle = c["text-secondary"];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, mid, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.globalAlpha = 1; // the count is text: always at full strength
      ctx.fillStyle = c["text-primary"];
      ctx.textAlign = "center";
      ctx.fillText(String(cluster.items.length), x, mid + 0.5);
      ctx.textAlign = "left";
      this.hits.push({ x0: x - 10, x1: x + 10, y0: row.y, y1: row.y + row.h, kind: "cluster", item: cluster.items, t: cluster.t });
    }
  }

  /** One event symbol: stop (outlined octagon), check, square, diamond, person, cross. */
  symbol(ctx, shape, x, y, colour, ink) {
    ctx.fillStyle = colour;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (shape === "stop") {
      for (let k = 0; k < 8; k += 1) {
        const a = Math.PI / 8 + (k * Math.PI) / 4;
        ctx[k ? "lineTo" : "moveTo"](x + 6.5 * Math.cos(a), y + 6.5 * Math.sin(a));
      }
      ctx.closePath();
      ctx.stroke();
    } else if (shape === "check") {
      ctx.moveTo(x - 5, y);
      ctx.lineTo(x - 1.5, y + 4);
      ctx.lineTo(x + 5.5, y - 4.5);
      ctx.stroke();
    } else if (shape === "square") {
      ctx.fillRect(x - 5.5, y - 5.5, 11, 11);
    } else if (shape === "diamond") {
      ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.closePath(); ctx.fill();
    } else if (shape === "person") {
      ctx.arc(x, y - 4, 2.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x - 5, y + 6); ctx.quadraticCurveTo(x, y - 3, x + 5, y + 6); ctx.closePath(); ctx.fill();
    } else if (shape === "cross") {
      ctx.lineWidth = 2.5;
      ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke();
    } else {
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    void ink;
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
    const line = (t, colour, dash, lineWidth = 1.5) => {
      const x = this.x(t, end, width);
      if (x < LEFT || x > width - RIGHT) return null;
      ctx.strokeStyle = colour;
      ctx.lineWidth = lineWidth;
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
    ctx.shadowColor = "rgba(167, 139, 250, 0.18)"; // the cursor: the selection's main emphasis, softly lit
    ctx.shadowBlur = 6;
    const x = line(t, TIMELINE.selected, [], 2);
    ctx.shadowBlur = 0;
    if (x === null) return;
    const sel = this.app.selection;
    const state = this.app.state;
    const title = sel?.intervention ? incidentCard(state, sel.intervention).title : sel?.incident?.title || "";
    const short = title.length > 32 ? `${title.slice(0, 31)}…` : title;
    const labelled = short ? `${short} · ${clock(t)}` : clock(t);
    const text = ctx.measureText(labelled).width + 12 < width / 3 ? labelled : clock(t); // a label only where it fits
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
      const counters = [item.detail.number ? `Attempt ${item.detail.number} of its goal (as in the Behavior Tree)` : "",
        item.detail.attempt ? `${item.label} call ${item.detail.attempt} in this goal` : "", item.detail.recovery ? "Recovery" : ""].filter(Boolean);
      return [`Skill: ${segmentLabel(item, "skill")}`, `${STATUS_TEXT[item.status] || item.status} · started ${clock(item.start)} · ${span(item)}`,
        goal ? `Node: ${goal}` : "", counters.join(" · "), item.detail.execution ? `Skill execution ${item.detail.execution}` : ""].filter(Boolean).join("\n");
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
      const peak = item.readings.reduce((top, r) => (r.risk > top.risk ? r : top));
      const requests = (state.interventions || []).filter((r) => peak.tool && r.tool === peak.tool);
      const signals = [...new Set(item.readings.map((r) => signalLabel(r.signal)))];
      const tool = state.tools.find((t) => t.id === peak.tool);
      return [`${span} · ${RISK_TEXT[item.state]}`,
        `${signalLabel(peak.signal)}: ${item.max.toFixed(2)} · ${item.count} reading${item.count > 1 ? "s" : ""} · aggregation: the second's maximum`,
        `Attempt: ${tool ? `${attemptLabel(tool)} (${tool.execution})` : "Unknown (the reading named no skill call)"}`,
        `Applies to: ${scopeText(peak, state)}`, inputWarning(peak) ? `${inputWarning(peak).text} at this observation` : "",
        signals.length > 1 ? `Signals in this second: ${signals.join(", ")}` : "",
        `Threshold ${item.threshold === null ? "none" : item.threshold.toFixed(2)} · ${item.source}${item.readings.some((r) => r.synthetic) ? " · synthetic" : ""}`,
        incident ? `Incident: ${incident.title}` : "",
        ...requests.map((r) => `Intervention ${r.id}: ${stateText(r.state)[1]}`)].filter(Boolean).join("\n");
    }
    if (kind === "crossing") {
      const r = item.block.readings.reduce((top, x) => (x.risk > top.risk ? x : top));
      const what = item.block.state === "high" ? `Threshold crossed (${r.risk.toFixed(2)} ≥ ${r.threshold?.toFixed(2) ?? "?"})` : `Rose to elevated (${r.risk.toFixed(2)})`;
      return `${what} at ${clock(item.block.start)}\n${signalLabel(r.signal)} after ${item.before.max?.toFixed(2) ?? "?"}`;
    }
    if (kind === "event") {
      const tool = state.tools.find((t) => t.id === item.tool);
      return [`${item.label} · ${clock(item.t)}`, item.detail || "", tool ? `Attempt: ${attemptLabel(tool)}` : "",
        "Click to inspect"].filter(Boolean).join("\n");
    }
    if (kind === "transition") {
      const name = (span) => attemptLabel({ skill: span.label, number: span.detail.number ?? span.detail.attempt });
      return `${name(item.from)} → ${name(item.to)} · ${clock(item.to.start)}\nRisk before this line belongs to the earlier attempt`;
    }
    if (kind === "cluster") {
      return [`${item.length} events · click to list them`, ...item.map((m) => `${clock(m.t)} ${m.label}`)].join("\n");
    }
    return `${item.title}\n${item.severity} · ${item.state} · ${clock(item.t)}${item.outcome ? `\nThen: ${item.outcome}` : ""}`;
  }

  click(event) {
    const { x, hit, t } = this.at(event);
    if (x < LEFT) return;
    if (hit && !hit.weak && hit.kind !== "transition") {
      if (!this.app.open(hit.kind, hit.item)) this.app.inspect(hit.t);
    } else {
      const at = hit?.kind === "transition" ? hit.t : t;
      if (this.app.investigation.active) this.app.selectMoment(at); // the investigated incident stays selected
      else this.app.inspect(at);
    }
  }

  keys(event) {
    const step = event.shiftKey ? 5 : 1;
    const at = this.app.inspectAt ?? this.app.timelineEnd().end;
    if (event.key === "ArrowLeft") this.app.selectMoment(at - step);
    else if (event.key === "ArrowRight") this.app.selectMoment(Math.min(this.app.timelineEnd().end, at + step));
    else if (event.key === "Escape" && this.app.inspectAt !== null) this.app.returnToLive(); // a frozen task end stays frozen
    else return;
    event.preventDefault();
  }
}
