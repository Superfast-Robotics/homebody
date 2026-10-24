// Pure state logic the panels share: no DOM, so node --test can check it (tests/js).

export const WINDOW = 60; // s on the main monitoring view
export const MIN_STALE = 3; // s without a reading before the monitor counts as stale
export const MIN_GAP = 1.5; // s a reading may stand for before the next one is missing
export const BUCKET = 1; // s per risk block
export const HOLD_MARGIN = 2; // s shown after a task's end when the timeline holds there

const ENDED = { model_done: "Run again", cancelled: "Run again", Stopped: "Run again" };

/** Seconds between a source's readings: the median of the recent intervals, or null with fewer than two. */
export function cadence(readings) {
  const times = readings.map((r) => r.t).sort((a, b) => a - b);
  const steps = times.slice(1).map((t, i) => t - times[i]).filter((s) => s > 0).sort((a, b) => a - b);
  return steps.length ? steps[Math.floor(steps.length / 2)] : null;
}

/** How long one reading stands for: three of its source's intervals, at least MIN_GAP. */
export function gapFor(readings) {
  const step = cadence(readings.slice(-40));
  return Math.max(MIN_GAP, step === null ? MIN_GAP : 3 * step);
}

/**
 * The monitor's connectivity (connected, stale, offline: the page lost the session, unknown: never heard from) and
 * the age of its newest observation. Heartbeats show connectivity; only an observation, by its own timestamp, counts
 * for "last observation".
 */
export function monitorStatus(readings, incidents, now, connected) {
  const latest = readings.at(-1) || null;
  const recent = readings.filter((r) => r.t >= now - WINDOW);
  const observations = recent.filter((r) => r.kind === "observation");
  const lastObservation = readings.findLast((r) => r.kind === "observation") || null;
  const withRisk = observations.filter((r) => r.risk !== null);
  const staleAfter = Math.max(MIN_STALE, gapFor(readings));
  const heard = latest ? Math.max(0, now - (latest.received ?? latest.t)) : null;
  const state = !connected ? "offline" : !latest ? "unknown" : heard > staleAfter ? "stale" : "connected";
  return {
    state, heard, staleAfter,
    observationAge: lastObservation ? Math.max(0, now - lastObservation.t) : null,
    synthetic: recent.some((r) => r.synthetic),
    observations: observations.length,
    risk: withRisk.at(-1) || null,
    anomalies: incidents.filter((i) => i.kind === "monitor" && i.state === "active").length,
  };
}

/** Task spans clipped to [start, end]. */
export function spans(tasks, start, end) {
  return tasks.map((t) => [Math.max(start, t.start), Math.min(end, t.end ?? end)]).filter(([a, b]) => b > a);
}

/**
 * The observation track: runs of one level, and "missing" stretches inside a task where no reading stands.
 * Only behavioural observations made during a task count; heartbeats only show the monitor is connected.
 */
export function observationSegments(readings, tasks, start, end) {
  const observed = readings.filter((r) => r.kind === "observation" && r.during_task).sort((a, b) => a.t - b.t);
  if (!observed.length) return [];
  const gap = gapFor(observed);
  const segments = [];
  observed.forEach((reading, index) => {
    const next = observed[index + 1];
    const until = Math.min(next ? Math.min(next.t, reading.t + gap) : reading.t + gap, end);
    if (until < start || reading.t > end) return;
    const last = segments.at(-1);
    if (last && last.level === reading.level && reading.t - last.end <= 1e-6) {
      last.end = until;
      last.readings.push(reading);
    } else {
      segments.push({ start: Math.max(start, reading.t), end: until, level: reading.level, readings: [reading] });
    }
  });
  const missing = [];
  for (const [a, b] of spans(tasks, start, end)) {
    let cursor = a;
    for (const s of segments) {
      if (s.end <= cursor || s.start >= b) continue;
      if (s.start - cursor >= 0.25) missing.push({ start: cursor, end: s.start, level: "missing", readings: [] });
      cursor = Math.max(cursor, s.end);
    }
    if (b - cursor >= 0.25) missing.push({ start: cursor, end: b, level: "missing", readings: [] });
  }
  return [...segments, ...missing].sort((x, y) => x.start - y.start);
}

/** A risk score judges behaviour, so only observations made during a task carry one onto the chart. */
function scored(r) {
  return r.risk !== null && r.kind === "observation" && r.during_task;
}

/** Sources that reported a risk score during a task, newest last. */
export function riskSources(readings) {
  const order = [];
  for (const r of readings) if (scored(r) && !order.includes(r.source)) order.push(r.source);
  return order;
}

/** A bucket's risk class from its readings: the monitor's threshold and levels decide, never a made-up cut. */
function riskClass(readings) {
  const high = readings.some((r) => r.level === "anomaly" || (r.threshold !== null && r.risk >= r.threshold));
  if (high) return "high";
  const elevated = readings.some((r) => r.level === "elevated" ||
    (r.elevated_threshold !== null && r.elevated_threshold !== undefined && r.risk >= r.elevated_threshold));
  return elevated ? "elevated" : "low";
}

/**
 * SOURCE's risk as one block per wall-clock second over [start, end]: the maximum risk in the second and its class.
 * A second of a task with no sample is "missing"; seconds outside every task get no block.
 */
export function riskBuckets(readings, source, tasks, start, end) {
  const groups = new Map();
  for (const r of readings) {
    if (!scored(r) || r.source !== source || r.t < start - BUCKET || r.t > end) continue;
    const key = Math.floor(r.t / BUCKET) * BUCKET;
    groups.set(key, [...(groups.get(key) || []), r]);
  }
  const ranges = spans(tasks, start, end);
  const blocks = [];
  for (let s = Math.floor(start / BUCKET) * BUCKET; s < end; s += BUCKET) {
    const inside = groups.get(s);
    if (inside) {
      const peak = inside.reduce((top, r) => (r.risk > top.risk ? r : top));
      blocks.push({ start: s, end: s + BUCKET, state: riskClass(inside), max: peak.risk, count: inside.length,
        threshold: peak.threshold, source, readings: inside });
    } else if (ranges.some(([a, b]) => s + BUCKET > a && s < b)) {
      blocks.push({ start: s, end: s + BUCKET, state: "missing", max: null, count: 0, threshold: null, source, readings: [] });
    }
  }
  return blocks;
}

/** The intervention track's marker for an incident, or null for a detection (failures and anomalies). */
export function interventionMarker(incident) {
  if (incident.kind !== "intervention" && incident.kind !== "recovery") return null;
  // human actions are the running blue, a pause amber; recoveries the violet of a recovery, green once it worked
  const markers = {
    stop: ["square", "running"], reset: ["reset", "running"], pause: ["pause", "warning"], resume: ["play", "running"],
    stop_skill: ["square", "running"], recovery_start: ["diamond", "violet"], recovery_ok: ["diamond", "success"],
    recovery_fail: ["diamond", "danger"], replan: ["replan", "violet"],
  };
  const [shape, tone] = markers[incident.tag] || ["circle", "running"];
  return { shape, tone };
}

/**
 * The header's primary action for the session state and the typed text: what pressing it does, never a guess.
 * Returns {kind: "start" | "reset" | "none", label, text, enabled, title}.
 */
export function primaryAction(state, typed, connected) {
  const none = { kind: "none", label: "", text: "", enabled: false, title: "" };
  if (!state || !connected) return none;
  const c = state.controls;
  if (c.busy) return none;
  if (state.status === "Reset required") {
    return { kind: "reset", label: "Reset scene", text: "", enabled: !c.reset_pending, title: "The physics needs a reset before another task" };
  }
  if (!c.accepting) return none;
  const text = typed.trim();
  if (text) return { kind: "start", label: "Start task", text, enabled: true, title: "Start this instruction as a new task" };
  const task = state.task;
  if (task && task.end !== null) {
    const label = ENDED[task.outcome] || "Retry";
    return { kind: "start", label, text: task.text, enabled: true, title: `${label}: ${task.text}` };
  }
  if (state.initial_task) {
    return { kind: "start", label: "Start task", text: state.initial_task, enabled: true, title: `Start: ${state.initial_task}` };
  }
  return { ...none, kind: "start", label: "Start task", title: "Write an instruction first" };
}

function covering(intervals, track, t, now) {
  return intervals.findLast((i) => i.track === track && i.start <= t && (i.end ?? now) >= t) || null;
}

/** What the record holds about moment T, for inspection: never interpolated between readings. */
export function momentAt(state, t, now) {
  const observed = state.readings.filter((r) => r.kind === "observation");
  const gap = gapFor(observed);
  const reading = observed.findLast((r) => r.t <= t && t - r.t <= gap) || null;
  const risky = state.readings.filter(scored);
  return {
    t,
    goal: covering(state.intervals, "goal", t, now),
    skill: covering(state.intervals, "skill", t, now),
    decision: state.decisions.findLast((d) => d.t <= t) || null,
    reading,
    risk: risky.findLast((r) => r.t <= t && t - r.t <= gapFor(risky)) || null,
    incidents: state.incidents.filter((i) => Math.abs(i.t - t) <= 2),
  };
}

/**
 * Where the live timeline ends: it follows the clock while a task runs, and holds once the run stops (at the task's
 * end, or at the last state received when the page lost the session), so nothing scrolls in that is not there. After
 * the operator returns to live (THAWED is that task's end), it follows the clock again until the next task ends.
 * Returns {end, held, reason}.
 */
export function liveEnd(state, now, connected, lastSeen, thawed = null) {
  if (!connected && lastSeen !== null && lastSeen !== undefined) {
    return { end: Math.min(now, lastSeen), held: true, reason: "disconnected" };
  }
  const task = state?.task;
  if (task && task.end !== null && !state.controls.busy && thawed !== task.end) {
    return { end: Math.min(now, task.end + HOLD_MARGIN), held: true, reason: "ended" };
  }
  return { end: now, held: false, reason: "" };
}

/**
 * How much of the task time in [start, end] the monitor's observations cover, from their own timestamps: a reading
 * stands for up to three of its source's intervals (gapFor), the rest is a gap. Without an established cadence (fewer
 * than two observations) there is no percentage, only counts. Readings on invalid camera frames are counted apart.
 * Returns {kind: "none" | "counts" | "coverage", text, covered, total, gaps, longest, invalid}.
 */
export function coverage(readings, tasks, start, end) {
  const total = spans(tasks, start, end).reduce((sum, [a, b]) => sum + (b - a), 0);
  if (total <= 0) return { kind: "none", text: "No task time in this window" };
  const observed = readings.filter((r) => r.kind === "observation" && r.during_task);
  const inWindow = observed.filter((r) => r.t >= start && r.t <= end);
  const invalid = inWindow.filter((r) => r.frame_quality === "invalid" || r.frame_quality === "suspected_invalid").length;
  const onInvalid = invalid ? ` · ${invalid} reading${invalid > 1 ? "s" : ""} on unusable camera frames (counted above)` : "";
  const s = (n) => `${Math.round(n)} s`;
  if (cadence(observed.slice(-40)) === null) {
    return { kind: "counts", total, invalid, text: `${inWindow.length} monitor reading${inWindow.length === 1 ? "" : "s"} in ${s(total)} of task time · no established cadence, so no coverage figure${onInvalid}` };
  }
  const gaps = observationSegments(readings, tasks, start, end).filter((g) => g.level === "missing").map((g) => g.end - g.start);
  const missing = gaps.reduce((sum, g) => sum + g, 0);
  const covered = Math.max(0, total - missing);
  const longest = gaps.length ? Math.max(...gaps) : 0;
  return { kind: "coverage", covered, total, gaps: gaps.length, longest, invalid,
    text: `Monitor data ${s(covered)} / ${s(total)} of task time (${Math.floor((100 * covered) / total)}%)` // never rounded up
      + (gaps.length ? ` · ${gaps.length} gap${gaps.length > 1 ? "s" : ""}, longest ${s(longest)}` : "") + onInvalid };
}
