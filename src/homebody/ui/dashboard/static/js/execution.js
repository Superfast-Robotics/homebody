// Execution, risk and system state at the level of one skill attempt: pure, so node --test checks it (tests/js).
// Attempts are the store's tool calls, each with its own identity (execution); readings, incidents and requests name
// the attempt they belong to (their "tool"), so nothing here correlates by skill name or by being close in time.
import { OPEN, requestStatus, signalLabel } from "./interventions.js";

const SKILLS = { navigate: "Navigate", pick: "Pick", place: "Place", call_coder: "Coding assistant" };
const DEFAULT_TIMING = { transition_s: 3, stale_s: 8, ack_timeout_s: 10, stop_s: 20, report_overdue_s: 15 };

/** "Pick attempt 4", or "Unknown attempt" for a record that names none. */
export function attemptLabel(tool) {
  if (!tool) return "Unknown attempt";
  return `${SKILLS[tool.skill] || tool.skill} attempt ${tool.number ?? tool.attempt ?? "?"}`;
}

export function toolById(state, id) {
  return id ? state.tools.find((t) => t.id === id) || null : null;
}

export function runningAttempt(state) {
  return state.tools.findLast((t) => t.state === "running") || null;
}

/** Where an incident or request stands against the attempt now running: current, previous (still open) or resolved. */
export function attemptRelation(state, toolId, resolved = false) {
  if (resolved) return "resolved";
  if (!toolId) return "unknown";
  return runningAttempt(state)?.id === toolId ? "current" : "previous";
}

/**
 * Whether the tree, the executor and the controller agree about what runs, by the configured tolerances
 * (state.timing): {state: consistent | transitioning | suspected | confirmed | unknown, severity, rule, attempt,
 * bt, executor, controller, controllerAge}. Stale or unreported controller status is unknown, never a mismatch.
 */
export function consistency(state, now) {
  const timing = { ...DEFAULT_TIMING, ...(state.timing || {}) };
  const tool = runningAttempt(state);
  const executing = ["Executing", "Paused"].includes(state.status);
  const since = now - (state.status_since ?? now);
  const controller = state.controller || null;
  const controllerAge = controller?.checked ? Math.max(0, now - controller.checked) : null;
  const base = {
    attempt: tool, severity: "info", rule: "",
    bt: tool ? `${attemptLabel(tool)} running` : "No attempt running",
    executor: `${state.status} · for ${Math.round(since)} s`,
    controller: !controller ? "Not reported" : controller.state === "in_process" ? "In process (no separate report)"
      : `${controller.state}${controllerAge !== null ? ` · checked ${Math.round(controllerAge)} s ago` : ""}`,
    controllerAge,
  };
  const out = (cls, extra = {}) => ({ ...base, state: cls, ...extra });
  if (tool && controller && controller.state !== "in_process") {
    if (controller.state === "unknown" || controllerAge === null || controllerAge > timing.stale_s) {
      return out("unknown", { rule: "Controller status unknown or stale: no mismatch can be judged" });
    }
    if (controller.state === "stopped" || controller.state === "error") {
      const stopOpen = (state.interventions || []).some((r) => r.tool === tool.id && OPEN.includes(r.state));
      return out("confirmed", { severity: stopOpen ? "critical" : "warning",
        rule: `Behavior Tree: ${attemptLabel(tool)} running · Controller: ${controller.state}` });
    }
  }
  if (tool) {
    const taken = (state.interventions || []).find((r) => r.tool === tool.id && r.state === "executing");
    const at = taken?.stages.find(([s]) => s === "accepted")?.[1];
    if (at !== undefined && now - at > timing.stop_s) {
      return out("suspected", { severity: "warning", rule: `The executor took a stop ${Math.round(now - at)} s ago; the attempt still runs` });
    }
    if (!executing) {
      const quiet = now - Math.max(tool.start, state.status_since ?? tool.start);
      return quiet <= timing.transition_s ? out("transitioning", { rule: "The attempt and the executor are catching up" })
        : out("suspected", { severity: "warning", rule: `Behavior Tree: ${attemptLabel(tool)} running · Executor: ${state.status}` });
    }
  } else if (state.status === "Executing") {
    return since <= timing.transition_s ? out("transitioning", { rule: "A decision is starting its skill" })
      : out("suspected", { severity: "warning", rule: `Executor: Executing · Behavior Tree: no attempt running for ${Math.round(since)} s` });
  }
  return out("consistent");
}

const high = (r) => r.level === "anomaly" || (r.risk !== null && r.threshold !== null && r.risk >= r.threshold);
const elevated = (r) => high(r) || r.level === "elevated";
const scored = (readings) => readings.filter((r) => r.kind === "observation" && r.risk !== null).sort((a, b) => a.t - b.t);

/** Whether readings A and B judge the same context, by B's scope: an attempt's readings belong to that attempt (an
 * unscoped one to its attempt too, never across), a goal's to that goal, a task's, robot's or environment's carry on. */
export function sameContext(a, b) {
  const scope = b.scope || "skill_attempt";
  if (["task", "robot", "environment"].includes(scope)) return (a.scope || "skill_attempt") === scope;
  if (scope === "goal") return a.scope === "goal" && a.scope_goal === b.scope_goal;
  return a.tool === b.tool && (a.scope || "skill_attempt") !== "goal";
}

/** The newest uninterrupted run of elevated-or-high readings of one context (sameContext): {seconds, high, tool, start,
 * end, signal, scope, scope_goal, latest}; a gap longer than GAP, a calmer reading or another context ends it. */
export function continuousRisk(readings, gap) {
  const rs = scored(readings);
  const latest = rs.at(-1);
  if (!latest || !elevated(latest)) return null;
  let start = latest;
  for (let i = rs.length - 2; i >= 0; i -= 1) {
    const r = rs[i];
    if (!sameContext(r, latest) || !elevated(r) || start.t - r.t > gap) break;
    start = r;
  }
  const run = rs.filter((r) => r.t >= start.t && sameContext(r, latest));
  return { seconds: latest.t - start.t, high: run.every(high), tool: latest.tool, start: start.t, end: latest.t, signal: latest.signal,
    scope: latest.scope || "skill_attempt", scope_goal: latest.scope_goal ?? null, latest };
}

/** Seconds of [start, end] (one-second buckets) with an elevated or high reading: not necessarily contiguous. */
export function cumulativeRisk(readings, start, end) {
  const seconds = new Set(scored(readings).filter((r) => r.t >= start && r.t <= end && elevated(r)).map((r) => Math.floor(r.t)));
  return seconds.size;
}

/** The longest elevated run of the attempt before TOOL_ID, if it had one: {seconds, tool, signal}. */
export function previousAttemptRisk(readings, toolId, gap) {
  const rs = scored(readings).filter((r) => r.tool && r.tool !== toolId);
  const last = rs.at(-1)?.tool;
  if (!last) return null;
  let best = 0, start = null, prev = null, signal = "";
  for (const r of rs.filter((x) => x.tool === last)) {
    if (elevated(r) && (prev === null || r.t - prev <= gap)) start ??= r.t;
    else start = elevated(r) ? r.t : null;
    if (start !== null) best = Math.max(best, r.t - start);
    if (elevated(r)) signal = r.signal;
    prev = r.t;
  }
  return best > 0 ? { seconds: best, tool: last, signal } : null;
}

/**
 * The System chip: unhealthy services that the running task depends on (impact "critical") are a visible
 * degraded state; unknown impact says so; an unused service down is quiet diagnostics.
 */
export function systemSummary(rows) {
  const bad = rows.filter((r) => r.health !== "healthy");
  const critical = bad.filter((r) => r.impact === "critical");
  const unknown = bad.filter((r) => r.impact === "unknown");
  const healthy = rows.length - bad.length;
  if (critical.length) {
    return { text: "System degraded · Required service unavailable", tone: "danger", degraded: true, required: true,
      names: critical.map((r) => r.name) };
  }
  if (unknown.length) return { text: "System degraded · impact unknown", tone: "warning", degraded: true, required: false, names: unknown.map((r) => r.name) };
  return { text: bad.length ? `System · ${bad.length} unused service${bad.length > 1 ? "s" : ""} unavailable` : `System · ${healthy}/${rows.length} healthy`,
    tone: "", degraded: false, required: false, names: [] };
}

const STOPS = { MONITOR_STOPPED: "monitor stop", OPERATOR_STOPPED: "operator stop" };

/** A goal's retry summary from its recorded attempts: "Attempt 4/8 · 3 previous attempts · 1 monitor stop · 2 other
 * failures". TOOLS are the goal's attempts in order. */
export function retrySummary(tools, budget) {
  if (!tools.length) return "";
  const current = tools.at(-1);
  const previous = current.state === "running" ? tools.slice(0, -1) : tools;
  const parts = [`Attempt ${current.number ?? tools.length}${budget ? `/${budget}` : ""}`];
  if (previous.length) {
    parts.push(`${previous.length} previous attempt${previous.length > 1 ? "s" : ""}`);
    const counts = {};
    for (const t of previous) {
      const key = t.state === "succeeded" ? "succeeded" : STOPS[t.code] || "other failure";
      counts[key] = (counts[key] || 0) + 1;
    }
    for (const key of ["monitor stop", "operator stop", "other failure", "succeeded"]) {
      if (counts[key]) parts.push(`${counts[key]} ${key}${counts[key] > 1 && key !== "succeeded" ? "s" : ""}`);
    }
  }
  return parts.join(" · ");
}

/** When the skill executor last reported (a stage, a result, a stop it took), from the recorded events only. */
export function lastExecutorReport(state) {
  const times = [];
  for (const tool of state.tools) {
    times.push(tool.start);
    for (const [t] of tool.stages) times.push(t);
    if (tool.end !== null) times.push(tool.end);
  }
  for (const row of state.interventions || []) {
    for (const [, t, source] of row.stages) if (t !== null && source.startsWith("Skill executor")) times.push(t);
  }
  return times.length ? Math.max(...times) : null;
}

/**
 * How certain the dashboard is about execution: confirmed executing (a running attempt, the executor reporting within
 * report_overdue_s, nothing disagreeing), transitioning, uncertain (an overdue executor report, a mismatch, or an
 * unknown controller under a running attempt), confirmed stopped (the last attempt ended by its own result and nothing
 * executes) or unknown. Returns {state, severity, reasons, attempt, overdue}. Uncertain never claims the robot moved
 * or stopped; severity grows with how overdue the report is, missing monitoring, and an open stop on the attempt.
 */
export function executionCertainty(state, now, { monitorFresh = true } = {}) {
  const timing = { ...DEFAULT_TIMING, ...(state.timing || {}) };
  const check = consistency(state, now);
  const tool = check.attempt;
  const report = lastExecutorReport(state);
  const overdue = tool && report !== null ? Math.max(0, now - report) : null;
  const out = (s, severity, reasons = []) => ({ state: s, severity, reasons, attempt: tool, overdue, check });
  if (check.state === "transitioning") return out("transitioning", "info", [check.rule]);
  if (tool) {
    const reasons = [];
    if (overdue !== null && overdue > timing.report_overdue_s) reasons.push(`Executor report overdue (${Math.round(overdue)} s)`);
    if (check.state === "suspected" || check.state === "confirmed") reasons.push(check.rule);
    if (check.state === "unknown") reasons.push("Controller state unknown");
    if (!reasons.length) return out("confirmed_executing", "info");
    const stopOpen = (state.interventions || []).some((r) => r.tool === tool.id && OPEN.includes(r.state));
    let severity = "info";
    if (check.state === "confirmed" || (overdue !== null && overdue > 2 * timing.report_overdue_s) || !monitorFresh) severity = "warning";
    if ((check.state === "confirmed" && stopOpen) || (stopOpen && overdue !== null && overdue > timing.stop_s)) severity = "critical";
    return out("uncertain", severity, reasons);
  }
  if (check.state === "suspected") return out("uncertain", "warning", [check.rule]);
  const last = state.task ? state.tools.findLast((t) => t.start >= (state.task.start ?? -Infinity)) : null;
  if (last && last.end !== null) return out("confirmed_stopped", "info", [`${attemptLabel(last)} ended (${last.code || last.state})`]);
  return out("unknown", "info", ["No attempt reported in this task"]);
}

const SIGNAL_NAMES = { stall: "stall", unsafe: "unsafe situation", human_needed: "needs a person" };
const signalWord = (signal) => SIGNAL_NAMES[signal] || (signal?.startsWith("failure:") ? signal.slice(8).replace(/_/g, " ") : "anomaly");

/**
 * The compact incident card's words for intervention ROW: {title, status, tone, attempt, outcome, severity}, e.g.
 * "Pick stall detected" · "Action pending" · "Pick · Attempt 4 · Stall score 1.00" · "Abort requested · No
 * controller acknowledgement". Every phrase follows a recorded stage; nothing is claimed past the last one.
 */
export function incidentCard(state, row) {
  const incident = state.incidents.find((i) => i.id === row.incident) || null;
  const tool = toolById(state, row.tool);
  const skill = tool ? (SKILLS[tool.skill] || tool.skill) : row.skill ? (SKILLS[row.skill] || row.skill) : "Skill";
  const signal = incident?.observation?.signal || "";
  const title = row.initiator === "Human operator" ? `${skill} stop requested by the operator` : `${skill} ${signalWord(signal)} detected`;
  const status = { requested: ["warning", "Action pending"], executing: ["warning", "Stopping"], timed_out: ["danger", "Ack overdue"],
    expired: ["danger", "Request expired"],
    confirmed: ["success", "Skill stopped"], failed: ["danger", "Stop failed"], rejected: ["danger", "Stop rejected"],
    cancelled: ["neutral", "Request withdrawn"] }[row.state] || ["neutral", row.state];
  const risk = incident?.evidence?.risk;
  const attempt = [skill, tool ? `Attempt ${tool.number}` : "Attempt unknown",
    risk !== undefined && risk !== null ? `${signalLabel(signal)} ${Number(risk).toFixed(2)}` : ""].filter(Boolean).join(" · ");
  const observed = incident?.evidence?.first_observation || incident?.observation || null;
  const quality = observed?.frame_quality;
  const warning = quality === "invalid" || quality === "suspected_invalid" ? (observed.quality_relation === "detector_input"
    ? `Detector image ${quality.replace("_", " ")} at detection` : `Robot camera quality warning at detection (${quality.replace("_", " ")}) · detector input unchecked`) : "";
  const outcome = requestStatus(row, state.now ?? -Infinity).headline;
  return { title, status: status[1], tone: status[0], attempt, outcome, warning, severity: incident?.severity || "warning", incident, tool };
}

const riskWord = (signal) => (signal?.startsWith("failure:") ? `Failure (${signal.slice(8).replace(/_/g, " ")})`
  : signalLabel(signal).replace(/ score$/, ""));
const SCOPED = { task: "Task-level", robot: "Robot-level", environment: "Environment" };

/** The context a reading judges, in words: its attempt, its goal, a wider scope, or "Attempt unknown". */
export function contextLabel(state, reading) {
  const scope = reading?.scope || "skill_attempt";
  if (scope === "goal") return reading.scope_goal ? `Goal ${reading.scope_goal}` : "Goal unknown";
  if (SCOPED[scope]) return SCOPED[scope];
  const tool = toolById(state, reading?.tool);
  return tool ? attemptLabel(tool) : "Attempt unknown";
}

/**
 * The monitoring panel's one summary: the current assessment and the context it applies to, e.g. "Stall risk high for
 * 9 s · Pick attempt 4". An attempt-scoped risk of an attempt that is no longer running is said to be the previous
 * attempt's, never a fresh assessment of the current one. A confirmed execution mismatch comes before elevated (not
 * high) risk; then a required service, or normal. Open interventions are counted, not described again (the incident card says what they are); a stale monitor
 * and unusable input frames (at the newest observation) are said in any case. Returns {level, text}.
 */
export function contextSummary(state, status, now, windowStart, { gap = 3, system = null, check = null } = {}) {
  void windowStart;
  const fresh = status.state === "connected" && status.observationAge !== null && status.observationAge <= Math.max(3, status.staleAfter || 3);
  const notes = [];
  const task = state.task?.id;
  const open = (state.interventions || []).filter((r) => r.task === task && (OPEN.includes(r.state) || r.state === "failed" || r.state === "rejected"));
  if (open.length) notes.push(`${open.length} open intervention${open.length > 1 ? "s" : ""}`);
  if (state.readings.length && !fresh) {
    notes.push(status.state === "offline" ? "Monitor disconnected" : `Monitor stale (last observation ${Math.round(status.observationAge ?? 0)} s ago)`);
  }
  const newest = state.readings.findLast((r) => r.kind === "observation") || null;
  const quality = newest?.frame_quality;
  if (fresh && (quality === "invalid" || quality === "suspected_invalid")) notes.push(`Robot camera quality warning: ${quality.replace("_", " ")} frames`);
  const said = (level, text) => ({ level, text: [text, ...notes].join(" · ") });
  if (!state.readings.length && !check && !system?.required) {
    return said("unavailable", "No monitoring service connected: session, agent and skill events only");
  }
  const tool = runningAttempt(state);
  const run = fresh ? continuousRisk(state.readings, gap) : null;
  if (check && check.state === "confirmed" && !run?.high) return said("anomaly", `Execution state mismatch · ${check.rule}`);
  if (run) {
    // a stall is a label, not an alert: the monitor flags it and keeps watching, the skill is not stopped
    const stall = run.signal === "stall";
    const word = stall ? "Stalled" : `${riskWord(run.signal)} risk ${run.high ? "high" : "elevated"}`;
    const level = run.high && !stall ? "anomaly" : "elevated";
    const previous = run.scope === "skill_attempt" && run.tool && tool && run.tool !== tool.id;
    if (previous) {
      return said(level, `Previous ${contextLabel(state, run.latest)}: ${word.toLowerCase()} · Current ${attemptLabel(tool)}: assessment unknown`);
    }
    const lasting = run.seconds >= 1 ? ` for ${Math.round(run.seconds)} s` : "";
    return said(level, `${word}${lasting} · ${contextLabel(state, run.latest)}${stall ? " · monitoring continues" : ""}`);
  }
  if (system?.required) return said("elevated", `Required service unavailable: ${system.names.join(", ")}`);
  if (!state.readings.length) return said("unavailable", "No monitoring service connected: session, agent and skill events only");
  if (!fresh) return said("unknown", "Current behaviour assessment unavailable");
  const where = newest && newest.risk !== null ? ` · ${contextLabel(state, newest)}` : "";
  return said("normal", `Monitoring active · No significant concerns${where}`);
}

// the interventions track's standard symbols: [shape, tone]
export const MARKERS = {
  abort_requested: ["stop", "danger"], acknowledged: ["check", "running"], stopped: ["square", "danger"],
  recovery_started: ["diamond", "violet"], recovery_completed: ["diamond", "success"], human: ["person", "running"],
  failed: ["cross", "danger"], deadline: ["cross", "event-warning"],
};
const MARKER_TEXT = { abort_requested: "Abort requested", acknowledged: "Controller acknowledged", stopped: "Skill stopped",
  recovery_started: "Recovery started", recovery_completed: "Recovery completed", human: "Human intervention",
  failed: "Intervention failed", deadline: "Acknowledgement deadline passed" };

/**
 * The interventions track's events in [start, end]: each {id, t, kind, label, tool, intervention, incident} with
 * KIND a MARKERS key, only from recorded stages and incidents (a timeout is shown where the deadline passed).
 */
export function eventMarkers(state, start, end) {
  const out = [];
  const add = (id, t, kind, extra) => { if (t !== null && t >= start && t <= end) out.push({ id, t, kind, label: MARKER_TEXT[kind], ...extra }); };
  for (const r of state.interventions || []) {
    const extra = { tool: r.tool, intervention: r.id, incident: r.incident };
    for (const [stage, t, source, note] of r.stages) {
      const ok = /ended with OK/.test(note);
      const kind = { requested: r.initiator === "Human operator" ? "human" : "abort_requested", accepted: "acknowledged",
        confirmed: "stopped", failed: "failed", rejected: "failed", expired: "failed", retry_started: "recovery_started",
        retry_finished: ok ? "recovery_completed" : "failed" }[stage];
      if (kind) add(`${r.id}:${stage}`, t, kind, { ...extra, detail: `${source}${note ? ` · ${note}` : ""}` });
    }
    const status = requestStatus(r, state.now ?? -Infinity); // the deadline stays marked after a late acknowledgement
    if (["deadline_exceeded", "acknowledged_late"].includes(status.ack) && status.times.deadline !== null) {
      add(`${r.id}:deadline`, status.times.deadline, "deadline", { ...extra, detail: `No acknowledgement within ${r.ack_timeout} s${status.late ? " (acknowledged later)" : ""}` });
    }
  }
  for (const i of state.incidents) {
    const kind = { recovery_start: "recovery_started", recovery_ok: "recovery_completed", recovery_fail: "failed", replan: "recovery_started",
      stop: "human", reset: "human", pause: i.initiator === "Live monitor" ? null : "human", resume: "human" }[i.tag];
    if (kind) add(i.id, i.t, kind, { incident: i.id, tool: i.tool, detail: i.title });
  }
  return out.sort((a, b) => a.t - b.t);
}

/** MARKERS at x positions (X(t)) grouped where they would overlap (closer than PX): [{x, t, items}], none hidden. */
export function clusterMarkers(markers, x, px = 12) {
  const clusters = [];
  for (const m of markers) {
    const at = x(m.t);
    const last = clusters.at(-1);
    if (last && at - last.x1 < px) {
      last.items.push(m);
      last.x1 = at;
      last.x = (last.x0 + at) / 2;
    } else clusters.push({ x: at, x0: at, x1: at, t: m.t, items: [m] });
  }
  return clusters;
}
