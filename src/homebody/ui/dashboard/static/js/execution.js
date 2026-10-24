// Execution, risk and system state at the level of one skill attempt: pure, so node --test checks it (tests/js).
// Attempts are the store's tool calls, each with its own identity (execution); readings, incidents and requests name
// the attempt they belong to (their "tool"), so nothing here correlates by skill name or by being close in time.
import { OPEN, signalLabel, stateText } from "./interventions.js";

const SKILLS = { navigate: "Navigate", pick: "Pick", place: "Place", call_coder: "Coding assistant" };
const DEFAULT_TIMING = { transition_s: 3, stale_s: 8, ack_timeout_s: 10, stop_s: 20 };

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

/** The newest uninterrupted run of elevated-or-high readings of one attempt: {seconds, high, tool, start, end};
 * a gap longer than GAP, a calmer reading or another attempt ends it. */
export function continuousRisk(readings, gap) {
  const rs = scored(readings);
  const latest = rs.at(-1);
  if (!latest || !elevated(latest)) return null;
  let start = latest;
  for (let i = rs.length - 2; i >= 0; i -= 1) {
    const r = rs[i];
    if (r.tool !== latest.tool || !elevated(r) || start.t - r.t > gap) break;
    start = r;
  }
  const run = rs.filter((r) => r.t >= start.t && r.tool === latest.tool);
  return { seconds: latest.t - start.t, high: run.every(high), tool: latest.tool, start: start.t, end: latest.t, signal: latest.signal };
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

const RESPONSE = { requested: "Abort requested · Awaiting executor", executing: "Executor stopping the skill",
  timed_out: "Abort requested · No executor acknowledgement" };

/**
 * The monitoring panel's summary, by priority: the current intervention, an anomaly or high risk (continuous on
 * this attempt, else cumulative in the window), an execution-state mismatch, a relevant system degradation, then
 * unknown or normal. Every phrase comes from a recorded event. Returns {level, text}.
 */
export function contextSummary(state, status, now, windowStart, { gap = 3, system = null, check = null } = {}) {
  const fresh = status.state === "connected" && status.observationAge !== null && status.observationAge <= Math.max(3, status.staleAfter || 3);
  const tool = runningAttempt(state);
  const open = (state.interventions || []).findLast((r) => OPEN.includes(r.state) && r.task === state.task?.id);
  if (open) {
    const what = state.incidents.find((i) => i.id === open.incident)?.observation?.signal;
    const head = open.initiator === "Human operator" ? "Operator stop" : `${what ? signalLabel(what).replace(" probability", "") : "Anomaly"} detected`;
    return { level: "intervention", text: `${head} on ${attemptLabel(toolById(state, open.tool))} · ${RESPONSE[open.state] || stateText(open.state)[1]}` };
  }
  if (!state.readings.length && !check && !system?.required) {
    return { level: "unavailable", text: "No monitoring service connected: session, agent and skill events only" };
  }
  const run = fresh ? continuousRisk(state.readings, gap) : null;
  if (run) {
    const runTool = toolById(state, run.tool);
    const where = runTool ? ` (${attemptLabel(runTool)})` : "";
    const response = run.tool && run.tool === tool?.id ? "Monitoring" : runTool ? "That attempt has ended" : "Attempt unknown";
    const label = signalLabel(run.signal);
    return { level: run.high ? "anomaly" : "elevated",
      text: `${label} ${run.high ? "high" : "elevated"} for ${Math.round(run.seconds)} s continuously${where} · ${response}` };
  }
  if (check && (check.state === "suspected" || check.state === "confirmed")) {
    return { level: check.severity === "critical" ? "anomaly" : "elevated", text: `Execution state mismatch · ${check.rule}` };
  }
  if (system?.required) return { level: "elevated", text: `Required service unavailable: ${system.names.join(", ")}` };
  if (!state.readings.length) return { level: "unavailable", text: "No monitoring service connected: session, agent and skill events only" };
  if (!fresh) return { level: "unknown", text: `Monitoring ${status.state} · Current behaviour assessment unavailable` };
  const total = cumulativeRisk(state.readings, windowStart, now);
  const previous = tool ? previousAttemptRisk(state.readings, tool.id, gap) : null;
  if (previous && previous.seconds >= 1) {
    return { level: "normal", text: `Previous ${attemptLabel(toolById(state, previous.tool))}: ${signalLabel(previous.signal).toLowerCase()} elevated for ${Math.round(previous.seconds)} s · New attempt running` };
  }
  if (total) return { level: "normal", text: `Current condition normal · elevated risk for ${total} s of the last 60 s` };
  return { level: "normal", text: "Monitoring active · No anomalies detected in the last minute" };
}
