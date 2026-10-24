// Execution consistency, attempt-level risk, system impact and retries: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MARKERS, attemptLabel, attemptRelation, clusterMarkers, consistency, contextSummary, continuousRisk, cumulativeRisk,
  eventMarkers, executionCertainty, incidentCard, previousAttemptRisk, retrySummary, systemSummary,
} from "../../src/homebody/ui/dashboard/static/js/execution.js";

const tool = (id, state = "running", extra = {}) => ({ id, skill: "pick", state, number: 1, start: 0, end: null, code: "", stages: [], ...extra });
const r = (t, level, risk, toolId, extra = {}) => ({ t, level, risk, threshold: 0.8, kind: "observation", tool: toolId, signal: "stall", ...extra });
const timing = { transition_s: 3, stale_s: 8, ack_timeout_s: 10, stop_s: 20 };
const live = { state: "connected", observationAge: 0.5, staleAfter: 3 };
const base = (extra = {}) => ({ timing, status: "Executing", status_since: 0, tools: [], interventions: [], incidents: [],
  readings: [], task: { id: "k1" }, controller: { state: "in_process" }, ...extra });

test("a brief disagreement is a transition, a lasting one a suspected mismatch", () => {
  const state = base({ status: "Thinking", status_since: 9, tools: [tool("t1", "running", { start: 8 })] });
  assert.equal(consistency(state, 10).state, "transitioning");
  const lasting = consistency(state, 20);
  assert.equal(lasting.state, "suspected");
  assert.match(lasting.rule, /Pick attempt 1 running · Executor: Thinking/);
  assert.equal(consistency(base({ status: "Executing", status_since: 0 }), 30).state, "suspected"); // executing, no attempt
  assert.equal(consistency(base({ tools: [tool("t1")] }), 30).state, "consistent");
});

test("a controller that stopped under a running attempt is a confirmed mismatch; stale status is unknown", () => {
  const running = { tools: [tool("t1")] };
  assert.equal(consistency(base({ ...running, controller: { state: "running", checked: 29 } }), 30).state, "consistent");
  const stopped = consistency(base({ ...running, controller: { state: "error", checked: 29 } }), 30);
  assert.equal(stopped.state, "confirmed");
  assert.equal(stopped.severity, "warning");
  const withStop = base({ ...running, controller: { state: "error", checked: 29 }, interventions: [{ tool: "t1", state: "requested", stages: [] }] });
  assert.equal(consistency(withStop, 30).severity, "critical"); // stop verification depends on it
  assert.equal(consistency(base({ ...running, controller: { state: "error", checked: 10 } }), 30).state, "unknown");
  const taken = base({ ...running, interventions: [{ tool: "t1", state: "executing", stages: [["accepted", 5, "x", ""]] }] });
  assert.equal(consistency(taken, 30).state, "suspected"); // stop taken 25 s ago, still running
});

test("continuous risk restarts per attempt and differs from cumulative risk", () => {
  const readings = [r(0, "elevated", 0.85, "t1"), r(1, "anomaly", 0.9, "t1"), r(2, "expected", 0.2, "t1"),
    r(10, "elevated", 0.85, "t2"), r(15, "anomaly", 0.95, "t2"), r(18, "anomaly", 0.97, "t2")];
  const run = continuousRisk(readings, 6);
  assert.deepEqual([run.seconds, run.tool, run.high], [8, "t2", true]);
  assert.equal(continuousRisk(readings, 3).seconds, 3); // the 5 s gap before 15 ends the run there
  assert.equal(cumulativeRisk(readings, 0, 20), 5); // seconds with elevated readings, not contiguous
  assert.equal(previousAttemptRisk(readings, "t2", 6).seconds, 1);
  assert.equal(continuousRisk([r(0, "elevated", 0.9, "t1"), r(5, "expected", 0.2, "t2")], 6), null);
});

test("the summary gives one conclusion by priority and still says when the monitor or camera is degraded", () => {
  const tools = [tool("t1", "failed", { code: "MONITOR_STOPPED", end: 5 }), tool("t2", "running", { number: 2, start: 6 })];
  const readings = [r(10, "elevated", 0.85, "t2"), r(28, "anomaly", 0.97, "t2")];
  assert.equal(contextSummary(base({ tools, readings }), live, 28.5, 0, { gap: 30 }).text, "Stall risk high · Pick attempt 2 · 18 s continuously");
  const incidents = [{ id: "x1", observation: { signal: "stall" }, evidence: { risk: 1 } }];
  const pending = { tools, readings, incidents, interventions: [{ id: "v1", tool: "t2", task: "k1", incident: "x1", initiator: "Live monitor", state: "requested", stages: [] }] };
  assert.equal(contextSummary(base(pending), live, 28.5, 0).text, "Pick stall detected · Abort requested · Awaiting controller");
  const stopped = { tools, readings: [r(9, "expected", 0.1, "t2")], incidents,
    interventions: [{ id: "v1", tool: "t1", task: "k1", incident: "x1", initiator: "Live monitor", state: "confirmed", stages: [["retry_started", 6, "", ""]] }] };
  assert.equal(contextSummary(base(stopped), live, 9.2, 0).text, "Pick stall detected · Skill stopped · Retry started");
  assert.equal(contextSummary(base({ tools, readings: [r(9, "expected", 0.1, "t2")] }), live, 9.2, 0).text, "Monitoring active · No significant concerns");
  const stale = { state: "stale", observationAge: 30, staleAfter: 3 };
  assert.equal(contextSummary(base({ tools, readings: [r(1, "anomaly", 1, "t2")] }), stale, 31, 0).text,
    "Current behaviour assessment unavailable · Monitor stale (last observation 30 s ago)");
  const camera = base({ tools, readings: [r(9, "expected", 0.1, "t2")], frame_quality: { state: "invalid" } });
  assert.equal(contextSummary(camera, live, 9.2, 0).text, "Monitoring active · No significant concerns · Visual input degraded");
  const mismatch = { state: "confirmed", rule: "Behavior Tree: Pick attempt 2 running · Controller: error" };
  assert.match(contextSummary(base({ tools, readings: [r(9, "elevated", 0.5, "t2")] }), live, 9.2, 0, { check: mismatch }).text, /^Execution state mismatch/);
  assert.equal(contextSummary(base({ tools, readings: [r(9, "anomaly", 0.9, "t2")] }), live, 9.2, 0).text, "Stall risk high · Pick attempt 2"); // one reading: no duration
});

test("execution certainty: confirmed, overdue (uncertain, never stopped), and confirmed stopped", () => {
  const running = base({ tools: [tool("t1", "running", { stages: [[25, "Approaching"]] })], controller: { state: "running", checked: 29 } });
  assert.equal(executionCertainty(running, 30).state, "confirmed_executing");
  const overdue = executionCertainty(running, 60);
  assert.deepEqual([overdue.state, overdue.severity], ["uncertain", "warning"]); // 35 s > 2 × 15 s
  assert.match(overdue.reasons[0], /Executor report overdue \(35 s\)/);
  assert.equal(executionCertainty(running, 45).severity, "info"); // overdue, but briefly and the monitor watches
  assert.equal(executionCertainty(running, 45, { monitorFresh: false }).severity, "warning");
  const ended = base({ status: "Thinking", tools: [tool("t1", "failed", { code: "NO_GRASP", end: 20 })] });
  assert.equal(executionCertainty(ended, 30).state, "confirmed_stopped");
  assert.equal(executionCertainty(base({ status: "Ready" }), 30).state, "unknown");
});

test("the compact card says the attempt and the outcome the stages establish", () => {
  const state = base({ tools: [tool("t4", "running", { number: 4 })], incidents: [{ id: "x1", severity: "critical", observation: { signal: "stall" }, evidence: { risk: 1 } }] });
  const card = incidentCard(state, { tool: "t4", incident: "x1", initiator: "Live monitor", state: "timed_out", stages: [] });
  assert.deepEqual([card.title, card.status, card.attempt, card.outcome],
    ["Pick stall detected", "Action pending", "Pick · Attempt 4 · Stall probability 1.00", "Abort requested · No controller acknowledgement"]);
  assert.equal(incidentCard(state, { tool: "t4", incident: "x1", initiator: "Live monitor", state: "executing", stages: [] }).outcome,
    "Controller acknowledged · Skill not yet stopped"); // acknowledgement is not a stop
  assert.equal(incidentCard(state, { tool: null, initiator: "Human operator", state: "requested", stages: [] }).attempt, "Skill · Attempt unknown");
});

test("event markers use the standard symbols and overlapping ones cluster without hiding any", () => {
  const rows = [{ id: "v1", tool: "t1", incident: "x1", initiator: "Live monitor", state: "confirmed", ack_timeout: 10, stages: [
    ["requested", 10, "Live monitor", ""], ["accepted", 30, "Skill executor", ""], ["confirmed", 31, "Skill executor", ""],
    ["retry_started", 33, "Plan goal", ""], ["retry_finished", 50, "Skill executor", "Attempt 2 ended with OK [t2]"]] },
    { id: "v2", tool: "t3", initiator: "Human operator", state: "timed_out", ack_timeout: 10, stages: [["requested", 60, "Human operator", ""]] }];
  const markers = eventMarkers(base({ interventions: rows, incidents: [{ id: "x9", t: 40, tag: "recovery_start", title: "Recovery" }] }), 0, 100);
  assert.deepEqual(markers.map((m) => m.kind), ["abort_requested", "acknowledged", "stopped", "recovery_started", "recovery_started",
    "recovery_completed", "human", "failed"]);
  assert.deepEqual(MARKERS.abort_requested, ["stop", "danger"]);
  const clusters = clusterMarkers(markers, (t) => t * 4);
  assert.deepEqual(clusters.map((c) => c.items.length), [1, 3, 1, 1, 1, 1]); // 30, 31 and 33 s overlap at 4 px/s
  assert.equal(clusters.reduce((n, c) => n + c.items.length, 0), markers.length);
});

test("system degradation says whether the task depends on it", () => {
  const rows = [{ name: "h", health: "healthy", impact: "critical" }, { name: "cart", health: "healthy", impact: "critical" },
    { name: "kitchen", health: "unavailable", impact: "noncritical" }];
  assert.deepEqual(systemSummary(rows), { text: "System · 1 unused service unavailable", tone: "", degraded: false, required: false, names: [] });
  rows[1].health = "unavailable";
  assert.equal(systemSummary(rows).text, "System degraded · Required service unavailable");
  rows[1].health = "healthy";
  rows[2].impact = "unknown";
  assert.equal(systemSummary(rows).text, "System degraded · impact unknown");
});

test("retries summarize the goal's attempts, and attempts relate to the one running", () => {
  const tools = [tool("t1", "failed", { code: "MONITOR_STOPPED", number: 1 }), tool("t2", "failed", { code: "NO_GRASP", number: 2 }),
    tool("t3", "failed", { code: "STALE_OBSERVATION", number: 3 }), tool("t4", "running", { number: 4 })];
  assert.equal(retrySummary(tools, 8), "Attempt 4/8 · 3 previous attempts · 1 monitor stop · 2 other failures");
  assert.equal(attemptLabel(tools[3]), "Pick attempt 4");
  assert.equal(attemptLabel(null), "Unknown attempt");
  const state = { tools };
  assert.deepEqual(["t4", "t1", null].map((id) => attemptRelation(state, id)), ["current", "previous", "unknown"]);
});
