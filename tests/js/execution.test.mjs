// Execution consistency, attempt-level risk, system impact and retries: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  attemptLabel, attemptRelation, consistency, contextSummary, continuousRisk, cumulativeRisk, previousAttemptRisk,
  retrySummary, systemSummary,
} from "../../src/homebody/ui/dashboard/static/js/execution.js";

const tool = (id, state = "running", extra = {}) => ({ id, skill: "pick", state, number: 1, start: 0, end: null, code: "", ...extra });
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

test("the summary names the attempt, its response, and never calls cumulative risk sustained", () => {
  const tools = [tool("t1", "failed", { code: "MONITOR_STOPPED" }), tool("t2", "running", { number: 2 })];
  const readings = [r(10, "elevated", 0.85, "t2"), r(28, "anomaly", 0.97, "t2")];
  const text = contextSummary(base({ tools, readings }), live, 28.5, 0, { gap: 30 }).text;
  assert.equal(text, "Stall probability high for 18 s continuously (Pick attempt 2) · Monitoring");
  const open = { tools, readings, incidents: [{ id: "x1", observation: { signal: "stall" } }],
    interventions: [{ id: "v1", tool: "t2", task: "k1", incident: "x1", initiator: "Live monitor", state: "requested" }] };
  assert.equal(contextSummary(base(open), live, 28.5, 0).text, "Stall detected on Pick attempt 2 · Abort requested · Awaiting executor");
  const calm = { tools, readings: [r(1, "elevated", 0.85, "t2"), r(2, "elevated", 0.85, "t2"), r(9, "expected", 0.1, "t2")] };
  assert.equal(contextSummary(base(calm), live, 9.2, 0, { gap: 1 }).text, "Current condition normal · elevated risk for 2 s of the last 60 s");
  const after = { tools, readings: [r(1, "elevated", 0.85, "t1"), r(5, "anomaly", 0.95, "t1"), r(9, "expected", 0.1, "t2")] };
  assert.match(contextSummary(base(after), live, 9.2, 0, { gap: 6 }).text, /^Previous Pick attempt 1: stall probability elevated for 4 s · New attempt running$/);
  const mismatch = { state: "suspected", severity: "warning", rule: "Behavior Tree: Pick attempt 2 running · Executor: Thinking" };
  assert.match(contextSummary(base({ tools, readings: [r(9, "expected", 0.1, "t2")] }), live, 9.2, 0, { check: mismatch }).text, /^Execution state mismatch/);
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
