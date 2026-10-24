// V8: a monitor result apart from how far its evidence holds, and one selection that every panel follows: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import { assessment, inputWarning, scopeText } from "../../src/homebody/ui/dashboard/static/js/assessment.js";
import { contextSummary, continuousRisk } from "../../src/homebody/ui/dashboard/static/js/execution.js";
import { steps } from "../../src/homebody/ui/dashboard/static/js/interventions.js";
import { recordedFrame, resolveSelection, stateAt, windowFor } from "../../src/homebody/ui/dashboard/static/js/selection.js";

const observation = (extra = {}) => ({ id: "o7", t: 100, level: "anomaly", kind: "observation", risk: 1, threshold: 0.8,
  confidence: 0.7, signal: "stall", source: "Live monitor", tool: "t4", frame_quality: "valid", scope: "skill_attempt",
  scope_goal: "g2", calibration: "uncalibrated", reliability: "unknown", sim_time: 50.2,
  semantics: "Probability the model answered 'no change' (uncalibrated)", quality_source: "Dashboard check at sim 50.20 s", ...extra });
const row = (key, rows) => rows.find((r) => r.key === key);

test("a high stall score with valid frames shows the score and a valid input", () => {
  const rows = assessment(observation());
  assert.equal(row("output", rows).label, "Stall score");
  assert.equal(row("output", rows).value, "1.00");
  assert.equal(row("threshold", rows).value, "0.80");
  assert.equal(row("input", rows).value, "Valid");
  assert.equal(inputWarning(observation()), null);
});

test("suspected invalid frames keep the score and add a warning with its reason", () => {
  const o = observation({ frame_quality: "suspected_invalid", quality_reason: "Nearly uniform frame (spread 1.2 < 2)" });
  const rows = assessment(o);
  assert.equal(row("output", rows).value, "1.00"); // the model's output is never altered
  assert.equal(row("input", rows).value, "Suspected invalid");
  assert.equal(row("input", rows).tone, "warning");
  assert.match(inputWarning(o).text, /suspected invalid/i);
  assert.match(inputWarning(o).reason, /Nearly uniform/);
});

test("uncalibrated confidence says so; unavailable reliability is Unknown, never confirmed", () => {
  const rows = assessment(observation());
  assert.equal(row("confidence", rows).value, "0.70 · Uncalibrated");
  assert.equal(row("reliability", rows).value, "Unknown");
  assert.equal(row("confidence", assessment(observation({ confidence: null }))).value, "Not reported");
  assert.equal(row("confidence", assessment(observation({ calibration: undefined }))).value, "0.70 · Calibration unknown");
});

test("a historical incident shows the quality at its observation, not the camera's quality now", () => {
  const then = observation({ frame_quality: "valid" });
  const rows = assessment(then, { current: { state: "invalid" } });
  assert.equal(row("input", rows).value, "Valid");
  const old = assessment(observation({ quality_source: "" })); // a record from before observation-time checks
  assert.match(row("input", old).note, /when the reading arrived/);
});

test("scope names the attempt, the goal or Unknown, never a guess", () => {
  const state = { tools: [{ id: "t4", skill: "pick", number: 4 }] };
  assert.equal(scopeText(observation(), state), "Pick attempt 4");
  assert.equal(scopeText(observation({ tool: null }), state), "A skill attempt the dashboard could not match");
  assert.equal(scopeText(observation({ scope: "goal", scope_goal: "g2" }), state), "Goal g2");
  assert.equal(scopeText(observation({ scope: "unknown" }), state), "Unknown");
  assert.equal(scopeText({}, state), "Unknown");
});

const history = [[1, 40], [2, 41], [3, 42], [4, 100.4], [5, 101.4]];
const state = (extra = {}) => ({
  now: 300, keep_s: 600, history, history_period: 1,
  task: { id: "k1", start: 0 }, tree: { nodes: { "goal:g2": { id: "goal:g2", children: ["tool:t4"] }, "tool:t4": { id: "tool:t4", children: [] } } },
  tools: [{ id: "t4", skill: "pick", number: 4, goal: "g2", start: 90, end: 112, state: "failed", code: "MONITOR_STOPPED", execution: "k1/call-7" }],
  incidents: [{ id: "x1", t: 100, tool: "t4", node: "g2", observation: observation({ id: "o9", t: 104 }),
    evidence: { first_observation: observation() } }],
  interventions: [{ id: "v1", tool: "t4", incident: "x1", task: "k1", t: 100,
    stages: [["detected", 100, "Monitor", ""], ["requested", 100.2, "Live monitor", ""]] }],
  ...extra,
});

test("selecting an incident resolves its attempt, node, observation and recorded frame by identity", () => {
  const s = resolveSelection(state(), { intervention: "v1" }, 300);
  assert.equal(s.tool.id, "t4");
  assert.equal(s.node, "tool:t4");
  assert.equal(s.goal, "goal:g2");
  assert.equal(s.observation.id, "o7"); // the detecting observation, not the newest one of the episode
  assert.equal(s.t, 100);
  assert.deepEqual([s.frame.id, s.frame.delta.toFixed(1)], [4, "0.4"]);
  assert.equal(s.historical, true);
});

test("an incident older than 60 s opens a window around it; a recent one keeps the live window", () => {
  assert.deepEqual(windowFor(100, 300, 60), { start: 70, end: 130, historical: true });
  assert.deepEqual(windowFor(280, 300, 60), { start: 240, end: 300, historical: false });
  assert.deepEqual(windowFor(295, 300, 60), { start: 240, end: 300, historical: false }); // never past now
});

test("camera evidence is the recorded frame near the moment, or explicitly unavailable", () => {
  assert.equal(recordedFrame(history, 101.5, 2).id, 5);
  assert.equal(recordedFrame(history, 70, 2), null); // no stored frame near: not the nearest one 28 s away
  assert.equal(recordedFrame([], 70, 2), null);
  const s = resolveSelection(state({ history: [] }), { intervention: "v1" }, 300);
  assert.equal(s.frame, null);
  assert.ok(s.missing.includes("Camera evidence unavailable for this timestamp"));
});

test("missing provenance is Unknown, and data older than the kept history says it expired", () => {
  const s = resolveSelection(state({ interventions: [{ id: "v2", tool: null, incident: null, stages: [["requested", 100, "Human operator", ""]] }] }),
    { intervention: "v2" }, 300);
  assert.equal(s.tool, null);
  assert.ok(s.missing.some((m) => /Attempt unknown/.test(m)));
  assert.ok(s.missing.some((m) => /No monitor observation/.test(m)));
  const old = resolveSelection(state(), { intervention: "v1" }, 900);
  assert.ok(old.missing.some((m) => /expired/.test(m)));
});

test("a node's state at the selected moment is told apart from its state now", () => {
  const t = state().tools[0];
  assert.equal(stateAt(t, 80), "Not started");
  assert.equal(stateAt(t, 100), "Running");
  assert.equal(stateAt(t, 120), "Failed (MONITOR_STOPPED)");
  assert.equal(stateAt(null, 100), "Unknown");
});

test("out-of-order stage events are put back in lifecycle order", () => {
  const arrived = { id: "v1", initiator: "Live monitor", state: "confirmed", ack_timeout: 10, stages: [
    ["confirmed", 72, "Skill executor", "The skill ended as asked"], ["accepted", 70, "Skill executor", ""],
    ["requested", 10, "Live monitor", ""], ["detected", 10, "Monitor", ""]] };
  assert.deepEqual(steps(arrived).slice(0, 5).map((s) => s.state), ["done", "unknown", "done", "done", "done"]);
});

const tool = (id, skill, number, start, end = null) => ({ id, skill, number, start, end, state: end === null ? "running" : "failed", stages: [] });
const r = (t, level, risk, toolId, extra = {}) => ({ t, level, risk, threshold: 0.8, kind: "observation", tool: toolId, signal: "stall",
  scope: "skill_attempt", frame_quality: "valid", ...extra });
const base = (extra = {}) => ({ timing: {}, status: "Executing", status_since: 0, tools: [], interventions: [], incidents: [],
  readings: [], task: { id: "k1" }, controller: { state: "in_process" }, ...extra });
const live = { state: "connected", observationAge: 0.5, staleAfter: 3 };

test("risk names its context; a previous attempt's risk is not a fresh assessment of the current one", () => {
  const tools = [tool("t3", "pick", 3, 0, 20), tool("t4", "navigate", 4, 21)];
  const current = base({ tools, readings: [r(10, "anomaly", 0.95, "t4"), r(19, "anomaly", 1, "t4")] });
  assert.equal(contextSummary(current, live, 19.5, 0, { gap: 30 }).text, "Stall risk high for 9 s · Navigate attempt 4");
  const previous = base({ tools, readings: [r(10, "anomaly", 0.95, "t3"), r(19, "anomaly", 1, "t3")] });
  assert.equal(contextSummary(previous, live, 19.5, 0, { gap: 30 }).text,
    "Previous Pick attempt 3: stall risk high · Current Navigate attempt 4: assessment unknown");
  const between = base({ tools, readings: [r(19, "anomaly", 1, null, { scope: "unknown" })] });
  assert.equal(contextSummary(between, live, 19.5, 0).text, "Stall risk high · Attempt unknown");
});

test("goal- or task-scoped risk continues across attempts; attempt-scoped risk restarts", () => {
  const goal = [r(0, "anomaly", 0.9, "t3", { scope: "goal", scope_goal: "g2" }), r(5, "anomaly", 0.9, "t4", { scope: "goal", scope_goal: "g2" })];
  assert.equal(continuousRisk(goal, 6).seconds, 5);
  const task = [r(0, "anomaly", 0.9, "t3", { scope: "task" }), r(5, "anomaly", 0.9, "t4", { scope: "task" })];
  assert.equal(continuousRisk(task, 6).seconds, 5);
  const attempt = [r(0, "anomaly", 0.9, "t3"), r(5, "anomaly", 0.9, "t4")];
  assert.equal(continuousRisk(attempt, 6).seconds, 0);
  const tools = [tool("t3", "pick", 3, 0, 4), tool("t4", "pick", 4, 4.5)];
  assert.equal(contextSummary(base({ tools, readings: goal }), live, 5.5, 0, { gap: 6 }).text, "Stall risk high for 5 s · Goal g2");
});

test("the summary is the current assessment, with an open intervention counted, not described again", () => {
  const tools = [tool("t4", "pick", 4, 0)];
  const pending = base({ tools, readings: [r(9, "anomaly", 1, "t4")], incidents: [{ id: "x1", observation: { signal: "stall" }, evidence: { risk: 1 } }],
    interventions: [{ id: "v1", tool: "t4", task: "k1", incident: "x1", initiator: "Live monitor", state: "requested", stages: [] }] });
  assert.equal(contextSummary(pending, live, 9.5, 0).text, "Stall risk high · Pick attempt 4 · 1 open intervention");
  const bad = base({ tools, readings: [r(9, "expected", 0.1, "t4", { frame_quality: "invalid" })] });
  assert.equal(contextSummary(bad, live, 9.5, 0).text, "Monitoring active · No significant concerns · Pick attempt 4 · Robot camera quality warning: invalid frames");
});

test("an incident recorded before its detecting observation was kept says it shows the newest one", () => {
  const s = resolveSelection(state(), { intervention: "v1" }, 300);
  assert.equal(s.detecting, true);
  const old = state({ incidents: [{ id: "x1", t: 100, tool: "t4", node: "g2", observation: observation({ id: "o9" }), evidence: {} }] });
  const legacy = resolveSelection(old, { intervention: "v1" }, 300);
  assert.deepEqual([legacy.detecting, legacy.observation.id], [false, "o9"]);
});

test("a reading without a scope is its attempt's everywhere, and Unknown without an attempt", () => {
  const st = { tools: [{ id: "t4", skill: "pick", number: 4 }] };
  assert.equal(scopeText({ tool: "t4" }, st), "Pick attempt 4");
  assert.equal(scopeText({ tool: null }, st), "Unknown");
});
