// The dashboard's state logic: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  gapFor, interventionMarker, liveEnd, momentAt, monitorStatus, observationSegments, primaryAction, riskBuckets,
  riskSources,
} from "../../src/homebody/ui/dashboard/static/js/model.js";

const obs = (t, level, extra = {}) => ({ t, level, kind: "observation", during_task: true, risk: null, threshold: null,
  source: "force", synthetic: false, ...extra });

test("one reading stands for three of its intervals, at least 1.5 s", () => {
  assert.equal(gapFor([obs(0, "expected"), obs(1, "expected"), obs(2, "expected")]), 3);
  assert.equal(gapFor([obs(0, "expected"), obs(0.1, "expected"), obs(0.2, "expected")]), 1.5);
  assert.equal(gapFor([]), 1.5);
});

test("the monitor is connected, stale, offline or unknown, and says when its data is synthetic", () => {
  const readings = [obs(0, "expected"), obs(1, "expected", { synthetic: true, risk: 0.2 })];
  assert.equal(monitorStatus(readings, [], 1.5, true).state, "connected");
  assert.equal(monitorStatus(readings, [], 9, true).state, "stale");
  assert.equal(monitorStatus(readings, [], 1.5, false).state, "offline");
  assert.equal(monitorStatus([], [], 1.5, true).state, "unknown");
  const status = monitorStatus(readings, [{ kind: "monitor", state: "active" }], 1.5, true);
  assert.equal(status.synthetic, true);
  assert.equal(status.risk.risk, 0.2);
  assert.equal(status.anomalies, 1);
});

test("observations merge into runs of one level and gaps inside a task are missing, not normal", () => {
  const readings = [0, 1, 2].map((t) => obs(t, "expected")).concat([obs(3, "anomaly"), obs(10, "expected")]);
  const segments = observationSegments(readings, [{ start: 0, end: 12 }], 0, 12);
  assert.deepEqual(segments.map((s) => [s.level, s.start, s.end]), [
    ["expected", 0, 3], ["anomaly", 3, 6], ["missing", 6, 10], ["expected", 10, 12]]);
  assert.equal(segments[0].readings.length, 3);
});

test("heartbeats and readings outside tasks leave the execution track empty", () => {
  const readings = [obs(0, "expected", { kind: "heartbeat" }), obs(1, "expected", { during_task: false })];
  assert.deepEqual(observationSegments(readings, [], 0, 60), []);
});

test("risk is one block per second at its peak, classed by the monitor's thresholds, missing when unsampled", () => {
  const points = [
    obs(0.1, "expected", { risk: 0.2, threshold: 0.7 }), obs(0.6, "expected", { risk: 0.9, threshold: 0.7 }), // spike kept
    obs(1.2, "elevated", { risk: 0.5, threshold: 0.7 }),
    obs(2.5, "expected", { risk: 0.55, threshold: 0.7, elevated_threshold: 0.5 }),
    obs(3.5, "expected", { risk: 0.1, threshold: 0.7 }),
  ];
  const blocks = riskBuckets(points, "force", [{ start: 0, end: 6 }], 0, 6);
  assert.deepEqual(blocks.map((b) => [b.start, b.state, b.max, b.count]), [
    [0, "high", 0.9, 2], [1, "elevated", 0.5, 1], [2, "elevated", 0.55, 1], [3, "low", 0.1, 1],
    [4, "missing", null, 0], [5, "missing", null, 0]]);
  assert.deepEqual(riskBuckets(points, "force", [], 5, 9), []); // outside every task: no blocks at all
  assert.deepEqual(riskSources([...points, obs(5, "expected", { risk: 0.1, source: "pose" })]), ["force", "pose"]);
});

test("detections are not interventions; actions have their own marker", () => {
  assert.equal(interventionMarker({ kind: "monitor", tag: "" }), null);
  assert.equal(interventionMarker({ kind: "skill", tag: "" }), null);
  assert.deepEqual(interventionMarker({ kind: "intervention", tag: "stop" }), { shape: "square", tone: "running" });
  assert.deepEqual(interventionMarker({ kind: "recovery", tag: "recovery_fail" }), { shape: "diamond", tone: "danger" });
});

const ready = (extra = {}) => ({ status: "Ready", initial_task: "", task: null,
  controls: { busy: false, accepting: true, reset_pending: false }, ...extra });

test("the primary action says what it will start, and offers nothing while a task runs", () => {
  assert.deepEqual(primaryAction(ready(), "  Pick the jar ", true).text, "Pick the jar");
  assert.equal(primaryAction(ready({ initial_task: "Tidy" }), "", true).title, "Start: Tidy");
  assert.equal(primaryAction(ready(), "", true).enabled, false);
  const done = ready({ task: { text: "Tidy", end: 5, outcome: "model_done" } });
  assert.equal(primaryAction(done, "", true).label, "Run again");
  const failed = ready({ task: { text: "Tidy", end: 5, outcome: "provider_error" } });
  assert.equal(primaryAction(failed, "", true).label, "Retry");
  assert.equal(primaryAction(ready({ controls: { busy: true } }), "x", true).kind, "none");
  assert.equal(primaryAction(ready({ status: "Reset required" }), "", true).kind, "reset");
  assert.equal(primaryAction(ready(), "x", false).kind, "none");
});

test("a moment shows what covered it and never a stale reading", () => {
  const state = {
    intervals: [{ track: "goal", start: 0, end: null, label: "g" }, { track: "skill", start: 2, end: 4, label: "pick" }],
    decisions: [{ t: 1, skill: "pick" }], incidents: [{ t: 3.5, title: "x" }],
    readings: [obs(0, "expected"), obs(1, "expected"), obs(3, "elevated", { risk: 0.6 })],
  };
  const at3 = momentAt(state, 3, 10);
  assert.equal(at3.skill.label, "pick");
  assert.equal(at3.reading.level, "elevated");
  assert.equal(at3.incidents.length, 1);
  const at9 = momentAt(state, 9.5, 10); // readings every 2 s stand for 6 s
  assert.equal(at9.reading, null);
  assert.equal(at9.skill, null);
  assert.equal(at9.goal.label, "g");
});

test("risk outside a task or from heartbeats is not charted", () => {
  const idle = [obs(0, "expected", { risk: 0.9, during_task: false }), obs(1, "expected", { risk: 0.9, kind: "heartbeat" })];
  assert.deepEqual(riskSources(idle), []);
  assert.deepEqual(riskBuckets(idle, "force", [], 0, 10), []);
});

test("the timeline follows a running task and holds once it stops or the session is lost", () => {
  const running = { task: { end: null }, controls: { busy: true } };
  assert.deepEqual(liveEnd(running, 100, true, 99), { end: 100, held: false, reason: "" });
  const ended = { task: { end: 50 }, controls: { busy: false } };
  assert.deepEqual(liveEnd(ended, 100, true, 99), { end: 52, held: true, reason: "ended" });
  assert.equal(liveEnd(ended, 51, true, 51).end, 51); // never ahead of the clock
  assert.deepEqual(liveEnd(running, 100, false, 90), { end: 90, held: true, reason: "disconnected" });
  assert.equal(liveEnd({ task: null, controls: { busy: false } }, 100, true, 99).held, false);
  assert.deepEqual(liveEnd(ended, 100, true, 99, 50), { end: 100, held: false, reason: "" }); // returned to live
});

test("the last observation's age comes from observations, not heartbeats", () => {
  const readings = [obs(1, "expected"), obs(9, "expected", { kind: "heartbeat", received: 9 })];
  const status = monitorStatus(readings, [], 10, true);
  assert.equal(status.state, "connected");
  assert.equal(status.observationAge, 9);
});
