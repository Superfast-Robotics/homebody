// V9: request states on separate axes, evidence attributed to its source, and a selection that holds still while live
// events arrive: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import { assessment, inputWarning } from "../../src/homebody/ui/dashboard/static/js/assessment.js";
import { cameraEvidence, qualityAttribution } from "../../src/homebody/ui/dashboard/static/js/evidence.js";
import { requestStatus, steps } from "../../src/homebody/ui/dashboard/static/js/interventions.js";
import { resolveSelection } from "../../src/homebody/ui/dashboard/static/js/selection.js";

const requested = ["requested", 100, "Live monitor", ""];
const row = (state, stages, extra = {}) => ({ id: "v1", initiator: "Live monitor", state, ack_timeout: 10, physical: "unverified",
  stages: [["detected", 100, "Monitor", ""], requested, ...stages], ...extra });

test("a passed acknowledgement deadline on a live request is overdue and still pending", () => {
  const s = requestStatus(row("timed_out", []), 115);
  assert.deepEqual([s.ack, s.command, s.physical], ["deadline_exceeded", "pending", "unverified"]);
  assert.equal(s.headline, "Acknowledgement overdue · Still awaiting controller");
  assert.deepEqual([s.times.requested, s.times.deadline, s.times.elapsed], [100, 110, 15]);
  assert.equal(steps(row("timed_out", []), 115)[3].state, "overdue");
  const waiting = requestStatus(row("requested", []), 104);
  assert.deepEqual([waiting.ack, waiting.headline], ["awaiting", "Abort requested · Awaiting controller acknowledgement"]);
});

test("an expired request is expired, not merely overdue", () => {
  const s = requestStatus(row("expired", [["expired", 120, "Controller", ""]]), 200);
  assert.deepEqual([s.ack, s.command, s.headline], ["deadline_exceeded", "expired", "Abort request expired · No controller acknowledgement"]);
  assert.equal(s.times.ended, 120);
});

test("a late acknowledgement updates the lifecycle and keeps the exceeded deadline", () => {
  const late = row("executing", [["accepted", 113, "Skill executor (session)", "taken"]]);
  const s = requestStatus(late, 114);
  assert.deepEqual([s.ack, s.command, s.late], ["acknowledged_late", "active", true]);
  assert.equal(s.headline, "Controller acknowledged after deadline · Physical stop unverified");
  const ack = steps(late, 114)[3];
  assert.equal(ack.state, "done");
  assert.match(ack.note, /after the deadline 10 s after the request/);
  const afterEnd = requestStatus(row("expired", [["expired", 112, "Controller", ""], ["accepted", 115, "Skill executor", ""]]), 120);
  assert.equal(afterEnd.headline, "Abort request expired · Controller acknowledged after it ended");
});

test("acceptance never reads as a physical stop", () => {
  const s = requestStatus(row("executing", [["accepted", 103, "Skill executor", ""]]), 104);
  assert.deepEqual([s.ack, s.command, s.physical], ["acknowledged", "active", "unverified"]);
  assert.equal(s.headline, "Abort accepted · Physical stop unverified");
  const done = requestStatus(row("confirmed", [["accepted", 103, "x", ""], ["confirmed", 105, "x", ""]]), 106);
  assert.equal(done.headline, "Skill stopped (executor result) · Physical stop unverified");
});

const observation = (extra = {}) => ({ id: "o7", t: 100, sim_time: 50.0, frame_quality: "suspected_invalid", quality_relation: "synchronized",
  quality_reason: "Nearly uniform frame", quality_source: "Dashboard check", risk: 1, threshold: 0.8, signal: "stall", ...extra });
const world = (history, extra = {}) => ({ history, history_period: 1, tools: [], incidents: [], interventions: [], ...extra });

test("a frame matched by simulation time is a synchronized observation, never detector input", () => {
  const e = cameraEvidence(world([[1, 99.0, 49.0], [2, 100.4, 50.05], [3, 101.4, 51.0]]), { t: 100, observation: observation(), incident: { id: "x1" } });
  assert.equal(e.relation, "synchronized");
  assert.equal(e.frame.id, 2);
  assert.match(e.label, /^Robot camera · Synchronized observation/);
  assert.equal(e.basis, "simulation");
  assert.equal(e.metadata.observation_id, "o7");
});

test("only a nearby recording is contextual, with its offset; none is unavailable", () => {
  const e = cameraEvidence(world([[1, 100.4, null]]), { t: 100, observation: observation(), incident: { id: "x1" } });
  assert.equal(e.relation, "contextual");
  assert.equal(e.label, "Contextual camera frame · 0.4 s after detection (dashboard clock)");
  assert.equal(cameraEvidence(world([[1, 140, 90]]), { t: 100, observation: observation() }).relation, "unavailable");
});

test("detector input is claimed only when the observation names the frame it consumed", () => {
  const e = cameraEvidence(world([[4, 100, 50]]), { t: 100, observation: observation({ detector_frame: 4 }) });
  assert.equal(e.relation, "detector_input");
  assert.match(e.label, /^Detector input/);
});

test("a camera quality warning on the dashboard feed does not claim the detector's input failed", () => {
  const q = qualityAttribution(observation());
  assert.equal(q.detector.text, "Detector input quality unknown");
  assert.equal(q.camera.text, "Robot camera quality warning: suspected invalid frames");
  assert.equal(inputWarning(observation()).text, "Robot camera quality warning: suspected invalid frames");
  const rows = assessment(observation());
  assert.equal(rows.find((r) => r.key === "detector").value, "Quality unknown");
  assert.equal(rows.find((r) => r.key === "input").value, "Suspected invalid");
  const checked = observation({ quality_relation: "detector_input", frame_quality: "invalid" });
  assert.equal(inputWarning(checked).text, "Detector image invalid");
  assert.equal(qualityAttribution(checked).camera, null);
});

test("new live events do not move a selected incident", () => {
  const tool = { id: "t4", skill: "pick", number: 4, goal: "g1", start: 90, end: null, state: "running", execution: "k1/call-4" };
  const base = world([[1, 100, 50]], { now: 120, keep_s: 600, tools: [tool],
    incidents: [{ id: "x1", t: 100, tool: "t4", evidence: { first_observation: observation() }, observation: observation() }],
    interventions: [{ id: "v1", tool: "t4", incident: "x1", stages: [requested] }] });
  const before = resolveSelection(base, { intervention: "v1" }, 120);
  const later = { ...base, now: 160, tools: [{ ...tool, end: 150, state: "failed", code: "MONITOR_STOPPED" }, { ...tool, id: "t5", number: 5, start: 151 }],
    incidents: [...base.incidents, { id: "x2", t: 155, tool: "t5", evidence: {} }] };
  const after = resolveSelection(later, { intervention: "v1" }, 160);
  assert.deepEqual([after.t, after.tool.id, after.observation.id, after.frame.id], [before.t, "t4", "o7", before.frame.id]);
  assert.equal(after.tool.state, "failed"); // the same attempt, with what is known about it now
});

test("headlines claim an acknowledgement only when one is recorded, in its order", () => {
  assert.equal(requestStatus(row("failed", [["failed", 120, "Skill executor", ""]]), 130).headline,
    "The skill did not stop as asked · No acknowledgement recorded");
  const ackedThenWithdrawn = row("cancelled", [["accepted", 103, "x", ""], ["cancelled", 105, "x", ""]]);
  assert.equal(requestStatus(ackedThenWithdrawn, 130).headline, "Request withdrawn after acknowledgement");
  const withdrawnThenAcked = row("cancelled", [["cancelled", 105, "x", ""], ["accepted", 109, "x", ""]]);
  assert.equal(requestStatus(withdrawnThenAcked, 130).headline, "Request ended before it was acknowledged · Acknowledged afterwards");
  const ackedThenExpired = row("expired", [["accepted", 104, "x", ""], ["expired", 130, "Controller", ""]]);
  assert.equal(requestStatus(ackedThenExpired, 140).headline, "Abort request expired after acknowledgement");
});

test("moving to another moment of an incident shows that moment's frame as context, not the detection's", () => {
  const tool = { id: "t4", skill: "pick", number: 4, goal: "g1", start: 90, end: null, state: "running", execution: "k1/call-4" };
  const state = world([[1, 100, 50.0], [2, 107, 57.0]], { now: 120, tools: [tool],
    incidents: [{ id: "x1", t: 100, tool: "t4", evidence: { first_observation: observation() }, observation: observation() }],
    interventions: [{ id: "v1", tool: "t4", incident: "x1", stages: [requested] }] });
  assert.equal(resolveSelection(state, { intervention: "v1" }, 120).evidence.relation, "synchronized");
  const later = resolveSelection(state, { intervention: "v1", t: 107 }, 120).evidence;
  assert.deepEqual([later.relation, later.frame.id], ["contextual", 2]);
  assert.match(later.label, /the selected moment/);
});

test("a frame more than half a storage period away in simulated time is not called synchronized", () => {
  const e = cameraEvidence(world([[1, 100.0, 50.9]]), { t: 100, observation: observation(), incident: { id: "x1" } });
  assert.equal(e.relation, "contextual");
});
