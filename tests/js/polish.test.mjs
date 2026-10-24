// V10: readable timeline segments in every state, one attempt number, honest coverage, and incident navigation:
// node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import { coverage } from "../../src/homebody/ui/dashboard/static/js/model.js";
import { incidentSequence, neighbours } from "../../src/homebody/ui/dashboard/static/js/selection.js";
import { TIMELINE, contrast, segmentLabel, segmentStyle } from "../../src/homebody/ui/dashboard/static/js/timeline_style.js";

const span = (status, extra = {}) => ({ label: "pick", status, detail: { number: 4, attempt: 3, ...extra } });
const RELATIONS = ["none", "related", "selected"];

test("segment text is one high-contrast colour in every state and relation", () => {
  for (const status of ["running", "failure", "success", "ended", "cancelled"]) {
    for (const recovery of [false, true]) {
      for (const track of ["goal", "skill"]) {
        for (const relation of RELATIONS) {
          const paint = segmentStyle(span(status, { recovery }), track, relation);
          assert.equal(paint.text, TIMELINE.text);
          assert.ok(contrast(paint.text, paint.fill) >= 7, `${status}/${track}/${relation}: ${contrast(paint.text, paint.fill).toFixed(1)}`);
        }
      }
    }
  }
});

test("selection never hides the execution state", () => {
  for (const relation of RELATIONS) {
    const failed = segmentStyle(span("failure"), "skill", relation);
    assert.deepEqual([failed.accent, failed.icon], [TIMELINE.accent.failure, "✗"]);
    const recovery = segmentStyle(span("success", { recovery: true }), "skill", relation);
    assert.deepEqual([recovery.accent, recovery.icon], [TIMELINE.accent.recovery, "↻"]);
    assert.equal(segmentStyle(span("running"), "goal", relation).accent, TIMELINE.accent.running);
  }
});

test("related intervals are a quiet tint, the exact selection a thin border, neither a bright outline", () => {
  const related = segmentStyle(span("ended"), "goal", "related");
  assert.deepEqual([related.fill, related.border], [TIMELINE.related, TIMELINE.relatedBorder]);
  const selected = segmentStyle(span("ended"), "skill", "selected");
  assert.deepEqual([selected.fill, selected.border], [TIMELINE.selectedFill, TIMELINE.selectedBorder]);
  assert.notEqual(selected.border, TIMELINE.selected); // the bright violet belongs to the cursor and the event marker
  assert.equal(segmentStyle(span("ended"), "skill").border, null);
});

test("the skill track names the goal's attempt, as the behavior tree does", () => {
  assert.equal(segmentLabel(span("running"), "skill"), "Pick · Attempt 4");
  assert.equal(segmentLabel({ label: "navigate", detail: {} }, "skill"), "Navigate");
  assert.equal(segmentLabel({ label: "Pick one jar", detail: {} }, "goal"), "Pick one jar");
});

const reading = (t) => ({ t, kind: "observation", during_task: true, level: "expected", risk: 0.1, frame_quality: "valid" });

test("coverage comes from the readings' own times and cadence, with the gaps named", () => {
  const readings = [0, 1, 2, 3, 4, 5, 20, 21, 22].map(reading); // a 1 s cadence, then 14 s without a reading
  const c = coverage(readings, [{ start: 0, end: 30 }], 0, 30);
  assert.equal(c.kind, "coverage");
  assert.ok(c.covered > 10 && c.covered < 16, String(c.covered));
  assert.equal(c.gaps, 2);
  assert.match(c.text, /^Monitor data \d+ s \/ 30 s of task time \(\d+%\) · 2 gaps, longest \d+ s$/);
});

test("without an established cadence there is no coverage percentage", () => {
  const c = coverage([reading(5)], [{ start: 0, end: 30 }], 0, 30);
  assert.equal(c.kind, "counts");
  assert.doesNotMatch(c.text, /%/);
  assert.equal(coverage([], [], 0, 30).kind, "none");
});

test("previous and next follow the incidents' own times and stop at either end", () => {
  const state = { incidents: [{ id: "x9", t: 50, alert: true }, { id: "x2", t: 10, alert: true }, { id: "x4", t: 30, alert: true, tool: "t1" }],
    interventions: [{ id: "v5", t: 31, incident: "x4", stages: [["detected", 30, "", ""]] }] };
  assert.deepEqual(incidentSequence(state).map((e) => e.id), ["x2", "v5", "x9"]); // x4 is represented by its intervention
  const middle = neighbours(state, { intervention: "v5" });
  assert.deepEqual([middle.previous.id, middle.next.id, middle.index, middle.total], ["x2", "x9", 1, 3]);
  assert.equal(neighbours(state, { incident: "x2" }).previous, null);
  assert.equal(neighbours(state, { incident: "x9" }).next, null);
});

test("an incident outside the sequence still has neighbours, by its own time", () => {
  const state = { incidents: [{ id: "x2", t: 10, alert: true }, { id: "x6", t: 20, tag: "recovery_ok" }, { id: "x9", t: 50, alert: true },
    { id: "x4", t: 30, alert: true }], interventions: [{ id: "v5", incident: "x4", stages: [["detected", 30, "", ""]] }] };
  const recovery = neighbours(state, { incident: "x6" });
  assert.deepEqual([recovery.previous.id, recovery.next.id, recovery.index], ["x2", "v5", -1]);
  const viaIncident = neighbours(state, { incident: "x4" }); // the intervention's own incident is its entry
  assert.deepEqual([viaIncident.previous.id, viaIncident.next.id], ["x2", "x9"]);
});

test("an observation bar says its level and its readings per second, a gap how long it lasted", async () => {
  const { observationLabel } = await import("../../src/homebody/ui/dashboard/static/js/timeline_style.js");
  const readings = Array.from({ length: 14 }, (_, i) => ({ t: i }));
  assert.equal(observationLabel({ level: "elevated", start: 0, end: 14, readings }), "Elevated · 14 readings · 1.0/s");
  assert.equal(observationLabel({ level: "expected", start: 0, end: 0.5, readings: readings.slice(0, 1) }), "Expected · 1 reading");
  assert.equal(observationLabel({ level: "missing", start: 2, end: 6.4, readings: [] }), "No reading · 4 s");
});
