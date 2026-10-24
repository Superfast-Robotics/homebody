// How the dashboard words events: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import { cameraFor, errorText, goalLabel, operatorText, shortLabel, summarize } from "../../src/homebody/ui/dashboard/static/js/story.js";

const incident = (id, t, kind, extra = {}) => ({ id, t, kind, tag: "", title: kind, node: "g1", skill: null, caused_by: null,
  initiator: "", ...extra });

test("declared failure codes read as text; unknown codes stay as they are", () => {
  assert.equal(errorText("INVALID_ARGUMENT"), "Invalid action argument");
  assert.equal(errorText("NO_GRASP"), "No grasp found");
  assert.equal(errorText("WEIRD_CODE"), "WEIRD_CODE");
});

test("long goal text becomes a short label at a word boundary", () => {
  assert.equal(shortLabel("pick a coffee bag"), "Pick a coffee bag");
  assert.equal(shortLabel("reposition and look again near the island counter", 30), "Reposition and look again…");
});

test("operator labels say what happened; raw codes stay in the detail", () => {
  const failed = { type: "action", label: "pick", status: "failure", detail: { code: "INVALID_ARGUMENT", attempt: 2 } };
  assert.deepEqual(operatorText(failed), { label: "Pick · attempt 2", meta: "Failed · Invalid action argument" });
  const stopped = { type: "action", label: "pick", status: "failure", detail: { code: "MONITOR_STOPPED", number: 1 } };
  assert.deepEqual(operatorText(stopped), { label: "Pick", meta: "Failed · Stopped by monitor" });
  const pending = { type: "action", label: "pick", status: "running", detail: { number: 4,
    interventions: [{ state: "requested", initiator: "Live monitor" }] } };
  assert.deepEqual(operatorText(pending, "9.0s"), { label: "Pick · attempt 4",
    meta: "Running · 9.0s · stop requested by the live monitor, awaiting the skill" }); // still running: the tree's own status
  const earlier = { type: "attempts", label: "Earlier attempts", status: "failure",
    detail: { count: 3, failed: 3, codes: ["MONITOR_STOPPED", "NO_GRASP", "STALE_OBSERVATION"] } };
  assert.deepEqual(operatorText(earlier), { label: "3 earlier attempts",
    meta: "All failed · Stopped by monitor, No grasp found, Observation too old" });
  const running = { type: "goal", label: "Pick one accessible jam jar from the shelf with the right hand, keeping the rest.",
    status: "running", detail: { used: 4, budget: 8 } };
  assert.deepEqual(operatorText(running, "1m"), { label: "Pick one accessible jam jar", meta: "Running · attempt 4 of 8 · 1m" });
  assert.equal(goalLabel("Carry the jam jar to the blue picking cart"), "Carry the jam jar");
  assert.equal(goalLabel("Go to the kitchen"), "Go to the kitchen"); // a cut would leave one word: kept whole
  const recovery = { type: "goal", label: "reposition and look again", status: "success", detail: { recovers: "g2", used: 1 } };
  assert.deepEqual(operatorText(recovery, "8.1s"), { label: "Recovery: Reposition and look again", meta: "Completed · 8.1s" });
  const goal = { type: "goal", label: "pick a coffee bag", status: "failure", detail: { used: 2 } };
  assert.equal(operatorText(goal).meta, "Failed · 2 attempts");
  assert.equal(operatorText({ type: "fallback", label: "Fallback", status: "running", detail: { selected: 1 } }).meta,
    "Recovering with option 2");
});

test("the summary groups repeated failures and links only recorded causes", () => {
  const state = {
    tree: { nodes: { "goal:g1": { label: "pick a coffee bag" } } },
    incidents: [
      incident("x1", 1, "skill", { skill: "pick" }), incident("x2", 2, "skill", { skill: "pick" }),
      incident("x3", 3, "goal"),
      incident("x4", 4, "recovery", { tag: "recovery_start", caused_by: "x3", initiator: "Behavior Tree fallback" }),
      incident("x5", 9, "recovery", { tag: "recovery_ok", caused_by: "x4" }),
      incident("x6", 9.5, "monitor", { title: "Grip force high" }),
    ],
    tasks: [{ id: "k1", end: 10, result: { state: "success_with_recoveries", label: "Success with recovery" } }],
  };
  const { blocks } = summarize(state, 0, 60);
  assert.deepEqual(blocks.map((b) => [b.label, b.kind, b.linked]), [
    ["Pick failed ×2", "failure", false],
    ["Goal failed: Pick a coffee bag", "failure", false], // nothing records that the attempts caused it
    ["Recovery initiated (BT fallback)", "recovery", true],
    ["Recovery succeeded", "success", true],
    ["Monitor: Grip force high", "detection", false], // concurrent, not causal
    ["Task finished · Success with recovery", "success", false],
  ]);
  assert.deepEqual(summarize(state, 20, 60).blocks, []);
});

test("the Auto camera prefers the activity's views that exist", () => {
  const auto = { pick: ["wrist_{hand}", "robot"], finished: ["scene3d", "robot"], idle: ["robot"] };
  const picking = { tools: [{ state: "running", skill: "pick", arguments: { hand: "left" } }], task: { end: null }, controls: { busy: true } };
  assert.equal(cameraFor(picking, auto, ["robot", "wrist_left"]), "wrist_left");
  assert.equal(cameraFor(picking, auto, ["robot", "depth"]), "robot"); // no wrist camera on this robot
  const finished = { tools: [], task: { end: 5 }, controls: { busy: false } };
  assert.equal(cameraFor(finished, auto, ["robot", "scene3d"]), "scene3d");
  assert.equal(cameraFor({ tools: [], task: null, controls: { busy: false } }, auto, ["robot"]), "robot");
});

test("repeated identical monitor detections are one milestone with a count", () => {
  const state = { tree: { nodes: {} }, tasks: [], incidents: [
    incident("m1", 1, "monitor", { title: "abort skill: stall p=1.00" }), incident("m2", 6, "monitor", { title: "abort skill: stall p=1.00" })] };
  const { blocks } = summarize(state, 0, 10);
  assert.equal(blocks.length, 1);
  assert.match(blocks[0].label, /×2$/);
});
