// The dropdown's option helpers, Session Configuration's server options and Apply rules, and the camera menu's groups.
import assert from "node:assert/strict";
import { test } from "node:test";

import { filterGroups, flatten, nextIndex } from "../../src/homebody/ui/dashboard/static/js/select.js";
import { applyState, healthStatus, serverOptions, sessionLabel } from "../../src/homebody/ui/dashboard/static/js/session.js";
import { AUTO, cameraGroups, triggerLabel, viewStatus } from "../../src/homebody/ui/dashboard/static/js/camera_views.js";

const ROWS = [
  { kind: "harness", name: "demo", health: "healthy", connected: true, detail: "Idle", last_ok: 99, latency_ms: 12.4 },
  { kind: "harness", name: "spare", health: "unavailable", connected: false, error: "connection refused" },
  { kind: "simulation", name: "warehouse_cart", health: "healthy", connected: true, controller: "running", video: "on", last_ok: 99 },
  { kind: "simulation", name: "kitchen", health: "degraded", connected: true, controller: "stopped", video: "ready", last_ok: 90 },
];

test("search matches labels and descriptions and drops empty groups", () => {
  const groups = [{ title: "A", options: [{ id: "1", label: "Robot camera" }, { id: "2", label: "Hand", description: "wrist" }] },
    { title: "B", options: [{ id: "3", label: "Map" }] }];
  assert.deepEqual(flatten(filterGroups(groups, "WRIST")).map((o) => o.id), ["2"]);
  assert.equal(filterGroups(groups, "").length, 2);
  assert.equal(filterGroups(groups, "zzz").length, 0);
});

test("the keyboard skips disabled options and clamps at the ends", () => {
  const options = [{ id: "a" }, { id: "b", disabled: true }, { id: "c" }];
  assert.equal(nextIndex(options, 0, 1), 2);
  assert.equal(nextIndex(options, 2, 1), 2);
  assert.equal(nextIndex(options, 2, -1), 0);
  assert.equal(nextIndex(options, -1, 1), 0);
  assert.equal(nextIndex(options, 0, Infinity), 2);
  assert.equal(nextIndex([{ id: "x", disabled: true }], -1, 1), -1);
});

test("server options carry health, readiness and the active mark", () => {
  assert.deepEqual(healthStatus({ checking: true, health: "healthy" }), { tone: "neutral", text: "Checking" });
  const sims = serverOptions(ROWS, "simulation", "warehouse_cart", 100);
  assert.deepEqual(sims.map((o) => o.id), ["warehouse_cart", "kitchen"]);
  assert.equal(sims[0].description, "Active · Controller running · Video on");
  assert.equal(sims[1].status.text, "Degraded");
  const harnesses = serverOptions(ROWS, "harness", null, 100);
  assert.equal(harnesses[1].description, "connection refused");
  assert.match(harnesses[0].meta, /just now · 12 ms/);
});

test("Apply is enabled only for a changed, reachable, idle pair", () => {
  const session = { harness: "demo", sim: "warehouse_cart", lifecycle: "Idle" };
  assert.equal(applyState(ROWS, session, { harness: "demo", sim: "warehouse_cart" }).reason, "No changes");
  assert.match(applyState(ROWS, session, { harness: "spare", sim: "warehouse_cart" }).reason, /spare is unavailable/);
  assert.match(applyState(ROWS, session, { harness: "demo", sim: "kitchen" }).reason, /Controller stopped/);
  assert.match(applyState(ROWS, session, { harness: "gone", sim: "warehouse_cart" }).reason, /no longer configured/);
  assert.equal(applyState(ROWS, session, { harness: "demo", sim: null }).enabled, false);
  assert.equal(applyState(ROWS, { ...session, applying: true }, { harness: "demo", sim: "kitchen" }).reason, "Applying…");
  const fresh = { harness: null, sim: null };
  assert.equal(applyState(ROWS, fresh, { harness: "demo", sim: "warehouse_cart" }).enabled, true);
  const rows = ROWS.map((r) => (r.name === "kitchen" ? { ...r, controller: "running" } : r));
  assert.match(applyState(rows, { ...session, lifecycle: "Executing" }, { harness: "demo", sim: "kitchen" }).reason, /A task is running/);
  assert.equal(applyState(rows, session, { harness: "demo", sim: "kitchen" }).enabled, true);
  assert.equal(sessionLabel(session), "demo → warehouse_cart");
  assert.equal(sessionLabel({}), "Session · not set");
});

test("the camera menu groups cameras apart from visualizations and reports what actually arrived", () => {
  const state = { scene_port: 8080, views: {
    robot: { label: "Robot camera (head)", group: "camera", available: true, last_frame: 99.5 },
    hand: { label: "Hand camera", group: "camera", available: false, last_frame: null },
    map: { label: "Map", group: "visualization", available: true, last_frame: null },
    decision: { label: "Decision", group: "visualization", available: false, last_frame: null } } };
  const groups = cameraGroups(state, 100, AUTO, { name: "robot", label: "Robot camera (head)", kind: "live", age: 0.5 });
  assert.deepEqual(groups.map((g) => g.title), ["Cameras", "Visualization & debug"]);
  assert.deepEqual(groups[0].options.map((o) => o.id), [AUTO, "robot", "hand"]);
  assert.equal(groups[0].options[0].description, "Showing Robot camera (head)");
  assert.equal(groups[0].options[1].status.text, "Live");
  assert.equal(groups[0].options[2].status.text, "No frames");
  assert.deepEqual(groups[1].options.map((o) => o.id), ["map", "decision", "scene3d"]);
  assert.equal(groups[1].options[1].description, "No current decision visualization");
  assert.equal(viewStatus("robot", { group: "camera", last_frame: 90 }, 100, null).text, "Stale");
  assert.equal(viewStatus("robot", {}, 100, { name: "robot", kind: "offline", age: 0 }).text, "Unavailable");
  assert.equal(triggerLabel(AUTO, "Robot camera (head)", null), "Auto · Robot camera (head)");
  assert.equal(triggerLabel("map", "Map", { label: "Map" }), "Map");
});
