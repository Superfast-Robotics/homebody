// The scenario list and servers chip's pure helpers, and the Rollouts drawer's choice.
import assert from "node:assert/strict";
import { test } from "node:test";

import { keepChoice } from "../../src/homebody/ui/dashboard/static/js/rollouts.js";
import { scenarioNamed, scenarioWhere, serversSummary } from "../../src/homebody/ui/dashboard/static/js/scenarios.js";

test("a scenario says where it runs", () => {
  const state = { scenarios: [{ name: "Jam", harness: "demo", sim: "warehouse_cart", task: "pick" }, { name: "Apple", scene: "kitchen" }] };
  assert.equal(scenarioWhere(scenarioNamed(state, "Jam")), "Runs on demo with warehouse_cart.");
  assert.equal(scenarioWhere(scenarioNamed(state, "Apple")), "Runs in kitchen.");
  assert.equal(scenarioNamed(state, "nope"), null);
  assert.equal(scenarioWhere(null), "");
});

test("the system chip is quiet unless a server in use is degraded, which it names", () => {
  const rows = [{ name: "harness", health: "up", active: true }, { name: "cart", health: "up", active: true },
    { name: "other", health: "down", active: false }];
  assert.deepEqual(serversSummary(rows), { text: "System · 2/3 answering", tone: "", degraded: false }); // an unused server down is diagnostics
  rows[1].health = "down";
  assert.deepEqual(serversSummary(rows), { text: "System degraded · cart not answering", tone: "danger", degraded: true });
  rows[1].health = "unknown";
  assert.equal(serversSummary(rows).tone, "warning");
});

test("the rollouts drawer keeps its choice while the list refreshes", () => {
  const list = [{ name: "imported/a", tasks: ["task-000", "task-001"] }, { name: "b", tasks: ["task-000"] }];
  assert.deepEqual(keepChoice(list, "b", "task-000"), { name: "b", task: "task-000" });
  assert.deepEqual(keepChoice(list, "gone", "task-009"), { name: "imported/a", task: "task-001" });
  assert.deepEqual(keepChoice([], "b", null), { name: null, task: null });
});
