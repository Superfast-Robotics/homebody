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

test("the servers chip counts the servers that answer and turns red when one does not", () => {
  assert.deepEqual(serversSummary([{ health: "up" }, { health: "up" }]), { text: "Servers 2/2", tone: "success" });
  assert.deepEqual(serversSummary([{ health: "up" }, { health: "unknown" }]), { text: "Servers 1/2", tone: "warning" });
  assert.equal(serversSummary([{ health: "up" }, { health: "down" }]).tone, "danger");
});

test("the rollouts drawer keeps its choice while the list refreshes", () => {
  const list = [{ name: "imported/a", tasks: ["task-000", "task-001"] }, { name: "b", tasks: ["task-000"] }];
  assert.deepEqual(keepChoice(list, "b", "task-000"), { name: "b", task: "task-000" });
  assert.deepEqual(keepChoice(list, "gone", "task-009"), { name: "imported/a", task: "task-001" });
  assert.deepEqual(keepChoice([], "b", null), { name: null, task: null });
});
