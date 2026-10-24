// The Extract dataset button's state and the modal's cards: shown in value order, empty low-value types hidden,
// empty incident-centred ones kept (disabled with why), the most valuable available types preselected.
import assert from "node:assert/strict";
import { test } from "node:test";

import { buttonState, countText, initialSelection, visibleTypes } from "../../src/homebody/ui/dashboard/static/js/dataset.js";

const type = (id, tone, count, preselected = false) =>
  ({ id, tone, count, available: count > 0, preselected: preselected && count > 0, reason: count ? "" : `No ${id}` });
const TYPES = [type("failure_recovery", "highest", 2, true), type("monitor_incidents", "high", 0, true),
  type("human_interventions", "high", 1, true), type("robot_state_actions", "core", 3), type("task_language", "core", 1),
  type("rgb", "common", 0)];

test("the button waits for data and is off for a session with nothing recorded", () => {
  assert.equal(buttonState({ status: "Loading" }).reason, "Extraction is unavailable while data is still loading");
  assert.deepEqual(buttonState({ status: "Ready", tasks: [], tools: [], readings: [], history: [] }),
    { disabled: true, reason: "No samples available for this session" });
  assert.equal(buttonState({ status: "Ready", tasks: [{}] }).disabled, false);
});

test("cards keep the value order, hide empty common types and keep empty incident types to explain them", () => {
  assert.deepEqual(visibleTypes(TYPES).map((t) => t.id),
    ["failure_recovery", "monitor_incidents", "human_interventions", "robot_state_actions", "task_language"]);
  assert.deepEqual([...initialSelection(TYPES)], ["failure_recovery", "human_interventions"]); // available only
  assert.equal(countText(TYPES[0]), "2 samples");
  assert.equal(countText(TYPES[4]), "1 sample");
  assert.equal(countText(TYPES[1]), "No monitor_incidents");
});
