// Interventions and the monitor's condition in words: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  activeIntervention, alertGroups, flowText, signalLabel, stateText, steps,
} from "../../src/homebody/ui/dashboard/static/js/interventions.js";

const row = (state, stages, extra = {}) => ({ id: "v1", task: "k1", incident: "x1", initiator: "Live monitor", state,
  ack_timeout: 10, stages, ...extra });
const detected = ["detected", 10, "Monitoring model (Live monitor)", "abort skill: stall p=1.00"];
const decided = ["decided", 10, "Monitor rules", "Abort the skill"];
const requested = ["requested", 10, "Live monitor", "stall p=1.00"];
const accepted = ["accepted", 70, "Skill executor (session)", "taken at the next check"];
const confirmed = ["confirmed", 72, "Skill executor (result MONITOR_STOPPED)", "The skill ended as asked"];
const obs = (t, level, extra = {}) => ({ t, level, kind: "observation", risk: null, threshold: null, signal: "", skill: null,
  reason: "", ...extra });
const live = { state: "connected", observationAge: 0.5, staleAfter: 3 };

test("abort requested and abort confirmed read differently, and acceptance is not a confirmed stop", () => {
  const pending = steps(row("requested", [detected, decided, requested]));
  assert.deepEqual(pending.map((s) => s.state), ["done", "done", "done", "pending", "pending", "unknown"]);
  const taken = steps(row("executing", [detected, decided, requested, accepted]));
  assert.deepEqual(taken.map((s) => s.state), ["done", "done", "done", "done", "pending", "unknown"]);
  const done = steps(row("confirmed", [detected, decided, requested, accepted, confirmed]));
  assert.deepEqual(done.map((s) => s.state), ["done", "done", "done", "done", "done", "unknown", "pending"]);
  assert.match(done.at(-2).note, /Unverified/); // the physical stop is never claimed
  assert.equal(done.at(-1).note, "None started yet"); // no retry claimed before one starts
  const retried = steps(row("confirmed", [detected, decided, requested, accepted, confirmed,
    ["retry_started", 80, "Plan goal", "pick attempt 2 [t2]"], ["retry_finished", 90, "Skill executor", "Attempt 2 ended with OK [t2]"]]));
  assert.deepEqual([retried.at(-1).state, retried.at(-1).note], ["done", "Attempt 2 ended with OK"]);
  assert.notEqual(stateText("requested")[1], stateText("confirmed")[1]);
});

test("a timed-out, rejected or failed request says so", () => {
  const late = steps(row("timed_out", [detected, requested]));
  assert.equal(late[3].state, "failed");
  assert.match(late[3].note, /No acknowledgement within 10 s/);
  const refused = steps(row("rejected", [detected, requested, ["rejected", 12, "Harness (monitor link)", "call 2 ended"]]));
  assert.equal(refused[3].state, "failed");
  assert.equal(refused[3].note, "call 2 ended");
  assert.equal(steps(row("failed", [requested, accepted, ["failed", 80, "Skill executor", "PHYSICS_INVALID"]]))[4].state, "failed");
  assert.equal(flowText(row("timed_out", [detected, decided, requested]), "Stall detected"), "Stall detected → Abort requested → Timed out");
  assert.equal(flowText(row("requested", [detected, decided, requested]), "Stall detected"), "Stall detected → Abort requested → Awaiting executor");
});

test("an operator's stop has no detection to show, and a request the dashboard never saw is not invented", () => {
  const operator = steps(row("requested", [["requested", 5, "Human operator", "pressed"]], { initiator: "Human operator" }));
  assert.equal(operator[0].state, "skipped");
  assert.equal(operator[1].note, "The operator's own decision");
  const unseen = steps(row("executing", [["requested", null, "Live monitor", "did not reach"], accepted]));
  assert.equal(unseen[2].state, "unknown");
});

test("the card shows an open request, or one that went wrong until its alert is resolved", () => {
  const state = (rows, incidentState = "active") => ({ task: { id: "k1" }, incidents: [{ id: "x1", state: incidentState }], interventions: rows });
  assert.equal(activeIntervention(state([row("requested", [requested])])).id, "v1");
  assert.equal(activeIntervention(state([row("confirmed", [requested, confirmed])])), null);
  assert.equal(activeIntervention(state([row("failed", [requested])])).id, "v1");
  assert.equal(activeIntervention(state([row("failed", [requested])], "resolved")), null);
  assert.equal(activeIntervention({ ...state([row("requested", [requested])]), task: { id: "k2" } }), null); // another run
});

test("signals are named, sustained risk is measured, and alerts group by their incident", () => {
  assert.equal(signalLabel("stall"), "Stall probability");
  assert.equal(signalLabel("failure:dropped_object"), "Failure probability: dropped object");
  assert.equal(signalLabel(""), "Monitor risk");
  const state = { incidents: [{ id: "x1", alert: true, state: "active" }, { id: "x2", alert: true, state: "resolved" },
    { id: "x3", alert: true, state: "acknowledged" }], interventions: [row("requested", [requested]), row("confirmed", [requested], { id: "v2", incident: "x3" })] };
  const groups = alertGroups(state);
  assert.deepEqual(groups.map((g) => [g.incident.id, g.interventions.map((r) => r.id)]), [["x3", ["v2"]], ["x1", ["v1"]]]);
});
