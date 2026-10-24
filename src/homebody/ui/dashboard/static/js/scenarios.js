// Scenarios, when this dashboard runs beside harnesses elsewhere (homebody connect, the monitor app): a scenario
// fills the instruction and runs on its harness and simulation server. Which servers the session uses, and their
// health, is the Session Configuration (session.js).
import { $ } from "./dom.js";
import { Select } from "./select.js";

const CUSTOM = "custom"; // the list's first entry: the instruction as typed

/** The scenario named NAME in STATE, or null. */
export function scenarioNamed(state, name) {
  return (state?.scenarios || []).find((s) => s.name === name) || null;
}

/** Where a scenario runs, in words: its harness and simulation server, or its scene. */
export function scenarioWhere(scenario) {
  if (!scenario) return "";
  if (scenario.harness) return `Runs on ${scenario.harness}${scenario.sim ? ` with ${scenario.sim}` : ""}.`;
  return scenario.scene ? `Runs in ${scenario.scene}.` : "";
}

export class Scenarios {
  constructor(app) {
    this.app = app;
    this.host = $("scenario");
    this.select = new Select(this.host, { label: "Scenario", onChange: (id) => this.choose(id),
      trigger: (option) => (option.id === CUSTOM ? "Custom" : option.label) });  // a compact prefix in the instruction bar
    this.host.dataset.menuWidth = "460";
    this.select.value = CUSTOM;
  }

  /** The chosen scenario's name, or null for a custom instruction. */
  chosen() {
    return this.host.hidden || this.select.value === CUSTOM ? null : this.select.value;
  }

  choose(name) {
    const scenario = scenarioNamed(this.app.state, name);
    if (scenario) {
      this.app.header.input.value = scenario.task;
      this.app.header.resize();
      this.app.header.show("", scenarioWhere(scenario));
    }
  }

  render(state) {
    const scenarios = state.scenarios || [];
    this.host.hidden = scenarios.length === 0 || Boolean(state.controls.busy); // task setup folds while a task runs
    this.select.setOptions([{ id: CUSTOM, label: "Custom instruction", description: "The instruction as typed" },
      ...scenarios.map((s) => ({ id: s.name, label: s.name, description: s.scene ? `Scene ${s.scene}` : "" }))]);
    if (!scenarios.some((s) => s.name === this.select.value)) this.select.value = CUSTOM;
  }
}
