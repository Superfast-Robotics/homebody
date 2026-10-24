// The header: task summary, status, elapsed time, the agent instruction input and the primary action.
import { confirmAction } from "./controls.js";
import { $, elapsed, setText, statusTone, toggleClass } from "./dom.js";
import { primaryAction } from "./model.js";

const STATES = {
  pending: ["pending", "Queued: the agent starts it next."],
  running: ["pending", "Accepted: the agent is working on it. That is not yet a result."],
  done: ["ok", "The agent reports the task done."],
  stopped: ["", "Stopped."],
  failed: ["bad", "Ended without success."],
  rejected: ["bad", "Rejected."],
  sent: ["pending", "Sent to the running agent."],
};
// the executor's lifecycle, apart from the task's outcome: finishing is not succeeding
const LIFECYCLE = {
  Loading: "", Ready: "", Planning: "running", Running: "running", Stopping: "warning", Paused: "warning",
  Resetting: "warning", Finished: "", Interrupted: "warning", Error: "danger",
};
const OUTCOME = {
  success: ["success", "✓"], success_with_recoveries: ["success", "✓"], partial: ["partial", "◐"],
  unverified: ["unverified", "?"], failed: ["failed", "✗"], cancelled: ["", "–"],
};
const BUSY_HINT = "The agent is working on a task. Stop it to give a new instruction.";
const NOTE_HINT = "Send to the running agent: it reads this before its next decision (Enter).";

/** Whether an instruction now goes to the running agent as a note (its controls take notes). */
const noting = (state) => Boolean(state?.controls.busy && state.controls.tell);

export class Header {
  constructor(app) {
    this.app = app;
    this.input = $("instruction");
    this.send = $("send");
    this.message = $("instruction-state");
    this.primary = $("primary-action");
    this.sending = false; // one submission at a time
    this.lastInstruction = null;
    this.input.addEventListener("input", () => this.resize());
    this.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        this.submit(this.input.value);
      }
    });
    $("command").addEventListener("submit", (event) => {
      event.preventDefault();
      this.submit(this.input.value);
    });
    this.primary.addEventListener("click", () => this.runPrimary());
    $("alert-chip").addEventListener("click", () => this.app.info.reveal());
  }

  resize() {
    this.input.style.height = "auto";
    this.input.style.height = `${Math.min(this.input.scrollHeight, 132)}px`;
    this.renderControls(this.app.state);
  }

  async submit(raw) {
    const text = raw.trim();
    const state = this.app.state;
    if (this.sending || !text) return;
    const scenario = this.app.scenarios?.chosen();
    if (scenario && !state?.controls.busy) return this.runScenario(scenario, text);
    if (!state?.controls.accepting && !noting(state)) return this.show("bad", state?.controls.busy ? BUSY_HINT : "The session cannot take a task now.");
    this.sending = true;
    this.renderControls(state);
    this.show("pending", "Sending…");
    const reply = await this.app.link.request("submit", { text });
    this.sending = false;
    if (reply.ok && this.input.value.trim() === text) {
      this.input.value = ""; // kept when the session refused it, so nothing typed is lost
      this.resize();
    }
    this.show(reply.ok ? "pending" : "bad", reply.message);
    this.renderControls(this.app.state);
  }

  async runScenario(name, text) {
    this.sending = true;
    this.renderControls(this.app.state);
    this.show("pending", `Starting ${name}…`);
    const reply = await this.app.link.request("run_scenario", { name, text });
    this.sending = false;
    this.show(reply.ok ? "pending" : "bad", reply.message || "The request failed.");
    this.renderControls(this.app.state);
  }

  async runPrimary() {
    const action = primaryAction(this.app.state, this.input.value, this.app.link.connected);
    if (!action.enabled) return;
    if (action.kind === "start") return this.submit(action.text);
    if (action.kind === "reset" && await confirmAction("Reset the scene?",
      "The physics became invalid. Resetting restores the scene to its start; the timeline keeps its history.", "Reset scene")) {
      this.app.controls.send("reset");
    }
  }

  show(kind, text) {
    this.message.className = `instruction-state ${kind}`;
    setText(this.message, text);
  }

  renderControls(state) {
    const connected = this.app.link.connected;
    const busy = Boolean(state?.controls.busy);
    const note = noting(state);
    const scenario = !busy && Boolean(this.app.scenarios?.chosen());
    const accepting = (Boolean(state?.controls.accepting) || note || scenario) && connected && !this.sending;
    this.send.disabled = !accepting || !this.input.value.trim();
    this.send.title = note ? NOTE_HINT : accepting ? "Start this instruction as a task (Enter). Shift+Enter adds a line."
      : busy ? BUSY_HINT : "The session cannot take a task now";
    this.send.setAttribute("aria-label", note ? "Send to the running agent" : accepting ? "Start task with this instruction" : this.send.title);
    this.input.placeholder = note ? "Tell the running agent something (it reads it before its next decision)…"
      : busy ? "The agent is working… stop the task to give a new instruction" : "Tell the agent what to do...";
    const action = primaryAction(state, this.input.value, connected);
    this.primary.hidden = action.kind === "none";
    this.primary.disabled = !action.enabled || this.sending;
    this.primary.title = action.title;
    setText(this.primary.querySelector("span"), this.sending ? "Starting…" : action.label);
    toggleClass(this.primary, "pending", this.sending);
  }

  render(state) {
    setText($("title"), state.scene);
    document.title = `${state.scene} · HomeBody`;
    const life = state.lifecycle;
    const pill = $("status");
    const tone = LIFECYCLE[life] ?? "";
    pill.className = `pill${tone ? ` tone-${tone}` : ""}`;
    setText(pill.querySelector(".pill-text"), this.app.link.connected ? life : "Disconnected");
    pill.title = `Executor: ${life} (session status: ${statusTone(state.status)[0]})`;
    if (!this.app.link.connected) pill.className = "pill tone-danger";
    this.renderOutcome(state.outcome);
    const ready = !state.task && state.controls.accepting && state.initial_task;
    setText($("summary"), state.task?.text || (ready ? `Ready to start: ${state.initial_task}` : state.text || "Ready for a task."));
    $("summary").title = state.task?.text || state.initial_task || "";
    this.renderControls(state);
    this.renderInstruction(state.instructions.at(-1));
    this.renderAlerts(state);
    this.tick(state);
  }

  renderInstruction(latest) {
    if (!latest || latest.id === this.lastInstruction?.id && latest.state === this.lastInstruction.state) return;
    this.lastInstruction = latest;
    const [kind, text] = STATES[latest.state] || ["", latest.state];
    this.show(kind, latest.reason && latest.state !== "running" ? `${text} ${latest.reason}` : text);
  }

  /** The task's outcome, shown only once the task has ended: never "done" as a stand-in for success. */
  renderOutcome(outcome) {
    const badge = $("outcome-badge");
    const shown = outcome && outcome.state !== "not_evaluated";
    badge.hidden = !shown;
    if (!shown) return;
    const [kind, mark] = OUTCOME[outcome.state] || ["", "?"];
    badge.className = `outcome-badge ${kind}`;
    setText(badge.querySelector(".mark"), mark);
    setText(badge.querySelector(".outcome-text"), outcome.label);
    badge.title = `Task outcome: ${outcome.reason}`;
  }

  renderAlerts(state) {
    const open = state.incidents.filter((i) => i.alert && i.state !== "resolved");
    const chip = $("alert-chip");
    chip.hidden = open.length === 0;
    toggleClass(chip, "critical", open.some((i) => i.severity === "critical" && i.state === "active"));
    setText(chip, `${open.length} unresolved alert${open.length === 1 ? "" : "s"}`);
  }

  /** The task clock: runs while the task does, then holds its final duration. */
  tick(state = this.app.state) {
    const task = state?.task;
    setText($("elapsed"), task ? elapsed((task.end ?? this.app.link.now()) - task.start) : "");
  }
}
