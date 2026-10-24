// Execution controls. The header and the camera overlay show the same buttons, bound to one state.
import { $, setIcon, setText, toggleClass } from "./dom.js";

const UNSUPPORTED = "Not supported by this session's controller";
const CONFIRM_TIMEOUT = 10; // s to wait for the controller to report a requested pause or resume

export function confirmAction(title, text, ok) {
  const dialog = $("confirm");
  setText($("confirm-title"), title);
  setText($("confirm-text"), text);
  setText($("confirm-ok"), ok);
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true });
    dialog.returnValue = "cancel";
    dialog.showModal();
  });
}

export class Controls {
  constructor(app) {
    this.app = app;
    this.pending = new Set(); // commands sent and not yet answered
    this.awaiting = null; // {paused, until}: a pause or resume the controller has not confirmed yet
    this.menu = $("more-menu");
    this.menuOwner = null;
    document.querySelectorAll("[data-control]").forEach((button) => {
      button.addEventListener("click", () => this.press(button.dataset.control, button));
    });
    this.menu.addEventListener("click", (event) => {
      const item = event.target.closest("[data-menu]");
      if (item && !item.disabled) {
        this.closeMenu();
        this.choose(item.dataset.menu);
      }
    });
    this.menu.addEventListener("keydown", (event) => this.menuKeys(event));
    document.addEventListener("click", (event) => {
      if (!this.menu.hidden && !this.menu.contains(event.target) && !event.target.closest("[data-control='more']")) {
        this.closeMenu();
      }
    });
  }

  async press(control, button) {
    const controls = this.app.state?.controls;
    if (!controls) return;
    if (control === "more") return this.toggleMenu(button);
    if (control === "pause") return this.send(controls.paused ? "resume" : "pause");
    if (control === "stop_skill") return this.send("stop_skill");
    if (control === "stop_task") {
      const ok = await confirmAction("Stop the task?",
        "The agent cancels the running skill and ends the task. This is a network task cancellation, " +
        "not an emergency stop: use the robot's safety-rated E-stop for that.", "Stop task");
      if (ok) this.send("stop_task");
    }
  }

  async choose(item) {
    if (item === "stop_skill") return this.send("stop_skill");
    if (item === "reset") {
      const ok = await confirmAction("Reset the scene?",
        "Cancels any task and restores the scene to its start. The timeline keeps its history.", "Reset scene");
      if (ok) this.send("reset");
    }
    if (item === "instructions") this.app.drawer.instructions();
    if (item === "rollouts") this.app.rollouts.open();
    if (item === "diagnostics") this.app.drawer.diagnostics();
  }

  async send(command) {
    if (this.pending.has(command)) return; // one request at a time
    this.pending.add(command);
    this.render(this.app.state);
    const reply = await this.app.link.request(command);
    this.pending.delete(command);
    if (reply.ok && (command === "pause" || command === "resume")) {
      this.awaiting = { paused: command === "pause", until: this.app.link.now() + CONFIRM_TIMEOUT };
    }
    this.app.notify(reply.ok ? "pending" : "bad", reply.ok ? reply.message : `${reply.message || "The request failed."}`);
    this.render(this.app.state);
  }

  /** A requested pause or resume stays pending until the controller's state shows it. */
  confirming(c) {
    if (!this.awaiting) return false;
    if (c.paused === this.awaiting.paused) {
      this.awaiting = null;
      return false;
    }
    if (this.app.link.now() > this.awaiting.until) {
      this.app.notify("bad", `The controller did not confirm the ${this.awaiting.paused ? "pause" : "resume"}.`);
      this.awaiting = null;
      return false;
    }
    return true;
  }

  render(state) {
    if (!state) return;
    const c = state.controls;
    const connected = this.app.link.connected;
    const pauseLabel = c.paused ? "Resume" : "Pause";
    const confirming = this.confirming(c);
    for (const button of document.querySelectorAll("[data-control]")) {
      const control = button.dataset.control;
      let disabled = !connected;
      let title = "";
      let pending = false;
      if (control === "pause") {
        disabled ||= !c.pause || !c.busy;
        title = !c.pause ? UNSUPPORTED : c.busy ? pauseLabel : "No task is running";
        pending = this.pending.has("pause") || this.pending.has("resume") || confirming;
        disabled ||= pending;
        setText(button.querySelector("span"), pending ? (c.paused ? "Resuming…" : "Pausing…") : pauseLabel);
        setIcon(button.querySelector("svg"), c.paused ? "play" : "pause");
        toggleClass(button, "is-paused", c.paused);
      } else if (control === "stop_skill") {
        pending = this.pending.has("stop_skill");
        disabled ||= !c.stop_skill || !c.busy || pending;
        title = !c.stop_skill ? UNSUPPORTED : "Stop the current skill gracefully; the task continues";
      } else if (control === "stop_task") {
        pending = this.pending.has("stop_task") || c.stop_pending;
        disabled ||= !c.busy || pending;
        title = c.busy ? "Cancel the task (not an emergency stop)" : "No task is running";
        setText(button.querySelector("span"), pending ? "Stopping…" : "Stop task");
      } else {
        title = "More actions";
      }
      button.disabled = disabled;
      button.title = title;
      toggleClass(button, "pending", pending);
      if (control !== "more") button.setAttribute("aria-label", title ? `${button.textContent.trim()}: ${title}` : button.textContent.trim());
    }
    const stopSkill = this.menu.querySelector("[data-menu='stop_skill']");
    stopSkill.disabled = !connected || !c.stop_skill || !c.busy;
    stopSkill.title = c.stop_skill ? "" : UNSUPPORTED;
    const reset = this.menu.querySelector("[data-menu='reset']");
    reset.disabled = !connected || !c.reset || c.reset_pending;
  }

  toggleMenu(button) {
    if (!this.menu.hidden && this.menuOwner === button) return this.closeMenu();
    this.menuOwner = button;
    this.menu.hidden = false;
    const rect = button.getBoundingClientRect();
    const width = this.menu.offsetWidth;
    const below = rect.bottom + 6 + this.menu.offsetHeight < innerHeight;
    this.menu.style.left = `${Math.max(8, Math.min(innerWidth - width - 8, rect.right - width))}px`;
    this.menu.style.top = below ? `${rect.bottom + 6}px` : `${rect.top - this.menu.offsetHeight - 6}px`;
    button.setAttribute("aria-expanded", "true");
    this.menu.querySelector("[role='menuitem']:not(:disabled)")?.focus();
  }

  closeMenu() {
    if (this.menu.hidden) return;
    this.menu.hidden = true;
    this.menuOwner?.setAttribute("aria-expanded", "false");
    this.menuOwner?.focus();
  }

  menuKeys(event) {
    const items = [...this.menu.querySelectorAll("[role='menuitem']:not(:disabled)")];
    const index = items.indexOf(document.activeElement);
    if (event.key === "Escape") this.closeMenu();
    else if (event.key === "ArrowDown") items[(index + 1) % items.length]?.focus();
    else if (event.key === "ArrowUp") items[(index - 1 + items.length) % items.length]?.focus();
    else return;
    event.preventDefault();
  }
}
