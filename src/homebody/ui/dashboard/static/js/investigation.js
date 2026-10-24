// Investigation Mode: opening an incident turns the workspace into recorded camera, behavior tree and the docked
// incident inspector side by side (the inspector takes the robot panel's place, so nothing covers the tree), with the
// monitoring timeline below and one toolbar that says what is inspected and holds the only Return to Live. Entering
// saves the live workspace (the camera's width and its Auto or chosen view) and leaving restores it. Nothing here
// sends a robot command: the robot keeps executing under the header's controls.
import { $, clock, setText, toggleClass } from "./dom.js";
import { attemptLabel, incidentCard } from "./execution.js";

export class Investigation {
  constructor(app) {
    this.app = app;
    this.active = false; // the investigation layout is shown
    this.saved = null; // the live workspace to restore: {cameraCol, cameraChoice}
    this.tab = "incident"; // on narrower screens the tree and the inspector share a column
    $("return-live").addEventListener("click", () => app.returnToLive());
    for (const button of $("investigation-tabs").querySelectorAll("[data-tab]")) {
      button.addEventListener("click", () => this.show(button.dataset.tab));
    }
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || event.defaultPrevented || app.inspectAt === null || $("confirm").open) return;
      if (["SELECT", "INPUT", "TEXTAREA"].includes(event.target?.tagName) || !$("legend").hidden) return; // their own Escape
      app.returnToLive();
    });
  }

  /** Enter the investigation layout (once; a second incident keeps the saved live workspace). */
  enter() {
    if (!this.active) {
      const workspace = $("workspace");
      this.saved = { cameraCol: workspace.style.getPropertyValue("--camera-col"), cameraChoice: this.app.camera.choice };
      workspace.style.removeProperty("--camera-col"); // the investigation's own proportions, until the divider moves
      this.active = true;
      this.app.drawer.dock(true);
    }
    toggleClass($("app"), "investigating", true);
    this.show("incident");
  }

  /** Leave it: the live workspace as it was (the camera's width and view), and the docked inspector cleared. */
  exit() {
    if (!this.active) return;
    this.active = false;
    toggleClass($("app"), "investigating", false);
    this.app.drawer.dock(false);
    $("investigation-panel").hidden = true;
    const workspace = $("workspace");
    if (this.saved?.cameraCol) workspace.style.setProperty("--camera-col", this.saved.cameraCol);
    else workspace.style.removeProperty("--camera-col");
    if (this.saved && this.app.camera.choice !== this.saved.cameraChoice) this.app.camera.choose(this.saved.cameraChoice);
    this.saved = null;
  }

  /** Which of the tree and the inspector a narrower screen shows. */
  show(tab) {
    this.tab = tab;
    $("app").dataset.investigationTab = tab;
    for (const button of $("investigation-tabs").querySelectorAll("[data-tab]")) button.setAttribute("aria-selected", String(button.dataset.tab === tab));
    $("investigation-panel").hidden = !this.active;
  }

  /** The one toolbar while anything is inspected: what (the incident and its attempt, or just a moment) and when. */
  render() {
    const app = this.app;
    const bar = $("investigation-bar");
    bar.hidden = app.inspectAt === null;
    toggleClass($("app"), "inspecting", app.inspectAt !== null);
    if (bar.hidden) return;
    const sel = app.selection;
    const title = sel?.intervention ? incidentCard(app.state, sel.intervention).title : sel?.incident?.title || "";
    const what = sel ? [title, sel.tool ? attemptLabel(sel.tool) : sel.incident || sel.intervention ? "Attempt unknown" : ""].filter(Boolean).join(" · ") : "";
    setText($("investigation-text"), `Inspecting ${what ? `${what} · ` : ""}${clock(app.inspectAt)}`);
    $("investigation-tabs").hidden = !this.active;
  }
}
