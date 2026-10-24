// Investigation Mode: opening an incident turns the workspace into recorded camera, behavior tree and the docked
// incident inspector side by side (the inspector takes the robot panel's place, so nothing covers the tree), with the
// monitoring timeline below and one toolbar that says what is inspected and holds the only Return to Live. Entering
// saves the live workspace (the camera's width and its Auto or chosen view) and leaving restores it. Nothing here
// sends a robot command: the robot keeps executing under the header's controls.
import { $, clock, setText, toggleClass } from "./dom.js";
import { attemptLabel, incidentCard } from "./execution.js";
import { neighbours, resolveSelection } from "./selection.js";

export class Investigation {
  constructor(app) {
    this.app = app;
    this.active = false; // the investigation layout is shown
    this.saved = null; // the live workspace to restore: {cameraCol, cameraChoice}
    this.tab = "incident"; // on narrower screens the tree and the inspector share a column
    this.key = null; // the incident whose evidence decided the layout
    this.noEvidence = false; // that incident has no recorded camera evidence: the camera column is compact
    this.expanded = false; // the operator gave the camera its width back for this incident
    $("return-live").addEventListener("click", () => app.returnToLive());
    $("investigation-prev").addEventListener("click", () => this.step("previous"));
    $("investigation-next").addEventListener("click", () => this.step("next"));
    $("camera-expand").addEventListener("click", () => { this.expanded = true; this.layout(); });
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

  /**
   * Settle the layout for the incident REF names: once per incident, from whether its own moment has recorded camera
   * evidence (known from the frames index, so nothing waits on a load), so moving the cursor never resizes panels. A
   * live camera that is merely stale does not count: only the recording decides.
   */
  settle(ref) {
    const row = ref.incident ? (this.app.state.interventions || []).find((r) => r.incident === ref.incident) : null;
    const key = ref.intervention || row?.id || ref.incident; // one key per incident, however it was reached
    if (!key || key === this.key) return;
    this.key = key;
    this.expanded = false;
    const own = resolveSelection(this.app.state, ref.intervention ? { intervention: key } : { incident: key }, this.app.timelineEnd().end);
    this.noEvidence = own.evidence.frame === null;
    this.layout();
  }

  layout() {
    toggleClass($("app"), "no-evidence", this.active && this.noEvidence && !this.expanded);
  }

  /** The compact camera lifts as soon as a recorded frame is shown for the inspected moment (a later event, a scrub, a
   * frame that was still arriving); it never compacts again for the same incident. */
  lift() {
    if (this.active && this.noEvidence && this.app.camera.recorded) {
      this.noEvidence = false;
      this.layout();
    }
  }

  /** The previous or next incident by its own time; it is selected like any other, and nothing reaches the robot. */
  step(direction) {
    const near = neighbours(this.app.state, this.app.selection?.ref)[direction];
    if (near) this.app.select(near.ref);
  }

  /** Leave it: the live workspace as it was (the camera's width and view), and the docked inspector cleared. */
  exit() {
    if (!this.active) return;
    this.active = false;
    this.key = null;
    this.noEvidence = this.expanded = false;
    this.layout();
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
    const inspecting = app.inspectAt !== null;
    if (inspecting !== $("app").classList.contains("inspecting")) {
      toggleClass($("app"), "inspecting", inspecting);
      if (app.state) app.controls.render(app.state); // their titles say they act on the live robot
    }
    if (bar.hidden) return;
    const sel = app.selection;
    const title = sel?.intervention ? incidentCard(app.state, sel.intervention).title : sel?.incident?.title || "";
    const what = sel ? [title, sel.tool ? attemptLabel(sel.tool) : sel.incident || sel.intervention ? "Attempt unknown" : ""].filter(Boolean).join(" · ") : "";
    setText($("investigation-text"), `Inspecting ${what ? `${what} · ` : ""}${clock(app.inspectAt)}`);
    $("investigation-tabs").hidden = !this.active;
    this.lift();
    const near = this.active ? neighbours(app.state, sel?.ref) : { previous: null, next: null };
    for (const [id, item] of [["investigation-prev", near.previous], ["investigation-next", near.next]]) {
      const button = $(id);
      button.hidden = !this.active;
      button.disabled = !item;
    }
    $("investigation-prev").title = near.previous ? `Previous incident (${clock(near.previous.t)})` : "No earlier incident";
    $("investigation-next").title = near.next ? `Next incident (${clock(near.next.t)})` : "No later incident"; // by their own times
  }
}
