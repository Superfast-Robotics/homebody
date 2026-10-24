// The camera panel: view choice, the JPEG stream, freshness, history frames for inspection, the 3D scene.
import { $, clock, duration, h, setText, storage, toggleClass } from "./dom.js";
import { cameraEvidence } from "./evidence.js";
import { cameraFor } from "./story.js";

const STALE = 2.5; // s without a frame of a live view before it is labelled stale
const SCENE = "scene3d";
const AUTO = "auto"; // follow the running skill (views.AUTO_CAMERAS); choosing a camera turns it off
const RECONNECTS = 3; // failed reconnections before the stream counts as offline

export class Camera {
  constructor(app) {
    this.app = app;
    this.panel = $("camera");
    this.image = $("camera-image");
    this.scene = $("camera-scene");
    this.empty = $("camera-empty");
    this.select = $("camera-select");
    this.choice = storage.get("camera", AUTO);
    this.view = "robot"; // the view shown: the choice, or Auto's pick
    this.url = null;
    this.blob = null;
    this.frameAt = null; // server time of the newest frame shown
    this.frameSim = null;
    this.rate = [];
    this.inspecting = null;
    this.recorded = null; // the recorded frame shown while inspecting ({id, t, sim}), or null when none is near
    this.evidence = null; // what that frame is to the detection (evidence.js)
    this.peek = false; // the live stream in this panel only, while the investigation stays
    this.select.addEventListener("change", () => this.choose(this.select.value));
    $("camera-fullscreen").addEventListener("click", () => this.fullscreen());
    $("camera-snapshot").addEventListener("click", () => this.snapshot());
    $("camera-settings").addEventListener("click", (event) => this.settings(event.currentTarget));
    const fit = $("camera-fit");
    fit.checked = storage.get("camera-fill", false);
    toggleClass(this.panel, "fill", fit.checked);
    fit.addEventListener("change", () => {
      toggleClass(this.panel, "fill", fit.checked);
      storage.set("camera-fill", fit.checked);
    });
    this.image.addEventListener("load", () => this.renderEmpty());
    $("camera-live-toggle").addEventListener("click", () => this.togglePeek());
  }

  views(state) {
    const views = Object.entries(state.views).map(([name, view]) => ({ name, ...view }));
    if (state.scene_port) views.push({ name: SCENE, label: "3D scene", available: true });
    return views;
  }

  renderOptions(state) {
    const views = this.views(state);
    if (this.choice !== AUTO && !views.some((v) => v.name === this.choice)) this.choice = AUTO;
    const live = views.filter((v) => v.available).map((v) => v.name);
    const view = this.choice !== AUTO ? this.choice
      : this.inspecting !== null ? this.view : cameraFor(state, state.auto_cameras || {}, live);
    if (view !== this.view) {
      this.view = view;
      this.clearFrame();
    }
    const shown = views.find((v) => v.name === this.view)?.label || this.view;
    const signature = `${views.map((v) => `${v.name}:${v.available}`).join("|")}:${shown}`;
    if (signature !== this.signature) {
      this.signature = signature;
      this.select.replaceChildren(h("option", { value: AUTO, text: `Auto · ${shown}` }), ...views.map((v) => h("option", {
        value: v.name, text: v.available ? v.label : `${v.label} (none yet)`,
      })));
    }
    if (this.select.value !== this.choice) this.select.value = this.choice;
    this.select.title = this.choice === AUTO ? "Auto: the camera follows the running skill; choose one to keep it" : "Chosen camera; Auto follows the skill";
    if (this.app.link.view !== this.wire()) this.app.link.setView(this.wire());
  }

  wire() {
    return this.view === SCENE ? "none" : this.view;
  }

  choose(choice) {
    this.choice = choice;
    storage.set("camera", choice);
    this.render(this.app.state);
  }

  clearFrame() {
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = this.blob = this.frameAt = null;
    this.image.removeAttribute("src");
    this.rate = [];
  }

  frame(meta, blob) {
    if (meta.view !== this.view) return;
    this.blob = blob;
    this.frameAt = meta.t;
    this.frameSim = meta.sim ?? this.frameSim;
    this.rate = [...this.rate.filter((t) => meta.t - t < 2), meta.t];
    if (this.inspecting !== null) $("camera-live-toggle").hidden = false; // a live stream now exists to peek at
    if (this.inspecting !== null && !this.peek) return;
    const previous = this.url;
    this.url = URL.createObjectURL(blob);
    this.image.src = this.url;
    if (previous) setTimeout(() => URL.revokeObjectURL(previous), 1000);
  }

  /** Show the recorded evidence for moment T (null returns to the live stream): the selected incident's attributed
   * frame at its own moment, else the recorded frame near T as context; never the live frame as a recording. */
  inspect(t, state) {
    this.inspecting = t;
    if (t === null) {
      this.recorded = this.evidence = null;
      this.peek = false;
      toggleClass(this.panel, "peeking", false);
      $("inspect-banner").hidden = true;
      if (this.url) this.image.src = this.url;
      return this.render(this.app.state);
    }
    this.peek = false; // a new moment or selection shows its recorded evidence
    const sel = this.app.selection;
    this.evidence = sel?.evidence && sel.t === t ? sel.evidence : cameraEvidence(state || { history: [] }, { t, observation: null });
    this.recorded = this.evidence.frame;
    $("inspect-banner").hidden = false;
    this.showInspected();
    this.render(this.app.state);
  }

  /** The banner and the image for the inspected moment: the evidence's label and capture time, or the live stream while
   * the operator peeks at it. */
  showInspected() {
    const toggle = $("camera-live-toggle");
    toggle.hidden = this.frameAt === null && !this.peek; // the live stream only when one has arrived
    toggleClass(this.panel, "peeking", this.peek);
    toggle.setAttribute("aria-pressed", String(this.peek));
    setText(toggle, this.peek ? "Recorded evidence" : "Live camera");
    const e = this.evidence;
    const text = this.peek ? "Live camera · the investigation continues"
      : !this.recorded ? `Inspecting ${clock(this.inspecting)} · Camera evidence unavailable for this timestamp`
        : `${e.short} · captured ${clock(this.recorded.t)}${this.recorded.sim !== null && this.recorded.sim !== undefined ? ` (sim ${this.recorded.sim.toFixed(2)} s)` : ""}`;
    setText($("inspect-text"), text);
    $("inspect-text").title = this.peek ? "" : e?.label || "";
    if (this.peek) {
      if (this.url) this.image.src = this.url;
    } else if (this.recorded) this.image.src = `history/${this.recorded.id}.jpg`;
    else this.image.removeAttribute("src");
  }

  /** The live stream in this panel only, or back to the recorded evidence; the investigation is unchanged. */
  togglePeek() {
    if (this.inspecting === null) return;
    this.peek = !this.peek;
    this.showInspected();
    this.render(this.app.state);
  }

  render(state) {
    if (!state) return;
    this.renderOptions(state);
    const scene = this.view === SCENE && this.inspecting === null; // inspection always shows the stored robot frame
    this.scene.hidden = !scene;
    this.image.hidden = scene;
    if (scene) {
      const src = `${location.protocol}//${location.hostname}:${state.scene_port}/`;
      if (this.scene.getAttribute("src") !== src) this.scene.src = src;
    } else if (this.scene.getAttribute("src")) {
      this.scene.removeAttribute("src");
    }
    this.renderFreshness(state);
    this.renderEmpty();
  }

  live() {
    return this.app.state?.views[this.view]?.label !== undefined && this.view !== SCENE &&
      !["decision", "topdown"].includes(this.view);
  }

  /** The camera's state from actual frame receipt: {kind: live | stale | reconnecting | offline | unknown, age}. A
   * connected session without recent frames is stale, never live. */
  freshness() {
    const age = this.frameAt === null ? null : Math.max(0, this.app.link.now() - this.frameAt);
    if (!this.app.link.connected) return { kind: this.app.link.attempt > RECONNECTS ? "offline" : "reconnecting", age };
    if (age === null) return { kind: "unknown", age };
    return { kind: age > STALE ? "stale" : "live", age };
  }

  /** The badge says transport, freshness and (for the robot camera) frame usability apart: "Live · 0.2s old",
   * "Live · Visual quality warning", "Stale · Last frame 5.2s ago", "Camera offline". */
  renderFreshness(state = this.app.state) {
    const badge = $("live-badge");
    const age = this.frameAt === null ? null : Math.max(0, this.app.link.now() - this.frameAt);
    const quality = this.view === "robot" ? state?.frame_quality : null;
    const warn = quality && (quality.state === "invalid" || quality.state === "suspected_invalid");
    let kind = "on";
    let text = "Live";
    if (!this.app.link.connected) [kind, text] = this.app.link.attempt > RECONNECTS ? ["off", "Camera offline"] : ["stale", "Reconnecting"];
    else if (this.inspecting !== null && !this.peek) {
      // the badge names the evidence's kind; the banner below it carries the full label and the capture time
      [kind, text] = ["inspect", !this.recorded ? "No recording" : { detector_input: "Recorded · detector input",
        synchronized: "Recorded · synchronized", contextual: "Recorded · context" }[this.evidence.relation] || "Recorded"];
    }
    else if (this.view === SCENE) [kind, text] = ["on", "Live 3D"];
    else if (!this.live()) [kind, text] = ["on", "Still"];
    else if (age === null) [kind, text] = ["stale", "No frames received"];
    else if (age > STALE) [kind, text] = ["stale", `Stale · Last frame ${duration(age)} ago`];
    else if (warn) [kind, text] = ["warn", "Live · Visual quality warning"];
    else text = `Live · ${duration(age)} old`;
    badge.className = `live-badge ${kind}`;
    badge.title = quality ? `Frames: ${quality.state.replace("_", " ")}${quality.reason ? ` · ${quality.reason}` : ""}` : "";
    setText(badge.querySelector("span:last-child"), text);
    const ended = state?.task && state.task.end !== null && !state.controls.busy;
    setText($("camera-freshness"), this.live() || this.view === SCENE || this.inspecting !== null ? (ended && kind === "on" ? "camera remains live" : "")
      : this.frameAt ? `captured ${clock(this.frameAt)}` : "");
    // a persistent invalid frame explains itself on the image; a brief one only in the badge
    const overlay = $("camera-quality");
    const lasting = quality?.state === "invalid" && quality.since !== null && this.app.link.now() - quality.since > 3;
    overlay.hidden = !lasting || this.inspecting !== null;
    setText(overlay, lasting ? `Camera frames invalid: ${quality.reason}` : "");
    if (!$("camera-popover").hidden) this.renderStats(state);
  }

  renderEmpty() {
    const state = this.app.state;
    let text = "";
    if (!this.app.link.connected) text = "Disconnected from the session. Reconnecting…";
    else if (this.inspecting !== null && !this.peek) text = this.recorded ? "" : "Camera evidence unavailable for this timestamp";
    else if (this.view === SCENE) text = "";
    else if (!this.image.getAttribute("src")) {
      text = state?.views[this.view]?.available === false
        ? (this.view === "decision" ? "No decision click yet. It appears when the agent chooses a target."
          : "No place target yet. It appears when the agent chooses a place.")
        : "Waiting for camera frames…";
    }
    this.empty.hidden = !text;
    setText(this.empty, text);
  }

  renderStats() {
    const fps = this.rate.length > 1 ? (this.rate.length - 1) / (this.rate.at(-1) - this.rate[0]) : null;
    const rows = [
      ["View", this.select.selectedOptions[0]?.textContent || this.view],
      ["Resolution", this.image.naturalWidth ? `${this.image.naturalWidth} × ${this.image.naturalHeight}` : "—"],
      ["Frame rate", fps ? `${fps.toFixed(1)} fps` : "—"],
      ["Last frame", this.frameAt ? clock(this.frameAt) : "—"],
      ["Simulation time", this.frameSim !== null && this.frameSim !== undefined ? `${this.frameSim.toFixed(1)} s` : "—"],
    ];
    $("camera-stats").replaceChildren(...rows.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]));
  }

  settings(button) {
    const popover = $("camera-popover");
    popover.hidden = !popover.hidden;
    button.setAttribute("aria-expanded", String(!popover.hidden));
    if (!popover.hidden) this.renderStats();
  }

  fullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else this.panel.requestFullscreen?.().catch(() => this.app.notify("bad", "Fullscreen is not available here."));
  }

  snapshot() {
    const source = this.inspecting === null ? this.url : this.image.src;
    if (!source || this.view === SCENE) return this.app.notify("bad", "No camera frame to save.");
    const link = h("a", { href: source, download: `homebody-${this.view}-${clock(this.frameAt || this.app.link.now()).replaceAll(":", "")}.jpg` });
    document.body.append(link);
    link.click();
    link.remove();
  }
}
