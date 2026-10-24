// The camera panel: view choice, the JPEG stream, freshness, history frames for inspection, the 3D scene.
import { $, clock, duration, h, setText, storage, toggleClass } from "./dom.js";
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
    if (this.inspecting !== null) return;
    const previous = this.url;
    this.url = URL.createObjectURL(blob);
    this.image.src = this.url;
    if (previous) setTimeout(() => URL.revokeObjectURL(previous), 1000);
  }

  /** Show the stored frame nearest before T, or return to the live stream when T is null. */
  inspect(t, history) {
    this.inspecting = t;
    if (t === null) {
      $("inspect-banner").hidden = true;
      if (this.url) this.image.src = this.url;
      return this.render(this.app.state);
    }
    const stored = [...history].reverse().find(([, at]) => at <= t) || history[0];
    $("inspect-banner").hidden = false;
    if (!stored) {
      setText($("inspect-text"), `Inspecting ${clock(t)} · no camera frame stored for that moment`);
      this.image.removeAttribute("src");
    } else {
      setText($("inspect-text"), `Inspecting ${clock(t)} · robot camera at ${clock(stored[1])}`);
      this.image.src = `history/${stored[0]}.jpg`;
    }
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

  renderFreshness(state = this.app.state) {
    const badge = $("live-badge");
    const age = this.frameAt === null ? null : this.app.link.now() - this.frameAt;
    let kind = "on";
    let text = "Live";
    if (!this.app.link.connected) [kind, text] = this.app.link.attempt > RECONNECTS ? ["off", "Offline"] : ["stale", "Reconnecting"];
    else if (this.inspecting !== null) [kind, text] = ["inspect", "Inspecting"];
    else if (this.view === SCENE) [kind, text] = ["on", "Live 3D"];
    else if (!this.live()) [kind, text] = ["on", "Still"];
    else if (age === null) [kind, text] = ["stale", "Waiting"];
    else if (age > STALE) [kind, text] = ["stale", `Stale · ${duration(age)}`];
    badge.className = `live-badge ${kind}`;
    setText(badge.querySelector("span:last-child"), text);
    const ended = state?.task && state.task.end !== null && !state.controls.busy;
    const fresh = this.view === SCENE || age === null || this.inspecting !== null ? "" :
      this.live() ? `${duration(Math.max(0, age))} old${ended && age <= STALE ? " · camera remains live" : ""}` : `captured ${clock(this.frameAt)}`;
    setText($("camera-freshness"), fresh);
    if (!$("camera-popover").hidden) this.renderStats(state);
  }

  renderEmpty() {
    const state = this.app.state;
    let text = "";
    if (!this.app.link.connected) text = "Disconnected from the session. Reconnecting…";
    else if (this.view === SCENE || this.inspecting !== null) text = "";
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
