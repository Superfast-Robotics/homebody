// Extract dataset: the Automatic Monitoring panel's button and its window. Left, the sample types this session
// holds, most valuable first (the backend's order, counts and previews: ui/dashboard/dataset.py), each with a
// thumbnail from its own samples; the most valuable are preselected. Right, a preview of the focused type: its most
// telling sample's key frames, its record and what it includes. It chooses samples only: no format, filter or
// split settings.
import { $, h, icon, setText } from "./dom.js";

const LOADING = ["Connecting", "Loading", "Resetting"];
const ALWAYS_SHOWN = ["highest", "high"]; // incident-centred types stay visible (disabled) when empty; the rest hide
const ICONS = { failure_recovery: "recovery", monitor_incidents: "shield", human_interventions: "hand",
  robot_state_actions: "action", task_language: "list", rgb: "camera", depth: "dataset" };
const INCLUDE_STYLE = { "camera frames": ["camera", "violet"], "robot state": ["follow", "blue"],
  actions: ["play", "green"], "monitor events": ["bell", "magenta"], "outcome label": ["tag", "amber"],
  language: ["list", "slate"] };

/** The button's state from the dashboard's STATE: {disabled, reason}. */
export function buttonState(state) {
  if (!state || LOADING.includes(state.status)) {
    return { disabled: true, reason: "Extraction is unavailable while data is still loading" };
  }
  const recorded = (state.tasks || []).length || (state.tools || []).length || (state.readings || []).length
    || (state.history || []).length;
  return recorded ? { disabled: false, reason: "Choose sample types to extract from this session" }
    : { disabled: true, reason: "No samples available for this session" };
}

/** The cards to show: every available type, and the empty incident-centred ones (disabled, with why). */
export function visibleTypes(types) {
  return (types || []).filter((t) => t.available || ALWAYS_SHOWN.includes(t.tone));
}

/** The types selected when the window opens: the backend's preselection, among those available. */
export function initialSelection(types) {
  return new Set((types || []).filter((t) => t.available && t.preselected).map((t) => t.id));
}

/** "3 samples", "1 sample", or why there is none. */
export function countText(type) {
  return type.available ? `${type.count} sample${type.count === 1 ? "" : "s"}` : type.reason;
}

/** The type the preview shows first: the first selected one, else the first available. */
export function firstFocus(types, selected) {
  const shown = visibleTypes(types);
  return (shown.find((t) => selected.has(t.id)) || shown.find((t) => t.available) || null)?.id ?? null;
}

/** RECORD as numbered lines of tokens ({cls, text}): one key per line, like a pretty-printed JSON object. */
export function recordLines(record) {
  const entries = Object.entries(record || {});
  const lines = [[{ cls: "p", text: "{" }]];
  entries.forEach(([key, value], i) => {
    lines.push([{ cls: "p", text: "  " }, { cls: "k", text: JSON.stringify(key) }, { cls: "p", text: ": " },
      ...valueTokens(value), { cls: "p", text: i < entries.length - 1 ? "," : "" }]);
  });
  lines.push([{ cls: "p", text: "}" }]);
  return lines;
}

function valueTokens(value) {
  if (Array.isArray(value)) {
    return [{ cls: "p", text: "[" }, ...value.flatMap((v, i) => [...valueTokens(v), ...(i < value.length - 1 ? [{ cls: "p", text: ", " }] : [])]),
      { cls: "p", text: "]" }];
  }
  if (value !== null && typeof value === "object") return [{ cls: "s", text: JSON.stringify(value) }];
  const cls = typeof value === "string" ? "s" : typeof value === "number" ? "n" : "b";
  return [{ cls, text: JSON.stringify(value) }];
}

export class DatasetModal {
  constructor(app) {
    this.app = app;
    this.button = $("dataset-open");
    this.dialog = $("dataset-modal");
    this.types = [];
    this.selected = new Set();
    this.focus = null;
    this.button.addEventListener("click", () => this.open());
    $("dataset-extract").addEventListener("click", () => this.extract());
    $("dataset-all").addEventListener("click", () => this.select(visibleTypes(this.types).filter((t) => t.available)));
    $("dataset-clear").addEventListener("click", () => this.select([]));
    $("dataset-copy").addEventListener("click", () => this.copy());
    this.dialog.addEventListener("click", (event) => { // a click on the dimmed backdrop (the dialog itself) closes it
      if (event.target === this.dialog && !this.busy) this.dialog.close();
    });
  }

  render(state) {
    const { disabled, reason } = buttonState(state);
    this.button.disabled = disabled;
    this.button.title = reason;
  }

  async open() {
    this.types = [];
    this.busy = false;
    setText($("dataset-result"), "");
    setText($("dataset-source"), "");
    $("dataset-cards").replaceChildren(h("p", { class: "dataset-loading", text: "Reading this session…" }));
    $("dataset-preview").hidden = true;
    $("dataset-empty").hidden = true;
    $("dataset-main").hidden = false;
    this.dialog.showModal();
    const reply = await this.app.link.request("dataset_types").catch((error) => ({ ok: false, message: String(error) }));
    if (!reply.ok) {
      $("dataset-cards").replaceChildren();
      setText($("dataset-result"), reply.message || "The session's samples could not be read.");
      return;
    }
    this.types = reply.types || [];
    setText($("dataset-source"), `Source: ${reply.source}`);
    this.selected = initialSelection(this.types);
    this.focus = firstFocus(this.types, this.selected);
    this.renderCards();
    this.renderPreview();
  }

  select(types) {
    this.selected = new Set(types.map((t) => t.id));
    this.renderCards();
  }

  renderCards() {
    const shown = visibleTypes(this.types);
    const any = shown.some((t) => t.available);
    $("dataset-empty").hidden = any;
    $("dataset-main").hidden = !any;
    $("dataset-cards").replaceChildren(...(any ? shown.map((type) => this.card(type)) : []));
    const extract = $("dataset-extract");
    extract.disabled = this.busy || this.selected.size === 0;
    extract.title = this.selected.size ? "" : "Choose at least one sample type";
    $("dataset-all").disabled = $("dataset-clear").disabled = !any || this.busy;
  }

  card(type) {
    const checked = this.selected.has(type.id);
    const input = h("input", { type: "checkbox", class: "dataset-check", "aria-describedby": `dataset-${type.id}-desc` });
    input.checked = checked;
    input.disabled = !type.available || this.busy;
    input.addEventListener("change", () => {
      if (input.checked) this.selected.add(type.id); else this.selected.delete(type.id);
      this.show(type.id);
      this.renderCards();
      this.dialog.querySelector(`[data-type="${type.id}"] input`)?.focus();
    });
    input.addEventListener("focus", () => this.show(type.id));
    const label = h("label", { class: `dataset-card tone-${type.tone}${checked ? " selected" : ""}${type.available ? "" : " disabled"}${this.focus === type.id ? " focused" : ""}`,
      "data-type": type.id, title: type.available ? "" : type.reason },
    input,
    h("span", { class: "dataset-icon", "aria-hidden": "true" }, icon(ICONS[type.id] || "dataset")),
    h("span", { class: "dataset-body" },
      h("span", { class: "dataset-title-row" },
        h("span", { class: "dataset-name", text: type.title }),
        h("span", { class: `dataset-badge tone-${type.tone}`, text: type.badge })),
      h("span", { class: "dataset-desc", id: `dataset-${type.id}-desc`, text: type.description }),
      h("span", { class: "dataset-count", text: countText(type) })),
    thumbnail(type),
    h("span", { class: "dataset-tick", "aria-hidden": "true" }, icon("check")));
    label.addEventListener("pointerenter", () => { if (type.available) this.show(type.id); });
    return label;
  }

  /** Show TYPE's preview (the card it belongs to is marked). */
  show(id) {
    if (this.focus === id) return;
    this.focus = id;
    for (const card of this.dialog.querySelectorAll(".dataset-card")) card.classList.toggle("focused", card.dataset.type === id);
    this.renderPreview();
  }

  renderPreview() {
    const type = this.types.find((t) => t.id === this.focus);
    const preview = type?.preview;
    $("dataset-preview").hidden = !preview;
    if (!preview) return;
    setText($("dataset-sample-name"), preview.name);
    setText($("dataset-sample-count"), countText(type));
    $("dataset-frames").replaceChildren(...(preview.frames.length ? preview.frames.map((f) => h("figure", { class: "dataset-frame" },
      h("img", { src: `history/${f.id}.jpg`, alt: f.caption, loading: "lazy" }),
      h("figcaption", { text: f.caption })))
      : [h("p", { class: "dataset-noframes", text: "No camera frames stored for this sample" })]));
    $("dataset-json").replaceChildren(...recordLines(preview.record).map((tokens) => h("li", {},
      ...tokens.map((t) => h("span", { class: `j-${t.cls}`, text: t.text })))));
    $("dataset-includes").replaceChildren(...preview.includes.map((name) => {
      const [glyph, tone] = INCLUDE_STYLE[name] || ["info", "slate"];
      return h("span", { class: `dataset-chip tone-${tone}` }, icon(glyph), name);
    }));
    this.record = preview.record;
  }

  async copy() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(this.record, null, 2));
      $("dataset-copy").classList.add("done");
      setTimeout(() => $("dataset-copy").classList.remove("done"), 1200);
    } catch {
      this.app.notify("bad", "The browser did not allow copying here.");
    }
  }

  async extract() {
    if (!this.selected.size || this.busy) return;
    this.busy = true;
    const extract = $("dataset-extract");
    setText(extract, "Extracting…");
    this.renderCards();
    const reply = await this.app.link.request("extract_dataset", { types: [...this.selected] })
      .catch((error) => ({ ok: false, message: String(error) }));
    this.busy = false;
    setText(extract, "Extract samples");
    this.renderCards();
    setText($("dataset-result"), reply.message || (reply.ok ? "Extracted." : "The extraction failed."));
    $("dataset-result").className = `dataset-result ${reply.ok ? "ok" : "bad"}`;
    this.app.notify(reply.ok ? "" : "bad", reply.message || "");
  }
}

/** A card's thumbnail: a camera frame from its own samples (with the record mark), the instruction for language
 * episodes, or nothing. */
function thumbnail(type) {
  const preview = type.preview;
  if (!preview) return h("span", { class: "dataset-thumb empty", "aria-hidden": "true" });
  const body = type.id === "task_language" || preview.thumbnail == null
    ? h("span", { class: "dataset-thumb-text", text: preview.text || type.title })
    : h("img", { src: `history/${preview.thumbnail}.jpg`, alt: "", loading: "lazy" });
  return h("span", { class: "dataset-thumb", "aria-hidden": "true" }, body, h("span", { class: "dataset-thumb-mark", text: "{ }" }));
}
