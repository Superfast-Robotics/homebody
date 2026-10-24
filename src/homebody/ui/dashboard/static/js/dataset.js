// Extract dataset: the Automatic Monitoring panel's button and its modal. The modal lists the sample types this
// session holds, most valuable first (the backend's order and counts: ui/dashboard/dataset.py), preselects the
// most valuable ones, and extracts the chosen types. It chooses samples only: no format, filter or split settings.
import { $, h, icon, setText } from "./dom.js";

const LOADING = ["Connecting", "Loading", "Resetting"];
const ALWAYS_SHOWN = ["highest", "high"]; // incident-centred types stay visible (disabled) when empty; the rest hide
const ICONS = { failure_recovery: "recovery", monitor_incidents: "shield", human_interventions: "hand",
  robot_state_actions: "robot", task_language: "list", rgb: "camera", depth: "dataset" };

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

/** The types selected when the modal opens: the backend's preselection, among those available. */
export function initialSelection(types) {
  return new Set((types || []).filter((t) => t.available && t.preselected).map((t) => t.id));
}

/** "3 samples", "1 sample", or why there is none. */
export function countText(type) {
  return type.available ? `${type.count} sample${type.count === 1 ? "" : "s"}` : type.reason;
}

export class DatasetModal {
  constructor(app) {
    this.app = app;
    this.button = $("dataset-open");
    this.dialog = $("dataset-modal");
    this.types = [];
    this.selected = new Set();
    this.button.addEventListener("click", () => this.open());
    $("dataset-extract").addEventListener("click", () => this.extract());
    $("dataset-all").addEventListener("click", () => this.select(visibleTypes(this.types).filter((t) => t.available)));
    $("dataset-clear").addEventListener("click", () => this.select([]));
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
    $("dataset-empty").hidden = true;
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
    this.renderCards();
  }

  select(types) {
    this.selected = new Set(types.map((t) => t.id));
    this.renderCards();
  }

  renderCards() {
    const shown = visibleTypes(this.types);
    const any = shown.some((t) => t.available);
    $("dataset-empty").hidden = any;
    $("dataset-cards").hidden = !any;
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
      this.renderCards();
      this.dialog.querySelector(`[data-type="${type.id}"] input`)?.focus();
    });
    return h("label", { class: `dataset-card tone-${type.tone}${checked ? " selected" : ""}${type.available ? "" : " disabled"}`,
      "data-type": type.id, title: type.available ? "" : type.reason },
    input,
    h("span", { class: "dataset-icon", "aria-hidden": "true" }, icon(ICONS[type.id] || "dataset")),
    h("span", { class: "dataset-body" },
      h("span", { class: "dataset-title-row" },
        h("span", { class: "dataset-name", text: type.title }),
        h("span", { class: `dataset-badge tone-${type.tone}`, text: type.badge })),
      h("span", { class: "dataset-desc", id: `dataset-${type.id}-desc`, text: type.description }),
      h("span", { class: "dataset-count", text: countText(type) })),
    h("span", { class: "dataset-tick", "aria-hidden": "true" }, icon("check")));
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
