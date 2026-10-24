// The dashboard's one dropdown: a button that opens a listbox of options in groups. An option may carry a
// description, metadata, an operational status (a dot and a word), an icon and a reason it cannot be chosen; the
// same component serves a short text list (SimpleSelect), servers with health (StatusSelect), cameras and
// visualizations in groups (GroupedSelect) and all of those at once (RichSelect). Violet marks only the selection
// and keyboard focus; status colours keep their operational meaning. The menu lives on the page's top layer so it
// opens over the video, flips above its button near the bottom of the window, scrolls inside itself, closes on
// Escape, an outside click or a choice, and is driven by the keyboard (Enter/Space/arrows/Home/End/typing to search).
import { h, setText } from "./dom.js";

const SEARCH_FROM = 8; // options from which the menu offers a search field
const MENU_MAX = 380; // px

/** GROUPS ([{title, options}]) with QUERY matched against each option's label, description and metadata; groups
 * left empty are dropped. */
export function filterGroups(groups, query) {
  const q = query.trim().toLowerCase();
  if (!q) return groups;
  return groups.map((g) => ({ ...g, options: g.options.filter((o) =>
    [o.label, o.description, o.meta, o.status?.text].filter(Boolean).some((text) => String(text).toLowerCase().includes(q))) }))
    .filter((g) => g.options.length);
}

/** The options of GROUPS in menu order. */
export function flatten(groups) {
  return groups.flatMap((g) => g.options);
}

/** The index STEP away from FROM among OPTIONS that can be chosen (disabled ones are skipped), clamped at the ends;
 * FROM -1 with a positive step starts at the first. */
export function nextIndex(options, from, step) {
  const usable = options.map((o, i) => (o.disabled ? -1 : i)).filter((i) => i >= 0);
  if (!usable.length) return -1;
  if (step === Infinity) return usable.at(-1);
  if (step === -Infinity) return usable[0];
  const position = usable.indexOf(from);
  if (position < 0) return step > 0 ? usable[0] : usable.at(-1);
  return usable[Math.max(0, Math.min(usable.length - 1, position + Math.sign(step)))];
}

let open = null; // the one open dropdown

export class Select {
  /**
   * HOST: the element the button goes into. LABEL: the accessible name. ON_CHANGE(id, option): a choice (not a
   * programmatic value change). TRIGGER(option): the button's text for the chosen option (default its label).
   * VARIANT: "dark" for the camera's glass button.
   */
  constructor(host, { label, onChange = () => {}, trigger = null, variant = "", placeholder = "Choose…" } = {}) {
    this.host = host;
    this.onChange = onChange;
    this.triggerText = trigger;
    this.placeholder = placeholder;
    this.groups = [];
    this._value = null;
    this.active = -1; // the keyboard's option, as an index into the shown options
    this.query = "";
    const id = host.id || `select-${Math.random().toString(36).slice(2, 8)}`;
    this.button = h("button", { class: `dropdown-trigger${variant ? ` ${variant}` : ""}`, type: "button", role: "combobox",
      "aria-haspopup": "listbox", "aria-expanded": "false", "aria-controls": `${id}-menu`, "aria-label": label });
    this.text = h("span", { class: "dropdown-text" });
    this.badge = h("span", { class: "dropdown-badge", hidden: true });
    this.button.append(this.text, this.badge, h("span", { class: "dropdown-chevron", "aria-hidden": "true" }));
    host.replaceChildren(this.button);
    host.classList.add("dropdown");
    this.menu = h("div", { class: "dropdown-menu", id: `${id}-menu`, hidden: true });
    this.search = h("input", { class: "dropdown-search", type: "search", placeholder: "Search", "aria-label": `Search ${label}` });
    this.list = h("ul", { class: "dropdown-list", role: "listbox", "aria-label": label });
    this.menu.append(this.search, this.list);
    document.body.append(this.menu);
    this.button.addEventListener("click", () => (this.isOpen() ? this.close() : this.openMenu()));
    this.button.addEventListener("keydown", (event) => this.keys(event));
    this.search.addEventListener("input", () => { this.query = this.search.value; this.active = -1; this.renderList(); });
    this.search.addEventListener("keydown", (event) => this.keys(event));
  }

  get value() { return this._value; }

  /** Set the value (no change callback: only an operator's choice calls it). */
  set value(id) {
    this._value = id;
    this.renderTrigger();
    if (this.isOpen()) this.renderList();
  }

  /** The options, as groups [{title, options: [{id, label, description, meta, status: {tone, text}, icon, disabled,
   * reason, hint}]}] (a flat list is one untitled group). The button is always redrawn (its text may depend on state
   * outside the options); the list only when the options changed. */
  setOptions(groups) {
    const list = Array.isArray(groups) && groups.length && !("options" in groups[0]) ? [{ title: "", options: groups }] : groups;
    const signature = JSON.stringify(list);
    if (signature === this.signature) { this.renderTrigger(); return; }
    this.signature = signature;
    this.groups = list;
    this.renderTrigger();
    if (this.isOpen()) this.renderList();
  }

  option(id = this._value) {
    return flatten(this.groups).find((o) => o.id === id) || null;
  }

  isOpen() { return open === this; }

  renderTrigger() {
    const option = this.option();
    setText(this.text, option ? (this.triggerText ? this.triggerText(option) : option.label) : this.placeholder);
    this.badge.hidden = !option?.status;
    if (option?.status) {
      this.badge.className = `dropdown-badge ${option.status.tone || ""}`;
      setText(this.badge, option.status.text);
    }
    this.button.title = option ? [option.label, option.description, option.status?.text].filter(Boolean).join(" · ") : "";
  }

  openMenu() {
    if (open && open !== this) open.close(false);
    open = this;
    this.query = this.search.value = "";
    this.search.hidden = flatten(this.groups).length < SEARCH_FROM;
    this.menu.hidden = false;
    this.button.setAttribute("aria-expanded", "true");
    const shown = flatten(this.groups);
    this.active = Math.max(0, shown.findIndex((o) => o.id === this._value));
    this.shown = shown;
    this.renderList();
    this.place();
    (this.search.hidden ? this.button : this.search).focus();
    this.list.querySelector("[aria-selected='true']")?.scrollIntoView({ block: "nearest" });
    // added at once (the opening click's pointerdown is already past), so close() always removes it
    document.addEventListener("pointerdown", this.outside = (event) => {
      if (!this.menu.contains(event.target) && !this.button.contains(event.target)) this.close(false);
    });
    addEventListener("resize", this.reposition = () => this.place());
    addEventListener("scroll", this.reposition, true);
  }

  close(focus = true) {
    if (open !== this) return;
    open = null;
    this.menu.hidden = true;
    this.button.setAttribute("aria-expanded", "false");
    this.button.removeAttribute("aria-activedescendant");
    document.removeEventListener("pointerdown", this.outside);
    removeEventListener("resize", this.reposition);
    removeEventListener("scroll", this.reposition, true);
    if (focus) this.button.focus();
  }

  /** Below the button, or above it when the window has more room there; never past the window's edges. */
  place() {
    const box = this.button.getBoundingClientRect();
    const width = Math.min(Math.max(box.width, Number(this.host.dataset.menuWidth) || 280), innerWidth - 16);
    const below = innerHeight - box.bottom - 8, above = box.top - 8;
    const up = below < Math.min(MENU_MAX, this.menu.scrollHeight) && above > below;
    const height = Math.min(MENU_MAX, up ? above : below);
    Object.assign(this.menu.style, { width: `${width}px`, maxHeight: `${Math.max(120, height)}px`,
      left: `${Math.max(8, Math.min(box.left, innerWidth - width - 8))}px`,
      top: up ? "" : `${box.bottom + 4}px`, bottom: up ? `${innerHeight - box.top + 4}px` : "" });
  }

  renderList() {
    const groups = filterGroups(this.groups, this.query);
    const shown = flatten(groups);
    const kept = this.shown?.[this.active]?.id; // the keyboard stays on its option when a push reorders the list
    if (kept !== undefined) this.active = shown.findIndex((o) => o.id === kept);
    let index = -1;
    const items = groups.flatMap((g) => [
      g.title ? h("li", { class: "dropdown-group", role: "presentation", text: g.title }) : null,
      ...g.options.map((o) => {
        index += 1;
        const at = index;
        const selected = o.id === this._value;
        const id = `${this.menu.id}-o${at}`;
        const item = h("li", { id, class: `dropdown-option${selected ? " selected" : ""}${at === this.active ? " active" : ""}${o.disabled ? " disabled" : ""}`,
          role: "option", "aria-selected": String(selected), "aria-disabled": o.disabled ? "true" : null,
          title: o.disabled ? o.reason || "Not available" : o.hint || null },
        h("span", { class: "dropdown-icon", "aria-hidden": "true", text: o.icon || "" }),
        h("span", { class: "dropdown-body" },
          h("span", { class: "dropdown-label", text: o.label }),
          o.description ? h("span", { class: "dropdown-description", text: o.description }) : null,
          o.disabled && o.reason ? h("span", { class: "dropdown-reason", text: o.reason }) : null),
        h("span", { class: "dropdown-side" },
          o.status ? h("span", { class: `dropdown-status ${o.status.tone || ""}` }, h("span", { class: "dot" }), o.status.text) : null,
          o.meta ? h("span", { class: "dropdown-meta", text: o.meta }) : null),
        h("span", { class: "dropdown-check", "aria-hidden": "true", text: selected ? "✓" : "" }));
        item.addEventListener("click", () => this.choose(o));
        item.addEventListener("pointermove", () => { if (this.active !== at) { this.active = at; this.markActive(); } });
        return item;
      })]).filter(Boolean);
    this.list.replaceChildren(...(items.length ? items : [h("li", { class: "dropdown-empty", text: this.query ? "No match" : "Nothing to choose" })]));
    this.shown = shown;
    this.markActive();
  }

  markActive() {
    for (const item of this.list.querySelectorAll(".dropdown-option")) item.classList.remove("active");
    const item = this.active >= 0 ? this.list.querySelectorAll(".dropdown-option")[this.active] : null;
    if (item) {
      item.classList.add("active");
      item.scrollIntoView({ block: "nearest" });
      this.button.setAttribute("aria-activedescendant", item.id);
    }
  }

  choose(option) {
    if (option.disabled) return;
    const changed = option.id !== this._value;
    this._value = option.id;
    this.renderTrigger();
    this.close();
    if (changed) this.onChange(option.id, option);
  }

  keys(event) {
    const opening = ["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key) && event.target === this.button;
    if (!this.isOpen()) {
      if (opening) { event.preventDefault(); this.openMenu(); }
      return;
    }
    const step = { ArrowDown: 1, ArrowUp: -1, Home: -Infinity, End: Infinity }[event.key];
    if (step !== undefined && !(event.target === this.search && ["Home", "End"].includes(event.key))) {
      event.preventDefault();
      this.active = nextIndex(this.shown || [], this.active, step);
      this.markActive();
    } else if (event.key === "Enter" || (event.key === " " && event.target === this.button)) {
      event.preventDefault();
      const option = (this.shown || [])[this.active];
      if (option) this.choose(option);
    } else if (event.key === "Escape") {
      event.preventDefault(); // handled here: Escape elsewhere (Return to Live) must not also fire
      event.stopPropagation();
      this.close();
    } else if (event.key === "Tab") {
      this.close(false);
    }
  }
}
