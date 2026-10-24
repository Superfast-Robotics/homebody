// Small DOM and formatting helpers. Text always goes in as textContent, never as HTML.
const SVG = "http://www.w3.org/2000/svg";

export const $ = (id) => document.getElementById(id);

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function icon(name, cls = "icon") {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

export function setText(node, text) {
  const value = text ?? "";
  if (node.textContent !== value) node.textContent = value;
}

export function setIcon(svg, name) {
  const use = svg.querySelector("use");
  if (use && use.getAttribute("href") !== `#i-${name}`) use.setAttribute("href", `#i-${name}`);
}

export function toggleClass(node, name, on) {
  if (node.classList.contains(name) !== Boolean(on)) node.classList.toggle(name, Boolean(on));
}

const pad = (n) => String(n).padStart(2, "0");

export function clock(t) {
  if (t === null || t === undefined) return "";
  const d = new Date(t * 1000);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function duration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "";
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${pad(s % 60)}s`;
  return `${Math.floor(m / 60)}h ${pad(m % 60)}m`;
}

export function elapsed(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}`;
}

export function ago(seconds) {
  if (seconds < 5) return "just now";
  return `${duration(seconds)} ago`;
}

export function json(value) {
  return JSON.stringify(value, null, 2);
}

// Per-viewer conveniences only; the page works when storage is unavailable.
export const storage = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem(`homebody.${key}`);
      return value === null ? fallback : JSON.parse(value);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`homebody.${key}`, JSON.stringify(value));
    } catch {
      /* storage unavailable: the choice lasts for this page only */
    }
  },
};

// Colours for the canvas, read from the design tokens.
let palette = null;
export function tokens() {
  if (palette) return palette;
  const style = getComputedStyle(document.documentElement);
  const read = (name) => style.getPropertyValue(`--${name}`).trim();
  palette = Object.fromEntries([
    "app-bg", "surface", "surface-2", "surface-3", "border", "border-subtle", "text-primary", "text-secondary",
    "text-muted", "primary", "on-primary", "violet", "violet-bg", "violet-edge", "cyan", "running", "running-light",
    "success", "success-light", "state-success-edge", "warning", "warning-light", "danger", "danger-light",
    "state-error-edge", "neutral", "font", "risk-low", "risk-elevated", "risk-high", "risk-missing", "event-high",
    "event-warning", "event-selected",
  ].map((name) => [name, read(name)]));
  return palette;
}

// The status tone shared by the header pill, tree nodes and tracks.
export const STATUS = {
  Loading: ["Loading", ""],
  Ready: ["Ready", ""],
  Thinking: ["Thinking", "running"],
  Executing: ["Running", "running"],
  Paused: ["Paused", "warning"],
  Stopped: ["Stopped", ""],
  Resetting: ["Resetting", "warning"],
  "Reset required": ["Reset required", "danger"],
  model_done: ["Done", "success"],
  step_limit: ["Step limit reached", "warning"],
  cancelled: ["Cancelled", ""],
  provider_error: ["Provider error", "danger"],
  error: ["Error", "danger"],
  interrupted: ["Interrupted", "warning"],
  physics_invalid: ["Physics invalid", "danger"],
};

export function statusTone(status) {
  return STATUS[status] || [status, ""];
}
