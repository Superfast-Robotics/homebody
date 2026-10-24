// The divider between the camera and the behavior tree on wide screens: dragging it (or the arrow keys) sets the
// camera column's width, kept for this browser session; double-click returns to the default proportions. Only the
// grid changes: the camera image and the 3D view stay connected while they resize.
import { $ } from "./dom.js";

const MIN_CAMERA = 420; // px that keep the video practical
const MIN_TREE = 280; // px that keep node labels readable
const STEP = 24; // px per arrow key
const KEY = "homebody-camera-width";

function remember(px) {
  try {
    if (px === null) sessionStorage.removeItem(KEY);
    else sessionStorage.setItem(KEY, String(px));
  } catch { /* storage blocked: the width lasts until reload */ }
}

function recalled() {
  try {
    const px = Number(sessionStorage.getItem(KEY));
    return Number.isFinite(px) && px > 0 ? px : null;
  } catch {
    return null;
  }
}

export function setupSplitter() {
  const handle = $("splitter"), workspace = $("workspace"), camera = $("camera"), info = $("info-panel");
  const gap = () => parseFloat(getComputedStyle(workspace).columnGap) || 0;
  const widest = () => { // what is left for the camera once the tree and an in-grid robot panel keep their minimum
    const inGrid = getComputedStyle(info).position !== "fixed" && getComputedStyle(info).display !== "none";
    return workspace.clientWidth - MIN_TREE - gap() - (inGrid ? Math.max(260, info.offsetWidth) + gap() : 0);
  };
  const apply = (px) => {
    if (px === null) {
      workspace.style.removeProperty("--camera-col");
      handle.removeAttribute("aria-valuenow");
      return;
    }
    const width = Math.round(Math.max(MIN_CAMERA, Math.min(px, widest())));
    workspace.style.setProperty("--camera-col", `${width}px`);
    handle.setAttribute("aria-valuenow", String(width));
  };
  apply(recalled());
  handle.addEventListener("pointerdown", (event) => {
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("dragging");
    event.preventDefault();
  });
  handle.addEventListener("pointermove", (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    apply(event.clientX - workspace.getBoundingClientRect().left);
  });
  handle.addEventListener("pointerup", (event) => {
    handle.releasePointerCapture(event.pointerId);
    handle.classList.remove("dragging");
    remember(camera.offsetWidth);
  });
  handle.addEventListener("dblclick", () => { apply(null); remember(null); });
  handle.addEventListener("keydown", (event) => {
    const step = { ArrowLeft: -STEP, ArrowRight: STEP }[event.key];
    if (event.key === "Home") { apply(null); remember(null); event.preventDefault(); return; }
    if (!step) return;
    apply(camera.offsetWidth + step);
    remember(camera.offsetWidth);
    event.preventDefault();
  });
  new ResizeObserver(() => { if (workspace.style.getPropertyValue("--camera-col")) apply(camera.offsetWidth); }).observe(workspace);
}

/** The drawer opens below the header, so Pause, Stop skill and Stop task stay in reach while it is open. */
export function keepHeaderClear() {
  const header = document.querySelector(".header");
  const place = () => document.documentElement.style.setProperty("--drawer-top", `${Math.max(0, Math.round(header.getBoundingClientRect().bottom))}px`);
  new ResizeObserver(place).observe(header);
  addEventListener("resize", place);
  place();
}
