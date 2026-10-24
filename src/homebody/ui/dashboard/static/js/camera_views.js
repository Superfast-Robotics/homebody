// The camera menu's options, from what the dashboard knows of each view: two groups (cameras; visualizations and
// debug views), each option's availability from frames that actually arrived (Live, Stale), from whether the view can
// be rendered or has data (Available, No frames, No data), never assumed. Pure, so node --test checks it.

export const AUTO = "auto";
export const SCENE = "scene3d";
export const STALE_S = 2.5; // s without a frame before a streamed view is stale
const DESCRIPTIONS = {
  depth: "Depth from the robot camera", labels: "Object segmentation from the robot camera", map: "Map and the robot's navigation",
  decision: "The agent's last click on the camera image", topdown: "Where the place skill releases, from above",
  [SCENE]: "The scene in 3D",
};
const EMPTY = { decision: "No current decision visualization", topdown: "No placement target available" };

/** One view's availability: {tone, text, detail}. SHOWN: the view on screen, whose freshness the camera knows. */
export function viewStatus(name, view, now, shown) {
  if (shown && shown.name === name) {
    const kind = shown.kind;
    if (kind === "live") return { tone: "success", text: "Live", detail: `Last frame ${shown.age.toFixed(1)}s ago` };
    if (kind === "stale") return { tone: "warning", text: "Stale", detail: `Last frame ${Math.round(shown.age)}s ago` };
    if (kind === "offline") return { tone: "danger", text: "Unavailable", detail: "Camera offline" };
  }
  if (view.last_frame !== null && view.last_frame !== undefined) {
    const age = Math.max(0, now - view.last_frame);
    return age <= STALE_S ? { tone: "success", text: "Live", detail: `Last frame ${age.toFixed(1)}s ago` }
      : { tone: "warning", text: "Stale", detail: `Last frame ${Math.round(age)}s ago` };
  }
  if (view.group === "camera") {
    return view.available ? { tone: "neutral", text: "Available", detail: "Rendered when shown" }
      : { tone: "neutral", text: "No frames", detail: "Waiting for camera data" };
  }
  return view.available ? { tone: "success", text: "Available", detail: DESCRIPTIONS[name] || "" }
    : { tone: "neutral", text: "No data", detail: EMPTY[name] || "No data yet" };
}

/** The menu's groups for STATE's views (with the 3D scene when served). CHOICE: Auto or a view; SHOWN: {name, label,
 * kind, age} of the view on screen. */
export function cameraGroups(state, now, choice, shown) {
  const views = Object.entries(state.views || {}).map(([name, view]) => ({ name, ...view }));
  if (state.scene_port) views.push({ name: SCENE, label: "3D scene", available: true, group: "visualization" });
  const option = (v) => {
    const status = viewStatus(v.name, v, now, shown);
    return { id: v.name, label: v.label, icon: v.group === "camera" ? "◉" : "◇", status: { tone: status.tone, text: status.text },
      description: status.detail };
  };
  const auto = { id: AUTO, label: "Auto — Best available view", icon: "✦",
    description: choice === AUTO && shown ? `Showing ${shown.label}` : "Follows the running skill" };
  return [
    { title: "Cameras", options: [auto, ...views.filter((v) => (v.group || "camera") === "camera").map(option)] },
    { title: "Visualization & debug", options: views.filter((v) => v.group === "visualization").map(option) },
  ].filter((g) => g.options.length);
}

/** The camera button's text: "Auto · Robot camera (head)" or the chosen view's label. */
export function triggerLabel(choice, shownLabel, option) {
  return choice === AUTO ? `Auto · ${shownLabel}` : option?.label || shownLabel;
}
