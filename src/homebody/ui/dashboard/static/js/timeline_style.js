// How the timeline paints a goal or skill interval, in three layers that never stand in for each other: the execution
// state (a left accent and an icon: running, failed, succeeded, recovery), the relationship to the selected incident
// (a quiet violet tint and a faint border) and the operator's exact selection (a slightly brighter tint and a thin
// border). Label text keeps one high-contrast colour in every state and nothing is drawn translucent over it; the
// strongest emphasis belongs to the cursor and the selected event marker, never to whole intervals. Pure, so node
// --test checks it.

export const TIMELINE = {
  background: "#12141D", track: "#1C2030", grid: "#292D3B",
  goal: "#243650", skill: "#26314A", running: "#304A78",
  related: "#302947", relatedBorder: "rgba(167, 139, 250, 0.25)",
  selectedFill: "#373052", selectedBorder: "#65569A", selected: "#A78BFA",
  failed: "#3A2532", succeededGoal: "#1F3A37", recovery: "#2D2944",
  text: "#F1F2FA", textSecondary: "#A6AABD", textMuted: "#8290A6",
  accent: { running: "#5C8DFF", failure: "#F06470", success: "#2CCB9A", recovery: "#A78BFA" },
};

const SKILLS = { navigate: "Navigate", pick: "Pick", place: "Place", call_coder: "Coding assistant" };
const ICONS = { running: "●", failure: "✗", success: "✓", recovery: "↻" };

/** The interval's execution state for painting: running, failure, success, recovery (a finished recovery interval) or
 * ended (cancelled or ended without a result). */
export function executionState(span) {
  if (span.status === "running" || span.status === "failure") return span.status;
  if (span.detail?.recovery) return "recovery";
  return span.status === "success" ? "success" : "ended";
}

/**
 * SPAN's paint on TRACK ("goal" or "skill") for RELATION ("none", "related" to the selected incident, or "selected":
 * the operator's exact selection): {fill, border, accent, icon, text}. The execution state keeps its accent and icon
 * whatever the relation, so a selected failure still reads as a failure; TEXT never changes.
 */
export function segmentStyle(span, track, relation = "none") {
  const state = executionState(span);
  const base = { running: TIMELINE.running, failure: TIMELINE.failed, recovery: TIMELINE.recovery,
    success: track === "goal" ? TIMELINE.succeededGoal : TIMELINE.skill, ended: track === "goal" ? TIMELINE.goal : TIMELINE.skill }[state];
  const fill = relation === "selected" ? TIMELINE.selectedFill : relation === "related" ? TIMELINE.related : base;
  const border = relation === "selected" ? TIMELINE.selectedBorder : relation === "related" ? TIMELINE.relatedBorder : null;
  const accent = state === "success" && track === "skill" ? null : TIMELINE.accent[state] || null; // a done skill stays quiet
  return { fill, border, accent, icon: ICONS[state] || "", text: TIMELINE.text, state };
}

/** A skill interval's operator label, with the same attempt number the behavior tree shows ("Pick · Attempt 4": the
 * goal's attempt count); a goal keeps its own text. */
export function segmentLabel(span, track) {
  if (track !== "skill") return span.label;
  const name = SKILLS[span.label] || span.label;
  const number = span.detail?.number;
  return number ? `${name} · Attempt ${number}` : name;
}

/** WCAG contrast ratio of two #RRGGBB colours. */
export function contrast(a, b) {
  const lum = (hex) => {
    const [r, g, bl] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
  };
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

const LEVELS = { expected: "Expected", unknown: "Unknown", elevated: "Elevated", anomaly: "Anomaly" };

// the monitor's own labels, as the operator reads them: "working" is the placeholder it decides with while the goal's
// labels are being written (or when they could not be), so it reads as Processing, not as something seen; none of the
// goal's labels fitting reads as Working (the robot busy with something the labels do not name)
const ACTIVITY_NAMES = { none_of_these: "Working", working: "Processing" };

/** An activity label as words: "grasp_jar" -> "grasp jar"; none_of_these -> "Working"; working -> "Processing". */
export function activityText(name) {
  if (!name) return "";
  return ACTIVITY_NAMES[name] || name.replaceAll("_", " ");
}

/** An observation bar's text: the activity the decision model chose (held over its window) and its probability in the
 * newest reading ("grasp jar · 0.82"); without one, its level and readings ("Elevated · 14 readings · 1.0/s"); a gap
 * says how long it lasted ("No reading · 4 s"). */
export function observationLabel(segment) {
  const seconds = Math.max(0, segment.end - segment.start);
  if (segment.level === "missing") return `No reading · ${Math.round(seconds)} s`;
  if (segment.activity) {
    const p = segment.readings.at(-1)?.activity_p;
    return `${activityText(segment.activity)}${p !== null && p !== undefined ? ` · ${p.toFixed(2)}` : ""}`;
  }
  const n = segment.readings.length;
  const rate = seconds >= 1 ? ` · ${(n / seconds).toFixed(1)}/s` : "";
  return `${LEVELS[segment.level] || segment.level} · ${n} reading${n === 1 ? "" : "s"}${rate}`;
}

/** The timeline rows a click inspects: only the interventions row, where the moments worth inspecting are marked. The
 * other rows (goal, skill, decisions, readings, risk) only highlight the bar under the cursor and show its full text. */
export const INSPECT_ROWS = ["interventions"];

/** The name of the row of ROWS ({name: {y, h}}) at Y, or null outside them. */
export function rowAt(rows, y) {
  return Object.entries(rows || {}).find(([, r]) => y >= r.y && y < r.y + r.h)?.[0] ?? null;
}
