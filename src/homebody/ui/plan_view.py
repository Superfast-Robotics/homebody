"""The console's main view while the monitor runs: a status strip that says what the robot is doing, and the
agent's behaviour tree drawn as a plan.

The tree reads the way a behaviour tree runs. A sequence of goals runs left to right; a sequence of larger parts is
drawn as numbered rows, top to bottom; a fallback's first child is drawn as usual and its alternatives beneath it,
dashed, because they run only when it fails. Colour means state and nothing else: blue is the goal being worked on
now, green done, red failed, amber a goal the monitor flagged; pending goals stay quiet. Pure HTML/SVG from a plan
snapshot (session/monitored.plan_snapshot) and the monitor's view.
"""
import textwrap
from dataclasses import dataclass, field
from html import escape

INK, MUTED, RULE, PAPER = "#1D2329", "#5E6A75", "#C7CFD6", "#F7F8F9"
LIVE, LIVE_SOFT, WATCH, WATCH_SOFT = "#2457C5", "#E3EAF8", "#D89A00", "#FBF1D9"
STOP, STOP_SOFT, DONE, DONE_SOFT = "#C2271D", "#F8E2E0", "#2E7A57", "#E2F0E8"
TOKENS = f"""<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@400;600&display=swap");
:root {{--op-ink:{INK}; --op-muted:{MUTED}; --op-rule:{RULE}; --op-paper:{PAPER}; --op-canvas:#ECEFF1;
 --op-live:{LIVE}; --op-live-soft:{LIVE_SOFT}; --op-watch:{WATCH}; --op-watch-soft:{WATCH_SOFT};
 --op-stop:{STOP}; --op-stop-soft:{STOP_SOFT}; --op-done:{DONE}; --op-done-soft:{DONE_SOFT};
 --op-panel:466px; --op-strip:58px; --op-recorder:300px;}}
</style>"""
STYLE = """<style>
.op-strip, .op-map {font-family:"IBM Plex Sans", system-ui, sans-serif; font-variant-numeric:tabular-nums;
 color:var(--op-ink); box-sizing:border-box;}
.op-strip {position:fixed; left:0; top:0; width:calc(100vw - var(--op-panel)); height:var(--op-strip); z-index:21;
 display:flex; align-items:center; gap:18px; padding:0 18px; background:var(--op-paper);
 border-bottom:1px solid var(--op-rule);}
.op-state {display:flex; align-items:center; gap:8px; font-weight:600; font-size:14px; white-space:nowrap;}
.op-state i {width:10px; height:10px; border-radius:50%;}
.op-now {display:flex; flex-direction:column; min-width:0; flex:1;}
.op-now b {font-weight:600; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.op-now span {color:var(--op-muted); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.op-clock {color:var(--op-muted); font-size:12px; text-align:right; white-space:nowrap;}
.op-clock strong {display:block; color:var(--op-ink); font-size:15px; font-weight:600;}
.op-map {position:fixed; left:0; top:var(--op-strip); bottom:var(--op-recorder); width:calc(100vw - var(--op-panel));
 z-index:19; overflow:auto; padding:18px 22px; background:var(--op-canvas);}
.op-map:not([open]) {bottom:auto; width:auto; background:transparent; padding:6px 22px;}
.op-map summary {cursor:pointer; list-style:none; font-size:13px; font-weight:600;}
.op-map summary::-webkit-details-marker {display:none;}
.op-map summary span {font-weight:400; color:var(--op-muted); margin-left:10px; font-size:12px;}
.op-map summary:focus-visible {outline:2px solid var(--op-live); outline-offset:2px;}
.op-map p {margin:4px 0 14px; color:var(--op-muted); font-size:12px;}
.op-map svg {display:block; max-width:100%; height:auto;}
.op-map svg text {font-family:"IBM Plex Sans", system-ui, sans-serif;}
</style>"""
GOAL_W, GOAL_H, ARROW, ROW_GAP, BADGE, ALT_H, ALT_GAP = 216, 104, 36, 30, 150, 34, 14
PER_LINE, LINE_GAP = 4, 40  # steps per line before a long sequence wraps, and the space between its lines
MAP_W, CARD_MAX = 1180, 340  # the main view's drawing width; cards widen into it up to CARD_MAX
TROUBLE_FAILS, TROUBLE_SHARE = 2, 0.6  # a goal is in trouble after this many failed calls, or this share of its budget
STATES = {  # the strip's state word -> its dot
    "Running": LIVE, "Thinking": LIVE, "Executing": LIVE, "Paused": WATCH, "Needs you": WATCH,
    "Ready": MUTED, "Stopped": STOP, "Cancelled": STOP, "Error": STOP, "Model reports done": DONE,
    "Provider error": STOP, "Interrupted": STOP, "Physics invalid": STOP, "Step limit reached": WATCH,
}
ENDED = {"model_done": "Model reports done", "step_limit": "Step limit reached", "provider_error": "Provider error",
         "cancelled": "Cancelled", "error": "Error", "interrupted": "Interrupted",
         "physics_invalid": "Physics invalid"}  # a task's final status -> the strip's word
LIVE_STATES = ("Thinking", "Executing", "Paused", "Needs you")
REASON_MAX = 180


@dataclass
class Block:
    """A laid-out subtree: its size and the SVG it draws at an offset."""
    width: float
    height: float
    parts: list = field(default_factory=list)  # (dx, dy, svg) drawn relative to the block's corner

    def at(self, x, y):
        return "".join(f'<g transform="translate({x + dx:.1f},{y + dy:.1f})">{svg}</g>' for dx, dy, svg in self.parts)


def check_text(check):
    if not check:
        return "the agent closes it"
    return {"holding": f"holding · {check.get('hand', '')} hand", "hand_free": f"{check.get('hand', '')} hand free",
            "placed": "placed", "arrived": "arrived"}.get(check.get("name"), str(check.get("name")))


def trouble(plan, node=None):
    """Why the active goal is in trouble ("2 navigate no route · 3 of 5 calls used"), or "" when it is not: repeated
    failed or refused calls, or most of its budget spent without its check holding."""
    if not plan or not plan.get("active"):
        return ""
    node = node or next((n for n in goals(plan["tree"]) if n["id"] == plan["active"]), {})
    results = plan.get("results") or []
    failed = [r for r in results if r.get("code") not in (None, "OK")]
    budget, used = node.get("budget", 8), plan.get("used", len(results))
    if len(failed) < TROUBLE_FAILS and used < TROUBLE_SHARE * budget:
        return ""
    counts = {}
    for r in failed:
        key = f"{r.get('skill')} {str(r.get('code')).lower().replace('_', ' ')}"
        counts[key] = counts.get(key, 0) + 1
    what = ", ".join(f"{n} {key}" for key, n in counts.items())
    return f"{what + ' · ' if what else ''}{used} of {budget} calls used"


def skipped(plan):
    """Goals that will not run: pending steps after a failed step of the same sequence (its fallback took over)."""
    found = set()

    def walk(node):
        if node["type"] == "goal":
            return
        failed = False
        for child in node["children"]:
            if node["type"] == "sequence" and failed:
                found.update(g["id"] for g in goals(child) if plan["status"].get(g["id"]) == "pending")
            elif child["type"] == "goal" and plan["status"].get(child["id"]) == "failed":
                failed = True
            walk(child)

    walk(plan["tree"])
    return found


def card_width(tree):
    """Cards as wide as the main view allows for its widest line of steps (at least GOAL_W, at most CARD_MAX)."""
    widest, rows = 1, tree["type"] == "sequence" and any(c["type"] != "goal" for c in tree["children"])

    def walk(node):
        nonlocal widest
        if node["type"] == "goal":
            return
        if node["type"] == "sequence" and all(c["type"] == "goal" for c in node["children"]):
            widest = max(widest, min(PER_LINE, len(node["children"])))
        for child in node["children"]:
            walk(child)

    walk(tree)
    room = MAP_W - (BADGE if rows else 0) - (widest - 1) * ARROW
    return max(GOAL_W, min(CARD_MAX, room / widest))


def goal_card(node, plan, phase, width=None):
    width = width or plan.get("card_w", GOAL_W)
    goal_id = node["id"]
    status = "skipped" if goal_id in plan.get("skipped", ()) else plan["status"].get(goal_id, "pending")
    active = goal_id == plan.get("active")
    flag = plan["flags"].get(goal_id)
    reason = plan["reasons"].get(goal_id, "")
    if active:
        fill, stroke, head, head_colour, stroke_width = LIVE_SOFT, LIVE, f"{goal_id} · now", LIVE, 2.5
    elif status == "done":
        fill, stroke, head, head_colour, stroke_width = DONE_SOFT, DONE, f"{goal_id} · done", DONE, 1.4
    elif status == "failed":
        fill, stroke, head, head_colour, stroke_width = STOP_SOFT, STOP, f"{goal_id} · failed", STOP, 1.4
    elif status == "skipped":
        fill, stroke, head, head_colour, stroke_width = PAPER, RULE, f"{goal_id} · skipped", MUTED, 1.0
    else:
        fill, stroke, head, head_colour, stroke_width = "#ffffff", RULE, goal_id, MUTED, 1.3
    explain = (status == "failed" and reason) or (flag and not active)  # the reason is what matters now
    lines = textwrap.wrap(node["goal"], int(width / 7.2)) or [""]
    keep = 1 if explain else 2
    if len(lines) > keep:
        lines = [*lines[:keep - 1], lines[keep - 1][: int(width / 7.2) - 1] + "…"]
    svg = []
    svg.append(f'<title>{escape(goal_id)}: {escape(node["goal"])}; done when {escape(check_text(node.get("check")))}'
               f'{"; " + escape(reason) if reason else ""}{"; monitor: " + escape(flag) if flag else ""}</title>'
               f'<rect x="0" y="0" width="{width}" height="{GOAL_H}" rx="10" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="{stroke_width}"/>'
               f'<text x="14" y="22" font-size="11" font-weight="700" fill="{head_colour}">{escape(head)}</text>')
    ink = MUTED if status == "skipped" else INK
    for index, line in enumerate(lines):
        weight = "600" if index == 0 else "400"
        svg.append(f'<text x="14" y="{42 + 16 * index}" font-size="{13 if index == 0 else 12}" font-weight="{weight}" '
                   f'fill="{ink}">{escape(line)}</text>')
    if active:
        budget = node.get("budget", 8)
        used = min(plan.get("used", 0), budget)
        worry = trouble(plan, node)
        bar = WATCH if worry else LIVE
        svg.append(f'<rect x="14" y="72" width="{width - 28}" height="6" rx="3" fill="#DDE3E8"/>'
                   f'<rect x="14" y="72" width="{(width - 28) * used / max(budget, 1):.1f}" height="6" rx="3" '
                   f'fill="{bar}"/>')
        running = phase is not None and phase.skill
        doing = (phase.skill + (f" · {phase.stage}" if phase.stage else "")) if running else \
            "the agent is choosing the next step"
        footer, colour = (doing, LIVE) if running or not worry else (worry.split(" · ")[0], "#8A6200")
        svg.append(f'<text x="{width - 14}" y="{GOAL_H - 10}" font-size="11" text-anchor="end" fill="{MUTED}">'
                   f'{used} of {budget} calls</text>')
    elif status == "skipped":
        footer, colour = "will not run: an earlier step failed", MUTED
    elif explain:
        text, colour = (f"monitor: {flag}", "#8A6200") if flag and status != "failed" else (reason, STOP)
        wrapped = textwrap.wrap(text, int(width / 6.4))
        if len(wrapped) > 3:
            wrapped = [*wrapped[:2], wrapped[2][: int(width / 6.4) - 1] + "…"]
        for index, line in enumerate(wrapped[:3]):
            svg.append(f'<text x="14" y="{62 + 14 * index}" font-size="11" fill="{colour}">{escape(line)}</text>')
        footer, colour = "", MUTED
    else:
        footer, colour = f"done when {check_text(node.get('check'))}", MUTED
    limit = int(width / 6.2) - (12 if active else 0)
    svg.append(f'<text x="14" y="{GOAL_H - 10}" font-size="11" fill="{colour}">'
               f'{escape(footer if len(footer) <= limit else footer[: limit - 1] + "…")}</text>')
    if flag:
        svg.append(f'<circle cx="{width - 14}" cy="14" r="9" fill="{WATCH}"/><text x="{width - 14}" y="18" '
                   f'font-size="12" font-weight="700" text-anchor="middle" fill="#fff">!</text>')
    return Block(width, GOAL_H, [(0, 0, "".join(svg))])


def alternative_card(node, plan, width):
    goal_id = node["id"]
    status = plan["status"].get(goal_id, "pending")
    colour = {"done": DONE, "failed": STOP}.get(status, MUTED)
    text = (f"if it fails: {goal_id} · {node['goal']}" if status == "pending" else
            f"Fallback ran: {goal_id} · {status} · {node['goal']}")
    limit = int(width / 6.6)
    svg = (f'<title>{escape(text)}</title><rect x="0" y="0" width="{width}" height="{ALT_H}" rx="8" fill="{PAPER}" '
           f'stroke="{RULE if status == "pending" else colour}" stroke-dasharray="4 3"/>'
           f'<text x="14" y="22" font-size="12" fill="{colour}">'
           f'{escape(text if len(text) <= limit else text[: limit - 1] + "…")}</text>')
    return Block(width, ALT_H, [(0, 0, svg)])


def arrow(x0, y0, x1, y1, dashed=False):
    dash = ' stroke-dasharray="3 3"' if dashed else ""
    head = "" if dashed else ' marker-end="url(#op-arrow)"'
    return f'<path d="M{x0:.1f},{y0:.1f} L{x1:.1f},{y1:.1f}" stroke="#9AA6B1" stroke-width="1.5" fill="none"{dash}{head}/>'


def layout(node, plan, phase):
    if node["type"] == "goal":
        return goal_card(node, plan, phase)
    children = node["children"]
    if node["type"] == "fallback":
        primary = layout(children[0], plan, phase)
        width, parts, y = primary.width, [(0, 0, primary.at(0, 0))], primary.height
        for child in children[1:]:
            if child["type"] == "goal" and child["id"] != plan.get("active"):
                alt = alternative_card(child, plan, max(width, plan.get("card_w", GOAL_W)))
            else:  # a running fallback is drawn as a full card, live
                alt = layout(child, plan, phase)
            parts.append((0, 0, arrow(22, y, 22, y + ALT_GAP, dashed=True)))
            parts.append((0, y + ALT_GAP, alt.at(0, 0)))
            y += ALT_GAP + alt.height
            width = max(width, alt.width)
        return Block(width, y, parts)
    blocks = [layout(child, plan, phase) for child in children]
    if all(child["type"] == "goal" for child in children):  # steps: left to right, wrapping after PER_LINE
        parts, width, width_above, y = [], 0.0, 0.0, 0.0
        for start in range(0, len(blocks), PER_LINE):
            line, x = blocks[start:start + PER_LINE], 0.0
            if start:  # carry on from the end of the line above to the start of this one
                half = blocks[0].width / 2
                parts.append((0, 0, (f'<path d="M{width_above - half:.1f},{y - LINE_GAP:.1f} '
                                     f'V{y - LINE_GAP / 2:.1f} H{half:.1f} V{y - 4:.1f}" stroke="#9AA6B1" '
                                     f'stroke-width="1.5" fill="none" marker-end="url(#op-arrow)"/>')))
            for index, block in enumerate(line):
                if index:
                    parts.append((0, 0, arrow(x - ARROW, y + GOAL_H / 2, x - 4, y + GOAL_H / 2)))
                parts.append((x, y, block.at(0, 0)))
                x += block.width + ARROW
            width_above, width = x - ARROW, max(width, x - ARROW)
            y += max(b.height for b in line) + LINE_GAP
        return Block(width, y - LINE_GAP, parts)
    parts, y = [], 0.0  # parts of the task: numbered rows, top to bottom
    for index, (child, block) in enumerate(zip(children, blocks, strict=True)):
        if index:
            parts.append((0, 0, arrow(12, y - ROW_GAP + 4, 12, y + 4)))
        label = row_label(child)
        parts.append((0, y, (f'<circle cx="12" cy="20" r="11" fill="{INK}"/><text x="12" y="24" font-size="11" '
                            f'font-weight="700" text-anchor="middle" fill="#fff">{index + 1}</text>'
                            f'<text x="30" y="24" font-size="12" font-weight="600" fill="{INK}">{escape(label)}</text>')))
        parts.append((BADGE, y, block.at(0, 0)))
        y += block.height + ROW_GAP
    return Block(BADGE + max(b.width for b in blocks), y - ROW_GAP, parts)


def row_label(node):
    """A short name for a row: what its goal ids share (jam_pick, jam_carry: "Jam"), else its first goal's text."""
    ids = [goal["id"] for goal in goals(node)]
    shared = ids[0].split("_")
    for goal_id in ids[1:]:
        parts = goal_id.split("_")
        shared = [a for a, b in zip(shared, parts, strict=False) if a == b][:len(shared)]
        shared = shared[:next((i for i, (a, b) in enumerate(zip(shared, parts, strict=False)) if a != b), len(shared))]
    if shared and len(ids) > 1 and "_".join(shared) not in ids:
        return " ".join(shared).capitalize()[:17]
    while node["type"] != "goal":
        node = node["children"][0]
    text = node["goal"]
    return text if len(text) <= 17 else text[:16].rstrip() + "…"


def tree_svg(plan, phase=None):
    """The behaviour tree of PLAN ({tree, status, flags, reasons, active, used, results}) as SVG."""
    plan = {**plan, "skipped": skipped(plan), "card_w": card_width(plan["tree"])}
    block = layout(plan["tree"], plan, phase)
    margin = 10
    return (f'<svg width="{block.width + 2 * margin:.0f}" height="{block.height + 2 * margin:.0f}" '
            f'viewBox="{-margin} {-margin} {block.width + 2 * margin:.0f} {block.height + 2 * margin:.0f}" role="img" '
            f'aria-label="The agent\'s behaviour tree"><defs><marker id="op-arrow" viewBox="0 0 8 8" refX="7" '
            f'refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9AA6B1"/>'
            f'</marker></defs>{block.at(0, 0)}</svg>')


def plan_view(plan, phase=None, task=""):
    """The main view: the plan, or what to expect before the agent has one; collapsible to show the 3D scene."""
    if not plan:
        what = escape(task) if task else "Enter a task in the panel and press Start task."
        return (f'<details class="op-map" open><summary>Plan</summary>'
                f'<p>The agent writes its plan as its first step. {what}</p></details>')
    counts = {state: sum(1 for s in plan["status"].values() if s == state) for state in ("done", "failed", "pending")}
    summary = f'{counts["done"]} done, {counts["failed"]} failed, {counts["pending"]} to go'
    return (f'<details class="op-map" open><summary>Plan<span>{summary}</span></summary>'
            f'<p>Steps run left to right; numbered rows run in order; dashed boxes are fallbacks, used only if the '
            f'step above them fails.</p>{tree_svg(plan, phase)}</details>')


def clock(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def status_strip(state, view=None, decisions=0, note=""):
    """The one line that answers whether the run is fine and what the robot is doing."""
    state = ENDED.get(state, state)
    dot = STATES.get(state, MUTED)
    goal, doing, elapsed = "No task running", note or "Enter a task in the panel and press Start task.", "00:00"
    if view is not None and view.plan:
        active = view.plan.get("active")
        node = next((n for n in goals(view.plan["tree"]) if n["id"] == active), None)
        goal = f"{active} · {node['goal']}" if node else "Between goals"
    elif view is not None and view.task:
        goal = view.task
    ended = state in ENDED.values()
    if ended and note:  # the last phase is stale once the task is over; its reason is what to read
        doing = note if len(note) <= REASON_MAX else note[:REASON_MAX - 1] + "…"
    elif view is not None and view.phase is not None and state in LIVE_STATES:
        phase = view.phase
        doing = (f"{phase.skill}: {phase.stage}" if phase.stage else phase.skill) if phase.skill else \
            "the agent is choosing the next step" if phase.name == "planning" else doing
    if view is not None and view.sim_start is not None and view.sim_now is not None:
        elapsed = clock(view.sim_now - view.sim_start)
    word = "Running" if state in ("Thinking", "Executing") else state
    worry = trouble(view.plan) if view is not None and word == "Running" else ""
    if worry:  # a goal burning its budget is the thing to see, whether or not the clock moves
        word, dot, doing = "Needs attention", WATCH, worry
    return (f'<div class="op-strip" role="status" aria-live="polite"><div class="op-state">'
            f'<i style="background:{dot}; box-shadow:0 0 0 4px {dot}22"></i>{escape(word)}</div>'
            f'<div class="op-now"><b>{escape(goal)}</b><span>{escape(doing)}</span></div>'
            f'<div class="op-clock"><strong>{elapsed}</strong>simulated · {decisions} decisions</div></div>')


def goals(tree):
    if tree["type"] == "goal":
        yield tree
        return
    for child in tree["children"]:
        yield from goals(child)
