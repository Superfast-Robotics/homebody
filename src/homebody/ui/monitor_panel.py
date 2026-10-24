"""The console's monitor views: in the side panel, the latest alert and what the monitor sees now against its
thresholds; along the bottom of the main view, the timeline of the task as the monitor saw it (goals, skill calls,
readings, risk against its threshold, and the monitor's and the operator's interventions). Pure HTML/SVG from one
monitor View; colours are the operator console's (ui/plan_view.py)."""
from html import escape
from itertools import pairwise

from homebody.monitor.targets import STATIC
from homebody.ui.plan_view import DONE, INK, LIVE, LIVE_SOFT, STOP, STOP_SOFT, WATCH, clock

STYLE = """<style>
.op-mon {font-family:"IBM Plex Sans", system-ui, sans-serif; font-variant-numeric:tabular-nums; color:#1D2329;
 font-size:12px; line-height:1.45; margin-bottom:10px;}
.op-mon h4 {margin:12px 0 6px; font-size:12px; font-weight:600; color:#5E6A75;}
.op-alert {border:1px solid var(--op-watch); background:var(--op-watch-soft); border-radius:6px; padding:10px 12px;}
.op-alert.is-stop {border-color:var(--op-stop); background:var(--op-stop-soft);}
.op-alert b {display:block; font-size:13px;}
.op-calm {color:#5E6A75;}
.op-read {display:grid; grid-template-columns:118px 1fr 36px; gap:6px 8px; align-items:center;}
.op-read span {overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.op-track {position:relative; height:8px; background:#DDE3E8; border-radius:4px;}
.op-fill {height:8px; border-radius:4px;}
.op-tau {position:absolute; top:-4px; width:2px; height:16px; background:#1D2329;}
.op-read output {text-align:right;}
.op-chips {display:flex; flex-wrap:wrap; gap:5px;}
.op-chips span {padding:2px 7px; border:1px solid #C7CFD6; border-radius:10px; background:#fff; font-size:11px;}
.op-chips .is-failure::before {content:"●"; color:var(--op-watch); margin-right:4px; font-size:9px;}
.op-chips .is-static {border-style:dashed;}
.op-context {color:#5E6A75; font-size:11px; margin:-2px 0 6px;}
.op-context.is-stale {color:#8A6200;}
.op-read.is-stale {opacity:.45;}
.op-chips .is-now {border-color:var(--op-live); color:var(--op-live); font-weight:600;}
.op-foot {margin-top:8px; color:#5E6A75; font-size:11px;}
/* the console's older pieces, in this console: red only for stopped or unsafe; the reasoning never fades */
.hb-bubble {animation:none !important;}
.hb-stage {border-left-color:var(--op-live) !important;}
.hb-steps li.is-current {border-color:var(--op-live) !important; color:var(--op-live) !important;
 background:var(--op-live-soft) !important;}
.hb-steps li.is-failed {border-color:var(--op-stop) !important; color:var(--op-stop) !important;}
.hb-wordmark {font-size:16px !important;}
.hb-wordmark span {color:var(--op-ink) !important;}
.hb-live::before {background:var(--op-done) !important; animation:none !important;}
.hb-room {white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:230px;}
.hb-title {font-size:11px !important; font-weight:400 !important; color:var(--op-muted) !important;}
.hb-message {font-size:14px !important;}
.hb-refused {border-left-color:var(--op-watch) !important; background:var(--op-watch-soft) !important;}
.hb-refused .hb-steps li {border-color:var(--op-watch) !important; color:#8A6200 !important; background:#fff !important;}
</style>"""
BAR_STYLE = """<style>
.op-recorder {position:fixed; left:0; bottom:0; width:calc(100vw - var(--op-panel, 480px));
 height:var(--op-recorder, 226px); z-index:20; box-sizing:border-box; padding:8px 18px 4px;
 background:var(--op-paper, #F7F8F9); border-top:1px solid var(--op-rule, #C7CFD6); color:#1D2329;
 font-family:"IBM Plex Sans", system-ui, sans-serif;}
.op-recorder header {display:flex; justify-content:space-between; align-items:baseline; gap:12px;}
.op-recorder h2 {margin:0; font-size:13px; font-weight:600;}
.op-recorder header span {color:#5E6A75; font-size:11px;}
.op-recorder header i {display:inline-block; width:9px; height:9px; margin:0 4px 0 10px; border-radius:2px;}
.op-recorder svg {display:block; width:100%; height:calc(100% - 26px);}
.op-recorder text {font-family:"IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-size:11px; fill:#5E6A75;}
.op-recorder .lane {fill:#1D2329; font-weight:600;}
</style>"""
KIND_FILL = {"progress": DONE, "success": LIVE, "failure": STOP, "static": "#B9C2CA"}
WORDS = {"abort_skill": "monitor stopped", "reopen_goal": "monitor reopened", "flag": "monitor flagged"}
OPERATOR = {"stop_skill": "you stopped", "note": "you told the agent", "dismiss": "you dismissed",
            "pause": "you paused", "resume": "you resumed"}


def kinds(view) -> dict:
    found = {label.name: label.kind for label in view.vocabulary.labels} if view.vocabulary else {}
    return {**found, **{name: "static" for name in STATIC}}


QUIET = "#A7B1BA"  # a reading below its threshold: it will not act, so it carries no alarm colour


def bar(name, p, colour, tau, acts=None):
    """One reading against its threshold; it takes COLOUR only when it would act (ACTS, else at or above TAU)."""
    acts = p >= tau if acts is None else acts
    return (f'<span title="{escape(name)}">{escape(name.replace("_", " "))}</span><div class="op-track">'
            f'<div class="op-fill" style="width:{100 * p:.0f}%; background:{colour if acts else QUIET}"></div>'
            f'<div class="op-tau" style="left:{100 * tau:.0f}%"></div></div><output>{p:.2f}</output>')


def alert_html(view):
    alert = view.alert
    if alert is None:
        return '<div class="op-calm">No alerts. The monitor stops a skill only when it is sure for several readings.</div>'
    stop = alert.action in ("abort_skill", "reopen_goal")
    what = {"abort_skill": f"Stopped the skill in {alert.goal_id}", "reopen_goal": f"Reopened {alert.goal_id}",
            "flag": f"Flagged {alert.goal_id}"}.get(alert.action, alert.action)
    label = str(alert.label).replace("_", " ")
    return (f'<div class="op-alert{" is-stop" if stop else ""}" role="alert"><b>{escape(what)} at '
            f'{alert.sim_time:.0f} s</b>{escape(label)} ({alert.p:.2f}){": " + escape(alert.reason) if alert.reason else ""}'
            f'</div>')


def reading_context(view):
    """When the reading was taken and during what, and whether it is stale (an earlier skill or goal)."""
    reading = view.reading
    origin = view.sim_start if view.sim_start is not None else reading.sim_time
    stale = reading.goal_id != view.goal_id or (reading.phase.name == "tool_call"
                                                and reading.phase.call != view.skill_call)
    during = (f"during {reading.phase.skill} in {reading.goal_id}" if reading.phase.skill else
              f"while the agent chose a step in {reading.goal_id}")
    note = "; from an earlier step, the world has not moved since" if stale else ""
    return stale, f"Reading at {clock(reading.sim_time - origin)}, {during}{note}"


def reading_html(view, tau, tau_unsafe):
    reading = view.reading
    if reading is None:
        return '<div class="op-calm">No reading yet.</div>'
    stale, context = reading_context(view)
    kind = kinds(view)
    ranked = sorted(reading.activity.items(), key=lambda item: -item[1])[:4]
    rows = [bar(name, p, KIND_FILL.get(kind.get(name, "static")), tau,
                acts=True if kind.get(name) in ("progress", "success") else None) for name, p in ranked]
    for name, p, threshold, colour in (("unsafe", reading.unsafe, tau_unsafe, STOP),
                                       ("needs a person", reading.human_needed, tau, WATCH)):
        if p is not None:
            rows.append(bar(name, p, colour, threshold))
    for goal_id, holds in reading.still_true.items():  # drawn as its loss, so every bar acts at the high end
        rows.append(bar(f"{goal_id} no longer holds", 1 - holds, STOP, tau))
    if reading.progress:
        rows.append(bar("no progress", reading.progress.get("no_change", 0.0), WATCH, tau))
    return (f'<div class="op-context{" is-stale" if stale else ""}">{escape(context)}</div>'
            f'<div class="op-read{" is-stale" if stale else ""}">{"".join(rows)}</div>'
            f'<div class="op-foot">The line on each bar is where the monitor would act.</div>')


def chips(view):
    if view.vocabulary is None:
        return ""
    now = view.reading.top()[0] if view.reading else None
    items = [(label.name, label.kind) for label in view.vocabulary.labels] + [(name, "static") for name in STATIC]
    spans = "".join(f'<span class="is-{kind}{" is-now" if name == now else ""}">{escape(name.replace("_", " "))}</span>'
                    for name, kind in items)
    source = "" if view.vocabulary.source == "model" else " (generic until the vocabulary model answers)"
    return (f'<h4>What it looks for in {escape(view.vocabulary.goal_id)}{escape(source)}</h4>'
            f'<div class="op-chips">{spans}</div>')


def monitor_html(view, tau: float = 0.8, tau_unsafe: float = 0.9) -> str:
    """The side panel's monitor section."""
    foot = (f'{escape(view.status)}; one reading every {view.period_s:.1f} s (p95 {view.p95_s:.1f} s); '
            f'{view.calls} readings, ${view.cost_usd:.3f}')
    return (f'<div class="op-mon">{alert_html(view)}<h4>What the monitor sees now</h4>'
            f'{reading_html(view, tau, tau_unsafe)}{chips(view)}<div class="op-foot">{foot}</div></div>')


LANES = (("Goal", 18), ("Skill", 48), ("Agent", 78), ("Monitor saw", 112), ("Risk", 156), ("Interventions", 206))
RISK_TOP, RISK_BOTTOM = 140, 180  # the risk lane: probability 1 at the top, 0 at the bottom
LABEL_MIN = 44  # px a segment needs before its label is drawn in it


def timeline_svg(view, width=1200, left=104, right=14, tau=0.8, ema=1.0):
    """The task as the monitor saw it, on a simulation time axis from its first event to now."""
    times = [t for t, *_ in view.history] + [t for t, _ in view.goals] + [t for t, *_ in view.marks] + \
            [t for span in view.spans for t in span[:2] if t is not None] + list(view.decisions)
    if not times:
        return (f'<svg viewBox="0 0 {width} 236"><text x="{left}" y="110">The timeline fills in once a task '
                f'starts.</text></svg>')
    start, end = min(times), max(times) + 1e-6
    origin = view.sim_start if view.sim_start is not None else start
    scale = (width - left - right) / max(end - start, 1.0)
    status = (view.plan or {}).get("status", {})

    def x(t):
        return left + (t - start) * scale

    def risk_y(p):
        return RISK_BOTTOM - (RISK_BOTTOM - RISK_TOP) * p

    parts = [f'<text class="lane" x="0" y="{y + 4}">{name}</text>' for name, y in LANES]
    tick = next(step for step in (5, 10, 20, 30, 60, 120, 300, 600, 1200) if (end - start) / step <= 10)
    for t in range(int(start // tick + 1) * tick, int(end) + 1, tick):
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="6" y2="218" stroke="#DDE3E8"/>'
                     f'<text x="{x(t):.1f}" y="232" text-anchor="middle">{clock(t - origin)}</text>')
    marks = list(view.goals) + [(end, None)]
    for (t0, goal), (t1, _) in pairwise(marks):
        active = bool(view.plan) and goal == view.plan.get("active")
        state = status.get(goal)
        fill, ink = ((LIVE_SOFT, LIVE) if active else (STOP_SOFT, STOP) if state == "failed" else
                     ("#E2F0E8", DONE) if state == "done" else ("#DDE3E8", INK))
        label = "planning" if goal == "task" else str(goal)
        span = x(t1) - x(t0)
        parts.append(f'<rect x="{x(t0):.1f}" y="8" width="{max(span, 1):.1f}" height="18" fill="{fill}">'
                     f'<title>{escape(label)} from {clock(t0 - origin)}{": " + state if state else ""}</title></rect>'
                     + (f'<text x="{x(t0) + 4:.1f}" y="21" fill="{ink}">{escape(label)}</text>' if span >= LABEL_MIN else ""))
    for t0, t1, skill, goal, code in view.spans:
        running = t1 is None
        t1 = end if running else t1
        word = "running" if running else "ok" if code == "OK" else str(code).lower().replace("_", " ")
        tip = f'<title>{clock(t0 - origin)} {escape(str(skill))} in {escape(str(goal))}: {escape(word)}</title>'
        if not running and code != "OK" and t1 - t0 < 0.25:  # refused before it moved: a tick, not a bar
            parts.append(f'<rect x="{x(t0) - 1.5:.1f}" y="36" width="3" height="22" fill="{STOP}">{tip}</rect>')
            continue
        colour, fill = (LIVE, LIVE_SOFT) if running else (DONE, "#E2F0E8") if code == "OK" else (STOP, STOP_SOFT)
        span = x(t1) - x(t0)
        parts.append(f'<rect x="{x(t0):.1f}" y="38" width="{max(span, 2):.1f}" height="18" rx="3" fill="{fill}" '
                     f'stroke="{colour}">{tip}</rect>'
                     + (f'<text x="{x(t0) + 4:.1f}" y="51" fill="{colour}">{escape(str(skill))} · {escape(word)}</text>'
                        if span >= LABEL_MIN else ""))
    stacks = {}
    for t in view.decisions:  # decisions at one simulation time: the world held still while the agent thought
        stacks[round(t, 1)] = stacks.get(round(t, 1), 0) + 1
    for t, count in stacks.items():
        colour = WATCH if count >= 3 else "#5E6A75"
        parts.append(f'<circle cx="{x(t):.1f}" cy="76" r="3.5" fill="{colour}"><title>{clock(t - origin)}: '
                     f'{count} decision{"s" if count != 1 else ""}</title></circle>'
                     + (f'<text x="{x(t) + 6:.1f}" y="80" fill="{colour}">×{count}</text>' if count > 1 else ""))
    smoothed, risk_points = None, []
    for t, top, p, _action, *rest in view.history:
        kind = rest[0] if rest else "static"
        height = 4 + 20 * float(p)
        parts.append(f'<rect x="{x(t):.1f}" y="{122 - height:.1f}" width="2.5" height="{height:.1f}" '
                     f'fill="{KIND_FILL.get(kind, KIND_FILL["static"])}"><title>{clock(t - origin)} '
                     f'{escape(str(top).replace("_", " "))} {float(p):.2f}</title></rect>')
        if len(rest) > 1:  # averaged as the rules average it, so the line crosses the threshold when they would act
            smoothed = float(rest[1]) if smoothed is None else ema * float(rest[1]) + (1 - ema) * smoothed
            risk_points.append(f"{x(t):.1f},{risk_y(smoothed):.1f}")
    for value, label in ((0.0, "0"), (tau, f"{tau:.1f} stops"), (1.0, "1")):
        parts.append(f'<text x="{left - 6}" y="{risk_y(value) + 4:.1f}" text-anchor="end">{label}</text>')
    parts.append(f'<rect x="{left}" y="{RISK_TOP}" width="{width - left - right}" height="{RISK_BOTTOM - RISK_TOP}" '
                 f'fill="none" stroke="#E6EAED"/><line x1="{left}" x2="{width - right}" y1="{risk_y(tau):.1f}" '
                 f'y2="{risk_y(tau):.1f}" stroke="{STOP}" stroke-dasharray="3 4"><title>the monitor acts above '
                 f'{tau:.2f}</title></line>')
    if len(risk_points) > 1:
        parts.append(f'<polyline points="{" ".join(risk_points)}" fill="none" stroke="{WATCH}" stroke-width="1.6"/>')
    events = [(v.sim_time, WORDS.get(v.action, v.action), str(v.label).replace("_", " "),
               STOP if v.action in ("abort_skill", "reopen_goal") else WATCH) for v in view.verdicts]
    events += [(t, OPERATOR.get(kind, kind), text, INK) for t, kind, text in view.marks]
    if not events:
        parts.append(f'<text x="{left}" y="210">None yet</text>')
    for t, word, text, colour in sorted(events, key=lambda event: event[0]):
        ex = x(t)
        parts.append(f'<path d="M{ex:.1f},198 l6,10 l-12,0 z" fill="{colour}"><title>{clock(t - origin)} '
                     f'{escape(word)}: {escape(text)}</title></path><text x="{ex + 8:.1f}" y="208" fill="{colour}">'
                     f'{escape(word)}: {escape(text[:40])}</text>')
    if view.sim_now is not None and start <= view.sim_now <= end:
        now = x(view.sim_now)
        parts.append(f'<line x1="{now:.1f}" x2="{now:.1f}" y1="4" y2="218" stroke="{LIVE}" stroke-width="2"/>'
                     f'<text x="{now - 4:.1f}" y="12" text-anchor="end" fill="{LIVE}">now</text>')
    return f'<svg viewBox="0 0 {width} 236" preserveAspectRatio="none">{"".join(parts)}</svg>'


def timeline_bar(view, tau=0.8, ema=1.0):
    """The bottom bar of the main view: the timeline, each key beside the lanes it explains."""
    saw = "".join(f'<i style="background:{colour}"></i>{name}' for name, colour in (
        ("expected", DONE), ("going wrong", STOP), ("nothing it knows", "#B9C2CA")))
    return (f'<section class="op-recorder" aria-label="Timeline"><header><h2>Timeline</h2>'
            f'<span>Agent: a dot per decision (×n: the world held still) · Monitor saw: {saw}, height is confidence · '
            f'Risk: <i style="background:{WATCH}"></i>averaged, dashed line where it acts</span></header>'
            f'{timeline_svg(view, tau=tau, ema=ema)}</section>')
