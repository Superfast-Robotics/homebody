"""The live exploration console that `homebody real2sim explore` starts on the workstation.

The console zeros the map at the start point, and again after the SLAM restart that Start
exploration performs. Under the controls: a status line, the map legend, the camera and the
model's decisions; the rest is under Details.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import numpy as np

from homebody.backends.g1.assets import VISUAL_URDF
from homebody.helpers.floor import departure_status
from homebody.primitives.navigation.errors import ObservationUnavailable
from homebody.ui import theme
from homebody.vlm.catalog import model_label

from . import vlm_pick
from .explorer import Explorer
from .livemap import RES_M, Boundary, rect_boundary
from .planner import EXECUTION_RADIUS_M
from .scene import EnvironmentView
from .settings import current, from_json, install
from .sources import RobotSource

REPO = Path(__file__).resolve().parents[4]
IDLE_PHASES = ("idle", "done", "stopped", "aborted", "error")
MAP_STALE_S = 3.0  # a map that fails to ingest this long stops exploration
RECORDER = "homebody_bag_ws"  # the recorder container record.sh starts
PURPLE, BLUE, TEAL, ORANGE = (124, 92, 255), (60, 130, 255), (13, 150, 135), (255, 140, 40)
GREY, SKY, RED, LILAC, PALE = (
    (150, 150, 160),
    (175, 205, 240),
    (255, 60, 60),
    (180, 160, 255),
    (170, 205, 255),
)
WALKED = (90, 80, 120)
PIN = (255, 210, 40)  # the start pose marker at the map origin
CONSOLE_FPS = 30.0  # the camera pane follows every new frame up to this rate
TICK_S = 0.3  # the status loop
SLOW_EVERY_S = 2.0  # skeleton, environment cloud, open frontier
NONE = np.zeros((1, 3), np.float32)  # an empty point cloud (viser wants at least one point)
NONE_SEG = np.zeros((1, 2, 3), np.float32)
HIDDEN = (0.0, 0.0, -5.0)  # under the floor: a marker with nothing to mark
BLANK_CAMERA = np.full((270, 480, 3), 13, np.uint8)  # the camera pane while no phone frame arrives
ODOM_STALE = {"odom_missing", "odom_stale", "odom_delayed"}  # freshness reasons naming odometry
LEGEND = (("Reachable floor", vlm_pick.REACHABLE_FLOOR, "area"), ("Walls", GREY, "area"),
          ("Task region", vlm_pick.TASK_REGION_COLOR, "line"), ("Planned route", BLUE, "line"),
          ("Travelled route", TEAL, "line"), ("Frontiers", RED, "dot"))
MAP_LEGEND = ('<div class="hb-legend" role="list">' + "".join(
    f'<span role="listitem"><i class="{shape}" style="--c:rgb{color}"></i>{name}</span>'
    for name, color, shape in LEGEND) + "</div>")
LAYERS_NOTE = ("Also drawn: the live LiDAR scan (orange points), the path network (purple), "
               "wall points (light blue), your boundary (orange line) and the start (yellow).")
NO_NEXT_STOP = "**Next stop:** none."
STREAM_WORDS = {"iphone_color": "camera", "ws_odometry": "odometry"}  # the recorder's streams


EXPLORE_STYLE = """<style>
.hb-exploration-status {margin:10px 0 12px; padding:12px;
 border:1px solid #e7e2dc; border-radius:8px; color:#2e2d29;}
.hb-exploration-status header {display:flex; flex-wrap:wrap; gap:4px 12px;
 align-items:baseline; font-size:12px; color:#585754; font-variant-numeric:tabular-nums;}
.hb-exploration-status header strong {font-size:14px; color:#2e2d29;}
.hb-exploration-status .is-stale {color:var(--hb-accent); font-weight:650;}
.hb-exploration-status p {margin:8px 0 0; font-size:12px; line-height:1.45;
 overflow-wrap:anywhere; white-space:pre-wrap;}
.hb-exploration-status p.hb-warning {color:var(--hb-accent); font-weight:650;}
.hb-exploration-folder-note {padding:0 10px;}
.hb-legend {display:grid; grid-template-columns:1fr 1fr; gap:6px 14px; margin:2px 0 10px;
 padding:0 10px; font-size:12px; color:#2e2d29;}
.hb-legend span {display:flex; align-items:center; gap:7px;}
.hb-legend i {flex:none; width:12px; height:12px; border-radius:3px; background:var(--c);
 box-shadow:inset 0 0 0 1px rgba(0,0,0,.15);}
.hb-legend i.line {height:3px; border-radius:2px; box-shadow:none;}
.hb-legend i.dot {width:8px; height:8px; margin:0 2px; border-radius:50%; box-shadow:none;}
.hb-decisions {padding:0 10px 6px; color:#2e2d29; font-size:12px;}
.hb-decisions > p {margin:0 0 6px; color:#585754;}
.hb-decisions ol {margin:0; padding:0; list-style:none;}
.hb-decisions li {padding:9px 0; border-top:1px solid #e7e2dc; overflow-wrap:anywhere;}
.hb-decisions header {display:flex; justify-content:space-between; gap:8px;
 align-items:baseline;}
.hb-decisions time {flex:none; color:#6d6c69; font-size:11px; font-variant-numeric:tabular-nums;}
.hb-decisions li p {margin:4px 0 0; line-height:1.45;}
.hb-decisions li p b {font-weight:650; color:#585754;}
/* Keep folder headings in normal flow, including long nested decision titles. */
body:has(.hb-exploration-status) .mantine-Paper-root:has(> [aria-hidden])
 > .mantine-Paper-root {position:static; transform:none; margin:0;
 padding:8px 10px; overflow-wrap:anywhere; line-height:1.45;}
body:has(.hb-exploration-status) .mantine-Paper-root > [aria-hidden] {overflow:hidden;}
body:has(.hb-exploration-status) .mantine-Paper-root:has(> [aria-hidden])
 {margin-bottom:8px !important;}
/* Viser's collapsed placeholder is unnecessary with an in-flow heading. */
body:has(.hb-exploration-status) .mantine-Paper-root > [aria-hidden]:has(> :empty)
 {display:none !important;}
</style>"""


def model_name() -> str:
    """The configured model as the console names it: its tool or API, model id and effort."""
    return model_label(current().agent)


def sentence(text) -> str:
    """TEXT with its whitespace collapsed and a full stop unless it already ends a sentence."""
    text = " ".join(str(text).split())
    return text if not text or text[-1] in ".!?…" else text + "."


def exploration_status(phase, elapsed_s, budget_s, paused_s, where, feeds, message, warning=""):
    """The escaped status card: phase, time, position and each feed's freshness on one line,
    then the current message and any warning."""
    timing = f"{elapsed_s / 60:.1f} of {budget_s / 60:g} min"
    if paused_s:
        timing += f" ({paused_s / 60:.1f} min paused)"
    parts = f"<span>{escape(timing)}</span><span>{escape(where)}</span>" + "".join(
        f'<span class="{"is-fresh" if fresh else "is-stale"}">'
        f'{escape(name)} {"fresh" if fresh else "stale"}</span>' for name, fresh in feeds)
    text = message.replace("**", "").replace("`", "")
    return ('<section class="hb-exploration-status" role="status">'
            f'<header><strong>{escape(phase)}</strong>{parts}</header>'
            + (f'<p>{escape(text)}</p>' if text else "")
            + (f'<p class="hb-warning">{escape(warning)}</p>' if warning else "")
            + '</section>')


def decision_map_label(explorer) -> str:
    return "The map it was given" + (" (your boundary)" if explorer.zone is not None else "")


def camera_caption(cam) -> str:
    """The camera block's one line: the latest view's age, or why there is none."""
    status, age = cam.get("status"), cam.get("age_s")
    if status not in ("fresh", "stale") or age is None:
        return "No camera image yet."
    return (f"Latest view, {age:.1f} s old." if status == "fresh"
            else f"Stale: the latest view is {age:.1f} s old.")


class EmptyHandshakeFilter(logging.Filter):
    """Ignore a probe closing before sending HTTP; retain malformed nonempty requests."""

    def filter(self, record):
        if record.getMessage() != "opening handshake failed" or not record.exc_info:
            return True
        exc = record.exc_info[1]
        seen, request_line_eof = set(), False
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            if isinstance(exc, EOFError):
                request_line_eof |= "connection closed while reading HTTP request line" in str(exc)
                if request_line_eof and "stream ends after 0 bytes, before end of line" in str(exc):
                    return False
            exc = exc.__cause__ or exc.__context__
        return True


def record_timing(timings, name, wall_start, cpu_start):
    """Replace stage NAME's timing record in TIMINGS with its updated totals."""
    wall, cpu = time.monotonic() - wall_start, time.thread_time() - cpu_start
    old = timings.get(name, {})
    timings[name] = {"count": old.get("count", 0) + 1, "wall_s": wall, "thread_cpu_s": cpu,
                     "max_wall_s": max(wall, old.get("max_wall_s", 0)),
                     "max_thread_cpu_s": max(cpu, old.get("max_thread_cpu_s", 0)),
                     "total_wall_s": old.get("total_wall_s", 0) + wall,
                     "total_thread_cpu_s": old.get("total_thread_cpu_s", 0) + cpu,
                     "completed_wall_s": time.time()}


def body_link_diagnostics(source):
    backend = getattr(source, "backend", None)
    host = getattr(backend, "_host", None)
    if host is None:
        return None
    age = None if not host.state_at else max(0, time.monotonic() - host.state_at)
    return {"state_age_s": age,
            "ticks": host.ticks, "interruptions": host.interruptions,
            "last_interruption": host.last_interruption,
            "callback_dropped": host.callback_dropped, "callback_error": host.callback_error,
            "control_error": str(getattr(backend, "_control_error", None) or "")}


def zone_preset(exploration) -> tuple[float, float, float, float] | None:
    """(length ahead, width, start x, start y) of the square start zone in the settings, the robot
    0.5 m in from its back edge and centred unless zone_start_xy_m says otherwise; None without one."""
    side = exploration.zone_size_m
    if not side:
        return None
    sx, sy = (float(v) for v in (exploration.zone_start_xy_m or f"0.5,{side / 2:g}").split(","))
    return side, side, sx, sy


def convex_zone(points) -> tuple:
    """The clicked corners as their convex hull, counter-clockwise: the zone is the shape they
    enclose whatever order they were clicked in, and a corner inside the others is dropped."""
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 3:
        return tuple((float(a), float(b)) for a, b in p)
    from scipy.spatial import ConvexHull, QhullError

    try:
        order = ConvexHull(p).vertices
    except QhullError:
        return ()  # Collinear or repeated corners do not define a manual region.
    ring = p[order]
    nxt = np.roll(ring, -1, axis=0)   # shoelace sum: negative when the ring runs clockwise
    if float((ring[:, 0] * nxt[:, 1] - nxt[:, 0] * ring[:, 1]).sum()) < 0.0:
        ring = ring[::-1]
    return tuple((float(a), float(b)) for a, b in ring)


def describe_rect(boundary: Boundary):
    """(length, width, robot x from the back edge, y from the right edge, turn in degrees) when
    the boundary is a rectangle around the robot (any orientation); None for other polygons."""
    p = np.array(boundary.points, float)
    if len(p) != 4:
        return None
    a, b = p[1] - p[0], p[2] - p[1]
    if a[0] * b[1] - a[1] * b[0] < 0:  # walk it counter-clockwise: the width then points left
        p = p[[0, 3, 2, 1]]
    e1, e2 = p[1] - p[0], p[2] - p[1]
    lx, ly = float(np.hypot(*e1)), float(np.hypot(*e2))
    if min(lx, ly) < 1e-6 or abs(e1 @ e2) > 1e-3 * lx * ly or np.hypot(*(p[3] - p[0] - e2)) > 1e-3:
        return None
    u1, u2 = e1 / lx, e2 / ly
    return lx, ly, float(-p[0] @ u1), float(-p[0] @ u2), math.degrees(math.atan2(u1[1], u1[0]))


class OnceLog:
    """Print an error's traceback once and count its repeats."""

    def __init__(self, where: str):
        self.where, self.last, self.n = where, "", 0

    def __call__(self, exc: BaseException) -> None:
        self.n += 1
        key = f"{type(exc).__name__}: {exc}"
        if key == self.last:
            return
        self.last = key
        print(f"[console] {self.where}: {key} — repeats are counted, not printed", flush=True)
        traceback.print_exc()


def set_md(handle, text: str) -> None:
    """Assign TEXT to HANDLE only when it changed."""
    if handle.content != text:
        handle.content = text


class OperatorQuestionPanel:
    """Derive the shared question UI from Explorer; only its answer API may resume."""

    def __init__(self, gui, explorer, stop):
        self.gui, self.explorer, self.stop = gui, explorer, stop
        self._lock = threading.RLock()
        self.question_id = self.submitted_id = None
        self.draft = ""
        self.modal = self.reply = self.send = self.feedback = None
        self.dismissed = False
        with gui.add_folder(f"{model_name()} has a question", visible=False) as self.panel:
            self.summary = gui.add_html("")
            self.open_button = gui.add_button("Answer")
            gui.add_markdown("Exploration stays paused until you answer or press STOP.")
        self.open_button.on_click(lambda _: self.open())

    def _close_modal(self):
        if self.modal is not None and not self.modal.closed:
            self.modal.close()
        self.modal = self.reply = self.send = self.feedback = None

    def sync(self):
        """The fast UI loop calls this without taking the map/control lock."""
        with self._lock:
            question = self.explorer.pending_question
            if question is None:
                self._close_modal()
                self.panel.visible = False
                self.question_id = self.submitted_id = None
                self.draft, self.dismissed = "", False
                return
            if question["id"] != self.question_id:
                self._close_modal()
                self.question_id, self.submitted_id = question["id"], None
                self.draft, self.dismissed = "", False
            self.panel.visible = True
            text = f"{question['question']}\n\n{question['why']}"
            if self.submitted_id == self.question_id:
                text += "\n\nAnswer sent. Exploration resumes once the sensors are fresh."
            set_md(self.summary, theme.note(text))
            ready = self.explorer.phase == "paused" and self.submitted_id is None
            self.open_button.disabled = not ready
            if self.send is not None:
                self.send.disabled = not ready
            if ready and not self.dismissed and self.modal is None:
                self.open()

    def open(self):
        with self._lock:
            question = self.explorer.pending_question
            if (question is None or question["id"] != self.question_id
                    or self.explorer.phase != "paused" or self.submitted_id is not None):
                return
            if self.modal is not None and not self.modal.closed:
                return
            self.dismissed = False
            question_id = question["id"]
            with self.gui.add_modal(f"{model_name()} has a question") as self.modal:
                self.gui.add_html(theme.note(f"{question['question']}\n\n{question['why']}"))
                self.reply = self.gui.add_text("Your answer", self.draft, multiline=True)
                reply = self.reply
                self.feedback = self.gui.add_markdown("Exploration is paused; the sensors keep running.")
                self.send = self.gui.add_button("Send answer and resume", color="green")
                dismiss = self.gui.add_button("Close and stay paused")
                stop = self.gui.add_button("STOP", color=(130, 0, 0))
            self.send.on_click(lambda _: self.submit(question_id, reply))
            dismiss.on_click(lambda _: self.dismiss(question_id))
            stop.on_click(lambda _: self.stop())

    def dismiss(self, question_id):
        with self._lock:
            if question_id != self.question_id:
                return
            self.draft = self.reply.value if self.reply is not None else self.draft
            self.dismissed = True
            self._close_modal()

    def submit(self, question_id, reply):
        with self._lock:
            if question_id != self.question_id or self.submitted_id is not None:
                return
            answer = reply.value.strip()
            if not answer:
                if self.feedback is not None:
                    set_md(self.feedback, "Enter an answer before resuming.")
                return
            if self.explorer.submit_operator_answer(question_id, answer):
                self.submitted_id = question_id
                self._close_modal()
                self.sync()
            else:
                self.sync()
                if self.feedback is not None and self.question_id == question_id:
                    set_md(self.feedback, "This question is no longer open.")


def visual_evidence_checklist(assessments, readable=0) -> str:
    """The model's own coverage judgement, one paragraph per area it named."""
    if not assessments:
        return (f"**The model's coverage judgement:** no area assessed yet. "
                f"Views it judged readable: {readable}.")
    lines = [f"**The model's coverage judgement** · views it judged readable: {readable}"]
    for area in assessments:
        evidence = ", ".join(area["evidence_views"]) or "none"
        extra = ", ".join(area["complementary_views"]) or "none"
        gaps = "; ".join(area["gaps"]) or "none"
        needed = "needed" if area["complementary_required"] else "not needed"
        lines.append(
            f"**{area['area']}** ({area.get('scope', 'included')}, {area['status']}): "
            f"{sentence(area['requirement'])}  \n"
            f"Views: {evidence}. Extra views ({needed}): {extra}.  \n"
            f"Gaps: {sentence(gaps)} Resolution: {sentence(area['resolution'] or 'none recorded')}")
    return "\n\n".join(lines)


def region_line(scope) -> str:
    """Who owns the task region, its revision and the edges the model is unsure of."""
    if scope["ownership"] == "manual":
        text = "**Region:** your boundary."
    elif scope["status"] != "accepted":
        text = "**Region:** the model has not proposed one yet."
    else:
        text = f"**Region:** proposed by the model, revision {scope['revision']}."
    edges = (scope.get("evidence") or {}).get("uncertain_edges", [])
    return text + (f"  \nUncertain edges: {sentence('; '.join(edges))}" if edges else "")


class DecisionPreview:
    """Keep a rejected selection visible without retaining any executable route."""

    def __init__(self, image, point, label, caption):
        self.image, self.point, self.label, self.caption = image, point, label, caption
        self.empty = self.shown = image.image
        self.marker = None

    def sync(self, explorer):
        view = dict(explorer.vlm_view)
        valid = (view.get("frame") is explorer.live
                 and view.get("picks") is explorer.vlm_picks)
        image = view.get("image") if valid else None
        completed = view.get("completed_preview") or {}
        record = (completed.get("record") or {}) if (
            valid and completed.get("frame") is explorer.live
            and completed.get("picks") is explorer.vlm_picks) else {}
        result = record.get("action_result") or {}
        rejected = ((not record.get("used") and bool(record.get("rejection")))
                    or (result.get("outcome") or record.get("outcome")) == "refused")
        marker = None
        if rejected and record.get("kind") in {"point", "seen"} and record.get("xy") is not None:
            marker = (record["n"], tuple(record["xy"]))
            image = completed["image"]
        image = self.empty if image is None else image
        if image is not self.shown:
            self.image.image = self.shown = image
        if marker != self.marker:
            self.marker = marker
            self.point.visible = self.label.visible = marker is not None
            if marker is not None:
                n, (x, y) = marker
                self.point.position = (x, y, .16)
                self.label.text = f"Rejected destination (decision {n})"
        set_md(self.caption, (f"The red marker is decision {marker[0]}'s rejected destination. "
                             "It clears after the next decision." if marker else ""))


ARTICLES = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "its", "each"}
RECOVERED = {"route_restored", "departure_restored", "repositioned_goal_unavailable"}
OUTCOME_WORDS = {"updated": "Region updated.", "accepted": "Accepted.",
                 "awaiting_answer": "Waiting for your answer.",
                 "answered": "You answered; the model decides again.",
                 "held": "Holding in place.", "cancelled": "Cancelled.",
                 "partial": "Partly completed."}


def decision_finished(record) -> bool:
    return bool(record.get("outcome") or (not record.get("used") and (
        record.get("rejection") or record.get("completion_refusal"))))


def decision_choice(record) -> str:
    """What the model chose, as a short instruction."""
    kind = record.get("kind")
    target = " ".join(str(record.get("target_area") or "").split())
    if target and target.split(" ")[0].lower() not in ARTICLES:
        target = f"the {target}"
    if kind in ("point", "seen"):
        return f"Walk to view {target}" if target else "Walk to a new viewpoint"
    if kind == "look":
        return f"Turn to look at {target}" if target else "Turn to look around"
    if kind == "update_region":
        return ("Change the task region" if (record.get("task_region_update") or {}).get(
            "base_revision") else "Set the task region")
    return {"ask_operator": "Ask you a question", "help": "Ask for help",
            "explored": "Declare the task done"}.get(kind, "Decide")


def decision_result(record) -> str:
    """What happened, in one or two sentences, with model and refusal text as plain text."""
    result = record.get("action_result") or {}
    kind = record.get("kind")
    outcome = result.get("outcome") or record.get("outcome")
    facing = record.get("facing_result") or {}
    refusal = record.get("rejection") or record.get("completion_refusal")
    back = " The robot backed out to where this step began."
    if record.get("rejection") and (record.get("departure_recovery") or {}).get(
            "outcome") in RECOVERED:
        return sentence(f"Rejected: {record['rejection']}") + back
    if kind == "look" and result.get("code") == "LOOK_REPOSITIONED":
        return sentence(f"The view was not completed: {result.get('why') or 'no reason given'}") + back
    if outcome == "arrived" and facing.get("outcome") in {"refused", "stopped"}:
        return sentence("Arrived, but the turn to face the view did not finish: "
                        f"{facing.get('why') or 'no reason given'}")
    if refusal and not result.get("outcome"):
        return sentence(f"{'Not accepted' if kind == 'explored' else 'Rejected'}: {refusal}")
    if outcome == "arrived":
        return "Turn completed." if kind == "look" else "Arrived."
    if outcome in ("refused", "stopped"):
        return sentence(f"{outcome.capitalize()}: {result.get('why') or 'no reason given'}")
    if kind == "explored" and record.get("used"):
        return "Accepted. Exploration is finished."
    if kind == "help" and record.get("used"):
        return "Holding in place until you help."
    return OUTCOME_WORDS.get(outcome) or sentence(
        str(outcome or "rejected").replace("_", " ").capitalize())


def decision_card(record, *, in_progress=False) -> str:
    """One decision as an escaped list item: what it chose, why, and the result."""
    def line(label, text, limit=240):
        full = " ".join(str(text).split())
        short = full if len(full) <= limit else full[:limit - 1].rstrip() + "…"
        return f'<p title="{escape(full, quote=True)}"><b>{label}:</b> {escape(short)}</p>'

    why = (record.get("why") or record.get("text")
           or (record.get("action_result") or {}).get("why") or "No reason was recorded.")
    when = f"{record['t'] / 60:.1f} min" if isinstance(record.get("t"), (int, float)) else ""
    return ('<li>'
            f'<header><strong>{escape(str(record["n"]))}. {escape(decision_choice(record))}'
            f'</strong><time>{escape(when)}</time></header>'
            + line("Why", why)
            + line("Result", "In progress." if in_progress else decision_result(record))
            + '</li>')


class DecisionHistory:
    """One card per decision, newest first, with the unfinished one on top; bounded pages, no
    image IO."""

    PAGE_SIZE = 15

    def __init__(self, gui):
        self.offset = 0
        self.anchor = None
        self._page_request = 0
        self._lock = threading.Lock()
        self.summary = gui.add_html("")
        self.pages = gui.add_button_group(
            "More decisions", options=("Newer", "Older"), visible=False)
        self.pages.on_click(lambda event: self._move_page(
            -1 if event.target.value == "Newer" else 1))
        self.sync([])

    def _move_page(self, direction):
        # Only queue navigation here; the status loop owns presentation updates.
        with self._lock:
            self._page_request += direction

    def sync(self, picks):
        completed = [p for p in picks if decision_finished(p)]
        records = {(p["n"], p.get("wall")): p for p in completed}
        keys = list(reversed(records))
        if self.offset and self.anchor in records:
            self.offset = keys.index(self.anchor)
        else:
            self.offset = 0
        with self._lock:
            requested, self._page_request = self._page_request, 0
        if requested:
            last_offset = self.offset + max(0, (len(keys) - 1 - self.offset)
                                            // self.PAGE_SIZE) * self.PAGE_SIZE
            self.offset = min(max(0, self.offset + requested * self.PAGE_SIZE), last_offset)
        if len(keys) <= self.PAGE_SIZE:
            self.offset = 0
        visible = keys[self.offset:self.offset + self.PAGE_SIZE]
        self.anchor = visible[0] if visible else None
        self.pages.visible = len(keys) > self.PAGE_SIZE
        options = (("Newer",) if self.offset else ()) + (
            ("Older",) if self.offset + self.PAGE_SIZE < len(keys) else ())
        if options and tuple(self.pages.options) != options:
            self.pages.options = options
        current_pick = (picks[-1] if picks and not self.offset
                        and not decision_finished(picks[-1]) else None)
        cards = ([decision_card(current_pick, in_progress=True)] if current_pick else []) + [
            decision_card(records[key]) for key in visible]
        body = ('<ol>' + "".join(cards) + '</ol>' if cards else '<p>No decisions yet.</p>')
        if self.pages.visible:
            body += (f'<p>Showing {self.offset + 1}–{self.offset + len(visible)} '
                     f'of {len(keys)}.</p>')
        set_md(self.summary, '<section class="hb-decisions" aria-label="Decisions">'
               f'<p>By {escape(model_name())}, newest first.</p>' + body + '</section>')


def rgba_overlay(img: np.ndarray) -> np.ndarray:
    a = np.dstack([img, np.full(img.shape[:2], 200, np.uint8)])
    a[np.all(img == 150, axis=2), 3] = 0
    return np.flipud(a)


def update_reachable_floor(scene, state, live, pose, radius_m, *, map_id,
                           visible=True, refresh=True):
    """Draw the reachable observed floor from a locked snapshot; a changed frame, extent or
    permission removes the drawn layer at once."""
    identity = (map_id, live.xmin, live.ymax, live.W, live.H,
                live.task_region_revision, live.scope_ready, radius_m)
    if state and state["identity"] != identity:
        state["handle"].remove()
        state.clear()
    if pose is None or not np.isfinite(pose[:2]).all():
        if state:
            state["handle"].visible = False
        return departure_status(live, None, radius_m)
    if not refresh:
        if state:
            state["handle"].visible = visible
        return state.get("departure")
    key = (identity, live.version, tuple(pose[:2]))
    if state and state["key"] == key:
        state["handle"].visible = visible
        return state["departure"]
    # reachable_network's geodesic footprint admission, without its skeleton
    reachable = np.isfinite(live.geodesic(*pose[:2], radius_m))
    departure = departure_status(live, pose, radius_m, reach=reachable)
    reachable &= live.known_free() & ~live.avoid()
    image = np.zeros((live.H, live.W, 4), np.uint8)
    image[reachable, :3] = vlm_pick.REACHABLE_FLOOR
    image[reachable, 3] = 255
    image = np.flipud(image)
    if state:
        state["handle"].image = image
        state["handle"].visible = visible
    else:
        state["handle"] = scene.add_image(
            "/reachable_floor", image, live.W * RES_M, live.H * RES_M,
            position=(live.xmin + (live.W - 1) * RES_M / 2,
                      live.ymax - (live.H - 1) * RES_M / 2, 0.006),
            format="png", visible=visible)
    state.update(identity=identity, key=key, departure=departure)
    return departure


WAYPOINT_WORDS = {"frontier": "the edge of the explored area",
                  "voronoi": "a point on the path network", "vlm": "the model's viewpoint",
                  "look": "a spot to look from", "home": "the start"}


def next_card(explorer) -> str:
    wp, plan = explorer.next_wp, explorer.plan
    if explorer.phase in IDLE_PHASES or wp is None or not plan:
        return NO_NEXT_STOP
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(plan))
    gx, gy = plan[-1]
    with explorer.lock:
        live = explorer.live
        u, v = live.uv(gx, gy)
        room = float(live.clearance_m()[v, u]) if 0 <= u < live.W and 0 <= v < live.H else 0.0
    turns = len(wp.headings)
    what = ("walks through it" if wp.through else "stops there" if not turns
            else f"stops and faces {turns} direction{'s' if turns != 1 else ''} there")
    return (
        f"**Next stop:** {WAYPOINT_WORDS.get(wp.kind, wp.kind)} at ({gx:.2f}, {gy:.2f}) m, "
        f"{length:.1f} m along the planned route; the robot {what}.  \n"
        f"Clearance at the goal is {room:.2f} m; the route needs {explorer.radius:.2f} m "
        f"and the body {EXECUTION_RADIUS_M:.2f} m."
    )


def clicks_left(explorer):
    """(unwalked known path metres or None, clicks that will take at their median leg length)."""
    with explorer.lock:
        planner = explorer.planner  # the model clicks on the seen network
        skel = planner.seen_skeleton if explorer.pick_hook else getattr(planner, "skeleton", None)
        visited = explorer.visited
        if skel is None or visited.shape != skel.shape:
            left_m = None
        else:
            todo = skel & explorer.live.known_free() & ~visited
            left_m = float(todo.sum()) * 0.05
        legs_m = [
            sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(r))
            for r in getattr(explorer, "routes", [])
            if len(r) > 1
        ]
    typical = sorted(legs_m)[len(legs_m) // 2] if legs_m else 3.0
    return left_m, None if left_m is None else math.ceil(left_m / max(typical, 0.5))


def progress_line(explorer) -> str:
    """Decisions so far and the walked share of the known path network, with roughly how many
    walks are left at the median walk length so far."""
    picks = list(explorer.vlm_picks)
    fell = sum(1 for p in picks if p.get("fallback"))
    replaced = f", {fell} replaced by the planner's choice" if fell else ""
    if current().exploration.observe_only:
        return (f"**Progress:** {len(picks)} proposals so far{replaced}. "
                "The robot does not move in a preview.")
    admitted = sum(1 for p in picks if p.get("used") and p.get("kind") == "point")
    arrived = sum(1 for p in picks if p.get("kind") == "point" and p.get("outcome") == "arrived")
    left_m, n_left = clicks_left(explorer)
    rest = ("" if left_m is None
            else f"; about {left_m:.0f} m left, roughly {n_left} more walks")
    return (f"**Progress:** {100 * float(explorer.walked or 0.0):.0f}% of the known paths "
            f"walked{rest}. {len(picks)} decisions so far: {admitted} walks accepted, "
            f"{arrived} arrived{replaced}.")


def update_solid_walls(server, handle, live):
    """Update the wall node in place, adding it on first use."""
    vertices, faces = live.solid_mesh()
    with server.atomic():
        if not len(faces):
            if handle is not None:
                handle.visible = False
        elif handle is None:
            handle = server.scene.add_mesh_simple(
                "/lidar/walls", vertices, faces, color=GREY, flat_shading=True, side="double")
        else:
            handle.vertices = vertices
            handle.faces = faces.astype(np.uint32, copy=False)
            handle.visible = True
    return handle


def update_task_region(server, handles, live):
    """Draw accepted start-frame geometry; retain concavity and clear retired regions."""
    points, revision = live.task_region_points, live.task_region_revision
    key = (points, revision)
    if handles.get("task_region_key") == key:
        return
    with server.atomic():
        if "task_region_line" not in handles:
            handles["task_region_line"] = server.scene.add_line_segments(
                "/task_region/outline", NONE_SEG, colors=vlm_pick.TASK_REGION_COLOR,
                line_width=4, visible=False)
            handles["task_region_label"] = server.scene.add_label(
                "/task_region/label", "Task region", visible=False)
        line, label = handles["task_region_line"], handles["task_region_label"]
        if points is None:
            line.visible = label.visible = False
            line.points = NONE_SEG
        else:
            boundary = np.c_[np.asarray(points), np.full(len(points), 0.06)].astype(np.float32)
            line.points = np.stack([boundary, np.roll(boundary, -1, axis=0)], axis=1)
            label.position = (*points[0], 0.25)
            label.text = "Task region"
            line.visible = label.visible = True
        handles["task_region_key"] = key


PHASE_WORDS = {"idle": "Ready", "starting": "Starting", "restarting SLAM": "Restarting the map",
               "arming": "Preparing the body", "explore": "Exploring", "preview": "Previewing",
               "done": "Done", "aborted": "Aborted", "error": "Stopped by an error",
               "stopped": "Stopped"}


def execution_status(explorer) -> str:
    """Describe the current operation; a selected point is not proof it is being driven."""
    stage = explorer.stage.lower()
    if explorer.phase in ("error", "aborted", "done"):
        return PHASE_WORDS[explorer.phase]
    if explorer.phase == "pausing":
        return "Pausing"
    if explorer.phase == "paused":
        return ("Waiting for your answer" if stage == "waiting for operator"
                else "Paused, holding position")
    if stage == "held":
        return "Holding; needs you"
    if explorer.phase == "stopped":
        return "Stopped"
    if stage == "verifying fresh data":
        return "Checking the sensors"
    if stage == "verifying arrival":
        return "Checking arrival"
    if "waiting" in stage and "fresh" in stage:
        return "Waiting for fresh sensor data"
    if "repair" in stage:
        return "Repairing the route"
    if stage.startswith("asking"):
        return "Choosing the next step"
    if stage in ("driving", "walking home"):
        return "Walking"
    if stage == "looking":
        return "Turning to look"
    return PHASE_WORDS.get(explorer.phase, explorer.phase.capitalize())


def preview_status(explorer) -> str:
    """The preview's standing note beside its controls, and its stop acknowledgement."""
    if explorer.phase not in IDLE_PHASES and explorer.source.stop_flag.is_set():
        return "Stop requested; the current step finishes first."
    return "Preview only: the robot does not move. STOP closes the console and holds the robot."


def request_shutdown(quit_now, explorer=None):
    """Latch process termination once, for the Stop button and signals alike."""
    if quit_now.is_set():
        return False
    quit_now.set()
    if explorer is not None:
        explorer.stop("console termination requested")
    return True


def close_console(quit_now, explorer, source, server, workers=(), env_view=None):
    """Close every owned resource; returns 1 when any close fails."""
    errors = []
    def finish(label, operation):
        try:
            operation()
        except Exception as exc:  # noqa: BLE001 -- one failed close must not strand the other owned workers
            errors.append(label)
            with contextlib.suppress(OSError, ValueError):
                print(f"[console] {label} failed: {type(exc).__name__}: {exc}", flush=True)
    finish("stop request", lambda: request_shutdown(quit_now, explorer))
    exploration_running = False
    if explorer is not None and explorer.thread is not None:
        finish("exploration worker", lambda: explorer.thread.join(timeout=15.0))
        if explorer.thread.is_alive():
            errors.append("exploration worker still running")
            exploration_running = True
    for worker in workers:
        if worker is not None:
            finish("display worker", lambda worker=worker: worker.join(timeout=2.0))
            if worker.is_alive():
                errors.append("display/startup worker still running")
    if env_view is not None:
        finish("robot view", env_view.close)
    if source is not None:
        finish("motion hold and source", source.close)
        if hasattr(source, "drive_log"):
            finish("motion log", source.drive_log.flush)
    session = getattr(explorer, "session", None)
    recorder = getattr(session, "recorder", None)
    if recorder is not None and not exploration_running:
        finish("session recorder", recorder.close)
    recording_error = getattr(session, "recording_error", None)
    if recording_error:
        errors.append("session recording")
        with contextlib.suppress(OSError, ValueError):
            print(f"[console] session recording failed: {recording_error}", flush=True)
    if server is not None:
        finish("UI server", server.stop)
    return int(bool(errors))


class StartupCancelled(Exception):
    """A shutdown request arrived before console construction completed."""


def ingest_loop(source, explorer, quit_now, pipeline_timings, ingest_log):
    """Keep sensor ingestion independent of the workflow's render cadence."""
    failing = {"since": None}
    while not quit_now.is_set():
        wall_start, cpu_start = time.monotonic(), time.thread_time()
        try:
            if not source.zeroed() and explorer.phase in IDLE_PHASES:
                try:  # a provisional frame, so the map shows before exploration starts
                    source.zero()
                    explorer.reset_frame()
                    explorer.message = (
                        "A provisional map starts where the robot stands; Start exploration "
                        "restarts it there."
                    )
                except RuntimeError:
                    pass
            if source.zeroed():  # the robot has no start frame before Zero
                explorer.ingest()
            failing["since"] = None
        except Exception as exc:  # noqa: BLE001 -- counted and shown; a map it keeps frozen stops the run
            ingest_log(exc)
            failing["since"] = failing["since"] or time.time()
        finally:
            record_timing(pipeline_timings, "ingest", wall_start, cpu_start)
        since = failing["since"]
        if (
            since is not None
            and time.time() - since > MAP_STALE_S
            and explorer.phase not in IDLE_PHASES
        ):
            explorer.stop(f"the live map froze ({ingest_log.last[:120]})")
        time.sleep(0.05)


@dataclass
class StatusDisplay:
    """Read-only scalar/control refresh, independent of expensive map rendering."""
    budget_s: float
    explorer: object
    source: object
    out: object
    quit_now: object
    config_latched: object
    ingest_log: object
    status_log: object
    pipeline_timings: object
    recording_state: object
    b_pause: object
    brief_box: object
    operator_questions: object
    preview_state: object
    progress: object
    card: object
    status: object
    run_status: object
    cam: dict

    def refresh(self, write_state=False):
        # display-only snapshots, read without explorer.lock
        x = y = yaw = None
        with contextlib.suppress(RuntimeError, ObservationUnavailable):
            x, y, yaw = getattr(self.source, "footprint_pose", self.source.pose)()
        el, cov = self.explorer.active_elapsed(), dict(self.explorer.coverage)
        self.b_pause.label = "Resume" if self.explorer.phase == "paused" else "Pause"
        self.brief_box.disabled = self.config_latched["started"] or self.quit_now.is_set()
        self.b_pause.disabled = (self.explorer.phase in (*IDLE_PHASES, "arming", "pausing")
                            or self.explorer.pending_question is not None)
        self.operator_questions.sync()
        freshness = self.source.freshness_status()
        odom_age = self.source.odom_age_s()
        if self.preview_state is not None:
            set_md(self.preview_state, preview_status(self.explorer))
        self.progress.value = round(min(100.0, 100.0 * el / self.budget_s), 1)
        ring, map_age = self.source.ring_fresh_m(), self.explorer.map_age_s()
        reasons = set(freshness.get("reasons") or [freshness.get("reason")])
        feeds = (("Map", map_age is not None and map_age <= MAP_STALE_S),
                 ("Odometry", math.isfinite(odom_age) and not reasons & ODOM_STALE),
                 ("Camera", self.cam.get("status") == "fresh"))
        where = (f"at ({x:.2f}, {y:.2f}) m, facing {math.degrees(yaw):.0f}°"
                 if x is not None else "position unknown")
        message = self.explorer.message
        if self.explorer.stage == "verifying fresh data":
            message = f"The sensors are ready; checking that they stay fresh. Waiting because: {message}"
        elif self.explorer.phase in ("stopped", "aborted", "error"):
            message = f"Reason: {message}"
        warning = (f"The odometry reset its frame ({self.source.frame_reset_why}), so every move "
                   "is refused. Press STOP and launch again with a fresh map."
                   if getattr(self.source, "frame_reset", False) else "")
        if self.explorer.phase in IDLE_PHASES:
            set_md(self.card, NO_NEXT_STOP)
        set_md(self.run_status, exploration_status(
            execution_status(self.explorer), el, self.budget_s,
            self.explorer.paused_elapsed(), where, feeds, message, warning))
        edges = ("no unexplored edges are left inside the region" if self.explorer.enclosure["enclosed"]
                 else f"{self.explorer.enclosure['open_m']:.1f} m of unexplored edges inside the region")
        lidar = ("its scan is stale" if ring == 0.0 else "no point nearby" if math.isinf(ring)
                 else f"closest point {ring:.2f} m")
        reason = freshness.get("reason")
        checks = [f"Sensor check: {reason.replace('_', ' ')}."] if reason not in (
            None, "fresh", "ready") else []
        checks += [f"{log.n} {what} errors; the last: {log.last[:80]}" for log, what in (
            (self.ingest_log, "map update"), (self.status_log, "console")) if log.n]
        set_md(self.status,
               f"**Map estimate:** {cov['>=1']:.0%} of the walls seen at least once; {edges}.\n\n"
               f"**Sensors:** LiDAR {self.source.scan_hz():.1f} Hz, {lidar}. "
               + ("Map empty. " if map_age is None else f"Map updated {map_age:.1f} s ago. ")
               + (f"Odometry {odom_age:.2f} s old. " if math.isfinite(odom_age)
                  else "No odometry. ")
               + f"Camera display: {self.cam['text']}."
               + "".join(f"  \n{line}" for line in checks))
        if write_state:
            route = self.explorer.plan
            state = {
                "t": time.time(),
                "phase": self.explorer.phase,
                "stage": self.explorer.stage,
                "message": self.explorer.message,
                "map_age_s": map_age,
                "odom_age_s": odom_age if math.isfinite(odom_age) else None,
                "freshness": freshness,
                "pipeline_timings": dict(self.pipeline_timings),
                "body_link": body_link_diagnostics(self.source),
                "elapsed_s": round(self.explorer.elapsed(), 1),
                "active_elapsed_s": round(el, 1),
                "paused_elapsed_s": round(self.explorer.paused_elapsed(), 1),
                "attempt_ended_wall_s": self.explorer.ended_wall_s,
                "recording_status": self.recording_state["text"],
                "pose": None if x is None else [round(x, 3), round(y, 3), round(yaw, 4)],
                "next": None if not route else list(route[-1]),
                "stops": self.explorer.stops,
                "coverage": cov,
                "boundary": self.explorer.boundary.points,
                "zone": None if self.explorer.zone is None else self.explorer.zone.points,
                "stop_reason": self.explorer.stop_reason,
            }
            (self.out / ".live_state.json").write_text(json.dumps(state))
            os.replace(self.out / ".live_state.json", self.out / "live_state.json")


@dataclass
class SceneLayers:
    """Handles for the current map, geometry and tracked motion display."""
    handles: object
    floor: object
    reachable_floor: object
    walls: object
    cloud: object
    skeleton: object
    waypoints: object
    trail: object
    plan: object
    route_dots: object
    goal: object
    goal_ring: object
    vlm_label: object
    sweep: object
    frontier: object
    robot: object
    stop_handles: object
    show_floor: object
    show_reachable: object


@dataclass
class DecisionPanels:
    """Recorded model context and outcome widgets; no motion callbacks."""
    ego_caption: object
    card: object
    decision_preview: object
    vlm_camera: object
    vlm_camera_status: object
    history_page: object
    history_image: object
    completed_history: object
    visual_checklist: object
    vlm_progress_md: object
    region: object
    departure: object


@dataclass
class RenderCache:
    last_slow: object = 0.0
    card_t: object = 0.0
    vlm_t: object = 0.0
    raster_key: object = None
    walls_seen: object = None
    scan_seen: object = None
    card_plan: object = None
    vlm_seen: object = -1
    trail_n: object = 0


@dataclass
class MapDisplay:
    """Snapshot shared map state, then update scene and evidence widgets in order."""
    explorer: object
    source: object
    server: object
    layers: SceneLayers
    panels: DecisionPanels
    floor_image: object
    show_history_page: object
    cam: dict
    cache: RenderCache = field(default_factory=RenderCache)

    def render(self):
        x = y = yaw = None
        with contextlib.suppress(RuntimeError, ObservationUnavailable):  # no pose before Zero
            x, y, yaw = getattr(self.source, "footprint_pose", self.source.pose)()
        if x is not None:
            self.layers.robot.position = (x, y, 0.0)
            self.layers.robot.wxyz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        with self.explorer.lock:
            live = self.explorer.live.snapshot()
            visited = self.explorer.visited.copy()
            skel = getattr(self.explorer.planner, "skeleton", None)
            skel = None if skel is None else skel.copy()
            hubs = tuple(self.explorer.planner.hubs)
            open_xy = self.explorer.planner.open_xy.copy()
            planning_radius = self.explorer.planner.radius_m
            key = (id(self.explorer.live), live.version)
            fresh = key != self.cache.raster_key
        update_task_region(self.server, self.layers.handles, live)
        if fresh:
            img, wall_mask = live.rgb(), live.solid()
            self.cache.raster_key = key
            if self.layers.floor["shape"] != img.shape[:2]:
                self.layers.floor["handle"].remove()
                self.layers.floor["handle"] = self.floor_image(live, self.layers.show_floor.value)
                self.layers.floor["shape"] = img.shape[:2]
            else:
                self.layers.floor["handle"].image = rgba_overlay(img)
            if self.cache.walls_seen is None or not np.array_equal(wall_mask, self.cache.walls_seen):
                self.cache.walls_seen = wall_mask
                self.layers.walls["handle"] = update_solid_walls(self.server, self.layers.walls["handle"], live)
        pts = self.source.live_points()
        if pts is not self.cache.scan_seen and pts is not None and len(pts):
            self.cache.scan_seen = pts
            self.layers.cloud.points = pts[:: max(1, len(pts) // 6000)].astype(np.float32)
        slow = time.time() - self.cache.last_slow > SLOW_EVERY_S
        departure = update_reachable_floor(
            self.server.scene, self.layers.reachable_floor, live, None if x is None else (x, y),
            planning_radius, map_id=key[0], visible=self.layers.show_reachable.value, refresh=slow)
        if departure is not None:
            warning = departure["status"] not in {"available", "region_required"}
            self.panels.departure.visible = warning
            set_md(self.panels.departure, '<p role="status" class="hb-exploration-folder-note">'
                   f'{escape(departure["message"])} Known floor remains on the map.</p>')
        if slow:
            with self.server.atomic():  # points and colours land together: no half-drawn frame
                if skel is not None and skel.any():
                    sv, su = np.nonzero(skel)
                    sxs, sys_ = live.xy(su, sv)
                    walked = (
                        visited[sv, su]
                        if visited.shape == skel.shape
                        else np.zeros(len(sv), bool)
                    )
                    self.layers.skeleton.points = np.c_[sxs, sys_, np.full(len(sv), 0.03)].astype(
                        np.float32
                    )
                    self.layers.skeleton.colors = np.where(
                        walked[:, None],
                        np.array(WALKED, np.uint8),
                        np.array(PURPLE, np.uint8),
                    ).astype(np.uint8)
                else:
                    self.layers.skeleton.points = NONE
                self.layers.waypoints.points = (
                    np.c_[np.array(hubs), np.full(len(hubs), 0.06)].astype(np.float32)
                    if hubs
                    else NONE
                )
        tr = list(self.explorer.trail)
        if len(tr) > 1 and len(tr) != self.cache.trail_n:
            self.cache.trail_n = len(tr)
            p = np.array([(a, b, 0.06) for a, b, _ in tr], np.float32)
            self.layers.trail.points = np.stack([p[:-1], p[1:]], 1)
        elif len(tr) <= 1 and self.cache.trail_n:  # the trail restarted: clear the drawn one
            self.cache.trail_n, self.layers.trail.points = 0, NONE_SEG
        route = list(self.explorer.plan)
        with self.server.atomic():
            if len(route) > 1:
                q = np.array([(a, b, 0.09) for a, b in route], np.float32)
                self.layers.plan.points = np.stack([q[:-1], q[1:]], 1)
                self.layers.route_dots.points = q + np.array([0, 0, 0.02], np.float32)
                gx, gy = route[-1]
                self.layers.goal.position = self.layers.goal_ring.position = (gx, gy, 0.16)
            else:
                self.layers.plan.points, self.layers.route_dots.points = NONE_SEG, NONE
                self.layers.goal.position = self.layers.goal_ring.position = HIDDEN
        stops = [s for s in self.explorer.stops if s["kind"] != "start spin"]
        if len(stops) < len(self.layers.stop_handles):  # fewer stops than drawn: clear the dots
            for h in self.layers.stop_handles.values():
                h.remove()
            self.layers.stop_handles.clear()
        for s in stops:
            if s["n"] not in self.layers.stop_handles:
                self.layers.stop_handles[s["n"]] = self.server.scene.add_icosphere(
                    f"/stops/{s['n']}", radius=0.08, color=TEAL, position=(*s["xy"], 0.1)
                )
        set_md(self.panels.ego_caption, camera_caption(self.cam))
        if self.explorer.plan is not self.cache.card_plan or time.time() - self.cache.card_t > 1.0:
            self.cache.card_plan, self.cache.card_t = self.explorer.plan, time.time()
            set_md(self.panels.card, next_card(self.explorer))
        self.panels.decision_preview.sync(self.explorer)
        view = self.explorer.vlm_view  # the model's picture, its words as they stream
        if view["version"] != self.cache.vlm_seen and view["image"] is not None:
            self.cache.vlm_seen = view["version"]
            context = view.get("context")
            self.panels.vlm_camera.visible = context is not None
            if context is not None:
                self.panels.vlm_camera.image = context
            camera_status = view.get("context_meta", {}).get("status", "unavailable")
            history = view.get("history") or {}
            count = len(history.get("view_ids", []))
            camera_words = {"fresh": "a fresh camera view", "stale": "a stale camera view"}
            set_md(self.panels.vlm_camera_status,
                   f"**What the model saw:** its latest decision had "
                   f"{camera_words.get(camera_status, 'no camera view')} and {count} earlier views.")
            pages = view.get("history_pages", [])
            self.panels.history_page.visible = self.panels.history_image.visible = bool(pages)
            if pages:
                self.panels.history_page.options = [f"Page {i + 1}" for i in range(len(pages))]
                self.show_history_page(None)
            last = self.explorer.vlm_picks[-1] if self.explorer.vlm_picks else None
            self.layers.vlm_label.visible = last is not None and last.get("goal") is not None
        if time.time() - self.cache.vlm_t > 1.0:
            self.cache.vlm_t = time.time()
            self.panels.completed_history.sync(list(self.explorer.vlm_picks))
            visual_history = self.explorer.visual_history
            set_md(self.panels.region, region_line(self.explorer.task_scope_context()))
            set_md(self.panels.visual_checklist, visual_evidence_checklist(
                visual_history.coverage_assessment, len(visual_history.readable_ids())))
            set_md(self.panels.vlm_progress_md, progress_line(self.explorer))
        if slow:
            columns = live.wall_points()
            oxy = open_xy
            self.layers.sweep.points = columns if len(columns) else NONE
            self.layers.frontier.points = (
                np.c_[oxy, np.full(len(oxy), 0.05)].astype(np.float32) if len(oxy) else NONE
            )
            self.cache.last_slow = time.time()


class ExplorationView:
    """The exploration console on one viser server: map scene, controls and evidence. `mount`
    builds the widgets; `run` keeps camera, status and map current until the console closes."""
    title, mode = "G1", "explore"

    def __init__(self, explorer, out, quit_now, *, bag_dir="", run_dir=None):
        self.explorer, self.source, self.out, self.quit_now = explorer, explorer.source, out, quit_now
        self.bag_dir, self.run_dir = bag_dir, Path(run_dir or out)
        self.startup_workers = []
        self.pipeline_timings = {"ingest": {}, "map_ui": {}}
        self.ingest_log = OnceLog("ingest")
        self.status_log = OnceLog("status loop")
        self.recording_state = {"text": "not started by this console"}
        self.recording_started = False
        self.camera_thread = self.ingest_thread = self.env_view = None
        self.server = None
        self._closed = False

    def mount(self, server):
        """Add the scene layers, controls and panels to SERVER and wire their callbacks."""
        self.server = server
        explorer, source, quit_now = self.explorer, self.source, self.quit_now
        settings = current()
        observe_only, corners = settings.exploration.observe_only, settings.map.zone_corners
        scene, gui = server.scene, server.gui
        gui.add_html(EXPLORE_STYLE)
        scene.set_up_direction("+z")
        scene.set_background_image(np.full((2, 2, 3), theme.SCENE_BACKGROUND, np.uint8), format="png")
        scene.add_grid("/grid", width=24, height=24, cell_size=1.0, position=(0, 0, -0.01))

        @server.on_client_connect
        def _camera(client):
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.position = (-4.0, -6.0, 7.0)
            client.camera.look_at = (1.5, 0.0, 0.0)

        handles: dict = {}

        def show_boundary() -> None:
            """Draw the manual zone's orange outline, or nothing without a zone."""
            zone = explorer.zone
            segments = NONE_SEG
            if zone is not None:
                b = np.array(zone.points + zone.points[:1])
                z = np.full(len(b) - 1, 0.04)
                segments = np.stack([np.c_[b[:-1], z], np.c_[b[1:], z]], 1).astype(np.float32)
            handles["range"] = scene.add_line_segments(
                "/labels/2_range/line", segments, colors=(230, 140, 20), line_width=4
            )

            label = handles.get("manual_boundary_label")
            if label is None:
                label = handles["manual_boundary_label"] = scene.add_label(
                    "/labels/2_range/label", "Your boundary", position=HIDDEN)
            label.visible = zone is not None
            if zone is not None:
                label.position = (*zone.points[0], .1)

        show_boundary()
        update_task_region(server, handles, explorer.live)
        # The start pose: a yellow ball and a heading frame at the origin of the start frame.
        scene.add_icosphere("/pin/spot", radius=0.12, color=PIN, position=(0.0, 0.0, 0.06))
        scene.add_frame("/pin/heading", axes_length=0.6, axes_radius=0.03, position=(0.0, 0.0, 0.06))

        def floor_image(live, visible: bool):
            return scene.add_image(
                "/floor_map",
                rgba_overlay(live.rgb()),
                live.W * RES_M,
                live.H * RES_M,
                position=(
                    live.xmin + (live.W - 1) * RES_M / 2,
                    live.ymax - (live.H - 1) * RES_M / 2,
                    0.005,
                ),
                visible=visible,
            )

        floor = {
            "handle": floor_image(explorer.live, False),
            "shape": (explorer.live.H, explorer.live.W),
        }
        cloud = scene.add_point_cloud("/lidar/scan", NONE, colors=ORANGE, point_size=0.012)
        walls = {"handle": None}  # every wall and object cell extruded to its height
        skeleton = scene.add_point_cloud("/voronoi/skeleton", NONE, colors=PURPLE, point_size=0.035)
        waypoints = scene.add_point_cloud("/voronoi/waypoints", NONE, colors=LILAC, point_size=0.09)
        trail = scene.add_line_segments("/trail", NONE_SEG, colors=TEAL, line_width=3)
        plan = scene.add_line_segments("/next/route", NONE_SEG, colors=BLUE, line_width=6)
        route_dots = scene.add_point_cloud("/next/intermediate", NONE, colors=BLUE, point_size=0.07)
        goal = scene.add_icosphere("/next/waypoint", radius=0.16, color=BLUE, position=HIDDEN)
        goal_ring = scene.add_icosphere(
            "/next/halo", radius=0.28, color=PALE, opacity=0.35, position=HIDDEN
        )
        # the route, its points and its labelled last node share one colour
        vlm_label = scene.add_label("/next/waypoint/label", "Next stop",
                                    position=(0.0, 0.0, 0.3))
        vlm_label.visible = False
        rejected_point = scene.add_icosphere(
            "/proposal/rejected", radius=0.14, color=RED, position=HIDDEN, visible=False)
        rejected_label = scene.add_label(
            "/proposal/rejected/label", "Rejected destination", position=(0.0, 0.0, .3),
            visible=False)
        sweep = scene.add_point_cloud("/lidar/sweep", NONE, colors=SKY, point_size=0.004)
        frontier = scene.add_point_cloud("/lidar/frontier", NONE, colors=RED, point_size=0.05)
        self.robot = robot = scene.add_frame("/robot", axes_length=0.45, axes_radius=0.02)
        # The live joint view owns the only URDF; this column stays if it cannot load.
        scene.add_cylinder("/robot/body", radius=0.18, height=1.3, color=(90, 95, 105),
                           position=(0, 0, 0.65))

        # robot controls first, so Stop and Pause stay in reach
        b_stop = gui.add_button("STOP", color=(130, 0, 0))
        b_pause = gui.add_button("Pause", color=(200, 78, 78), disabled=True)
        operator_questions = OperatorQuestionPanel(gui, explorer, lambda: _stop(None))
        start_label = "Start preview" if observe_only else "Start exploration"
        run_status = gui.add_html(exploration_status(
            "Ready", 0, explorer.budget_s, 0, "position unknown", (),
            f"Enter the task, then press {start_label}."))
        progress = gui.add_progress_bar(0.0, color="red")
        config_latched = self.config_latched = {"started": False}
        task_status = gui.add_html("", visible=False)
        brief_box = gui.add_text("Task", initial_value=settings.exploration.task, multiline=True)

        def update_brief(_):
            if not config_latched["started"] and not quit_now.is_set():
                explorer.brief = brief_box.value
            else:
                brief_box.value = explorer.brief

        brief_box.on_update(update_brief)
        explorer.brief = brief_box.value
        b_explore = gui.add_button(start_label)
        b_clear = gui.add_button("Let the model set the region",
                                 color=(109, 40, 217), visible=explorer.zone is not None)
        b_zone = gui.add_button("Draw the region myself", color=(180, 83, 9),
                                visible=explorer.zone is None)
        zone_spacing = gui.add_html('<div style="height:16px" aria-hidden="true"></div>')
        draw_md = gui.add_markdown("", visible=False)
        preview_state = gui.add_markdown(preview_status(explorer)) if observe_only else None

        with gui.add_folder("Map"):
            gui.add_html(MAP_LEGEND)
            departure_note = gui.add_html("", visible=False)
            with gui.add_folder("Layers", expand_by_default=False):
                show_reachable = gui.add_checkbox("Reachable floor", initial_value=True)
                show_floor = gui.add_checkbox("Discovered floor", initial_value=False)
                show_floor.on_update(lambda _: setattr(floor["handle"], "visible", show_floor.value))
                show_skel = gui.add_checkbox("Path network", initial_value=True)

                def _skel_vis(_):
                    skeleton.visible = waypoints.visible = show_skel.value

                show_skel.on_update(_skel_vis)
                show_sweep = gui.add_checkbox("Wall points", initial_value=True)
                sweep.visible = True
                show_sweep.on_update(lambda _: setattr(sweep, "visible", show_sweep.value))
                show_front = gui.add_checkbox("Frontiers", initial_value=True)
                show_front.on_update(lambda _: setattr(frontier, "visible", show_front.value))
                gui.add_html('<div class="hb-exploration-folder-note">'
                             + theme.note(LAYERS_NOTE) + '</div>')
        with gui.add_folder("Camera"):
            self.ego = gui.add_image(np.full((270, 480, 3), 228, np.uint8),
                                     format="jpeg", jpeg_quality=82)
            ego_caption = gui.add_markdown(camera_caption({}))
        with gui.add_folder("Decisions"):
            completed_history = DecisionHistory(gui)
            rejected_caption = gui.add_markdown("")
        with gui.add_folder("Details", expand_by_default=False):
            gui.add_markdown(f"**Model:** {model_name()}, {settings.agent.service_tier} service tier.")
            restart_slam = self.restart_slam = gui.add_checkbox(
                "Start a new map where the robot stands", initial_value=True)
            region = gui.add_markdown("")
            card = gui.add_markdown(NO_NEXT_STOP)
            vlm_progress_md = gui.add_markdown("")
            status = gui.add_markdown("")
            visual_checklist = gui.add_markdown(visual_evidence_checklist([]))
            vlm_camera_status = gui.add_markdown("**What the model saw:** no decision yet.")
            vlm_image = gui.add_image(
                np.full((48, 96, 3), 228, np.uint8), label=decision_map_label(explorer),
                format="jpeg", jpeg_quality=88)
            vlm_camera = gui.add_image(
                np.full((48, 96, 3), 228, np.uint8), label="The camera view it was given",
                format="jpeg", jpeg_quality=80, visible=False)
            history_page = gui.add_dropdown("Earlier views", options=["Page 1"], visible=False)
            history_image = gui.add_image(
                np.full((48, 96, 3), 228, np.uint8), visible=False,
                label="Earlier views (labels match the map above)", format="jpeg", jpeg_quality=90)

            @history_page.on_update
            def show_history_page(_):
                pages = explorer.vlm_view.get("history_pages", [])
                index = int(history_page.value.split()[-1]) - 1
                if 0 <= index < len(pages):
                    history_image.image = pages[index]

            self.rec = gui.add_markdown(self.recording_line())
        decision_preview = DecisionPreview(vlm_image, rejected_point, rejected_label, rejected_caption)

        # the range: a rectangle of four numbers, or the hull of clicked corners
        mode = {"zone": False, "points": [], "yaw": 0.0}

        def rect_text() -> str:
            if explorer.zone is None:
                return ("Region: the model sets it from its first survey, within "
                        f"{current().map.extent_m:g} m of the start.")
            d = describe_rect(explorer.zone)
            if d is None:
                return f"Region: your boundary, {len(explorer.zone.points)} corners."
            lx, ly, sx, sy, yaw = d
            turn = f", turned {yaw:+.0f}° from the robot's heading" if abs(yaw) > 0.5 else ""
            return (f"Region: your boundary, {lx:g} × {ly:g} m, the robot {sx:.2f} m from its "
                    f"back edge and {sy:.2f} m from its right edge{turn}.")

        def sync_rect() -> None:
            """The zone is its corners; only its heading is read, for the size text."""
            d = None if explorer.zone is None else describe_rect(explorer.zone)
            mode["yaw"] = 0.0 if d is None else d[4]

        def show_range(note: str = "") -> None:
            show_boundary()
            sync_rect()
            b_clear.visible = not config_latched["started"] and explorer.zone is not None
            b_zone.visible = not config_latched["started"] and explorer.zone is None
            zone_spacing.visible = not config_latched["started"]
            draw_md.visible = not config_latched["started"]
            vlm_image.label = decision_map_label(explorer)
            set_md(draw_md, rect_text() + (f" {note}" if note else ""))

        def leave_modes() -> None:
            """Leaving the zone mode drops its clicks and their markers."""
            mode.update(zone=False, points=[])
            for k in [k for k in handles if k.startswith("vtx")]:
                handles.pop(k).remove()

        show_range()

        def idle(what: str) -> bool:
            if explorer.phase in IDLE_PHASES:
                return True
            explorer.message = f"{what} is not available during a run. Press STOP first."
            return False

        @b_explore.on_click
        def _explore(_):
            with explorer.lock:  # one click, one restart
                if quit_now.is_set() or config_latched["started"] or not idle("Start exploration"):
                    return
                config_latched["started"] = True
                leave_modes()
                brief_box.disabled = restart_slam.disabled = b_clear.disabled = b_zone.disabled = True
                b_clear.visible = b_zone.visible = False
                zone_spacing.visible = False
                draw_md.visible = False
                b_explore.disabled = True
                b_explore.visible = brief_box.visible = False
                task_status.content = theme.note(f"Task: {explorer.brief}")
                task_status.visible = True
                if restart_slam.value:
                    explorer.phase = "restarting SLAM"
                    source.stop_flag.clear()
                    worker = threading.Thread(target=self.restart_then_explore, daemon=True)
                    self.startup_workers.append(worker)
                    worker.start()
                    return
                explorer.phase = "starting"
            why = self.ensure_recording()
            if why:
                explorer.phase = "idle"
                explorer.message = f"Could not start: {why}"
                return
            self.launch()

        @b_stop.on_click
        def _stop(_):
            request_shutdown(quit_now, explorer)
            b_stop.label = "Stopping…"
            b_stop.disabled = True
            b_pause.disabled = True
            b_explore.disabled = True

        @b_pause.on_click
        def _pause(_):
            if explorer.phase == "paused":
                explorer.resume()
            else:
                explorer.request_pause()

        @b_clear.on_click
        def _clear(_):
            with explorer.lock:
                if not quit_now.is_set() and not config_latched["started"] and idle("Changing the region"):
                    leave_modes()
                    explorer.set_boundary(None)
                    show_range()

        @b_zone.on_click
        def _zone(_):
            with explorer.lock:
                if not quit_now.is_set() and not config_latched["started"] and idle("Drawing the region"):
                    leave_modes()
                    mode["zone"] = True
                    draw_md.visible = True
                    set_md(draw_md, f"Click {corners} corners of the region on the map.")

        @scene.on_click()
        def _click(event):
            with explorer.lock:
                if (quit_now.is_set() or config_latched["started"] or not mode["zone"]
                        or explorer.phase not in IDLE_PHASES):
                    return
                o, d = np.array(event.ray_origin), np.array(event.ray_direction)
                if abs(d[2]) < 1e-6:
                    return
                p = o + (-o[2] / d[2]) * d
                px, py = float(p[0]), float(p[1])
                mode["points"].append((px, py))
                k = f"vtx{len(mode['points'])}"
                handles[k] = scene.add_icosphere(
                    f"/draw/{k}", radius=0.08, color=(230, 140, 20), position=(px, py, 0.05)
                )
                if len(mode["points"]) < corners:
                    left = corners - len(mode["points"])
                    set_md(draw_md, f"{len(mode['points'])} of {corners} corners placed; "
                                    f"click {left} more.")
                    return
                # the last corner closes the zone as the convex hull of the clicks
                clicked = list(mode["points"])
                leave_modes()
                hull = convex_zone(clicked)
                if len(hull) < 3:
                    set_md(draw_md, "Those corners are in a line. Press Draw the region "
                                    "myself and try again.")
                    return
                explorer.set_boundary(Boundary(hull))
                inside = len(clicked) - len(hull)
                dropped = ("" if not inside else
                           f"; {inside} inside the others {'was' if inside == 1 else 'were'} dropped")
                show_range(f"Drawn from {len(clicked)} corners{dropped}.")

        self.cam = {"text": "connecting"}
        self.status_display = StatusDisplay(
            budget_s=explorer.budget_s, explorer=explorer, source=source,
            out=self.out, quit_now=quit_now, config_latched=config_latched,
            ingest_log=self.ingest_log, status_log=self.status_log,
            pipeline_timings=self.pipeline_timings, recording_state=self.recording_state,
            b_pause=b_pause, brief_box=brief_box, operator_questions=operator_questions,
            preview_state=preview_state, progress=progress, card=card, status=status,
            run_status=run_status, cam=self.cam,
        )
        layers = SceneLayers(
            handles=handles, floor=floor, reachable_floor={},
            walls=walls, cloud=cloud, skeleton=skeleton,
            waypoints=waypoints, trail=trail, plan=plan,
            route_dots=route_dots, goal=goal, goal_ring=goal_ring,
            vlm_label=vlm_label, sweep=sweep, frontier=frontier,
            robot=robot, stop_handles={}, show_floor=show_floor,
            show_reachable=show_reachable,
        )
        panels = DecisionPanels(
            ego_caption=ego_caption, card=card, decision_preview=decision_preview,
            vlm_camera=vlm_camera, vlm_camera_status=vlm_camera_status,
            history_page=history_page, history_image=history_image, completed_history=completed_history,
            visual_checklist=visual_checklist, vlm_progress_md=vlm_progress_md, region=region,
            departure=departure_note,
        )
        self.map_display = MapDisplay(explorer, source, server, layers, panels,
                                      floor_image, show_history_page, self.cam)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=2.0)
            if self.camera_thread.is_alive():
                errors.append("camera/status worker still running")
        if self.env_view is not None:
            try:
                self.env_view.close()
            except Exception as exc:  # noqa: BLE001 -- outer cleanup must continue through source/server
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    # -- startup: SLAM restart, recording, then the explorer's own start -------------------------
    @staticmethod
    def recorder_running(*, unknown: bool) -> bool:
        """Whether the recorder container is up; `unknown` when docker cannot answer."""
        try:
            ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                                capture_output=True, text=True, check=False)
        except OSError:
            return unknown
        return RECORDER in ps.stdout.split() if ps.returncode == 0 else unknown

    def ensure_recording(self) -> str:
        """Start the bag recorder: '' when recording or when recording is off, else why not."""
        source = self.source
        if self.quit_now.is_set() or source.stop_flag.is_set():
            return "console is stopping"
        if not self.bag_dir:
            self.recording_state["text"] = "off — no sensor bag; live evidence is temporary"
            return ""
        if self.recorder_running(unknown=False):
            self.recording_started = True
            self.recording_state["text"] = "recorder running; STOP closes and checks capture"
            return ""
        if (Path(self.bag_dir) / "ws").exists():
            return "this run's bag is already closed: launch again for a new one"
        if self.quit_now.is_set() or source.stop_flag.is_set():
            return "console is stopping"
        self.explorer.message = "Starting the recorder."
        done = subprocess.run(
            [str(Path(__file__).parent / "record/record.sh"), "start", self.bag_dir],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            self.recording_state["text"] = "recorder start failed"
            return f"the recorder did not start: {(done.stderr or done.stdout).strip()[-200:]}"
        self.recording_started = True
        self.recording_state["text"] = "recorder running; STOP closes and checks capture"
        return ""

    def launch(self) -> None:
        """Start exploration; the caller has already claimed the "starting" phase."""
        explorer, source = self.explorer, self.source
        if explorer.thread is not None and explorer.thread.is_alive():
            explorer.phase = "idle"
            explorer.message = ("The last run is still closing. Press Start exploration again "
                                "in a moment.")
            return
        if self.quit_now.is_set() or source.stop_flag.is_set():
            return
        try:
            print(f"[console] {source.hold_watchdog()}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- the run still goes; the watchdog stays the robot's own
            explorer.message = f"Note: {exc}"
            print(f"[console] watchdog not frozen: {exc}", flush=True)
        if not self.quit_now.is_set():
            explorer.start()

    def restart_then_explore(self) -> None:
        """Restart SuperOdom where the robot stands, zero there, then record and explore."""
        explorer, source, quit_now = self.explorer, self.source, self.quit_now
        explorer.message = "Restarting the map; this spot becomes its origin."
        try:
            for line in source.restart_slam():
                explorer.message = f"Restarting the map: {line}"
                if quit_now.is_set():
                    return
                if source.stop_flag.is_set():
                    raise RuntimeError(explorer.stop_why or "exploration startup stopped")
            if quit_now.is_set() or source.stop_flag.is_set():
                return
            source.zero()
            why = self.ensure_recording()
            if why:
                raise RuntimeError(why)
            if quit_now.is_set() or source.stop_flag.is_set():
                return
            explorer.reset_frame()
            explorer.phase = "starting"
            self.restart_slam.value = False  # a later start keeps this frame and zone
            explorer.message = "Map restarted here. Exploring from the start point."
            self.launch()
        except Exception as exc:  # noqa: BLE001 -- shown on the console; the run never starts blind
            if quit_now.is_set():
                return
            explorer.phase = "idle"
            explorer.message = (f"The map restart failed: {exc}. Fix it, then press Start "
                                "exploration again.")
            traceback.print_exc()

    # -- the running console ---------------------------------------------------------------------
    def camera_loop(self) -> None:
        """Refresh the camera pane and status apart from map rendering; no image is encoded
        without a browser tab open."""
        source, quit_now, cam = self.source, self.quit_now, self.cam
        last_gen, shown, t_rate, log = None, 0, time.time(), OnceLog("camera pane")
        last_status = last_state = 0.0
        while not quit_now.is_set():
            if time.monotonic() - last_status >= TICK_S:
                try:
                    write_state = time.monotonic() - last_state >= 1.0
                    self.status_display.refresh(write_state)
                    if write_state:
                        last_state = time.monotonic()
                except Exception as exc:  # noqa: BLE001 -- counted and shown; STOP stays available
                    self.status_log(exc)
                last_status = time.monotonic()
            phone, camera_meta = source.context_image()
            cam["status"], cam["age_s"] = camera_meta.get("status"), camera_meta.get("receive_age_s")
            gen = camera_meta.get("sequence")
            viewers = len(self.server.get_clients())
            if not viewers:
                cam["text"], last_gen = "no browser tab open", None
            elif gen != last_gen:
                last_gen = gen
                try:
                    self.ego.image = phone if phone is not None else BLANK_CAMERA
                    shown += 1
                except Exception as exc:  # noqa: BLE001 -- a drawing bug must not take the pane away
                    log(exc)
            if time.time() - t_rate >= 2.0:
                if viewers:
                    cam["text"] = f"{shown / (time.time() - t_rate):.0f} fps"
                shown, t_rate = 0, time.time()
            time.sleep(1.0 / CONSOLE_FPS)

    def recording_line(self) -> str:
        """Recording on or off, and the output folder."""
        if not self.bag_dir:
            return f"**Recording:** off.  \nRun folder: {self.run_dir}"
        folder = "" if Path(self.bag_dir) == self.run_dir else f"  \nRun folder: {self.run_dir}"
        return f"**Recording:** on, {self.bag_dir}.{folder}"

    def bag_lines(self) -> str:
        if not self.bag_dir or not Path(self.bag_dir).exists():
            return self.recording_line() + (
                "  \nIt starts with exploration and closes on STOP." if self.bag_dir else "")
        closed = (Path(self.bag_dir) / "ws/metadata.yaml").is_file()
        if closed:
            self.recording_state["text"] = "ROS bag closed; saved files are in the run folder"
        total = sum(f.stat().st_size for f in Path(self.bag_dir).rglob("*") if f.is_file())
        free = shutil.disk_usage(self.bag_dir).free
        return (self.recording_line() + f"  \n{total / 1e9:.2f} GB written, {free / 1e9:.0f} GB free"
                + ("; the sensor bag is closed." if closed else "."))

    def streams_line(self, dead_since: dict) -> tuple[str, str]:
        """The bridge's live fps per stream (bag_dir/bridge_live.json), bold when a required stream
        is under 80 % of its rate, and the name of a stream dead for more than 10 s, else ''."""
        if not self.bag_dir:
            return "", ""
        try:
            live = json.loads((Path(self.bag_dir) / "bridge_live.json").read_text())
        except (OSError, ValueError):
            return "  \nThe recorder has not reported its inputs yet.", ""
        if time.time() - float(live.get("wall_s", 0)) > 10.0:
            return "  \n**The recorder has been silent for 10 s.**", ""
        parts, dead = [], ""
        for name, hz in {"iphone_color": 5, "ws_odometry": 10}.items():
            got = float(live.get("fps", {}).get(name, 0.0))
            ok = got >= 0.8 * hz
            words = f"{STREAM_WORDS[name]} {got:.0f} of {hz} per second"
            parts.append(words if ok else f"**{words}**")
            if got < 0.2 * hz:
                dead_since.setdefault(name, time.time())
                if time.time() - dead_since[name] > 10.0:
                    dead = dead or name
            else:
                dead_since.pop(name, None)
        return "  \nRecorder input: " + ", ".join(parts) + ".", dead

    def run(self) -> None:
        """Start ingestion, the camera pane and the robot view, then render until the console closes."""
        explorer, source, quit_now = self.explorer, self.source, self.quit_now
        self.ingest_thread = threading.Thread(
            target=lambda: ingest_loop(source, explorer, quit_now, self.pipeline_timings,
                                       self.ingest_log), daemon=True)
        self.ingest_thread.start()
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        try:  # a read-only URDF view; the map, scans and routes do not depend on it
            if not current().exploration.observe_only:
                source.listen_body(current().body.address)
            self.env_view = EnvironmentView(self.server, source.feeds(), urdf_path=VISUAL_URDF, hz=4.0)
            self.robot.visible = False  # the URDF stands where the marker did
        except Exception as exc:  # noqa: BLE001 -- the console runs without the 3-D robot
            print(f"[console] no 3-D room: {type(exc).__name__}: {exc}", flush=True)
        dead_since: dict = {}
        last_bag = last_flush = last_art = 0.0
        stream_warned = ""
        while not quit_now.is_set():
            wall_start, cpu_start = time.monotonic(), time.thread_time()
            try:
                if time.time() - last_art > 1.0:
                    explorer.save_map()
                    last_art = time.time()
                if time.time() - last_flush > 1.0:
                    source.drive_log.flush()
                    last_flush = time.time()
                self.map_display.render()
                if explorer.phase in IDLE_PHASES:
                    source.release_watchdog()  # released once exploration ends
                if time.time() - last_bag > 3.0:
                    streams, dead = self.streams_line(dead_since)
                    set_md(self.rec, self.bag_lines() + streams)
                    last_bag = time.time()
                    if explorer.phase not in IDLE_PHASES:
                        # a dead stream is reported, never a stop
                        if dead and dead != stream_warned:
                            stream_warned = dead
                            explorer.message = (
                                f"Note: the recorder receives no {STREAM_WORDS[dead]} frames; "
                                "exploration continues."
                            )
                            print(f"[console] {explorer.message}", flush=True)
                        if (self.bag_dir and Path(self.bag_dir).exists()
                                and shutil.disk_usage(self.bag_dir).free < 3e9):
                            explorer.stop("the bag disk is under 3 GB free")
                        if self.recording_started and not self.recorder_running(unknown=True):
                            explorer.stop("the recorder exited")
            except Exception as exc:  # noqa: BLE001 -- a drawing bug must not take the STOP button away
                self.status_log(exc)
            finally:
                record_timing(self.pipeline_timings, "map_ui", wall_start, cpu_start)
            time.sleep(TICK_S)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m homebody.real2sim.exploration.app",
                                 description="The exploration console; homebody real2sim explore starts it.")
    ap.add_argument("--settings-json", type=Path, help="the run's resolved settings (meta/settings.json)")
    ap.add_argument("--out", type=Path, help="explorer output directory (default: a new run under the output root)")
    ap.add_argument("--bag-dir", default="", help="run directory the recorder writes; empty records nothing")
    ap.add_argument("--run-dir", type=Path, help="run directory the console reports")
    args = ap.parse_args(argv)
    if args.settings_json is not None:
        try:
            install(from_json(args.settings_json.read_text()))
        except (OSError, ValueError) as error:
            ap.error(f"--settings-json: {error}")
    settings = current()

    quit_now = threading.Event()
    source = explorer = server = view = None

    def on_signal(_signum, _frame):
        request_shutdown(quit_now, explorer)

    # installed before the map and URDF load, so a signal during loading is not lost
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    output = Path(settings.exploration.output or REPO / "runs/explore")
    out = args.out or output / time.strftime("%Y%m%dT%H%M%S") / "explore"
    out.mkdir(parents=True, exist_ok=True)

    try:
        if not settings.robot.iface:
            raise SystemExit("Robot interface is unset: launch homebody real2sim explore --iface INTERFACE")
        source = RobotSource(settings.robot.iface, str(out / "live_map.npz"),
                             observe_only=settings.exploration.observe_only)
        boundary, preset = None, zone_preset(settings.exploration)
        if preset is not None:
            try:
                boundary = rect_boundary(*preset)
            except ValueError as bad:
                raise SystemExit(f"exploration.zone_size_m / zone_start_xy_m: {bad}") from bad
        print(
            "[console] no manual zone: the model proposes the task region before walking; "
            f"map extent {current().map.extent_m:g} m from the start"
            if boundary is None
            else f"[console] manual zone preset: {boundary.points}",
            flush=True,
        )
        explorer = Explorer(source, boundary, out, budget_s=settings.exploration.budget_s)
        explorer.pick_hook = vlm_pick.Picker(workspace=out / "provider")
        if quit_now.is_set():
            raise StartupCancelled
        view = ExplorationView(explorer, out, quit_now, bag_dir=args.bag_dir, run_dir=args.run_dir)
        logging.getLogger("websockets.server").addFilter(EmptyHandshakeFilter())
        server = theme.open_server(settings.exploration.host, settings.exploration.port,
                                   view.title, view.mode)
        view.mount(server)
        view.run()
    except StartupCancelled:
        pass
    finally:
        workers = () if view is None else (view.ingest_thread, *view.startup_workers)
        cleanup_code = close_console(quit_now, explorer, source, server, workers, view)
    return cleanup_code


if __name__ == "__main__":
    code = main()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError):
            stream.flush()
    # exit without interpreter teardown, which DDS, ZMQ and cv2 daemon threads would abort
    os._exit(code)
