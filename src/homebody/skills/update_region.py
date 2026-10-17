"""Admit an evidence-grounded task region as one transaction in the current start frame."""
import time
from copy import copy

import numpy as np

from homebody.helpers.raster import grid_to_px, to_metres, valid_grid
from homebody.helpers.region import TaskRegionRejected, contains, validate_polygon
from homebody.primitives.navigation.errors import Abort, ActionRefused, Stopped
from homebody.primitives.navigation.live import LIVE_FAILURES, LiveActions

from .contract import Result

SURFACE = LiveActions
NEEDS = ("hold_position",)
FAILURES = LIVE_FAILURES
PROMPT = ('update_region: base_revision, the task region\'s current revision; vertices, 3 to 64 '
          '[x, y] clicks on the overview; reason; evidence_ids, the view ids that justify it; '
          'uncertain_edges. The region must contain the robot, stay inside the map extent and be one '
          'connected area. Replaces the task region without moving; walking outside the old region '
          'waits for its acceptance.')
PARAMETERS = {"type": "object", "properties": {
    "base_revision": {"type": "integer", "minimum": 0},
    "vertices": {"type": "array", "minItems": 3, "maxItems": 64,
                 "items": {"type": "array", "minItems": 2, "maxItems": 2,
                           "items": {"type": "number", "minimum": 0, "maximum": 1000}}},
    "reason": {"type": "string", "minLength": 1},
    "evidence_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
    "uncertain_edges": {"type": "array", "items": {"type": "string"}}},
    "required": ["base_revision", "vertices", "reason", "evidence_ids", "uncertain_edges"],
    "additionalProperties": False}


class Plan:
    def __init__(self, inputs, frame_id):
        self._inputs = inputs
        self.summary = {"action": "update_region", "frame_id": frame_id}


def prepare(arguments, selected, current, planning):
    return Plan(dict(arguments), selected.frame_id)


def execute(plan, ctx):
    try:
        refusal = _apply_task_region(ctx.world, ctx.actions, **plan._inputs)
        return Result("update_region", "INVALID_ARGUMENT" if refusal else "OK",
                      refusal or "The task region was updated.")
    except (ActionRefused, Stopped, Abort) as error:
        ctx.fail(error)


def action_refusal(answer):
    if not isinstance(answer.task_region_update, dict):
        return "update_region requires a polygon object"
    if answer.coverage_updates is not None or answer.image_assessments is not None:
        return "region updates cannot also change evidence assessments"
    return ""


def admitted(record):
    record.update(used=True, outcome="updated")
    return ("update_region", None, None)


def _apply_task_region(world, actions, update, transform, shape, supplied_ids, frame, context=None):
    if update is None:
        return ""
    initial = world.region_snapshot()
    admission_generation = initial["decision_generation"]
    if initial["manual"]:
        return "manual override is authoritative; the model cannot change the boundary"
    if not isinstance(update, dict):
        return "update_region requires a polygon object"
    ids, reason = update.get("evidence_ids"), update.get("reason")
    edges = update.get("uncertain_edges")
    if (not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids)
            or not set(ids).issubset(supplied_ids) or not isinstance(reason, str)
            or not reason.strip() or not isinstance(edges, list)
            or not all(isinstance(e, str) for e in edges)):
        return "task boundary needs supplied evidence IDs, a reason and uncertain_edges"
    vertices = update.get("vertices")
    if (not isinstance(vertices, list) or not 3 <= len(vertices) <= 64
            or not all(valid_grid(v) for v in vertices)):
        return "task boundary requires 3..64 finite vertices on the map's 0..1000 grid"
    started = time.perf_counter()
    try:
        points = validate_polygon([
            to_metres(transform, *grid_to_px(v, shape)) for v in vertices])
    except TaskRegionRejected as exc:
        return str(exc)
    revision = update.get("base_revision")
    def current_context_refusal():
        state = world.region_snapshot()
        if state["frame"] is not frame:
            raise Abort("start frame changed during task boundary admission")
        if state["decision_generation"] != admission_generation or state["stopped"]:
            return "operator interrupted region admission; request a fresh decision"
        if state["manual"]:
            return "manual override is authoritative; the model cannot change the boundary"
        if (isinstance(revision, bool) or not isinstance(revision, int)
                or revision != frame.task_region_revision):
            return "task region revision changed; use the current revision"
        if context is not None and any(
                context.get(k) != state["context"].get(k)
                for k in ("revision", "generation", "visual_epoch", "brief", "ownership")):
            return "task context changed during region admission; request a fresh decision"
        if state["brief"] != state["task_brief"]:
            return "task brief changed during region admission"
        return ""
    with world.lock:
        refusal = current_context_refusal()
    if refusal:
        return refusal
    if not world.observe_only:
        actions.hold_position()
    _x, _y, _ = world.decision_pose()
    authority = world.stationary_guard()
    base_xy = world.base_xy()
    if not np.isfinite(base_xy).all() or not contains(points, [base_xy])[0]:
        return "task region excludes the current base position"
    try:
        # Allocation and raster admission stay off ingestion and motion callbacks.
        growth = frame.prepare_growth(points)
        candidate = growth.candidate if growth is not None else copy(frame)
        refusal = candidate.set_task_region(points, base_xy, revision)
        if refusal:
            return refusal
        if np.any(candidate._task_region_mask & ~candidate.inside_inset):
            return "task region is clipped by the automatic extent"
        world.check()
        world.decision_pose()
        base_xy = world.base_xy()
        world.check()
        world.stationary_guard(authority)
        with world.lock:
            refusal = current_context_refusal()
            if refusal:
                return refusal
            if not candidate.task_region_contains(*base_xy):
                return "task region excludes the current base position after admission"
            evidence = {"reason": reason, "evidence_ids": ids, "uncertain_edges": edges}
            accepted = world.publish_region(candidate, growth, revision, evidence)
    except MemoryError:
        return "map allocation refused: insufficient memory; previous region retained"
    except TaskRegionRejected as exc:
        return str(exc)
    metrics = None
    if growth is not None:
        metrics = growth.metrics | {"wall_s": time.time(),
                    "revision": frame.task_region_revision,
                    "generation": world.region_snapshot()["generation"],
                    "resize_s": time.perf_counter() - started}
    world.persist_region(accepted, metrics)
    return ""
