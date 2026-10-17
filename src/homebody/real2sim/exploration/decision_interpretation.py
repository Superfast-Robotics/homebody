"""Interpret one model response against its captured exploration evidence."""
import json
import math

from homebody.helpers.floor import FACE_MIN_M, MAX_GOAL_ADJUST_M
from homebody.helpers.raster import grid_to_px, to_metres, to_px, valid_grid
from homebody.primitives.navigation.errors import Abort, DecisionInterrupted
from homebody.skills import ask_operator, update_region

from . import vlm_pick
from .decision_types import ActionRequest, PrepareAction
from .planner import Waypoint


def interpret(explorer, evidence, answer, *, clock, terminal):
    time = clock
    request = evidence.request
    n, decision_generation = request.number, request.generation
    proposal_context = request.permission
    x, y, _yaw = evidence.position
    vlm_dir, decision_live, decision_brief = evidence.directory, evidence.frame, evidence.brief
    rgb, T = evidence.rgb, evidence.transform
    context_meta, history = evidence.camera_metadata, evidence.history
    history_owner, decision_history = evidence.history_owner, evidence.history_snapshot
    selection, walked_legs = evidence.selection, evidence.walked_legs
    map_views, scope_context = evidence.map_views, evidence.scope
    rejection, continuation = evidence.rejection, evidence.continuation
    explorer.check()
    if decision_generation != explorer._decision_generation:
        cancelled = {"n": n, "kind": answer.kind, "used": False, "outcome": "cancelled",
                     "why": "operator pause invalidated this response",
                     "seconds": answer.seconds,
                     "provider_response": answer.provider_response,
                     "context_manifest": f"context_leg_{n}.json", "T_map_px": T.tolist()}
        (vlm_dir / f"leg_{n}.json").write_text(json.dumps(cancelled, indent=1))
        explorer.vlm_picks.append(cancelled)
        raise DecisionInterrupted()
    explorer.stage = "planning the route"
    coordinate_view = answer.map_view if answer.kind in {"point", "look"} else "overview"
    rec = {
        "n": n,
        "t": round(explorer.elapsed(), 1),
        "wall": time.time(),
        "kind": answer.kind,
        "proposal_context": proposal_context,
        "why": answer.why,
        "text": answer.text,
        "provider_response": answer.provider_response,
        "seconds": answer.seconds,
        "model": answer.model,
        "grid": answer.grid,
        "face_grid": answer.face_grid,
        "camera_view": context_meta,
        "visual_history": history,
        "completion_assessment": answer.completion,
        "coverage_updates": answer.coverage_updates,
        "image_assessments": answer.image_assessments,
        "task_region_update": answer.task_region_update,
        "target_area": answer.target_area, "expected_gain": answer.expected_gain,
        "question": answer.question,
        "context_manifest": f"context_leg_{n}.json",
        "selection_basis": "rendered_network" if selection else "current_network",
        "px": None,
        "xy": None,
        "goal": None,
        "goal_px": None,
        "used": False,
        "click_on": None,
        "snap_xy": None,
        "snap_m": None,
        "face_yaw": None,
        "fallback": None,
        "rejection": None,
        "rejection_code": None,
        "rejection_details": None,
        "retry_of": explorer.vlm_picks[-1]["n"] if rejection else None,
        "goal_adjustment_m": None,
        "max_goal_adjustment_m": MAX_GOAL_ADJUST_M,
        "proposed_goal": None,
        "image": f"leg_{n}.png",
        "map_view": coordinate_view,
    }
    leg, near = None, (x, y)
    why = answer.why if answer.kind == "error" else ""
    brief_changed = explorer._current_brief() != decision_brief
    action_refusal = ""
    if coordinate_view not in map_views:
        action_refusal = "selected coordinate map was not supplied for this decision"
    if answer.task_region_update is not None and answer.kind != "update_region":
        action_refusal = "region updates must be a standalone update_region action"
    if answer.kind == "update_region":
        action_refusal = update_region.action_refusal(answer)
    if answer.kind == "ask_operator":
        action_refusal = ask_operator.action_refusal(answer, explorer.ask_operator_enabled)
    coverage_refusal = ""
    if not brief_changed and not action_refusal and answer.kind != "update_region":
        if (explorer.live is not decision_live or explorer.visual_history is not history_owner):
            coverage_refusal = "start frame changed during coverage assessment"
        else:
            assessed = decision_history.snapshot()
            coverage_refusal = assessed.apply_image_assessments(
                answer.image_assessments, history["view_ids"], decision_n=n, persist=False)
            if not coverage_refusal:
                coverage_refusal = assessed.apply_coverage(
                    answer.coverage_updates, history["view_ids"], persist=False)
            if not coverage_refusal:
                history_owner = explorer._commit_visual_history(
                    history_owner, assessed, decision_live, persist=True)
        rec["coverage_assessment"] = explorer.visual_history.manifest()["coverage_assessment"]
    scope_refusal = ("user brief changed during decision; reinterpret scope before moving"
                     if brief_changed else action_refusal)
    if not coverage_refusal and not brief_changed and not action_refusal:
        if answer.task_region_update is not None:
            result = yield ActionRequest("update_region", {
                "update": answer.task_region_update, "transform": T, "shape": rgb.shape,
                "supplied_ids": history["view_ids"], "frame": decision_live,
                "context": scope_context})
            scope_refusal = "" if result.ok else result.message
        if answer.task_region_update is not None and not scope_refusal:
            selection = None  # the pictured network predates this permission revision
        if (not scope_refusal and not explorer.live.scope_ready
                and answer.kind not in {"look", "help", "ask_operator", "update_region"}):
            scope_refusal = "initial task boundary is required before walking or completion"
        rec["task_region"] = explorer.task_scope_context()
    # motion coordinates use the selected view and its transform; regions use the overview
    rgb, T, rec["image"] = map_views.get(coordinate_view, map_views["overview"])
    continuation_expired = False
    if continuation:
        explorer.decision_pose()
        if explorer.live is not decision_live:
            raise Abort("map frame changed during model decision; holding position")
        continuation_expired = not explorer._proposal_budget_available()
    if coverage_refusal:
        why = coverage_refusal
        rec.update(rejection=why, rejection_code="invalid_coverage_update")
        if answer.kind == "explored":
            rec["completion_refusal"] = why
    elif scope_refusal:
        why = scope_refusal
        rec.update(rejection=why, rejection_code="task_scope")
    elif continuation_expired:
        why = "original exploration budget expired during the proposal"
        rec.update(rejection=why, rejection_code="proposal_budget_expired")
    elif answer.kind == "look" and (
        not valid_grid(answer.grid) or
        (answer.face_grid is not None and not valid_grid(answer.face_grid))
    ):
        why = "invalid map coordinates; use the map's 0..1000 grid"
    elif answer.kind == "look":
        x, y, _yaw = explorer.decision_pose()
        if explorer.live is not decision_live:
            raise Abort("map frame changed during model decision; holding position")
        px = grid_to_px(answer.grid, rgb.shape)
        fx, fy = to_metres(T, *px)
        if math.hypot(fx - x, fy - y) < FACE_MIN_M:
            why = "viewing point too close to the base to define a heading"
        else:
            face = math.atan2(fy - y, fx - x)
            leg = ("look", Waypoint("look", (x, y), [face]), None)
            rec.update(used=True, face_yaw=round(face, 3),
                       face_px=[round(float(v), 1) for v in px],
                       face_xy=[round(fx, 3), round(fy, 3)])
    elif answer.kind == "point":
        attempt = yield PrepareAction("pick_point", {"admission": {
            "answer": answer, "record": rec, "frame": decision_live,
            "history": history, "transform": T, "image": rgb,
            "selection": selection, "continuation": continuation,
            "near": near}})
        if attempt.finished:
            why = attempt.result.message
            near = attempt.result.details.get("near", near)
        else:
            near = attempt.plan.near
            leg = ("vlm", attempt.plan.waypoint, None)
    elif answer.kind == "update_region":
        leg = update_region.admitted(rec)
    elif answer.kind == "ask_operator":
        leg, why = ask_operator.admit(answer, rec)
    elif answer.kind == "help":
        yield ActionRequest("request_help", {"why": answer.why, "record": rec})
    elif answer.kind == "explored":
        result = yield ActionRequest("explored", {
            "answer": answer, "history": history, "record": rec,
            "walked_legs": walked_legs,
            "continuation": continuation})
        why = "" if terminal(result) is not None else result.message
    if leg is None and not explorer.stop_reason and answer.kind != "explored":
        rec["rejection"] = why or "the model gave no usable destination"
    if leg is not None and leg[1] is not None:
        rec["goal"] = [round(float(v), 3) for v in leg[1].xy]
        rec["goal_px"] = [round(v, 1) for v in to_px(T, leg[1].xy)]
    (vlm_dir / f"leg_{n}.json").write_text(json.dumps(rec | {"T_map_px": T.tolist()}, indent=1))
    explorer.vlm_picks.append(rec)
    outcome = vlm_pick.outcome_words(rec)
    marked = vlm_pick.mark(rgb, rec["px"], rec["used"], rec["goal_px"],
                           face_px=rec.get("face_px"))
    explorer.vlm_view.update(
        image=marked,
        T=T.tolist(),
        completed_preview={"image": marked, "record": rec, "frame": decision_live,
                           "picks": explorer.vlm_picks},
        head=f"{vlm_pick.model_label_text()} · decision {n} · {answer.seconds:.0f} s · {outcome}",
        words=(answer.why or answer.text) + (
            f"\nCompletion refused: {rec['completion_refusal']}"
            if rec.get("completion_refusal") else ""),
        since=None,
        version=explorer.vlm_view["version"] + 1,
    )
    print(f"[vlm_pick] decision {n}: {answer.kind} grid={answer.grid} xy={rec['xy']} "
          f"goal={rec['goal']} {outcome} ({answer.seconds:.1f} s) — {answer.why}", flush=True)
    return leg
