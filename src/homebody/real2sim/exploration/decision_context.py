"""Build and persist frozen exploration evidence before a provider request."""
import json
import math
from dataclasses import dataclass

import numpy as np

from homebody.helpers.floor import departure_status
from homebody.primitives.navigation.errors import Abort, DecisionInterrupted, Stopped

from . import vlm_pick
from .decision_types import DecisionContext
from .planner import RADIUS_M
from .settings import current
from .skills import ACTIONS, REGISTRY

# the skill that ran each motion the explorer records (planner.Waypoint.kind)
SKILL_OF_KIND = {"vlm": "pick_point", "look": "look_at", "recovery": "departure_recovery",
                 "frontier": "walk_route", "voronoi": "walk_route", "home": "walk_route"}


@dataclass(frozen=True)
class DecisionEvidence:
    """Captured views and frame bindings retained for interpreting one response."""
    request: DecisionContext
    position: tuple
    directory: object
    frame: object
    brief: str
    rgb: np.ndarray
    transform: np.ndarray
    camera_metadata: dict
    history: object
    history_owner: object
    history_snapshot: object
    selection: object
    walked_legs: int
    map_views: dict
    scope: object
    rejection: str
    continuation: object


def build_context(explorer, x, y, yaw, *, rejection="", continuation=None, clock):
    time = clock
    explorer._sync_task_brief()
    decision_generation = explorer._decision_generation
    decision_brief = explorer._task_brief
    decision_live = getattr(explorer, "live", None)
    n = len(explorer.vlm_picks) + 1
    vlm_dir = explorer.out_dir / "vlm"
    vlm_dir.mkdir(exist_ok=True)
    while (vlm_dir / f"leg_{n}.json").exists():
        n += 1
    sample, selection = explorer._camera_context()
    context, context_meta = sample["rgb"], sample["metadata"] | {
        "selection": selection, "robot_pose_xyyaw": list(sample["pose"]),
        "pose_sampled_at_s": sample["sampled_at"]}
    if context is None:
        raise Stopped("camera view is unavailable or stale; refusing a map-only decision")
    age_at_selection = (context_meta.get("receive_age_s", math.inf)
                        + time.time() - sample["sampled_at"])
    selected_at = time.monotonic()

    def check_current_camera():
        # the age of this frozen receipt, not of the stream's newest image
        elapsed = time.monotonic() - selected_at
        age = age_at_selection + elapsed
        limit = current().iphone.max_age_s
        if (context_meta.get("status") != "fresh" or not math.isfinite(age)
                or elapsed < 0. or not 0. <= age <= limit):
            raise Stopped(f"selected current camera view expired while preparing the decision "
                          f"({age:.2f} s old; limit {limit:g} s); holding position")
        context_meta["receive_age_s"] = age

    # observe the pose again, outside the ingestion lock
    x, y, yaw = explorer.decision_pose()
    with explorer.lock:
        history_owner = explorer.visual_history
        decision_history = history_owner.snapshot()
    gallery = explorer._gallery_sample or sample
    key, admission = decision_history.capture(
        gallery["rgb"], gallery["metadata"], gallery["pose"],
        sampled_at=gallery["sampled_at"], odom_age_s=gallery["odom_age_s"])
    history_owner = explorer._commit_visual_history(history_owner, decision_history, decision_live)
    explorer._decision_gallery_key = key
    selected_ids = decision_history.selected_ids(**explorer._recent_route_evidence)
    explorer._recent_route_evidence = {}
    if key is not None and key not in selected_ids:
        selected_ids.append(key)
    pages = decision_history.pages(ids=selected_ids)
    history = decision_history.manifest(ids=selected_ids) | {
        "current_viewpoint": key, "admission": admission, "current_rgb_is_separate": True}
    check_current_camera()
    explorer.refresh_planning_radius()
    decision_radius = explorer.planner.radius_m
    rgb, T, selection = vlm_pick.render(explorer, (x, y, yaw))
    frozen_map = selection[0].live if selection is not None else explorer.live
    network = selection[1] if selection is not None else None
    departure = departure_status(frozen_map, (x, y, yaw), decision_radius,
                                 reach=network if network is not None and network.any() else None)
    local_rgb, local_T = vlm_pick.local_map(rgb, T, (x, y, yaw), departure=departure)
    names = [f"leg_{n}_camera.png"] + [f"leg_{n}_history_{k}.png" for k in range(1, len(pages) + 1)]
    names += [f"leg_{n}.png", f"leg_{n}_local.png"]
    images = [vlm_pick.png(picture) for picture in (context, *pages, rgb, local_rgb)]
    for name, data in zip(names, images, strict=True):
        (vlm_dir / name).write_bytes(data)
    image_paths = names
    map_views = {"overview": (rgb, T, f"leg_{n}.png"),
                 "local": (local_rgb, local_T, f"leg_{n}_local.png")}
    walked_legs = sum(1 for s in explorer.stops if s["kind"] == "vlm")
    scope_context = explorer.task_scope_context(T, rgb.shape)
    proposal_context = {k: v for k, v in scope_context.items()
                        if k in {"generation", "visual_epoch", "brief", "ownership", "revision"}}
    proposal_context["map_version"] = decision_live.version
    current_readiness = explorer._observation_readiness(context_meta)
    readiness = {key: value for key, value in current_readiness.items()
                 if key not in {"freshness", "blocking_observation"}}
    blocked = current_readiness["blocking_observation"]
    readiness["blocking_observation"] = None if blocked is None else blocked["reason"]
    enclosure = explorer.enclosure
    observation = {
        "decision": n, "legs_walked": walked_legs,
        "frontier_inside_region": ({"enclosed": True} if enclosure.get("enclosed") else {
            "open_m": round(enclosure.get("open_m", 0.0), 1),
            "unknown_m2": round(enclosure.get("unknown_m2", 0.0), 1)}),
        "network_walked_fraction": round(explorer.walked, 2),
        "stood_at_xy_m": [[round(v, 1) for v in s["xy"]] for s in explorer.stops
                          if s.get("kind") == "vlm"][-20:],
        "planning_radius_m": decision_radius,
        "task_region": {k: scope_context[k] for k in
                        ("revision", "status", "frozen", "ownership", "vertices_grid")
                        if k in scope_context},
        "readiness": readiness, "departure": departure}
    observation["current_camera"] = {
        "image_index": 0,
        "selection": {k: context_meta["selection"][k] for k in ("status", "reason")},
    }
    recent_refusals = explorer._recent_proposal_refusals()
    results = {}
    if explorer.last_skill_result is not None:
        results["previous_action"] = dict(explorer.last_skill_result)
    elif explorer.previous_action_result is not None:
        record = explorer.previous_action_result
        action = {"skill": SKILL_OF_KIND.get(record["kind"], record["kind"]),
                  "code": record["code"] or "OK", "message": record["why"]}
        action.update((k, v) for k, v in record.items() if k not in {"kind", "code", "why", "details"})
        action["details"] = {
            k: v for k, v in record.get("details", {}).items()
            if k in {"retry_count", "completed", "observations_recovered",
                     "original_reason", "elapsed_s", "remaining_s", "departure_recovery"}}
        results["previous_action"] = action
    if explorer.previous_action_result is not None:
        results["failed_destinations_since_arrival"] = explorer.failed_goals
        details = explorer.previous_action_result.get("details", {})
        recovery = details.get("departure_recovery", details)
        if recovery.get("departure_verdict") == "admitted":
            results["departure_handoff"] = departure | {
                "usable_now": departure["status"] == "available",
                "checked_after_capture": True,
                "pose_xyyaw": [float(v) for v in (x, y, yaw)],
                "note": "Departure re-checked at this decision."}
    if rejection:
        results["rejected_proposal"] = rejection
    if continuation:
        results["final_reassessment"] = continuation
    if recent_refusals:
        results["recent_proposal_refusals"] = recent_refusals
    answers = [{k: q[k] for k in ("id", "question", "why", "answer") if k in q}
               for q in explorer.operator_questions if q["status"] == "answered"]
    if answers:
        results["operator_answers"] = answers
    manual = scope_context["ownership"] == "manual"
    offered = [name for name in ACTIONS if not (manual and name == "update_region")
               and (explorer.ask_operator_enabled or name != "ask_operator")]
    tools = tuple(vlm_pick.tool(name, REGISTRY[name]) for name in offered)
    prompt = vlm_pick.prompt(decision_brief, camera=explorer.source.camera_description,
                             actions=[REGISTRY[name] for name in offered],
                             observation=observation, history=history, results=results)
    (vlm_dir / f"context_leg_{n}.json").write_text(json.dumps({
        "images": image_paths, "history": history, "prompt": prompt,
        "map_views": {name: {"image": path, "T_map_px": matrix.tolist(),
                             "shape": list(view.shape)}
                      for name, (view, matrix, path) in map_views.items()},
        "robot_pose_xyyaw": [float(v) for v in (x, y, yaw)],
        "robot_pose_reference": "navigation_footprint", "current_departure": departure,
        "pose_frame": "exploration_start", "context_built_at_s": time.time(),
        "planning_radius_m": decision_radius, "requested_radius_m": RADIUS_M,
        "previous_action_result": explorer.previous_action_result,
        "current_observation_readiness": current_readiness,
        "failed_goals": explorer.failed_goals,
        "recent_proposal_refusals": recent_refusals,
        "task_region": scope_context,
        "ask_operator_enabled": explorer.ask_operator_enabled,
        "proposal_continuation": continuation,
        "previous_proposal_rejection": ({
            "code": explorer.vlm_picks[-1].get("rejection_code"),
            "why": explorer.vlm_picks[-1]["rejection"],
            "details": explorer.vlm_picks[-1].get("rejection_details"),
        } if rejection else None),
    }, indent=2))
    explorer.vlm_view.update(
        n=n,
        frame=decision_live,
        picks=explorer.vlm_picks,
        image=rgb,
        context=context,
        context_meta=context_meta,
        history_pages=pages,
        history=history,
        T=T.tolist(),
        head=f"{vlm_pick.model_label_text()} · decision {n} · thinking",
        words="",
        since=time.time(),
        version=explorer.vlm_view["version"] + 1,
    )
    explorer.message = f"Decision {n}: {vlm_pick.model_label_text()} is choosing the next move or view."
    explorer.stage = f"asking {vlm_pick.model_name()}"
    with explorer.lock:
        if explorer.live is not decision_live:
            raise Abort("map frame changed while building visual context; holding position")
        if decision_history is not None and (
            explorer.visual_history is not history_owner
            or explorer.visual_history.epoch != decision_history.epoch
        ):
            raise Stopped("start frame changed while building visual context")
    explorer.check()
    if continuation:
        if explorer.live is not decision_live:
            raise Abort("map frame changed during final proposal context; holding position")
        explorer.decision_pose()
        if explorer.live is not decision_live:
            raise Abort("map frame changed during final proposal context; holding position")
        if not explorer._proposal_budget_available():
            raise Stopped("original exploration budget expired before the final proposal")
    decision_limits = ({"remaining_s": max(0., explorer.budget_s - explorer.active_elapsed()),
                        "cancel": lambda: (explorer.source.stop_flag.is_set() or
                                            explorer.source.pause_flag.is_set())}
                       if isinstance(explorer.pick_hook, vlm_pick.Picker) else {})
    if decision_generation != explorer._decision_generation:
        raise DecisionInterrupted()
    check_current_camera()
    request = DecisionContext(
        tuple(images), prompt, lambda words: (
            explorer.vlm_view.__setitem__("words", words)
            if decision_generation == explorer._decision_generation else None), n,
        tuple(tools), decision_limits,
        tuple((name, tuple(np.asarray(view[1]).ravel())) for name, view in map_views.items()),
        proposal_context, decision_generation,
    )

    return DecisionEvidence(
        request=request, position=(x, y, yaw), directory=vlm_dir,
        frame=decision_live, brief=decision_brief, rgb=rgb,
        transform=T, camera_metadata=context_meta, history=history,
        history_owner=history_owner, history_snapshot=decision_history,
        selection=selection, walked_legs=walked_legs, map_views=map_views,
        scope=scope_context, rejection=rejection, continuation=continuation)
