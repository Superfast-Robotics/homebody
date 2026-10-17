"""The model's exploration decision: its prompt, the map pictures it clicks on, and its answer.

`prompt` composes explore_prompt.md with the offered actions and the decision's data; `render`
and `local_map` draw the two maps; `parse_decision` turns a provider Decision into an Answer.
Clicks become start-frame metres through `helpers.raster`.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import time
import uuid
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from homebody.helpers.floor import departure_status, reachable_network
from homebody.helpers.raster import to_px, world_to_px
from homebody.vlm.catalog import make, model_label
from homebody.vlm.provider import ProviderError

from .livemap import OBSTACLE_BAND_M, RES_M, LiveMap
from .settings import current
from .visual_history import LEDGER

MARGIN_M = 1.0  # drawn around the manual limit, or around the seen map without one
FRAME_MIN_M = 3.0  # least picture reach around the robot without a manual limit
GRID_M, LABEL_M = 0.5, 1.0  # the metre grid over the picture, and its labels
LABEL_RGBA = (120, 120, 120, 150)  # the decision map's axis labels
# map colours; dark grey is an object's inside or an obstacle outside the drawn band
UNKNOWN, FLOOR, OBSTACLE, INTERIOR = (150, 150, 150), (255, 255, 255), (0, 0, 0), (72, 72, 72)
REACHABLE_FLOOR = (238, 220, 220)  # light red: floor reachable now
OUTSIDE_CONNECTED = (250, 238, 215)
OUTSIDE_HATCH = (190, 125, 30)
FRONTIER, TRAIL, VORONOI, RANGE, GOAL, INK = (
    (230, 30, 30),
    (13, 150, 135),
    (124, 92, 255),
    (255, 215, 0),  # the limit: a yellow border, always drawn
    (60, 130, 255),
    (0, 90, 200),
)
CLICK = (0, 190, 60)  # where the model pointed
TASK_REGION_COLOR = (190, 30, 170)
SNAP = (0, 110, 30)  # where that click snapped to
GRIDLINE = (214, 214, 214)
LEGEND_PX = 80
LEGEND = (
    "white other floor · grey unknown · black occupied · dark grey object inside",
    "violet paths · teal walked · red frontier edges · grid 0.5 m",
    "blue triangle YOU · yellow map extent/manual limit · magenta task region",
)
MAP_IMAGE_MAX_PX = 2048
STANDALONE = ("update_region",)  # actions decided alone: they carry no coverage ledger


@dataclass(frozen=True)
class Tool:
    """An offered action: its name and the JSON schema of its arguments."""
    name: str
    parameters: Mapping


def tool(name: str, skill) -> Tool:
    """The action NAME offered with SKILL's argument schema; every action outside STANDALONE also
    carries the coverage ledger fields."""
    parameters = skill.PARAMETERS
    if name not in STANDALONE:
        parameters = parameters | {"properties": parameters["properties"] | LEDGER}
    return Tool(name, parameters)


def model_name() -> str:
    """The configured agent's model id."""
    return current().agent.model_id


def model_label_text() -> str:
    """The configured model's display label: its tool or API, model id and effort."""
    return model_label(current().agent)


def provider():
    """The configured agent's provider: any model in the catalog can explore."""
    return make(current().agent)


def prompt(task: str, *, camera: str, actions, observation: dict, history: dict,
           results: dict) -> str:
    """explore_prompt.md, one prompt line per offered action, the task, then this decision's
    observation, the supplied views with the coverage ledger, and the recent results."""
    instructions = (Path(__file__).parent / "explore_prompt.md").read_text()
    return "\n\n".join((
        instructions.format_map({"camera": camera, "grid_m": GRID_M}).strip(),
        "Actions:\n" + "\n".join(skill.PROMPT for skill in actions),
        "Current task:\n" + task,
        "Current observation:\n" + json.dumps(observation),
        "Views:\n" + json.dumps(model_history(history)),
        "Recent results:\n" + json.dumps(results)))


def model_history(history):
    """The supplied views with approximate poses and the coverage ledger; exact provenance stays
    in the views index."""
    state = {k: history[k] for k in ("epoch", "view_ids", "readable_ids", "current_viewpoint",
             "admission", "current_rgb_is_separate", "coverage_assessment", "association")
             if k in history}
    state["images"] = []
    for record in history.get("image_records", []):
        item = {k: record[k] for k in ("id", "source_kind", "capture_role", "viewpoint_group",
                "usability", "advisory_only", "association_kind", "exposure_pose_known")
                if k in record}
        if record.get("robot_pose_xyyaw") is not None:
            item["robot_pose_xyyaw"] = [round(v, 2) for v in record["robot_pose_xyyaw"]]
        selection = record.get("phone", {}).get("selection", {})
        if selection:
            item["selection"] = {k: selection[k] for k in ("status", "reason") if k in selection}
        route_context = record.get("route_context")
        if route_context is not None:
            item["route_context"] = {k: route_context[k] for k in (
                "route_id", "revision", "segment", "nearest_knot") if k in route_context}
        assessment = record.get("assessment") or (record.get("assessment_history") or [{}])[-1]
        item["assessment"] = {k: assessment[k] for k in ("reason", "compared_to", "relationship")
                              if k in assessment}
        if record.get("source_kind") == "transit":
            interval = record.get("action_interval", {})
            item["action"] = interval.get("context", {}).get("kind", "motion")
        state["images"].append(item)
    return state


@dataclass
class Answer:
    kind: str  # point | look | update_region | ask_operator | help | explored | error
    grid: tuple[int, int] | None  # the click on the 0..1000 grid
    why: str
    text: str = ""
    seconds: float = 0.0
    model: str | None = None
    face_grid: tuple[int, int] | None = None  # the second point: what to face from the spot
    completion: dict | None = None
    provider_response: dict | None = None  # returned text/actions, not internal model reasoning
    coverage_updates: list[dict] | None = None
    image_assessments: list[dict] | None = None
    task_region_update: dict | None = None
    target_area: str = ""
    expected_gain: str = ""
    question: str = ""
    map_view: str = "overview"


# -- the pictures -------------------------------------------------------------------------------
def render(explorer, pose):
    """(RGB uint8 image, its T_map_px, (planner, network)) drawn from one snapshot of the
    explorer's map; `to_metres(T_map_px, u, v)` grounds a pixel."""
    cell_px = current().map.picture_cell_px
    with explorer.lock:
        live = explorer.live.snapshot()
        planner = copy(explorer.planner)
        planner.live = live
        open_xy = np.asarray(planner.open_xy, float).reshape(-1, 2).copy()
        visited = explorer.visited.copy()
        trail = [(a, b) for a, b, _ in explorer.trail]
        limit = explorer.zone if explorer.zone is not None else explorer.boundary
        zone = None if limit is None else np.array(limit.points, float)
        manual_zone = zone if explorer.zone is not None else None
    known, assumed, avoid = live.known_free(), live.assumed, live.avoid()
    black = _thicker(live.obstacle_slice(*OBSTACLE_BAND_M))
    interior = live.interior()
    skel, reach = reachable_network(planner, pose[0], pose[1])
    visited = visited if visited.shape == known.shape else np.zeros_like(known)
    frame = _frame(live, known | avoid, manual_zone, pose)
    u_lo, v_hi = live.uv(frame[0], frame[2])
    u_hi, v_lo = live.uv(frame[1], frame[3])
    u0, u1 = max(int(u_lo), 0), min(int(u_hi) + 1, live.W)
    v0, v1 = max(int(v_lo), 0), min(int(v_hi) + 1, live.H)
    win = (slice(v0, v1), slice(u0, u1))
    free = (known[win] | assumed[win]) & ~avoid[win]
    height, width = v1 - v0, u1 - u0
    scale = min(cell_px, MAP_IMAGE_MAX_PX / width,
                (MAP_IMAGE_MAX_PX - LEGEND_PX) / height)
    pixels = (max(1, int(height * scale)), max(1, int(width * scale)))
    img = np.full((*pixels, 3), UNKNOWN, np.uint8)
    img[_blocks(free, pixels)] = FLOOR
    # a display pixel is reachable only when all its cells are
    reachable = reach[win] & known[win] & ~avoid[win]
    img[~_blocks(~reachable, pixels)] = REACHABLE_FLOOR
    # hatch connected floor outside the task region
    if explorer.zone is None and live.task_region_required:
        checker = live.collision_checker(planner.radius_m, task_scope=False)
        unscoped = np.isfinite(live.geodesic(*pose[:2], planner.radius_m, checker=checker))
        outside = unscoped[win] & known[win] & ~avoid[win] & ~live.center_allowed[win]
        outside_pixels = ~_blocks(~outside, pixels)
        img[outside_pixels] = OUTSIDE_CONNECTED
        yy, xx = np.indices(pixels)
        img[outside_pixels & ((xx + yy) % 12 < 2)] = OUTSIDE_HATCH
    img[_blocks(interior[win] | (avoid[win] & ~black[win]), pixels)] = INTERIOR
    img[_blocks(black[win], pixels)] = OBSTACLE
    # the transform uses the rounded output dimensions
    metres_x, metres_y = RES_M * width / pixels[1], RES_M * height / pixels[0]
    T = np.array(
        [
            [metres_x, 0.0, live.xmin + RES_M * (u0 - 0.5) + metres_x / 2],
            [0.0, -metres_y, live.ymax - RES_M * (v0 - 0.5) - metres_y / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    xs, ys = _grid_metres(frame)
    for u in np.round(world_to_px(T, np.c_[xs, np.zeros_like(xs)])[:, 0]).astype(int):
        if 0 <= u < img.shape[1]:
            img[:, u] = GRIDLINE
    for v in np.round(world_to_px(T, np.c_[np.zeros_like(ys), ys])[:, 1]).astype(int):
        if 0 <= v < img.shape[0]:
            img[v, :] = GRIDLINE
    seen_skel = skel[win] & known[win]  # the seen network is seen floor by construction
    img[_blocks(seen_skel & ~visited[win], pixels)] = VORONOI
    img[_blocks(seen_skel & visited[win], pixels)] = TRAIL
    pil = Image.new("RGB", (img.shape[1], img.shape[0] + LEGEND_PX), FLOOR)
    pil.paste(Image.fromarray(img), (0, 0))
    draw = ImageDraw.Draw(pil)
    r = cell_px
    for u, v in world_to_px(T, open_xy) if len(open_xy) else ():
        draw.ellipse([u - r, v - r, u + r, v + r], fill=FRONTIER)
    if len(trail) > 1:
        draw.line([tuple(p) for p in world_to_px(T, np.array(trail))], fill=TRAIL, width=cell_px)
    if zone is not None:
        draw.polygon([tuple(p) for p in world_to_px(T, zone)], outline=RANGE, width=6)
        (lu, lv), = world_to_px(T, zone[:1])
        draw.text((lu + 8, min(lv + 6, img.shape[0] - 22)),
                  "MANUAL LIMIT" if explorer.zone is not None else
                  "MAP EXTENT", fill=RANGE, font=_font(16))
    pil = _grid_labels(pil, T, xs, ys)
    draw = ImageDraw.Draw(pil)
    _robot(draw, T, pose)
    h = img.shape[0]
    legend = f"OVERVIEW · light red: reachable now · planning radius {planner.radius_m:g} m"
    departure = departure_status(live, pose, planner.radius_m, reach=reach)
    if departure["status"] not in {"available", "region_required"}:
        legend = departure["message"]
    extra = ("amber hatch: connected floor outside the region; not permission to walk"
             if explorer.zone is None else "MANUAL LIMIT: region changes are unavailable")
    for row, text in enumerate((legend, extra, *LEGEND)):
        draw.text((6, h + 2 + row * 15), text, fill=(60, 60, 60), font=_font(11))
    if live.task_region_points is not None:
        vertices = world_to_px(T, np.asarray(live.task_region_points))
        draw.line([tuple(v) for v in vertices] + [tuple(vertices[0])],
                  fill=TASK_REGION_COLOR, width=3)
        draw.text(tuple(vertices[0]), f" TASK REGION v{live.task_region_revision}",
                  fill=TASK_REGION_COLOR, font=_font(13))
    return np.asarray(pil), T, (planner, skel)


def local_map(image, transform, pose, *, departure=None):
    """Enlarge the local reachable patch from the same snapshot; return its own pixel transform."""
    height, width = image.shape[0] - LEGEND_PX, image.shape[1]
    centre = to_px(transform, pose[:2])
    step = np.abs([transform[0, 0], transform[1, 1]])
    vv, uu = np.nonzero(np.all(image[:height] == REACHABLE_FLOOR, axis=-1))
    nearby = (np.abs(uu - centre[0]) * step[0] <= 4.) & (
        np.abs(vv - centre[1]) * step[1] <= 4.)
    points = np.vstack([np.array(centre), np.c_[uu[nearby], vv[nearby]]])
    lo = np.maximum(points.min(axis=0) - 1.5 / step, np.array(centre) - 4. / step)
    hi = np.minimum(points.max(axis=0) + 1.5 / step, np.array(centre) + 4. / step)
    lo = np.minimum(lo, np.array(centre) - 2. / step)
    hi = np.maximum(hi, np.array(centre) + 2. / step)
    x0, y0 = np.maximum(np.floor(lo).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(hi).astype(int) + 1, [width, height])
    crop = Image.fromarray(image[y0:y1, x0:x1])
    scale = max(1., min(3., 1024. / max(crop.size)))
    size = tuple(max(1, round(v * scale)) for v in crop.size)
    enlarged = crop.resize(size, Image.Resampling.NEAREST)
    result = Image.new("RGB", (size[0], size[1] + LEGEND_PX), FLOOR)
    result.paste(enlarged)
    draw = ImageDraw.Draw(result)
    reach_label = "light red: reachable now · amber hatch: needs region permission"
    if departure is not None and departure["status"] not in {"available", "region_required"}:
        reach_label = departure["message"]
    for row, label in enumerate(("LOCAL · set map_view=local for coordinates in this image",
                                 reach_label,
                                 "thin teal line: past travel, not clearance · blue triangle: YOU",
                                 "Use OVERVIEW coordinates for update_region vertices.")):
        draw.text((6, size[1] + 3 + row * 16), label, fill=(60, 60, 60), font=_font(11))
    sx, sy = crop.width / size[0], crop.height / size[1]
    # PIL nearest-neighbour resize samples pixel centres
    crop_to_original = np.array([[sx, 0., x0 + (sx - 1.) / 2.],
                                 [0., sy, y0 + (sy - 1.) / 2.], [0., 0., 1.]])
    return np.asarray(result), np.asarray(transform) @ crop_to_original


def _frame(live: LiveMap, seen: np.ndarray, zone, pose) -> tuple[float, float, float, float]:
    """The picture's extent: the manual limit, or the robot, the seen map and the task region."""
    if zone is not None:
        lo, hi = zone.min(axis=0), zone.max(axis=0)
    else:
        here = np.array([float(pose[0]), float(pose[1])])
        lo, hi = here - FRAME_MIN_M, here + FRAME_MIN_M
        if live.task_region_points is not None:
            region = np.asarray(live.task_region_points)
            lo, hi = np.minimum(lo, region.min(axis=0)), np.maximum(hi, region.max(axis=0))
        vv, uu = np.nonzero(seen)
        if len(vv):
            sx, sy = live.xy(uu, vv)
            lo = np.minimum(lo, [sx.min(), sy.min()])
            hi = np.maximum(hi, [sx.max(), sy.max()])
    lo = np.maximum(lo - MARGIN_M, [live.xmin, live.ymax - (live.H - 1) * RES_M])
    hi = np.minimum(hi + MARGIN_M, [live.xmin + (live.W - 1) * RES_M, live.ymax])
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _blocks(cells: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize directly at display size; retain thin boolean features when reducing."""
    result = cells
    if cells.dtype == bool:
        for axis, size in enumerate(shape):
            if size < result.shape[axis]:
                starts = np.maximum(0, np.ceil(np.arange(size) * result.shape[axis] / size
                                               - .5).astype(int))
                result = np.logical_or.reduceat(result, starts, axis=axis)
    v = np.minimum(((np.arange(shape[0]) + .5) * result.shape[0] / shape[0]).astype(int),
                   result.shape[0] - 1)
    u = np.minimum(((np.arange(shape[1]) + .5) * result.shape[1] / shape[1]).astype(int),
                   result.shape[1] - 1)
    return result[v[:, None], u]


def _grid_metres(frame) -> tuple[np.ndarray, np.ndarray]:
    """The metre grid's x and y lines (every GRID_M) over the picture's extent."""
    x_lo, x_hi, y_lo, y_hi = frame
    return (
        np.arange(math.floor(x_lo), x_hi + 1e-9, GRID_M),
        np.arange(math.floor(y_lo), y_hi + 1e-9, GRID_M),
    )


def _grid_labels(pil: Image.Image, T, xs, ys) -> Image.Image:
    """Grey see-through metre labels, one every LABEL_M, on their own layer."""
    labels = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(labels)
    for x in xs:
        if abs(round(x / LABEL_M) * LABEL_M - x) < 1e-6:
            (u, _v), = world_to_px(T, np.array([[float(x), 0.0]]))
            label_draw.text((u + 2, 2), f"x {x:+.0f}", fill=LABEL_RGBA, font=_font(11))
    for y in ys:
        if abs(round(y / LABEL_M) * LABEL_M - y) < 1e-6:
            (_u, v), = world_to_px(T, np.array([[0.0, float(y)]]))
            label_draw.text((2, v - 12), f"y {y:+.0f}", fill=LABEL_RGBA, font=_font(11))
    return Image.alpha_composite(pil.convert("RGBA"), labels).convert("RGB")


def _thicker(mask: np.ndarray) -> np.ndarray:
    """The mask grown by one cell in every direction."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _robot(draw: ImageDraw.ImageDraw, T, pose) -> None:
    """The robot as a small orientation triangle."""
    x, y, yaw = (float(v) for v in pose[:3])
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([[x + c * dx - s * dy, y + s * dx + c * dy]
                        for dx, dy in ((.18, 0.), (-.10, .10), (-.10, -.10))])
    draw.polygon([tuple(p) for p in world_to_px(T, corners)], fill=INK)


def _font(size: int):
    return ImageFont.load_default(size=size)


def png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def mark(rgb: np.ndarray, px, used: bool, goal_px=None, label: str | None = None,
         face_px=None) -> np.ndarray:
    """The picture with the click, its snapped goal and the facing arrow drawn; LABEL names the
    model, by default the configured one."""
    label = model_label_text() if label is None else label
    pil = Image.fromarray(np.asarray(rgb, np.uint8).copy())
    draw = ImageDraw.Draw(pil)
    if face_px is not None and goal_px is not None:
        gu, gv = goal_px
        du, dv = face_px[0] - gu, face_px[1] - gv
        n = math.hypot(du, dv) or 1.0
        tu, tv = gu + du / n * min(n, 40.0), gv + dv / n * min(n, 40.0)
        draw.line([(gu, gv), (tu, tv)], fill=GOAL, width=3)
        draw.ellipse([tu - 4, tv - 4, tu + 4, tv + 4], fill=GOAL)
    if px is not None:
        u, v = px
        colour = CLICK if used else FRONTIER
        for radius in (14, 4):
            draw.ellipse([u - radius, v - radius, u + radius, v + radius], outline=colour, width=4)
        text = (f"{label} pointed" if used else f"{label} pointed (not walked)")
        draw.text((u + 18, v - 8), text, fill=colour, font=_font(13))
    apart = goal_px is not None and (
        px is None or math.hypot(goal_px[0] - px[0], goal_px[1] - px[1]) > 6)
    if goal_px is not None and apart:
        u, v = goal_px
        if px is not None:
            draw.line([tuple(px), (u, v)], fill=CLICK, width=3)
        draw.ellipse([u - 9, v - 9, u + 9, v + 9], fill=SNAP, outline=(255, 255, 255), width=2)
        draw.text((u + 14, v + 4), "snapped here", fill=SNAP, font=_font(13))
    return np.asarray(pil)


def route_changed(previous, current, tolerance_m):
    """Compare the remaining route, ignoring sampling, travelled prefix and arrival noise."""
    previous, current = np.asarray(previous, float), np.asarray(current, float)
    if any(line.ndim != 2 or line.shape[1] != 2 or len(line) < 2
           or not np.isfinite(line).all() for line in (previous, current)):
        return False
    start, delta = previous[:-1], np.diff(previous, axis=0)
    length2 = (delta * delta).sum(axis=1)
    along = np.clip(((current[0] - start) * delta).sum(axis=1)
                    / np.maximum(length2, 1e-12), 0, 1)
    projections = start + along[:, None] * delta
    segment = int(np.argmin(np.linalg.norm(projections - current[0], axis=1)))
    previous = np.vstack([projections[segment], previous[segment + 1:]])

    def departure(vertices, polyline):
        start, delta = polyline[:-1], np.diff(polyline, axis=0)
        offset = vertices[:, None, :] - start
        length2 = (delta * delta).sum(axis=1)
        along = np.clip((offset * delta).sum(axis=2) / np.maximum(length2, 1e-12), 0, 1)
        distance = np.linalg.norm(offset - along[:, :, None] * delta, axis=2)
        return bool(np.any(distance.min(axis=1) > tolerance_m))

    return departure(previous, current) or departure(current, previous)


# -- the decision -------------------------------------------------------------------------------
# JSON schema types of the skills' declared arguments
SCHEMA_TYPES = {"string": (str,), "integer": (int, float), "number": (int, float),
                "boolean": (bool,), "object": (Mapping,), "array": (list, tuple)}


def schema_mismatch(name, args, parameters) -> str:
    """Why ARGS do not satisfy the skill's PARAMETERS (required keys, value types), or ''."""
    missing = [key for key in parameters.get("required", ()) if key not in args]
    if missing:
        return f"{name} requires {', '.join(missing)}"
    for key, value in args.items():
        kind = parameters["properties"][key].get("type")
        if kind in SCHEMA_TYPES and (not isinstance(value, SCHEMA_TYPES[kind])
                                     or (kind != "boolean" and isinstance(value, bool))):
            return f"{name}: {key} must be a JSON {kind}"
    return ""


def parse_decision(decision, provider, tools, seconds) -> Answer:
    """The Answer for one provider Decision; an unoffered action or field, a missing required
    argument or a mistyped one is an error Answer."""
    returned = asdict(decision)
    text, name = decision.text, decision.skill
    model = getattr(getattr(provider, "settings", None), "model_id", None) or model_name()

    def invalid(reason):
        return Answer("error", None, reason, text, seconds, model, provider_response=returned)

    offered = next((tool for tool in tools if tool.name == name), None)
    if offered is None:
        return invalid(f"action {name!r} was not offered for this decision")
    if not isinstance(decision.arguments, Mapping):
        return invalid("action arguments must be an object")
    args = dict(decision.arguments)
    if set(args) - offered.parameters["properties"].keys():
        return invalid(f"{name} cannot include other actions or unoffered fields")
    mismatch = schema_mismatch(name, args, offered.parameters)
    if mismatch:
        return invalid(mismatch)
    if name == "ask_operator" and any(not isinstance(args.get(key), str) or not args[key].strip()
                                      for key in ("question", "why")):
        return invalid("ask_operator requires a nonempty question and explanation")
    map_view = args.get("map_view", "overview")
    if not isinstance(map_view, str) or map_view not in {"overview", "local"}:
        return invalid("map_view must identify overview or local")
    coverage = args.get("coverage_updates")
    context = {"image_assessments": args.get("image_assessments"),
                   "target_area": args.get("target_area", ""),
                   "expected_gain": args.get("expected_gain", "")}
    why = str(args.get("why") or text or "").strip()
    if name == "update_region":
        if not args["reason"].strip():
            return invalid("update_region requires a nonempty reason")
        return Answer("update_region", None, args["reason"].strip(), text, seconds, model,
                      task_region_update=args, provider_response=returned)
    if name == "ask_operator":
        return Answer("ask_operator", None, why, text, seconds, model,
                      question=args["question"].strip(), provider_response=returned,
                      coverage_updates=coverage, **context)
    if name == "request_help":
        return Answer("help", None, why, text, seconds, model, provider_response=returned,
                      coverage_updates=coverage, **context)
    if name == "explored":
        return Answer("explored", None, why, text, seconds, model, completion=args,
                      provider_response=returned, coverage_updates=coverage, **context)
    if name == "look_at":
        grid = (round(float(args["x"])), round(float(args["y"])))
        return Answer("look", grid, why, text, seconds, model, provider_response=returned,
                      coverage_updates=coverage, map_view=map_view, **context)
    if name == "pick_point":
        grid = (round(float(args["x"])), round(float(args["y"])))
        face = None
        if "face_x" in args and "face_y" in args:
            face = (round(float(args["face_x"])), round(float(args["face_y"])))
        return Answer("point", grid, why, text, seconds, model, face,
                      provider_response=returned, coverage_updates=coverage,
                      map_view=map_view, **context)
    return invalid(f"unsupported exploration action {name!r}")


OFFERED = ("\nSelect exactly one offered action. Return its name as skill and its fields as "
           "arguments_json in the required Decision response. Offered actions:\n")


class DecisionSession:
    """The exploration model's provider and its decisions, without a robot, map or recorder."""

    def __init__(self, provider=None):
        self.provider = provider

    def pick_exploration(self, config, images, prompt, on_partial, leg, tools, remaining_s=None,
                         cancel=None):
        """One decision of CONFIG (a vlm_pick.Picker) among TOOLS. A provider that fails is built
        again for the next decision, and its failure is the decision's error answer."""
        if leg <= 1 or self.provider is None:
            self.provider = make(current().agent)
        try:
            budget = config.timeout_s if remaining_s is None else min(config.timeout_s, remaining_s)
            if budget <= 0:
                raise TimeoutError("exploration decision budget expired")
            return self.generate_exploration(self.provider, images, prompt, budget, tools, cancel,
                                             workspace=config.workspace)
        # the provider failed to answer
        except (ProviderError, TimeoutError, OSError) as exc:
            self.provider = None
            detail = f"{type(exc).__name__}: {exc}"
            return Answer("error", None, detail[:300], text=detail)

    @staticmethod
    def generate_exploration(provider, images, prompt_text, timeout_s, tools, cancel=None,
                             workspace=None):
        """One provider decision among TOOLS within TIMEOUT_S, as a vlm_pick.Answer. The decision's
        files go under WORKSPACE, by default a temporary directory."""
        started = time.monotonic()
        deadline = started + timeout_s

        def interrupted():
            return (cancel is not None and cancel()) or time.monotonic() >= deadline

        tools = tuple(tools)
        prompt = prompt_text + OFFERED + json.dumps([asdict(tool) for tool in tools])
        with tempfile.TemporaryDirectory(prefix="homebody-exploration-inputs-") as temporary:
            paths = []
            for index, image in enumerate(images):
                path = Path(temporary) / f"image-{index:02d}.png"
                path.write_bytes(bytes(image))
                paths.append(path)
            root = Path(temporary) if workspace is None else Path(workspace)
            decision = provider.generate(
                prompt, images=tuple(paths), names=tuple(tool.name for tool in tools),
                attempt=root / f"decision-{uuid.uuid4().hex}", tick=lambda: time.sleep(.02),
                cancelled=interrupted)
        if interrupted():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no answer in {timeout_s:.0f} s")
            raise ProviderError("exploration decision cancelled while held")
        return parse_decision(decision, provider, tools, round(time.monotonic() - started, 1))


class Picker:
    """The decision hook: one exploration decision through the ExplorationSession's provider."""

    def __init__(self, workspace=None):
        self.timeout_s = current().agent.timeout
        self.workspace = Path(workspace) if workspace is not None else None
        self.session = None

    def __call__(self, images, prompt_text: str, on_partial, leg: int, tools,
                 remaining_s=None, cancel=None) -> Answer:
        if self.session is None:
            self.session = DecisionSession()
        return self.session.pick_exploration(self, images, prompt_text, on_partial, leg,
                                             tools, remaining_s, cancel)


# -- what the console shows ---------------------------------------------------------------------
def outcome_words(rec: dict) -> str:
    """Describe decision selection; a selected click does not establish movement."""
    if rec.get("rejection"):
        recovered = (rec.get("departure_recovery") or {}).get("outcome")
        if recovered in {"route_restored", "departure_restored", "repositioned_goal_unavailable"}:
            return "destination rejected; the robot was moved back to a safe starting spot"
        if rec.get("kind") == "explored":
            return f"completion refused: {rec['rejection']}"
        return f"rejected: {rec['rejection']}"
    result = rec.get("action_result") or {}
    if rec.get("kind") == "look" and result.get("code") == "LOOK_REPOSITIONED":
        return "view incomplete; the robot was moved back to a safe starting spot"
    facing = rec.get("facing_result") or {}
    if (result.get("outcome") or rec.get("outcome")) == "arrived" and facing.get(
            "outcome") in {"refused", "stopped"}:
        return "arrived; facing incomplete"
    if (result.get("outcome") or rec.get("outcome")) == "arrived":
        return "completed viewing turn" if rec.get("kind") == "look" else "arrived"
    if result.get("outcome") in ("refused", "stopped"):
        return f"{result['outcome']}: {result['why']}"
    if rec.get("kind") == "explored" and rec.get("used"):
        return "exploration complete"
    if rec.get("kind") == "help" and rec.get("used"):
        return "holding, unfinished; the model asked the operator for help"
    if rec.get("kind") == "ask_operator" and rec.get("used"):
        return ("operator answered; preparing a new decision"
                if rec.get("outcome") == "answered" else "holding; waiting for the operator's answer")
    if rec.get("kind") == "update_region" and rec.get("used"):
        return "updated the task region; no movement"
    if rec.get("kind") == "look" and rec.get("used"):
        return "selected a viewing turn at the current position"
    if rec.get("used") and rec.get("selection_basis") == "requested_observed_floor":
        return "selected the requested observed-floor viewpoint"
    if rec.get("used") and rec.get("selection_basis") == "nearby_reachable_floor":
        return f"selected nearby reachable floor ({rec['snap_m']:.2f} m away)"
    if rec.get("used") and rec.get("snap_m") is not None:
        return f"selected the click (moved {rec['snap_m']:.2f} m onto the path network)"
    if rec.get("used"):
        return "selected the click"
    return rec.get("fallback") or rec.get("kind") or "?"
