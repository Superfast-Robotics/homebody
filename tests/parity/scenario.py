"""Seeded synthetic world for the parity harness: room grid, camera frames, clock, provider."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass

import numpy as np
from PIL import Image, ImageDraw

SEED = 7
CELL_M = 0.05
SIZE = 120  # cells per side: the grid covers [ORIGIN_M, ORIGIN_M + 6 m] in x and y
ORIGIN_M = -2.0
START = (0.0, 0.0, 0.0)
CLEAR_GOAL = (1.6, 0.0)
BLOCKED_GOAL = (1.0, 0.0)
BLOCKED_FACING = (1.0, 1.0)
DOOR_Y_M = (0.7, 1.5)  # the partition at x = 2.6 m is open between these y values


def cells(x, y):
    return round((x - ORIGIN_M) / CELL_M), round((y - ORIGIN_M) / CELL_M)


def room(seed: int = SEED, *, blocked: bool = False) -> np.ndarray:
    """Occupancy (rows = y): outer walls, a partition with a doorway, one box on the
    clear route (extended over BLOCKED_GOAL when blocked) and two seeded boxes."""
    grid = np.zeros((SIZE, SIZE), bool)
    grid[:3], grid[-3:], grid[:, :3], grid[:, -3:] = True, True, True, True
    u, _ = cells(2.6, 0.)
    _, d0 = cells(0., DOOR_Y_M[0])
    _, d1 = cells(0., DOOR_Y_M[1])
    grid[:d0, u:u + 2] = grid[d1:, u:u + 2] = True
    grid[37:44, 53:58] = True  # x 0.65..0.85, y -0.15..0.15: forces a detour
    if blocked:
        grid[37:44, 57:64] = True  # x 0.85..1.15 around BLOCKED_GOAL
    rng = np.random.default_rng(seed)
    placed = 0
    while placed < 2:
        x, y = rng.uniform(-1.4, 2.2), rng.uniform(-1.4, 3.4)
        if min(math.dist((x, y), p) for p in (START[:2], CLEAR_GOAL, (1.6, 1.0),
                                                (0.5, 0.), (1.2, 0.))) < 1.0:
            continue
        cu, cv = cells(x, y)
        grid[cv - 2:cv + 3, cu - 2:cu + 3] = True
        placed += 1
    return grid


def transform() -> np.ndarray:
    """Pixel (u, v, 1) to start-frame metres."""
    return np.array([[CELL_M, 0, ORIGIN_M], [0, CELL_M, ORIGIN_M], [0, 0, 1.]])


def world(grid: np.ndarray, x, y) -> np.ndarray:
    """Occupancy sampled at world points; outside the grid counts as occupied."""
    u = np.round((np.asarray(x) - ORIGIN_M) / CELL_M).astype(int)
    v = np.round((np.asarray(y) - ORIGIN_M) / CELL_M).astype(int)
    inside = (u >= 0) & (u < SIZE) & (v >= 0) & (v < SIZE)
    out = np.ones(u.shape, bool)
    out[inside] = grid[v[inside], u[inside]]
    return out


def scan(grid: np.ndarray, pose, rays: int = 360, height_m: float = 0.5) -> np.ndarray:
    """Synthetic registered LiDAR: the first occupied cell along each ray, as (N, 3) points."""
    x, y = pose[0], pose[1]
    angles = np.linspace(0., 2 * math.pi, rays, endpoint=False)
    ranges = np.arange(CELL_M, 8., CELL_M / 2)
    xs = x + np.cos(angles)[:, None] * ranges
    ys = y + np.sin(angles)[:, None] * ranges
    hit = world(grid, xs, ys)
    first = hit.argmax(axis=1)
    keep = hit[np.arange(rays), first]
    return np.c_[xs[keep, first[keep]], ys[keep, first[keep]], np.full(keep.sum(), height_m)]


def frame(pose, sequence: int, shape=(90, 160)) -> np.ndarray:
    """A camera frame drawn with PIL that changes with heading and position, never wall time."""
    h, w = shape
    image = Image.new("RGB", (w, h), (236, 232, 224))
    draw = ImageDraw.Draw(image)
    shift = int((pose[2] % (2 * math.pi)) / (2 * math.pi) * w)
    for i in range(0, w, 16):
        left = (i + shift) % w
        draw.rectangle([left, h // 3, left + 7, h - 10], fill=(90 + i % 120, 110, 140))
    draw.rectangle([0, h - 10, w, h], fill=(150, 120, 90))
    draw.text((4, 2), f"{pose[0]:+.1f} {pose[1]:+.1f}", fill=(20, 20, 20))
    rgb = np.asarray(image).copy()
    rgb[0, 0] = sequence % 256
    return rgb


class Clock:
    """Fake time for every patched module; sleeps advance it."""

    def __init__(self, start: float = 1000.):
        self.value = start

    def time(self):
        return self.value

    monotonic = perf_counter = time

    def monotonic_ns(self):
        return int(self.value * 1e9)

    def sleep(self, seconds):
        self.value = round(self.value + seconds, 9)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (tuple, list, set)):
        return [plain(v) for v in value]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else str(value)
    return value


def digest(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(
        plain(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def percentiles(samples) -> dict:
    values = np.asarray(samples, float)
    if not len(values):
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {"n": len(values), "p50_ms": round(float(np.percentile(values, 50)) * 1e3, 4),
            "p95_ms": round(float(np.percentile(values, 95)) * 1e3, 4),
            "max_ms": round(float(values.max()) * 1e3, 4)}


SCRIPT = ("update_region", "pick_point", "look_at", "pick_point", "explored")


class ScriptedProvider:
    """Returns SCRIPT in order; coordinates are chosen in metres and mapped through the
    overview transform the decision context recorded."""

    def __init__(self, explorer, px_to_grid, world_to_px, decision):
        self.explorer, self.px_to_grid, self.world_to_px = explorer, px_to_grid, world_to_px
        self.decision = decision
        self.requests, self.provider_s = [], []

    def grid(self, xy, context):
        view = context["map_views"]["overview"]
        pixels = self.world_to_px(np.asarray(view["T_map_px"]), np.asarray([xy]))[0]
        return [round(v) for v in self.px_to_grid(*pixels, view["shape"])]

    def generate(self, prompt, *, images, names, attempt, tick, cancelled):
        import time
        started = time.perf_counter()
        n = len(self.requests) + 1
        if n > len(SCRIPT):
            raise AssertionError("scripted decisions exhausted")
        context = json.loads((self.explorer.out_dir / f"vlm/context_leg_{n}.json").read_text())
        offered = {tool["name"] for tool in json.loads(prompt.split("Offered actions:\n", 1)[1])}
        history = context["history"] or {}
        ids = history.get("view_ids", [])
        skill = SCRIPT[n - 1]
        assert skill in offered, f"{skill} is not offered: {sorted(offered)}"
        if skill == "update_region":
            args = {"base_revision": 0, "vertices": [self.grid(xy, context) for xy in
                    [(-1.8, -1.8), (3.8, -1.8), (3.8, 3.8), (-1.8, 3.8)]],
                    "reason": "Synthetic room enclosed by walls.",
                    "evidence_ids": ids, "uncertain_edges": []}
        elif skill == "pick_point":
            goal, face = ((1.6, 0.), (1.6, 1.0)) if n == 2 else ((0.2, 1.2), (0.2, 2.4))
            x, y = self.grid(goal, context)
            fx, fy = self.grid(face, context)
            args = {"x": x, "y": y, "face_x": fx, "face_y": fy, "why": "Inspect the far wall.",
                    "target_area": "synthetic wall", "expected_gain": "readable edges"}
        elif skill == "look_at":
            x, y = self.grid((3.4, 1.1), context)
            args = {"x": x, "y": y, "why": "Look through the doorway.",
                    "target_area": "doorway", "expected_gain": "different angle"}
        else:
            args = {"why": "The synthetic room has readable evidence.",
                    "coverage_summary": "Walls and doorway visible.", "evidence_views": ids,
                    "remaining_gaps": [],
                    "image_assessments": [{"id": i, "usability": "readable",
                                           "reason": "Synthetic edges readable."} for i in ids],
                    "coverage_updates": [{"area": "synthetic room", "scope": "included",
                        "requirement": "readable walls", "complementary_required": False,
                        "status": "covered", "evidence_views": ids, "complementary_views": [], "gaps": [],
                        "resolution": "walls visible"}]}
        self.requests.append({"n": n, "skill": skill, "names": sorted(names),
                              "images": len(images), "pose": context["robot_pose_xyyaw"],
                              "region_revision": context["task_region"]["revision"],
                              "history_ids": len(ids), "arguments": args,
                              "prompt_sha256": digest(prompt.encode())})
        self.provider_s.append(time.perf_counter() - started)
        return self.decision("Scripted decision.", skill, args)
