# SPDX-License-Identifier: Apache-2.0
# The occupancy grid checker, grid A* and RRT-Connect planners and footprint dilation are derived from
# Isaac Lab Arena, Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
"""Planar geometry for exploration: the occupancy-grid collision checkers and swept-footprint
checks, deterministic grid A* with an RRT-Connect fallback, and footprint dilation."""

from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]


def _spiral_nearest_safe(is_safe, x, y, radius, max_search=2.0, step=0.1):
    """The point itself when safe, else the nearest safe ring sample within ``max_search`` m,
    or None."""
    if is_safe(x, y, radius):
        return (x, y)
    r = step
    while r <= max_search + 1e-9:
        n = max(8, round(2.0 * math.pi * r / step))
        for k in range(n):
            th = 2.0 * math.pi * k / n
            cx, cy = x + r * math.cos(th), y + r * math.sin(th)
            if is_safe(cx, cy, radius):
                return (cx, cy)
        r += step
    return None


@dataclass
class GridCollisionChecker:
    """Pre-sampled occupancy grid with O(1) ``is_safe`` lookups; the footprint radius is baked
    in, the per-query ``radius`` is ignored and out-of-grid points are unsafe."""

    xmin: float
    ymin: float
    cell: float
    nx: int
    ny: int
    grid: list  # nx lists of ny bools
    radius_baked: float = 0.0

    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` of the sampled cell centres."""
        return (self.xmin, self.xmin + self.cell * (self.nx - 1),
                self.ymin, self.ymin + self.cell * (self.ny - 1))

    def contains(self, x: float, y: float) -> bool:
        """Whether ``(x, y)`` lies on the sampled grid; `is_safe` is also False off the grid."""
        i = round((x - self.xmin) / self.cell)
        j = round((y - self.ymin) / self.cell)
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_safe(self, x: float, y: float, radius: float = 0.0) -> bool:
        i = round((x - self.xmin) / self.cell)
        j = round((y - self.ymin) / self.cell)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return False
        return self.grid[i][j]

    def nearest_safe(self, x: float, y: float, radius: float = 0.0, max_search: float = 2.0, step: float = 0.1):
        return _spiral_nearest_safe(self.is_safe, x, y, radius, max_search, step)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return round((x - self.xmin) / self.cell), round((y - self.ymin) / self.cell)


def _point_segment_distances(points, a, b):
    delta = b - a
    fractions = np.clip((points - a) @ delta / max(float(delta @ delta), 1e-30), 0., 1.)
    return np.linalg.norm(points - (a + fractions[:, None] * delta), axis=1)


def _segment_square_distances(a, b, centers, half_cell):
    """Exact 2-D distance from a segment to each closed, axis-aligned cell square."""
    lower, upper = centers - half_cell, centers + half_cell
    start = np.linalg.norm(np.maximum(np.maximum(lower - a, a - upper), 0.), axis=1)
    end = np.linalg.norm(np.maximum(np.maximum(lower - b, b - upper), 0.), axis=1)
    distance = np.minimum(start, end)
    for corner in (lower, upper, np.column_stack((lower[:, 0], upper[:, 1])),
                   np.column_stack((upper[:, 0], lower[:, 1]))):
        distance = np.minimum(distance, _point_segment_distances(corner, a, b))
    enter, leave = np.zeros(len(centers)), np.ones(len(centers))
    delta = b - a
    for axis in (0, 1):
        if abs(delta[axis]) < 1e-30:
            leave[(a[axis] < lower[:, axis]) | (a[axis] > upper[:, axis])] = -1.
        else:
            first = (lower[:, axis] - a[axis]) / delta[axis]
            last = (upper[:, axis] - a[axis]) / delta[axis]
            enter = np.maximum(enter, np.minimum(first, last))
            leave = np.minimum(leave, np.maximum(first, last))
    distance[enter <= leave] = 0.
    return start, distance


_EGRESS_NUMERICAL_TOLERANCE_M = 1e-6  # clearance roundoff allowed when leaving an overlap


def swept_circle_points(a, b, points, radius_m, *, allow_egress=False):
    """Check a circular footprint swept from A to B against metric points; with egress, an
    existing overlap may be left when separation from each overlapping point never decreases."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    points = np.asarray(points, dtype=float)
    if (a.shape != (2,) or b.shape != (2,) or points.ndim != 2
            or points.shape[1] < 2 or not np.isfinite(radius_m) or radius_m < 0
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.isfinite(points).all()):
        raise ValueError("swept footprint requires finite planar geometry and nonnegative radius")
    result = {"safe": True, "reason": "", "radius_m": float(radius_m)}
    if not len(points):
        return result
    points = points[:, :2]
    nearby = np.all((points >= np.minimum(a, b) - radius_m)
                    & (points <= np.maximum(a, b) + radius_m), axis=1)
    points = points[nearby]
    if not len(points):
        return result
    start = np.linalg.norm(points - a, axis=1)
    minimum = _point_segment_distances(points, a, b)
    overlap = start <= radius_m + 1e-10
    refused = minimum <= radius_m + 1e-10
    if allow_egress:
        refused &= (~overlap | (minimum < start - _EGRESS_NUMERICAL_TOLERANCE_M)
                    | (minimum <= 1e-10))
    if refused.any():
        index = int(np.flatnonzero(refused)[np.argmin(minimum[refused])])
        result.update(safe=False, reason="fresh obstacle intersects swept footprint",
                      point_xy=points[index].tolist(), clearance_m=float(minimum[index]),
                      start_clearance_m=float(start[index]))
    return result


def twist_sweep_envelope(x, y, yaw, vx, vy, wz, duration_s):
    """The constant-twist endpoint and the padding (sagitta) that makes the chord capsule
    contain the swept circular arc."""
    if not all(math.isfinite(v) for v in (x, y, yaw, vx, vy, wz, duration_s)) or duration_s < 0:
        raise ValueError("twist sweep requires finite values and nonnegative duration")
    angle = wz * duration_s
    if wz == 0.:
        dx, dy, padding = vx * duration_s, vy * duration_s, 0.
    else:
        sin_angle, one_minus_cos = math.sin(angle), 2. * math.sin(angle / 2.) ** 2
        dx = (vx * sin_angle - vy * one_minus_cos) / wz
        dy = (vx * one_minus_cos + vy * sin_angle) / wz
        turning_radius = math.hypot(vx, vy) / abs(wz)
        padding = (2. * turning_radius if abs(angle) >= 2. * math.pi else
                   2. * turning_radius * math.sin(angle / 4.) ** 2)
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * dx - s * dy, y + s * dx + c * dy), padding


@dataclass
class ExactGridCollisionChecker(GridCollisionChecker):
    """Exact circle-versus-cell-square sweeps; the grid is only the A* graph. ``obstacles``
    marks cells eligible for non-worsening egress; ``center_allowed`` bounds the centre."""

    occupied: np.ndarray | None = None
    obstacles: np.ndarray | None = None
    center_allowed: np.ndarray | None = None

    @property
    def exact_footprint(self):
        return True

    def is_safe(self, x, y, radius=0.):
        return self.check_motion((x, y), (x, y))["safe"]

    def graph_cell(self, point):
        """Connect the actual point to a local graph vertex without moving the endpoint."""
        i, j = self._cell(*point)
        cells = [(a, b) for a in range(i - 1, i + 2) for b in range(j - 1, j + 2)
                 if 0 <= a < self.nx and 0 <= b < self.ny and self.grid[a, b]]
        cells.sort(key=lambda ij: math.dist(point, (self.xmin + ij[0] * self.cell,
                                                  self.ymin + ij[1] * self.cell)))
        return next((ij for ij in cells if self.check_motion(
            point, (self.xmin + ij[0] * self.cell, self.ymin + ij[1] * self.cell))["safe"]), None)

    def _near_cells(self, mask, a, b, radius, *, invert=False):
        origin = np.array((self.xmin, self.ymin))
        padding = radius + self.cell / 2.
        lo = np.maximum(0, np.floor((np.minimum(a, b) - padding - origin) / self.cell).astype(int))
        hi = np.minimum((self.nx, self.ny),
                        np.ceil((np.maximum(a, b) + padding - origin) / self.cell).astype(int) + 1)
        window = mask[lo[0]:hi[0], lo[1]:hi[1]]
        return np.argwhere(~window if invert else window) + lo

    def check_motion(self, a, b, *, allow_egress=False, extra_radius_m=0.):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if not math.isfinite(extra_radius_m) or extra_radius_m < 0:
            raise ValueError("swept footprint padding must be finite and nonnegative")
        radius = self.radius_baked + extra_radius_m
        result = {"safe": True, "reason": "", "radius_m": radius}
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
            return result | {"safe": False, "reason": "non-finite or non-planar motion"}
        origin = np.array((self.xmin, self.ymin))
        lower = origin - self.cell / 2. + radius
        upper = origin + (np.array((self.nx, self.ny)) - .5) * self.cell - radius
        if (np.minimum(a, b) < lower).any() or (np.maximum(a, b) > upper).any():
            return result | {"safe": False, "reason": "footprint crosses map extent"}
        if self.center_allowed is not None:
            boundary = GridCollisionChecker(self.xmin, self.ymin, self.cell, self.nx, self.ny,
                                             self.center_allowed)
            if not _grid_segment_free(boundary, tuple(a), tuple(b)):
                return result | {"safe": False, "reason": "centre crosses operator boundary"}
            if extra_radius_m:
                outside = self._near_cells(self.center_allowed, a, b, extra_radius_m, invert=True)
                if len(outside):
                    _, distance = _segment_square_distances(
                        a, b, origin + outside * self.cell, self.cell / 2.)
                    if (distance <= extra_radius_m + 1e-10).any():
                        return result | {"safe": False,
                                         "reason": "curved centre sweep crosses operator boundary"}
        indices = self._near_cells(self.occupied, a, b, radius)
        if not len(indices):
            return result
        centers = origin + indices * self.cell
        start, minimum = _segment_square_distances(a, b, centers, self.cell / 2.)
        refused = minimum <= radius + 1e-10
        if allow_egress and self.obstacles is not None:
            known = self.obstacles[indices[:, 0], indices[:, 1]]
            existing = known & (start <= radius + 1e-10)
            refused &= (~existing | (minimum < start - _EGRESS_NUMERICAL_TOLERANCE_M)
                        | (minimum <= 1e-10))
        if refused.any():
            index = int(np.flatnonzero(refused)[np.argmin(minimum[refused])])
            known = self.obstacles is not None and self.obstacles[tuple(indices[index])]
            result.update(safe=False, reason="obstacle intersects swept footprint" if known else
                          "blocked or unobserved cell intersects swept footprint",
                          cell_ij=indices[index].tolist(), cell_xy=centers[index].tolist(),
                          clearance_m=float(minimum[index]), start_clearance_m=float(start[index]))
        return result


def inflate_exact_footprint(blocked, cell, radius):
    """Circle versus cell-square admission at graph vertices, without radius rounding."""
    reach = math.ceil(radius / cell + .5)
    out = np.zeros_like(blocked, dtype=bool)
    nx, ny = blocked.shape
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            distance = math.hypot(max(abs(di) - .5, 0.), max(abs(dj) - .5, 0.)) * cell
            if distance > radius + 1e-10:
                continue
            si0, si1 = max(0, di), min(nx, nx + di)
            di0, di1 = max(0, -di), min(nx, nx - di)
            sj0, sj1 = max(0, dj), min(ny, ny + dj)
            dj0, dj1 = max(0, -dj), min(ny, ny - dj)
            out[di0:di1, dj0:dj1] |= blocked[si0:si1, sj0:sj1]
    margin = np.minimum(np.arange(nx) + .5, nx - np.arange(nx) - .5) * cell < radius
    out[margin, :] = True
    margin = np.minimum(np.arange(ny) + .5, ny - np.arange(ny) - .5) * cell < radius
    out[:, margin] = True
    return out


# extend()/connect() status codes
_TRAPPED, _ADVANCED, _REACHED = 0, 1, 2


@dataclass
class RRTConfig:
    """Tunables for :func:`plan_rrt`."""

    step_size: float = 0.5
    """Tree-extension step [m]: each extend adds a node at most this far from its parent."""
    edge_step: float = 0.1
    """Edge collision-sampling resolution [m]."""
    collision_margin: float = 0.12
    """Extra footprint inflation [m] added to ``robot_radius`` for ALL collision queries."""
    goal_bias: float = 0.1
    """Probability of sampling the goal directly (speeds convergence on open maps)."""
    max_iters: int = 4000
    """Max dual-tree iterations before returning None."""
    robot_radius: float = 0.3
    """Footprint radius passed to ``checker.is_safe`` for every collision query."""
    bounds: tuple[float, float, float, float] | None = None
    """Sampling box ``(xmin, xmax, ymin, ymax)``. None = start/goal bbox + ``bounds_margin``."""
    bounds_margin: float = 3.0
    """Margin [m] added around the start/goal bbox when ``bounds`` is not given."""
    smooth_iters: int = 150
    """Shortcut-smoothing passes over the raw path (0 = no smoothing)."""
    seed: int = 0
    """RNG seed — makes planning deterministic for a given (start, goal, checker)."""


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def segment_collision_free(checker, a: Point, b: Point, radius: float, edge_step: float) -> bool:
    """Check every touched grid cell, or sample other checkers every ``edge_step`` m."""
    if isinstance(checker, GridCollisionChecker):
        return _grid_segment_free(checker, a, b)
    d = _dist(a, b)
    n = max(1, math.ceil(d / max(edge_step, 1e-6)))
    for i in range(n + 1):
        t = i / n
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        if not checker.is_safe(x, y, radius):
            return False
    return True


def _steer(frm: Point, to: Point, step: float) -> Point:
    d = _dist(frm, to)
    if d <= step or d < 1e-12:
        return to
    s = step / d
    return (frm[0] + (to[0] - frm[0]) * s, frm[1] + (to[1] - frm[1]) * s)


class _Tree:
    """Minimal RRT tree: parallel node/parent arrays with linear nearest-neighbour scan."""

    def __init__(self, root: Point):
        self.nodes: list[Point] = [root]
        self.parent: list[int] = [-1]

    def nearest(self, q: Point) -> int:
        best_i, best_d = 0, _dist(self.nodes[0], q)
        for i in range(1, len(self.nodes)):
            d = _dist(self.nodes[i], q)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def add(self, q: Point, parent: int) -> int:
        self.nodes.append(q)
        self.parent.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, i: int) -> list[Point]:
        out: list[Point] = []
        while i != -1:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out


def _extend(tree: _Tree, q: Point, checker, cfg: RRTConfig) -> tuple[int, int]:
    i = tree.nearest(q)
    near = tree.nodes[i]
    new = _steer(near, q, cfg.step_size)
    if not segment_collision_free(checker, near, new, cfg.robot_radius, cfg.edge_step):
        return _TRAPPED, i
    j = tree.add(new, i)
    return (_REACHED if _dist(new, q) < 1e-9 else _ADVANCED), j


def _connect(tree: _Tree, q: Point, checker, cfg: RRTConfig, cancelled=None) -> tuple[int, int]:
    status, j = _ADVANCED, tree.nearest(q)
    while status == _ADVANCED:
        if cancelled is not None and cancelled():
            return _TRAPPED, j
        status, j = _extend(tree, q, checker, cfg)
    return status, j


def _ensure_safe(checker, p: Point, radius: float) -> Point | None:
    if checker.is_safe(p[0], p[1], radius):
        return p
    nearest_safe = getattr(checker, "nearest_safe", None)
    if nearest_safe is not None:
        s = nearest_safe(p[0], p[1], radius)
        if s is not None and checker.is_safe(s[0], s[1], radius):
            return (float(s[0]), float(s[1]))
    return None


def _default_bounds(start: Point, goal: Point, margin: float) -> tuple[float, float, float, float]:
    return (
        min(start[0], goal[0]) - margin,
        max(start[0], goal[0]) + margin,
        min(start[1], goal[1]) - margin,
        max(start[1], goal[1]) + margin,
    )


def _shortcut_smooth(path: list[Point], checker, cfg: RRTConfig, rng: random.Random,
                     cancelled=None) -> list[Point] | None:
    pts = list(path)
    for _ in range(cfg.smooth_iters):
        if cancelled is not None and cancelled():
            return None
        is_long_enough = len(pts) > 2
        if not is_long_enough:
            break
        i = rng.randint(0, len(pts) - 2)
        j = rng.randint(i + 2, len(pts) - 1) if i + 2 <= len(pts) - 1 else -1
        if j < 0:
            continue
        if segment_collision_free(checker, pts[i], pts[j], cfg.robot_radius, cfg.edge_step):
            pts = pts[: i + 1] + pts[j:]
    return pts


def plan_rrt(
    start: Point,
    goal: Point,
    checker,
    *,
    seed: int = 0,
    config: RRTConfig | None = None,
    cancelled=None,
    **overrides,
) -> list[Point] | None:
    """A collision-free polyline ``[start, ..., goal]`` by seeded RRT-Connect, ``overrides``
    setting :class:`RRTConfig` fields; None when unsafe, not found or cancelled."""
    if cancelled is not None and cancelled():
        return None
    cfg = config or RRTConfig()
    # the seed argument overrides config and overrides
    merged = {**cfg.__dict__, **overrides, "seed": seed}
    cfg = RRTConfig(**merged)
    rng = random.Random(cfg.seed)

    margin = max(cfg.collision_margin, 0.75 * cfg.edge_step)
    cfg = RRTConfig(**{**cfg.__dict__, "robot_radius": cfg.robot_radius + margin})
    radius = cfg.robot_radius

    s = _ensure_safe(checker, (float(start[0]), float(start[1])), radius)
    g = _ensure_safe(checker, (float(goal[0]), float(goal[1])), radius)
    is_start_or_goal_unsafe = s is None or g is None
    if is_start_or_goal_unsafe:
        return None

    is_straight_shot_clear = segment_collision_free(checker, s, g, radius, cfg.edge_step)
    if cancelled is not None and cancelled():
        return None
    if is_straight_shot_clear:
        return [s, g]

    bounds = cfg.bounds or _default_bounds(s, g, cfg.bounds_margin)
    xmin, xmax, ymin, ymax = bounds

    tree_a, tree_b = _Tree(s), _Tree(g)
    for _ in range(cfg.max_iters):
        if cancelled is not None and cancelled():
            return None
        if rng.random() < cfg.goal_bias:
            q = g if tree_a.nodes[0] == s else s  # bias toward the other tree's root
        else:
            q = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
        status, ja = _extend(tree_a, q, checker, cfg)
        if status != _TRAPPED:
            q_new = tree_a.nodes[ja]
            cstatus, jb = _connect(tree_b, q_new, checker, cfg, cancelled=cancelled)
            if cstatus == _REACHED:
                half_a = tree_a.path_to_root(ja)
                half_b = tree_b.path_to_root(jb)
                full = list(reversed(half_a)) + half_b[1:]
                if _dist(full[0], s) > _dist(full[-1], s):
                    full = list(reversed(full))
                return _shortcut_smooth(full, checker, cfg, rng, cancelled=cancelled)
        tree_a, tree_b = tree_b, tree_a
    return None


_SQRT2 = math.sqrt(2.0)


def plan_grid_astar(start: Point, goal: Point, grid: GridCollisionChecker, *,
                    robot_radius: float = 0.0, cancelled=None,
                    clearance_margin_m: float = 0.0) -> list[Point] | None:
    """8-connected A* over the free cells, shortcut by line of sight into a sparse polyline
    ``[start, ..., goal]``; None when the endpoints cannot connect or on cancel."""
    if cancelled is not None and cancelled():
        return None
    if not math.isfinite(clearance_margin_m) or clearance_margin_m < 0:
        raise ValueError("clearance_margin_m must be finite and nonnegative")
    penalty = None
    if clearance_margin_m:
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        if cancelled is not None and cancelled():
            return None
        # a soft cost outside the baked footprint, not occupancy
        distance = distance_transform_edt(np.pad(grid.grid, 1))[1:-1, 1:-1] * grid.cell
        distance = np.maximum(0., distance - grid.cell / math.sqrt(2.))
        penalty = 2. * np.maximum(0., 1. - distance / clearance_margin_m) ** 2

    def to_cell(x, y):
        return round((x - grid.xmin) / grid.cell), round((y - grid.ymin) / grid.cell)

    def to_world(i, j):
        return (grid.xmin + i * grid.cell, grid.ymin + j * grid.cell)

    s = grid.nearest_safe(float(start[0]), float(start[1]), robot_radius) if not grid.is_safe(float(start[0]), float(start[1])) else (float(start[0]), float(start[1]))
    g = grid.nearest_safe(float(goal[0]), float(goal[1]), robot_radius) if not grid.is_safe(float(goal[0]), float(goal[1])) else (float(goal[0]), float(goal[1]))
    if s is None or g is None:
        return None
    if isinstance(grid, ExactGridCollisionChecker):
        start_cell, goal_cell = grid.graph_cell(s), grid.graph_cell(g)
        if start_cell is None or goal_cell is None:
            return None
        si, sj = start_cell
        gi, gj = goal_cell
    else:
        si, sj = to_cell(*s)
        gi, gj = to_cell(*g)
    if not (0 <= si < grid.nx and 0 <= sj < grid.ny and grid.grid[si][sj]):
        return None
    if not (0 <= gi < grid.nx and 0 <= gj < grid.ny and grid.grid[gi][gj]):
        return None

    NEI = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
           (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2)]

    def h(i, j):
        return math.hypot(i - gi, j - gj)

    open_heap = [(h(si, sj), 0.0, si, sj)]
    came: dict = {}
    best = {(si, sj): 0.0}
    found = False
    while open_heap:
        if cancelled is not None and cancelled():
            return None
        _, gc, i, j = heapq.heappop(open_heap)
        if (i, j) == (gi, gj):
            found = True
            break
        if gc > best.get((i, j), 1e18):
            continue
        for di, dj, w in NEI:
            ni, nj = i + di, j + dj
            if not (0 <= ni < grid.nx and 0 <= nj < grid.ny and grid.grid[ni][nj]):
                continue
            if di != 0 and dj != 0 and not (grid.grid[i][nj] and grid.grid[ni][j]):
                continue  # no diagonal squeeze through a wall corner
            ng = gc + w * (1. if penalty is None else
                           1. + .5 * (penalty[i, j] + penalty[ni, nj]))
            if ng < best.get((ni, nj), 1e18):
                best[(ni, nj)] = ng
                came[(ni, nj)] = (i, j)
                heapq.heappush(open_heap, (ng + h(ni, nj), ng, ni, nj))
    if not found:
        return None

    cells = [(gi, gj)]
    while cells[-1] != (si, sj):
        cells.append(came[cells[-1]])
    cells.reverse()
    world = [s] + [to_world(i, j) for i, j in cells] + [g]

    def segment_cost(a, b):
        length = _dist(a, b)
        n = max(1, math.ceil(length / (grid.cell / 2.)))
        cost = 0.
        for k in range(n):
            t = (k + .5) / n
            i, j = to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            cost += 1. + penalty[i, j]
        return length * cost / n

    costs = ([0.] if penalty is not None else None)
    if costs is not None:
        for a, b in itertools.pairwise(world):
            costs.append(costs[-1] + segment_cost(a, b))

    # exact endpoints can lie on cell edges
    if not (_grid_segment_free(grid, s, world[1])
            and _grid_segment_free(grid, world[-2], g)):
        return None

    if penalty is not None:
        # split at the largest bend until each shortcut keeps occupancy and the soft cost
        pending, kept = [(0, len(world) - 1)], {0, len(world) - 1}
        while pending:
            if cancelled is not None and cancelled():
                return None
            lo, hi = pending.pop()
            if hi <= lo + 1:
                continue
            a, b = world[lo], world[hi]
            if (_grid_segment_free(grid, a, b)
                    and segment_cost(a, b) <= 1.01 * (costs[hi] - costs[lo]) + 1e-9):
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            mid = max(range(lo + 1, hi), key=lambda k:
                      abs(dx * (world[k][1] - a[1]) - dy * (world[k][0] - a[0])))
            kept.add(mid)
            pending.extend(((lo, mid), (mid, hi)))
        result = []
        for k in sorted(kept):
            p = world[k]
            if result and _dist(result[-1], p) < 1e-10:
                result[-1] = p
                continue
            while len(result) >= 2:
                a, b = result[-2:]
                cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                if abs(cross) > 1e-10 or not _grid_segment_free(grid, a, p):
                    break
                result.pop()  # collinear: the same segment
            result.append(p)
        return result

    # line-of-sight shortcut: keep a waypoint only where the straight segment would be blocked
    smoothed = [world[0]]
    anchor = 0
    for k in range(1, len(world)):
        if cancelled is not None and cancelled():
            return None
        if not _grid_segment_free(grid, world[anchor], world[k]):
            smoothed.append(world[k - 1])
            anchor = k - 1
    smoothed.append(world[-1])
    if all(_grid_segment_free(grid, a, b) for a, b in itertools.pairwise(smoothed)):
        return smoothed
    return world


def _grid_segment_free(grid: GridCollisionChecker, a: Point, b: Point) -> bool:
    """Whether every cell the segment touches, edges and corner tangencies included, is free
    and on the grid."""
    if isinstance(grid, ExactGridCollisionChecker):
        return grid.check_motion(a, b)["safe"]
    eps = 1e-10
    x, y, bx, by = [(float(p[axis]) - origin) / grid.cell + 0.5
                    for p in (a, b) for axis, origin in ((0, grid.xmin), (1, grid.ymin))]
    if not all(math.isfinite(v) for v in (x, y, bx, by)):
        return False
    dx, dy = bx - x, by - y
    cells, nx, ny = grid.grid, grid.nx, grid.ny
    first, last = math.floor(min(y, by) - eps), math.floor(max(y, by) + eps)
    if first < 0 or last >= ny:
        return False
    # intersect the segment with each touched row's closed slab, widened by eps
    for j in range(first, last + 1):
        lo, hi = 0., 1.
        if dy:
            enter, leave = (j - eps - y) / dy, (j + 1 + eps - y) / dy
            lo, hi = max(0., min(enter, leave)), min(1., max(enter, leave))
        ax, ex = x + lo * dx, x + hi * dx
        left, right = math.floor(min(ax, ex) - eps), math.floor(max(ax, ex) + eps)
        if left < 0 or right >= nx:
            return False
        for i in range(left, right + 1):
            if not cells[i][j]:
                return False
    return True


def plan_path(start: Point, goal: Point, checker, *, cancelled=None,
              **kwargs) -> list[Point] | None:
    """A collision-free path: grid A* for a :class:`GridCollisionChecker`, RRT-Connect when A*
    finds nothing or for other checkers; takes ``plan_rrt``'s keyword arguments."""
    if cancelled is not None and cancelled():
        return None
    if isinstance(checker, GridCollisionChecker):
        path = plan_grid_astar(start, goal, checker,
                               robot_radius=float(kwargs.get("robot_radius", 0.0)),
                               cancelled=cancelled)
        if cancelled is not None and cancelled():
            return None
        if path is not None:
            return path
    return plan_rrt(start, goal, checker, cancelled=cancelled, **kwargs)


def path_len(path: list[Point]) -> float:
    """Sum of consecutive Euclidean segment lengths along ``path``."""
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))


def inflate(blocked: np.ndarray, cell: float, radius: float) -> np.ndarray:
    """Dilate the blocked grid by a disc of ``radius`` metres."""
    r = math.ceil(radius / cell - 1e-9)
    if r <= 0:
        return blocked.copy()
    nx, ny = blocked.shape
    out = blocked.copy()
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r * r:
                continue
            si0, si1 = max(0, di), min(nx, nx + di)
            di0, di1 = max(0, -di), min(nx, nx - di)
            sj0, sj1 = max(0, dj), min(ny, ny + dj)
            dj0, dj1 = max(0, -dj), min(ny, ny - dj)
            out[di0:di1, dj0:dj1] |= blocked[si0:si1, sj0:sj1]
    return out
