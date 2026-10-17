"""Exploration goals from the discovered map, start-frame metres.

Frontier views are chosen by unknown floor revealed per metre walked; once the map is enclosed,
the Voronoi skeleton's unwalked branches are walked. Surface patches and their viewing angles
score camera coverage."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from homebody.helpers.floor import MAX_GOAL_ADJUST_M, RADIUS_M, TOO_CLOSE_M, Waypoint, path_checker
from homebody.helpers.geometry import wrap
from homebody.motion.exploration.nav import plan_grid_astar
from homebody.skills.navigate_real.controller import EXPLORATION_CLEARANCE_MARGIN_M, _resample

from .livemap import RES_M, LiveMap
from .settings import current

EXECUTION_RADIUS_M = 0.25  # footprint radius for tracking and local recovery
PADDED_RADII_M = (RADIUS_M,)  # hard planning radii
VIEW_LEG_MIN_M = 1.2  # view spots nearer than this are penalised
VISITED_PENALTY_M = 4.0  # cost added for spots and edges already visited
FRONTIER_PROGRESS_M2 = 0.5  # known floor gained near an edge that makes a look a success
PATCH_M = 0.10
CAM_HFOV_DEG, CAM_Z_M = 69.0, 1.17
VIEW_RANGE_M = (0.4, 2.5)
INCIDENCE_MAX_DEG = 75.0
N_BINS, SPREAD_BINS = 36, 3
LIP_SHADOW_M = 0.25  # unknown this close to an obstacle is not an opening
OPEN_FRACTION = 0.5  # least walkable share of a pocket's rim for it to be explorable
MIN_UNKNOWN_M2 = 1.0  # smallest unknown pocket worth exploring
FRONTIER_GAIN_CAP_M2 = 6.0  # cap on the unknown area credited to one edge
SLIVER_M = 0.30  # pockets narrower than this everywhere are slivers
OPENING_MIN_M = 0.60  # a free rim at least this long makes a pocket explorable
VIEW_REACH_M = 4.0  # farthest view spot from an open edge
VIEW_MIN_EXTRA_M = 0.3  # nearest view spot, past lidar.self_clear_m
VIEW_AIM_EXTRA_M = 0.6  # preferred view distance, past lidar.self_clear_m
SPUR_M = 0.60  # skeleton branches shorter than this are pruned
CORNER_ROOM_M = 0.90  # branch tips with less room than this are pruned
FRONTIER_TIP_M = 0.60  # seen-network tips this near the frontier are kept
HUB_ROOM_M = 0.90  # least clearance for a junction to be a hub
MIN_BRANCH_M = 0.4
VISITED_FRACTION = 0.8  # share of a branch's known cells passed that marks it walked
FRONTIER_BONUS_M = 3.0  # extra value for a branch running into unknown
LEG_ALONG_M = 4.0  # longest walk along one branch
EDGE_TRIES_PLANNER = 2


def route_refusal(live: LiveMap, start, goal=None, radius_m: float = RADIUS_M, *, cancelled=None):
    """Why a route from `start` (to `goal`, when given) is refused, or None when it is not."""
    if cancelled is not None and cancelled():
        return None
    details = {"radius_m": radius_m, "start_xy": list(map(float, start)),
               "goal_xy": None if goal is None else list(map(float, goal)),
               "outside_map": False}
    if not live.scope_ready:
        return details | {"code": "task_scope_unresolved", "why":
                          "translation awaits an accepted task region"}
    checker = path_checker(live, radius_m)
    refusal = _route_refusal(checker, start, goal, details)
    if refusal is None or live.task_region_points is None:
        return refusal
    unscoped = path_checker(live, radius_m, task_scope=False)
    physical_refusal = _route_refusal(unscoped, start, goal, details)
    reachable = (physical_refusal is None if goal is None else
                 plan_path(live, start, goal, radius_m, task_scope=False,
                           cancelled=cancelled) is not None)
    if cancelled is not None and cancelled():
        return None
    if reachable:
        return details | {"code": "task_boundary", "task_region_revision":
                          live.task_region_revision, "unscoped_route_available": True,
                          "scoped_refusal": refusal["code"], "why":
                          "task boundary excludes this route; the same physical map allows it"}
    return physical_refusal | {"unscoped_route_available": False,
                               "scoped_refusal": refusal["code"]}


def goal_adjustment_diagnostics(live: LiveMap, start, requested, proposed,
                                radius_m: float = RADIUS_M, *, cancelled=None):
    """Why `requested` was adjusted to `proposed` beyond the limit, on a map snapshot."""
    live = live.snapshot()
    if cancelled is not None and cancelled():
        return None
    details = {"radius_m": radius_m, "start_xy": list(map(float, start)),
               "requested_goal_xy": list(map(float, requested)),
               "proposed_goal_xy": list(map(float, proposed)),
               "goal_adjustment_m": math.dist(requested, proposed),
               "map_version": live.version,
               "task_region_revision": live.task_region_revision}
    route, _ = plan_padded(live, start, requested, radius_m, cancelled=cancelled)
    if route is not None:
        return details | {"code": "network_selection", "scoped_route_available": True,
                          "unscoped_route_available": True, "why":
                          "the original destination is reachable, but network selection "
                          "could not preserve it within the adjustment limit"}
    refusal = route_refusal(live, start, requested, radius_m, cancelled=cancelled)
    if cancelled is not None and cancelled():
        return None
    why = refusal["why"].replace("accepted destination", "requested destination").replace(
        "accepted goal", "requested destination")
    physical = refusal.get("unscoped_route_available")
    if physical is None and live.scope_ready:
        physical = False  # without a task region the failed plan was unscoped
    return refusal | details | {"scoped_route_available": False,
                                 "unscoped_route_available": physical, "why": why}


def _route_refusal(checker, start, goal, details):
    for name, point in (("start", start), ("goal", goal)):
        if point is not None and not checker.is_safe(*point):
            outside = not checker.contains(*point)
            subject = "actual starting footprint" if name == "start" else "accepted destination"
            why = (f"{subject} is outside the map" if outside else
                   f"{subject} is blocked at minimum {details['radius_m']:.2f} m clearance")
            return details | {"code": f"{name}_blocked", "outside_map": outside, "why": why}
    if goal is None:
        return None
    return details | {"code": "disconnected", "why":
                      f"start and accepted goal are disconnected at minimum "
                      f"{details['radius_m']:.2f} m clearance"}


def plan_path(live: LiveMap, start, goal, radius_m: float = RADIUS_M, *, task_scope=True,
              cancelled=None):
    """The drive's A* route from `start` to `goal`, or None when either end is blocked."""
    if cancelled is not None and cancelled():
        return None
    checker = (path_checker(live, radius_m) if task_scope else
               path_checker(live, radius_m, task_scope=False))
    if not checker.is_safe(*start) or not checker.is_safe(*goal):
        return None
    path = plan_grid_astar(tuple(map(float, start)), tuple(map(float, goal)), checker,
                          clearance_margin_m=EXPLORATION_CLEARANCE_MARGIN_M,
                          cancelled=cancelled)
    return None if path is None else [(float(p[0]), float(p[1])) for p in path]


def route_points(path) -> list[tuple[float, float]]:
    """The points the drive walks for PATH, start and goal included, in the start frame."""
    if not path:
        return []
    inner = [tuple(map(float, q)) for q in _resample(path, current().drive.waypoint_spacing_m)]
    return [tuple(map(float, path[0])), *inner, tuple(map(float, path[-1]))]


def plan_padded(live: LiveMap, start, goal, radius_m: float = RADIUS_M, *, cancelled=None):
    """(route, radius) at RADIUS_M, or (None, None)."""
    if cancelled is not None and cancelled():
        return None, None
    path = plan_path(live, start, goal, radius_m, cancelled=cancelled)
    return (path, radius_m) if path is not None else (None, None)


def padded_spot(live: LiveMap, xy, reach, search_m: float = 0.30):
    """Within search_m of xy, the reachable cell farthest from every lidar point."""
    clear = live.clearance_m()
    u0, v0 = live.uv(*xy)
    r = round(search_m / RES_M)
    v1, v2 = max(v0 - r, 0), min(v0 + r + 1, live.H)
    u1, u2 = max(u0 - r, 0), min(u0 + r + 1, live.W)
    if v1 >= v2 or u1 >= u2:
        return xy
    window = np.where(reach[v1:v2, u1:u2], clear[v1:v2, u1:u2], -1.0)
    j = np.unravel_index(int(np.argmax(window)), window.shape)
    if window[j] <= 0:
        return xy
    return tuple(float(c) for c in live.xy(u1 + j[1], v1 + j[0]))


def nearest_reachable_floor(planner, xy, x, y, *, reach, cancelled=None):
    """The nearest reachable known floor to a refused click within MAX_GOAL_ADJUST_M, and its
    route from (x, y); None when there is none."""
    live = planner.live
    if (cancelled is not None and cancelled()) or not live.task_region_contains(*xy):
        return None
    u, v = live.uv(*xy)
    radius = math.ceil(MAX_GOAL_ADJUST_M / RES_M) + 1
    u0, u1 = max(0, u - radius), min(live.W, u + radius + 1)
    v0, v1 = max(0, v - radius), min(live.H, v + radius + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    candidates = reach[v0:v1, u0:u1] & live.known_free()[v0:v1, u0:u1]
    vv, uu = np.nonzero(candidates)
    xs, ys = live.xy(uu + u0, vv + v0)
    adjustment = np.hypot(xs - xy[0], ys - xy[1])
    eligible = ((adjustment <= MAX_GOAL_ADJUST_M)
                & (np.hypot(xs - x, ys - y) >= TOO_CLOSE_M))
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return None
    j = indices[np.argmin(adjustment[indices])]
    goal = (float(xs[j]), float(ys[j]))
    route, _ = plan_padded(live, (x, y), goal, planner.radius_m, cancelled=cancelled)
    if route is None or (cancelled is not None and cancelled()):
        return None
    return goal, route


class Planner:
    def __init__(self, live: LiveMap):
        self.live = live
        self.radius_m = RADIUS_M
        self.edge_tries: dict = {}
        self.edges: list = []
        self.hubs: list = []
        self.open_xy = np.zeros((0, 2), np.float32)
        self.last_leg: dict = {}
        self.look_trees: dict[str, cKDTree] = {}
        self.views = None
        self.blacklist: list[tuple[float, float]] = []
        self.done: set[int] = set()
        self.seen_skeleton = None  # cached seen_network
        self._seen_key = None

    def publish_growth(self, growth, visited: np.ndarray) -> np.ndarray:
        """Publish a map expansion with its view history; returns VISITED padded to match."""
        growth.check_source(self.live)
        expanded_visited = growth.pad_layer(visited)
        factor = round(PATCH_M / RES_M)
        expanded_views = (np.zeros((growth.candidate.H // factor,
                                   growth.candidate.W // factor), np.uint64)
                          if self.views is None else
                          growth.pad_layer(self.views, cell_factor=factor))
        empty_frontier = np.zeros((0, 2), np.float32)
        empty_edges, empty_hubs = [], []
        self.live.publish_growth(growth)
        self.views = expanded_views
        self.edges, self.hubs = empty_edges, empty_hubs
        self.open_xy = empty_frontier
        self.seen_skeleton, self._seen_key = None, None
        if hasattr(self, "skeleton"):
            del self.skeleton
        return expanded_visited

    # -- stage 1 -----------------------------------------------------------------------------
    def explorable_unknown(self) -> np.ndarray:
        """Unknown floor a walk can reveal: pockets that open onto free floor, excluding object
        interiors, slivers and the shadow at obstacle feet."""
        m = self.live
        obst = m.avoid()
        unk = m.unknown() & m.task_inside & ~m.interior()
        unk &= ndimage.distance_transform_edt(~obst) * RES_M > LIP_SHADOW_M
        labels, n = ndimage.label(unk, structure=np.ones((3, 3)))
        if n == 0:
            return unk
        ring = ndimage.maximum_filter(labels, size=3) * (labels == 0)
        touch_free = np.bincount(ring[m.known_free() & m.center_allowed], minlength=n + 1)
        # the boundary closes a pocket like a wall
        touch_closed = np.bincount(ring[obst | ~m.center_allowed], minlength=n + 1)
        area = np.bincount(labels.ravel(), minlength=n + 1) * RES_M * RES_M
        width = ndimage.maximum(ndimage.distance_transform_edt(unk), labels, index=np.arange(n + 1))
        width = np.nan_to_num(np.asarray(width, float)) * 2 * RES_M
        opening = (touch_free >= OPEN_FRACTION * np.maximum(touch_free + touch_closed, 1)) | (
            touch_free * RES_M >= OPENING_MIN_M
        )
        keep = opening & (area >= MIN_UNKNOWN_M2) & (width >= SLIVER_M)
        keep[0] = False
        return keep[labels]

    def enclosure(self) -> dict:
        """Whether the range is enclosed: no known floor borders explorable unknown. `open_m` is
        the length of edge still open."""
        m = self.live
        unk = self.explorable_unknown()
        front = m.known_free() & ndimage.binary_dilation(
            unk, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        )
        looked = np.zeros(front.shape, bool)
        if self.blacklist:  # pockets near blacklisted edges count as looked at
            labels, _ = ndimage.label(unk, structure=np.ones((3, 3)))
            uv_ = np.argwhere(labels > 0)
            ux_, uy_ = m.xy(uv_[:, 1], uv_[:, 0])
            near = np.zeros(len(uv_), bool)
            for bx, by in self.blacklist:
                near |= np.hypot(ux_ - bx, uy_ - by) < 1.0
            pockets = np.unique(labels[uv_[near, 0], uv_[near, 1]])
            given_up = np.isin(labels, pockets[pockets > 0])
            looked = front & ndimage.binary_dilation(
                given_up, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            )
        open_cells = int((front & ~looked).sum())
        ov, ou = np.nonzero(front & ~looked)
        ox, oy = m.xy(ou[::2], ov[::2])
        self.open_xy = np.c_[ox, oy].astype(np.float32)  # the open edge, for the console
        return {
            "enclosed": open_cells == 0,
            "open_m": round(open_cells * RES_M, 2),
            "looked_unresolved_m": round(float(looked.sum()) * RES_M, 2),
            "unknown_m2": round(float(unk.sum()) * RES_M * RES_M, 2),
        }

    def mapped_fraction(self) -> float:
        """The known share of the range against what could still be explored."""
        m = self.live
        known = (m.known_free() | m.avoid()) & m.center_allowed
        todo = self.explorable_unknown() & m.center_allowed
        total = int(known.sum()) + int(todo.sum())
        return 0.0 if total == 0 else float(known.sum()) / total

    def walked_fraction(self, visited: np.ndarray, skel: np.ndarray | None = None) -> float:
        """The share of the skeleton's known cells in VISITED; `skel` replaces the Voronoi
        skeleton."""
        if skel is None:
            if getattr(self, "skeleton", None) is None:
                self.voronoi()
            skel = self.skeleton
        cells = skel & self.live.known_free()
        n = int(cells.sum())
        return 0.0 if n == 0 else float((cells & visited).sum()) / n

    def frontier_clusters(self):
        m = self.live
        unk = self.explorable_unknown()
        front = m.known_free() & ndimage.binary_dilation(
            unk, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        )
        labels, n = ndimage.label(front, structure=np.ones((3, 3)))
        out = []
        for k in range(1, n + 1):
            cells = np.argwhere(labels == k)
            if len(cells) >= 6:
                out.append(tuple(float(c) for c in m.xy(cells[:, 1].mean(), cells[:, 0].mean())))
        return out, unk

    def next_frontier(
        self, x: float, y: float, visited: np.ndarray | None = None
    ) -> tuple[Waypoint, tuple[float, float], float] | None:
        """The best reachable spot within VIEW_REACH_M of an open edge with a clear line of sight,
        scored by unknown revealed per metre walked; (waypoint, edge centre, gain) or None."""
        m = self.live
        clear_m = current().lidar.self_clear_m
        view_min_m, view_aim_m = clear_m + VIEW_MIN_EXTRA_M, clear_m + VIEW_AIM_EXTRA_M
        clusters, unk = self.frontier_clusters()
        if not clusters:
            return None
        geo = m.geodesic(x, y, self.radius_m)
        vv, uu = np.nonzero(np.isfinite(geo) & (geo >= 0.5))
        if not len(vv):
            return None
        rx, ry = m.xy(uu, vv)
        room_all = m.clearance_m()
        avoid = m.avoid()
        uv = np.argwhere(unk)
        ux, uy = m.xy(uv[:, 1], uv[:, 0])
        rng = np.random.default_rng(0)
        found = []
        for cx, cy in clusters:
            if any(math.hypot(cx - bx, cy - by) < 0.6 for bx, by in self.blacklist):
                continue
            gain = float((np.hypot(ux - cx, uy - cy) <= 2.5).sum() * RES_M * RES_M)
            if gain < MIN_UNKNOWN_M2:
                continue
            d = np.hypot(rx - cx, ry - cy)
            idx = np.nonzero((d <= VIEW_REACH_M) & (d >= view_min_m))[0]
            if not len(idx):
                continue
            if len(idx) > 600:
                idx = rng.choice(idx, 600, replace=False)
            ts = np.linspace(0.0, 1.0, 40)[None, :]
            qx = rx[idx, None] + (cx - rx[idx, None]) * ts
            qy = ry[idx, None] + (cy - ry[idx, None]) * ts
            qu, qv = m.uv(qx, qy)
            qu, qv = np.clip(qu, 0, m.W - 1), np.clip(qv, 0, m.H - 1)
            short = (d[idx, None] * (1.0 - ts)) > 0.30  # ignore the last 30 cm, the edge itself
            sight = ~(avoid[qv, qu] & short).any(axis=1)
            idx = idx[sight]
            if not len(idx):
                continue
            room = room_all[vv[idx], uu[idx]]
            cost = geo[vv[idx], uu[idx]] + 0.5 * np.abs(d[idx] - view_aim_m)
            cost -= 2.0 * np.minimum(room, 1.0)
            # penalise short legs and visited spots and edges
            cost += 1.5 * np.maximum(0.0, VIEW_LEG_MIN_M - geo[vv[idx], uu[idx]])
            if visited is not None and visited.shape == m.floor.shape:
                cost += VISITED_PENALTY_M * visited[vv[idx], uu[idx]]
                cu, cv = m.uv(cx, cy)
                if 0 <= cu < m.W and 0 <= cv < m.H and visited[cv, cu]:
                    cost += VISITED_PENALTY_M
            i = idx[int(np.argmin(cost))]
            found.append(
                (
                    min(gain, FRONTIER_GAIN_CAP_M2),
                    float(geo[vv[i], uu[i]]),
                    (float(rx[i]), float(ry[i])),
                    (cx, cy),
                    gain,
                )
            )
        if not found:
            return None
        top = max(f[0] for f in found)
        # the nearest edge with at least half the largest gain
        _, _, goal, centre, gain = min((f for f in found if f[0] >= 0.5 * top), key=lambda f: f[1])
        face = math.atan2(centre[1] - goal[1], centre[0] - goal[0])
        return Waypoint("frontier", goal, [face]), centre, gain

    def edge_still_there(self, centre) -> bool:
        return any(
            math.hypot(c[0] - centre[0], c[1] - centre[1]) < 0.5
            for c in self.frontier_clusters()[0]
        )

    def known_near(self, centre, radius_m: float = 2.5) -> float:
        """Known floor (m²) within `radius_m` of a point."""
        m = self.live
        known = m.known_free()
        u, v = m.uv(centre[0], centre[1])
        r = round(radius_m / RES_M)
        v0, v1, u0, u1 = max(v - r, 0), min(v + r + 1, m.H), max(u - r, 0), min(u + r + 1, m.W)
        if v0 >= v1 or u0 >= u1:
            return 0.0
        vv, uu = np.ogrid[v0:v1, u0:u1]
        disk = (vv - v) ** 2 + (uu - u) ** 2 <= r * r
        return float((known[v0:v1, u0:u1] & disk).sum()) * RES_M * RES_M

    # -- stage 2: surfaces and viewpoints ----------------------------------------------------
    def patches(self):
        """Surface patches (10 cm): wall cells bordering known floor, with heights and normals."""
        m = self.live
        f = round(PATCH_M / RES_M)
        H2, W2 = m.H // f, m.W // f
        occ = m.walls()[: H2 * f, : W2 * f].reshape(H2, f, W2, f).any(axis=(1, 3))
        free = m.known_free()[: H2 * f, : W2 * f].reshape(H2, f, W2, f).mean(axis=(1, 3)) > 0.5
        maxz = m.maxz[: H2 * f, : W2 * f].reshape(H2, f, W2, f).max(axis=(1, 3))
        inside = m.task_inside[: H2 * f, : W2 * f].reshape(H2, f, W2, f).any(axis=(1, 3))
        k = np.ones((3, 3))
        pu, pv = np.meshgrid(np.arange(W2), np.arange(H2))
        px = m.xmin + (pu + 0.5) * PATCH_M - RES_M / 2
        py = m.ymax - (pv + 0.5) * PATCH_M + RES_M / 2
        cnt = ndimage.convolve(free.astype(float), k, mode="constant")
        nx = ndimage.convolve(free * px, k, mode="constant") / np.maximum(cnt, 1) - px
        ny = ndimage.convolve(free * py, k, mode="constant") / np.maximum(cnt, 1) - py
        nn = np.hypot(nx, ny) + 1e-9
        skin = occ & (cnt > 0) & inside
        if self.views is None or self.views.shape != skin.shape:
            self.views = np.zeros(skin.shape, np.uint64)
        return skin, px, py, maxz, nx / nn, ny / nn

    def see(self, xy, headings, credit: bool) -> tuple[np.ndarray, np.ndarray]:
        """Patches a camera at xy would see over `headings` (2.5-D line of sight on the map)."""
        skin, px2, py2, maxz2, nx2, ny2 = self.patches()
        pv, pu = np.nonzero(skin)
        px, py = px2[pv, pu], py2[pv, pu]
        pz = np.clip(maxz2[pv, pu] * 0.6, 0.3, 1.2)
        sx, sy = xy
        dx, dy, dz = sx - px, sy - py, CAM_Z_M - pz
        h = np.hypot(dx, dy)
        d3 = np.sqrt(h * h + dz * dz)
        cosi = (nx2[pv, pu] * dx + ny2[pv, pu] * dy) / (d3 + 1e-9)
        ok = (
            (d3 >= VIEW_RANGE_M[0])
            & (d3 <= VIEW_RANGE_M[1])
            & (cosi >= math.cos(math.radians(INCIDENCE_MAX_DEG)))
        )
        idx = np.nonzero(ok)[0]
        m = self.live
        if len(idx):
            start = np.minimum(0.25 / (h[idx] + 1e-9), 0.9)
            ts = start[:, None] + (1 - start[:, None]) * np.linspace(0, 0.95, 20)[None, :]
            qu, qv = m.uv(px[idx, None] + dx[idx, None] * ts, py[idx, None] + dy[idx, None] * ts)
            qu, qv = np.clip(qu, 0, m.W - 1), np.clip(qv, 0, m.H - 1)
            ray_z = pz[idx, None] + dz[idx, None] * ts
            idx = idx[~((m.avoid()[qv, qu]) & (m.maxz[qv, qu] > ray_z + 0.10)).any(1)]
        az = np.arctan2(py[idx] - sy, px[idx] - sx)
        vis = np.zeros(len(idx), bool)
        for hd in headings:
            vis |= np.degrees(np.abs(wrap(az - hd))) <= CAM_HFOV_DEG / 2
        idx = idx[vis]
        bins = (
            np.floor(
                (np.arctan2(sy - py[idx], sx - px[idx]) % (2 * np.pi)) / (2 * np.pi) * N_BINS
            ).astype(np.int64)
            % N_BINS
        )
        if credit and len(idx):
            np.bitwise_or.at(
                self.views, (pv[idx], pu[idx]), np.left_shift(np.uint64(1), bins.astype(np.uint64))
            )
        return np.stack([pv[idx], pu[idx]], 1), bins

    @staticmethod
    def spread(mask: int) -> int:
        bits = [i for i in range(N_BINS) if mask >> i & 1]
        best = 0
        for s in bits:
            picked = [s]
            for k in range(1, N_BINS):
                i = (s + k) % N_BINS
                if (
                    mask >> i & 1
                    and (i - picked[-1]) % N_BINS >= SPREAD_BINS
                    and (s - i) % N_BINS >= SPREAD_BINS
                ):
                    picked.append(i)
            best = max(best, len(picked))
        return best

    def coverage(self) -> dict:
        skin = self.patches()[0]
        cache: dict[int, int] = {}
        counts = np.zeros(skin.shape, np.int8)
        for pv, pu in np.argwhere(skin & (self.views > 0)):
            key = int(self.views[pv, pu])
            if key not in cache:
                cache[key] = self.spread(key)
            counts[pv, pu] = cache[key]
        n = max(int(skin.sum()), 1)
        return {f">={k}": float(((counts >= k) & skin).sum()) / n for k in (1, 2, 3)} | {
            "patches": int(skin.sum())
        }

    def gain(self, wp: Waypoint) -> float:
        cells, bins = self.see(wp.xy, wp.headings, credit=False)
        if not len(cells):
            return 0.0
        mask = self.views[cells[:, 0], cells[:, 1]].astype(np.int64)
        near = np.zeros(len(cells), np.int64)
        for k in range(-(SPREAD_BINS - 1), SPREAD_BINS):
            near |= np.left_shift(1, (bins + k) % N_BINS)
        return float(((mask & near) == 0).sum())

    # -- continuous Voronoi exploration -------------------------------------------------------
    def voronoi(self) -> list[dict]:
        """The Voronoi skeleton of known floor plus explorable unknown; returns its branches
        (ordered cells, length, unknown share) and sets the skeleton and hubs."""
        m = self.live
        unk = self.explorable_unknown()
        skel, clear = _skeleton((m.known_free() | unk) & m.center_allowed, unk, self.radius_m)
        k = np.ones((3, 3))
        degree = ndimage.convolve(skel.astype(int), k, mode="constant") - skel
        junction = skel & (degree >= 3)
        branches = skel & ~ndimage.binary_dilation(junction, structure=k, iterations=2)
        labels, n = ndimage.label(branches, structure=k)
        edges = []
        for b in range(1, n + 1):
            chain = _order_chain(np.argwhere(labels == b))
            if len(chain) * RES_M < MIN_BRANCH_M:
                continue
            edges.append(
                {
                    "cells": chain,
                    "length_m": len(chain) * RES_M,
                    "unknown": float(unk[chain[:, 0], chain[:, 1]].mean()),
                }
            )
        hubs = []
        jl, jn = ndimage.label(junction, structure=k)
        for j in range(1, jn + 1):
            cells = np.argwhere(jl == j)
            best = cells[np.argmax(clear[cells[:, 0], cells[:, 1]])]
            if clear[best[0], best[1]] >= HUB_ROOM_M:
                hubs.append(tuple(float(c) for c in m.xy(best[1], best[0])))
        self.skeleton, self.edges, self.hubs = skel, edges, hubs
        return edges

    def seen_network(self) -> np.ndarray:
        """The skeleton of seen floor only, tips kept within FRONTIER_TIP_M of the frontier;
        cached per map version."""
        m = self.live
        key = (id(m), m.version, self.radius_m)
        if self._seen_key != key or self.seen_skeleton is None:
            near_frontier = ndimage.binary_dilation(
                self.explorable_unknown(), iterations=round(FRONTIER_TIP_M / RES_M)
            )
            skel, _ = _skeleton(m.known_free() & m.center_allowed, near_frontier, self.radius_m)
            self.seen_skeleton, self._seen_key = skel & m.known_free(), key
        return self.seen_skeleton

    def next_leg(self, x: float, y: float, visited: np.ndarray):
        """The next walk onto and along the best unwalked branch: (route, branch, why) in the
        start frame, or None when none is left."""
        m = self.live
        edges = self.voronoi()
        geo = m.geodesic(x, y, self.radius_m)
        known = m.known_free()
        best = None
        for e in edges:
            cells = e["cells"]
            key = tuple(np.round(m.xy(*cells[len(cells) // 2][::-1]), 1))
            if self.edge_tries.get(key, 0) >= EDGE_TRIES_PLANNER:
                continue
            walkable = known[cells[:, 0], cells[:, 1]] & np.isfinite(geo[cells[:, 0], cells[:, 1]])
            if not walkable.any():
                continue
            unwalked = walkable & ~visited[cells[:, 0], cells[:, 1]]
            walked_share = 1.0 - unwalked.sum() / max(walkable.sum(), 1)
            frontier = e["unknown"] > 0.0
            if walked_share >= VISITED_FRACTION and not frontier:
                continue
            idx = np.nonzero(unwalked if unwalked.any() else walkable)[0]
            cost = geo[cells[idx, 0], cells[idx, 1]]
            entry = int(idx[int(np.argmin(cost))])
            value = unwalked.sum() * RES_M + (FRONTIER_BONUS_M if frontier else 0.0)
            score = value / (float(cost.min()) + 1.0)
            if best is None or score > best[0]:
                best = (score, e, entry, key, frontier)
        if best is None:
            return None
        _, e, entry, key, frontier = best
        cells = e["cells"]
        # from the entry toward the end with more unwalked or unknown cells
        fwd = cells[entry:]
        back = cells[: entry + 1][::-1]

        def todo(seq):
            ok = known[seq[:, 0], seq[:, 1]]
            return int((ok & ~visited[seq[:, 0], seq[:, 1]]).sum()) + (5 if not ok.all() else 0)

        along = fwd if todo(fwd) >= todo(back) else back
        along = along[known[along[:, 0], along[:, 1]].cumprod().astype(bool)]  # stop at unknown
        steps = round(LEG_ALONG_M / RES_M)
        along = along[:steps]
        ex, ey = m.xy(cells[entry][1], cells[entry][0])
        approach, radius = plan_padded(m, (x, y), (float(ex), float(ey)), self.radius_m)
        if approach is None:
            self.edge_tries[key] = self.edge_tries.get(key, 0) + 1
            return (
                self.next_leg(x, y, visited) if self.edge_tries[key] < EDGE_TRIES_PLANNER else None
            )
        ax, ay = m.xy(along[:, 1], along[:, 0])
        route = [*approach, *[(float(a), float(b)) for a, b in zip(ax[::4], ay[::4])]]
        self.last_leg = {"key": key, "frontier": frontier, "radius": radius}
        why = "into unknown" if frontier else "unwalked branch"
        return route, e, why

    def face(self, key: str | None, x: float, y: float, default: float) -> float:
        tree = self.look_trees.get(key) if key else None
        if tree is None:
            return default
        _, j = tree.query([x, y])
        qx, qy = tree.data[j]
        return math.atan2(qy - y, qx - x)


def _skeleton(room: np.ndarray, keep: np.ndarray,
              radius_m: float = RADIUS_M) -> tuple[np.ndarray, np.ndarray]:
    """(skeleton, clearance in metres): the medial axis of `room` with `radius_m` of clearance,
    tips and spurs pruned except tips in `keep`."""
    from skimage.morphology import medial_axis, remove_small_holes, remove_small_objects

    room = ndimage.binary_opening(room, structure=np.ones((5, 5)))
    room = remove_small_holes(room, max_size=int(0.5 / RES_M**2))
    room = remove_small_objects(room, max_size=int(1.0 / RES_M**2))
    clear = ndimage.distance_transform_edt(room) * RES_M
    skel = medial_axis(room) & (clear >= radius_m)
    skel = _prune_tips(skel, clear, keep, CORNER_ROOM_M)
    return _prune_spurs(skel, round(SPUR_M / RES_M)), clear


def _prune_tips(skel: np.ndarray, clear: np.ndarray, keep: np.ndarray, room_m: float) -> np.ndarray:
    """Remove tips with less than `room_m` of clearance cell by cell, except tips in `keep`."""
    k = np.ones((3, 3))
    skel = skel.copy()
    while True:
        degree = ndimage.convolve(skel.astype(int), k, mode="constant") - skel
        tips = skel & (degree <= 1) & (clear < room_m) & ~keep
        if not tips.any():
            return skel
        skel &= ~tips


def _prune_spurs(skel: np.ndarray, length: int) -> np.ndarray:
    """Remove dead-end branches shorter than `length` cells, keeping longer branches whole."""
    k = np.ones((3, 3))
    pruned = skel.copy()
    for _ in range(length):
        degree = ndimage.convolve(pruned.astype(int), k, mode="constant") - pruned
        pruned &= ~(pruned & (degree <= 1))
    grown = pruned.copy()
    for _ in range(length):
        grown |= ndimage.binary_dilation(grown, structure=k) & skel
    return grown


def _order_chain(cells: np.ndarray) -> np.ndarray:
    """A branch's pixels in walking order, starting from one of its ends."""
    if len(cells) <= 2:
        return cells
    tree = cKDTree(cells)
    degree = np.array([len(tree.query_ball_point(c, 1.5)) - 1 for c in cells])
    start = int(np.argmin(degree))
    order, seen = [start], {start}
    for _ in range(len(cells) - 1):
        near = [j for j in tree.query(cells[order[-1]], k=min(9, len(cells)))[1] if j not in seen]
        if not near:
            break
        j = min(near, key=lambda q: np.hypot(*(cells[q] - cells[order[-1]])))
        if np.hypot(*(cells[j] - cells[order[-1]])) > 3.0:
            break
        order.append(j)
        seen.add(j)
    return cells[order]
