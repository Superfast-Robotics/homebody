"""The explorer's occupancy map from lidar returns, in the start frame (x forward of the robot's
heading at Zero, y left, z up from the floor), 5 cm cells.

Each body-band cell keeps an evidence count: a scan that hits it adds one, a beam that sees through
it at its hit heights takes one away, capped at EVIDENCE_CAP. A cell is avoided from AVOID_SCANS
and becomes a wall after WALL_AFTER_S avoided; the blind disk around the robot is routable."""

from __future__ import annotations

import math
import os
import tempfile
import time
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy import ndimage

from homebody.helpers.raster import cell_transform, cell_uv, cell_xy
from homebody.helpers.region import TaskRegionRejected, contains, rasterize
from homebody.motion.exploration.gait import occupancy_grid
from homebody.primitives.localization.lidar import FLOOR_BAND_M

from .settings import current

RES_M = 0.05
OBSTACLE_BAND_M = (0.25, 1.35)  # heights that count as obstacles
FOOTPRINT_M = 0.35  # radius of the footprint marked as known floor
VIEW_BAND_TOP_M = 2.30  # surfaces the cameras should see, above the body band
WALL_AFTER_S = 10.0  # avoided this long before a cell is drawn as wall
SOLID_CLOSE_CELLS = 2   # closing radius for the drawn solids
SOLID_MIN_CELLS = 6     # smallest drawn solid
SOLID_SMOOTH_CELLS = 5  # median window for the drawn height
AVOID_SCANS = 3  # net hit scans that make a cell avoided
EVIDENCE_CAP = 6  # cap on a cell's evidence count

SPECK_CELLS = 3  # largest blob that can be a speck
SPECK_SPAN_M = 0.30  # a speck's height span is below this
FORGET_S = 2.0  # unconfirmed hits are cleared after this long without a hit
PASS_UNDER_M = 0.05  # a beam this far below a cell's lowest hit still sees through it
RECENT_HIT_S = 1.0  # a cell hit twice this recently blocks a move
BOUNDARY_INSET_M = 0.10  # the robot's centre stays this far inside the drawn line
CARVE_RAYS = 1500  # beams sampled per scan for see-through updates
INTERIOR_GAP_M = 0.15  # hits this close together enclose the space between them
INTERIOR_MAX_M2 = 8.0  # largest enclosed unknown treated as an object interior
INTERIOR_RIM_M = 0.30  # free rim shorter than this is no opening into a pocket
# Scan levelling (LiveMap._level)
LEVEL_FLOOR_M = (0.35, 4.0)  # floor candidates: height band, horizontal range
LEVEL_MIN_POINTS = 300
LEVEL_RESIDUAL_M = 0.08  # residual beyond which a return is refitted out
LEVEL_MAX_DEG = 8.0  # steeper fitted tilts leave the scan unchanged


@dataclass(frozen=True)
class Boundary:
    """The boundary polygon, start-frame metres."""

    points: tuple[tuple[float, float], ...]

    def contains(self, xy: np.ndarray) -> np.ndarray:
        return PolygonPath(self.points).contains_points(np.atleast_2d(xy))

    def distance_outside(self, x: float, y: float) -> float:
        if self.contains(np.array([[x, y]]))[0]:
            return 0.0
        e = np.array(self.points + self.points[:1], dtype=float)
        a, b = e[:-1], e[1:]
        t = np.clip(np.sum((np.array([x, y]) - a) * (b - a), 1) / np.sum((b - a) ** 2, 1), 0, 1)
        return float(np.min(np.hypot(*(a + t[:, None] * (b - a) - [x, y]).T)))

SQUARE_START_MARGIN_M = 0.30  # least distance from the start to a range edge


def rect_boundary(length_m: float, width_m: float, start_x_m: float, start_y_m: float) -> Boundary:
    """A rectangular range in the start frame: `length_m` along x, `width_m` along y, the start
    `start_x_m` from its back edge and `start_y_m` from its right edge."""
    lx, ly, sx, sy = float(length_m), float(width_m), float(start_x_m), float(start_y_m)
    m = SQUARE_START_MARGIN_M
    if lx <= 2 * m or ly <= 2 * m or not (m <= sx <= lx - m) or not (m <= sy <= ly - m):
        raise ValueError(
            f"{lx:g} x {ly:g} m range with the start at ({sx:g}, {sy:g}) m from its corner: the "
            f"start must be at least {m:g} m inside every edge"
        )
    return Boundary(((-sx, -sy), (lx - sx, -sy), (lx - sx, ly - sy), (-sx, ly - sy)))


GROWTH_STEP_M = 4.0
MAX_MAP_CELLS = 4_000_000


@dataclass
class MapGrowth:
    """An enlarged grid prepared for publication."""

    source: LiveMap
    candidate: LiveMap
    source_geometry: tuple
    source_revision: int
    offset_vu: tuple[int, int]
    metrics: dict

    def check_source(self, live: LiveMap) -> None:
        if (live is not self.source or live._geometry() != self.source_geometry
                or live.task_region_revision != self.source_revision):
            raise TaskRegionRejected("map extent or task region changed while preparing growth")

    def pad_layer(self, layer: np.ndarray, *, cell_factor: int = 1) -> np.ndarray:
        """Copy current 5 cm cells or aligned coarse patches into unknown-filled space."""
        old_h, old_w = self.source_geometry[:2]
        v, u = self.offset_vu
        if (cell_factor < 1 or v % cell_factor or u % cell_factor
                or layer.shape != (old_h // cell_factor, old_w // cell_factor)):
            raise TaskRegionRejected("map growth history does not match its source grid")
        target = np.zeros((self.candidate.H // cell_factor,
                           self.candidate.W // cell_factor), dtype=layer.dtype)
        v, u = v // cell_factor, u // cell_factor
        target[v:v + layer.shape[0], u:u + layer.shape[1]] = layer
        return target


def default_boundary(extent_m: float | None = None) -> Boundary:
    """The square boundary of half-side `extent_m` (default map.extent_m) around the start."""
    e = float(current().map.extent_m if extent_m is None else extent_m)
    return Boundary(((-e, -e), (e, -e), (e, e), (-e, e)))


def _disk(radius_cells: int) -> np.ndarray:
    r = int(radius_cells)
    vv, uu = np.ogrid[-r : r + 1, -r : r + 1]
    return uu**2 + vv**2 <= r * r


class LiveMap:
    def __init__(self, boundary: Boundary, margin_m: float = 5.0):
        p = np.array(boundary.points, dtype=float)
        self.boundary = boundary
        self.margin_m = float(margin_m)
        self.xmin = float(p[:, 0].min() - margin_m)
        self.ymax = float(p[:, 1].max() + margin_m)
        self.W = math.ceil((p[:, 0].max() + margin_m - self.xmin) / RES_M)
        self.H = math.ceil((self.ymax - (p[:, 1].min() - margin_m)) / RES_M)
        self._allocate_grid()
        self.task_region_points = None
        self.task_region_revision = 0
        self.task_region_required = False
        self._task_region_mask = None
        self.now = 0.0
        self.version = 0  # bumped on every change; keys the layer cache
        self._cache: dict = {}

    def _allocate_grid(self) -> None:
        """Allocate empty evidence layers and the boundary masks."""
        shape = (self.H, self.W)
        self.floor = np.zeros(shape, np.int32)
        self.swept = np.zeros(shape, bool)  # stood on: known free
        self.assumed = np.zeros(shape, bool)  # inside the lidar's blind disk: routable only
        self.hits = np.zeros(shape, np.int32)  # obstacle evidence count
        self.first_hit = np.full(shape, np.nan)
        self.last_hit = np.full(shape, -np.inf)
        self.maxz = np.zeros(shape, np.float32)
        self.hit_zmin = np.full(shape, np.inf, np.float32)  # lowest and highest hit heights
        self.hit_zmax = np.full(shape, -np.inf, np.float32)
        self.ever_hit = np.zeros(shape, bool)
        vv, uu = np.mgrid[0 : self.H, 0 : self.W]
        centres = np.stack(self.xy(uu, vv), -1).reshape(-1, 2)
        self.inside = self.boundary.contains(centres).reshape(shape)
        inset = ndimage.distance_transform_edt(self.inside) * RES_M
        self.inside_inset = inset > BOUNDARY_INSET_M

    def _geometry(self) -> tuple:
        return self.H, self.W, self.xmin, self.ymax, self.boundary

    def prepare_growth(self, points) -> MapGrowth | None:
        """An enlarged copy of the map covering POINTS, or None when it already does."""
        started = time.perf_counter()
        proposed = np.asarray(points, dtype=float)
        if (proposed.ndim != 2 or proposed.shape[1] != 2 or not len(proposed)
                or not np.isfinite(proposed).all()):
            raise TaskRegionRejected("task region vertices must be finite x/y pairs")
        candidate = copy(self)
        source_geometry, source_revision = candidate._geometry(), candidate.task_region_revision
        old_h, old_w, old_xmin, old_ymax, _ = source_geometry
        old = np.asarray(candidate.boundary.points)
        lo, hi = old.min(axis=0), old.max(axis=0)
        # grow so the whole polygon lies inside the inset line
        padding = BOUNDARY_INSET_M + RES_M
        lower = tuple(math.ceil(max(float(a - b + padding), 0.) / GROWTH_STEP_M - 1e-12)
                      for a, b in zip(lo, proposed.min(axis=0), strict=True))
        upper = tuple(math.ceil(max(float(a - b + padding), 0.) / GROWTH_STEP_M - 1e-12)
                      for a, b in zip(proposed.max(axis=0), hi, strict=True))
        if not (any(lower) or any(upper)):
            return None
        step = round(GROWTH_STEP_M / RES_M)
        left, bottom = (n * step for n in lower)
        right, top = (n * step for n in upper)
        height, width = old_h + bottom + top, old_w + left + right
        if height * width > MAX_MAP_CELLS:
            raise TaskRegionRejected(f"map capacity: proposed extent needs {height * width:,} cells; "
                             f"limit is {MAX_MAP_CELLS:,}")
        lo, hi = lo - np.array(lower) * GROWTH_STEP_M, hi + np.array(upper) * GROWTH_STEP_M
        candidate.boundary = Boundary(((lo[0], lo[1]), (hi[0], lo[1]),
                                       (hi[0], hi[1]), (lo[0], hi[1])))
        candidate.H, candidate.W = height, width
        candidate.xmin, candidate.ymax = old_xmin - left * RES_M, old_ymax + top * RES_M
        candidate._allocate_grid()
        candidate._cache = {}
        growth = MapGrowth(self, candidate, source_geometry, source_revision,
                           (top, left), {})
        if candidate._task_region_mask is not None:
            candidate._task_region_mask = growth.pad_layer(candidate._task_region_mask)
            candidate._task_region_mask.flags.writeable = False
        growth.metrics.update(old_shape=[old_h, old_w], shape=[height, width],
                              cells=height * width, offset_vu=[top, left],
                              allocated_bytes=sum(a.nbytes for a in vars(candidate).values()
                                                  if isinstance(a, np.ndarray)),
                              prepare_s=time.perf_counter() - started)
        return growth

    def publish_growth(self, growth: MapGrowth) -> None:
        """Copy the current evidence into the grown grid and adopt it in place."""
        growth.check_source(self)
        started = time.perf_counter()
        target = growth.candidate
        v, u = growth.offset_vu
        for name in ("floor", "swept", "assumed", "hits", "first_hit", "last_hit",
                     "maxz", "hit_zmin", "hit_zmax", "ever_hit"):
            getattr(target, name)[v:v + self.H, u:u + self.W] = getattr(self, name)
        target.now, target.version = self.now, self.version + 1
        self.__dict__.update(vars(target))
        growth.metrics["publish_s"] = time.perf_counter() - started

    def publish_task_region(self, candidate: LiveMap, expected_revision: int) -> None:
        """Adopt the task region of a validated candidate, keeping this map's evidence."""
        if (isinstance(expected_revision, bool) or not isinstance(expected_revision, int)
                or self.task_region_revision != expected_revision
                or candidate._geometry() != self._geometry()
                or candidate.task_region_revision != expected_revision + 1):
            raise TaskRegionRejected("map extent or task region changed while preparing permission")
        self.task_region_points = candidate.task_region_points
        self._task_region_mask = candidate._task_region_mask
        self.task_region_revision = candidate.task_region_revision
        self.version += 1

    @property
    def scope_ready(self) -> bool:
        return not self.task_region_required or self.task_region_points is not None

    @property
    def task_inside(self) -> np.ndarray:
        return (self.inside if self._task_region_mask is None else
                self.inside & self._task_region_mask)

    @property
    def center_allowed(self) -> np.ndarray:
        return (self.inside_inset if self._task_region_mask is None else
                self.inside_inset & self._task_region_mask)

    def task_region_contains(self, x: float, y: float) -> bool:
        if self.task_region_points is None:
            return not self.task_region_required
        u, v = self.uv(x, y)
        return bool(self._in_grid(u, v) and self.center_allowed[v, u]
                    and contains(self.task_region_points, [(x, y)])[0])

    def set_task_region(self, points, current_xy, expected_revision) -> str:
        """Validate and set the task region; returns the refusal reason or ''."""
        if (isinstance(expected_revision, bool) or not isinstance(expected_revision, int)
                or expected_revision != self.task_region_revision):
            return "task region revision changed; use the current revision"
        try:
            pose = np.asarray(current_xy, float)
            if pose.shape != (2,) or not np.isfinite(pose).all():
                return "task region needs a finite current base position"
            vertices, mask = rasterize(points, xmin=self.xmin, ymax=self.ymax,
                                       shape=self.inside.shape, resolution=RES_M)
        except (TypeError, ValueError) as exc:
            return str(exc)
        allowed = mask & self.inside_inset
        if ndimage.label(allowed)[1] != 1:
            return "task region permission is empty or disconnected within the run range"
        u, v = self.uv(*pose)
        if (not self._in_grid(u, v) or not allowed[v, u]
                or not contains(vertices, [pose])[0]):
            return "task region excludes the current base position"
        mask.flags.writeable = False
        self.task_region_points, self._task_region_mask = vertices, mask
        self.task_region_revision += 1
        self.version += 1
        return ""

    def clear_task_region(self) -> None:
        """Clear the task region, keeping the map evidence."""
        self.task_region_points, self._task_region_mask = None, None
        self.task_region_revision += 1
        self.version += 1

    def snapshot(self) -> LiveMap:
        """A read-only copy of the map's layers."""
        frozen = copy(self)
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                layer = value.copy()
                layer.flags.writeable = False
                setattr(frozen, name, layer)
        frozen._cache = {}
        return frozen

    # -- frames ------------------------------------------------------------------------------
    def uv(self, x, y):
        return cell_uv(self.xmin, self.ymax, RES_M, x, y)

    def xy(self, u, v):
        return cell_xy(self.xmin, self.ymax, RES_M, u, v)

    def T_map_px(self) -> np.ndarray:
        return cell_transform(self.xmin, self.ymax, RES_M)

    def _in_grid(self, u, v):
        return (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)

    # -- ingest ------------------------------------------------------------------------------
    def add_scan(
        self,
        points: np.ndarray,
        origin: np.ndarray,
        t: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        """One lidar scan, start frame: points (N,3) and the sensor origin (3,)."""
        self.now = float(t)
        self.version += 1
        if not len(points):
            return
        points = self._level(np.asarray(points, float), np.asarray(origin, float))
        u, v = self.uv(points[:, 0], points[:, 1])
        ok = self._in_grid(u, v)
        pts, u, v = points[ok], u[ok], v[ok]
        z = pts[:, 2]
        floor = np.abs(z) < FLOOR_BAND_M
        np.add.at(self.floor, (v[floor], u[floor]), 1)
        body = (z >= OBSTACLE_BAND_M[0]) & (z <= OBSTACLE_BAND_M[1])
        view = (z >= OBSTACLE_BAND_M[0]) & (z <= VIEW_BAND_TOP_M)
        np.maximum.at(self.maxz, (v[view], u[view]), z[view].astype(np.float32))
        hv, hu = v[body], u[body]
        cells = np.unique(hv * self.W + hu)
        cv, cu = cells // self.W, cells % self.W
        self.hits[cv, cu] = np.minimum(self.hits[cv, cu] + 1, EVIDENCE_CAP)  # once per scan
        np.minimum.at(self.hit_zmin, (hv, hu), z[body].astype(np.float32))
        np.maximum.at(self.hit_zmax, (hv, hu), z[body].astype(np.float32))
        self.ever_hit[hv, hu] = True
        fresh = np.isnan(self.first_hit[cv, cu])
        self.first_hit[cv[fresh], cu[fresh]] = t
        self.last_hit[cv, cu] = t
        self._forget(t)
        self._carve(pts, np.asarray(origin, float), rng, cells)

    @staticmethod
    def _level(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
        """The scan levelled by a plane fit through its near floor returns; heights keep their
        offset, and the scan is unchanged without enough floor."""
        dx, dy, z = points[:, 0] - origin[0], points[:, 1] - origin[1], points[:, 2]
        near = (np.abs(z) < LEVEL_FLOOR_M[0]) & (dx * dx + dy * dy < LEVEL_FLOOR_M[1] ** 2)
        if near.sum() < LEVEL_MIN_POINTS:
            return points
        A = np.c_[dx[near], dy[near], np.ones(int(near.sum()))]
        coef = np.linalg.lstsq(A, z[near], rcond=None)[0]
        floor = np.abs(A @ coef - z[near]) < LEVEL_RESIDUAL_M
        if floor.sum() < LEVEL_MIN_POINTS:
            return points
        a, b, _ = np.linalg.lstsq(A[floor], z[near][floor], rcond=None)[0]
        if math.hypot(a, b) > math.tan(math.radians(LEVEL_MAX_DEG)):
            return points
        out = points.copy()
        out[:, 2] = z - (a * dx + b * dy)
        return out

    def _forget(self, t: float) -> None:
        """Clear cells with fewer than AVOID_SCANS hits and no hit for FORGET_S."""
        noise = (self.hits > 0) & (self.hits < AVOID_SCANS) & (t - self.last_hit > FORGET_S)
        if noise.any():
            self._clear(noise)
            self.ever_hit[noise] = False

    def _clear(self, mask: np.ndarray) -> None:
        self.hits[mask] = 0
        self.first_hit[mask] = np.nan
        self.hit_zmin[mask], self.hit_zmax[mask] = np.inf, -np.inf

    def _carve(self, points: np.ndarray, origin: np.ndarray, rng, hit_cells: np.ndarray) -> None:
        """Take one count of evidence off each body-band cell a beam crosses within its hit
        heights, except the cells this scan hit."""
        if len(points) > CARVE_RAYS:
            pick = (rng or np.random.default_rng()).choice(len(points), CARVE_RAYS, replace=False)
            points = points[pick]
        d = points - origin
        length = np.hypot(d[:, 0], d[:, 1])
        steps = np.arange(RES_M, length.max(), RES_M)
        if not len(steps):
            return
        frac = steps[None, :] / np.maximum(length[:, None], 1e-6)
        before_end = steps[None, :] < (length[:, None] - RES_M)
        qx = origin[0] + d[:, 0:1] * frac
        qy = origin[1] + d[:, 1:2] * frac
        qz = origin[2] + d[:, 2:3] * frac
        through = before_end & (qz >= OBSTACLE_BAND_M[0]) & (qz <= OBSTACLE_BAND_M[1])
        u, v = self.uv(qx[through], qy[through])
        z = qz[through]
        ok = self._in_grid(u, v)
        u, v, z = u[ok], v[ok], z[ok]
        # only crossings within a cell's hit heights count
        at_height = (z >= self.hit_zmin[v, u] - PASS_UNDER_M) & (z <= self.hit_zmax[v, u])
        cells = np.setdiff1d(np.unique(v[at_height] * self.W + u[at_height]), hit_cells)
        cv, cu = cells // self.W, cells % self.W
        self.hits[cv, cu] -= 1
        gone = self.hits[cv, cu] <= 0
        if gone.any():
            self.hits[cv[gone], cu[gone]] = 0
            self.first_hit[cv[gone], cu[gone]] = np.nan
            self.hit_zmin[cv[gone], cu[gone]] = np.inf
            self.hit_zmax[cv[gone], cu[gone]] = -np.inf

    def mark_obstacle(self, points: np.ndarray) -> None:
        """Mark the cells of POINTS as avoided obstacles until beams see through them."""
        if not len(points):
            return
        u, v = self.uv(points[:, 0], points[:, 1])
        ok = self._in_grid(u, v)
        u, v, z = u[ok], v[ok], points[ok, 2].astype(np.float32)
        self.hits[v, u] = np.maximum(self.hits[v, u], AVOID_SCANS)
        np.minimum.at(self.hit_zmin, (v, u), z)
        np.maximum.at(self.hit_zmax, (v, u), z)
        self.ever_hit[v, u] = True
        fresh = np.isnan(self.first_hit[v, u])
        self.first_hit[v[fresh], u[fresh]] = self.now
        self.last_hit[v, u] = self.now
        self.version += 1

    def sweep(self, x: float, y: float, footprint_m: float, blind_m: float) -> None:
        """Mark the footprint as known floor and the blind disk as assumed floor, except cells
        ever hit."""
        for layer, radius_m in ((self.swept, footprint_m), (self.assumed, blind_m)):
            u, v = self.uv(x, y)
            r = math.ceil(radius_m / RES_M)
            vv, uu = np.ogrid[-r : r + 1, -r : r + 1]
            disk = uu**2 + vv**2 <= r * r
            v0, v1, u0, u1 = (
                max(v - r, 0),
                min(v + r + 1, self.H),
                max(u - r, 0),
                min(u + r + 1, self.W),
            )
            if v0 < v1 and u0 < u1:
                window = disk[v0 - (v - r) : v1 - (v - r), u0 - (u - r) : u1 - (u - r)]
                layer[v0:v1, u0:u1] |= window & ~self.ever_hit[v0:v1, u0:u1]
        self.version += 1

    # -- layers ------------------------------------------------------------------------------
    def _cached(self, key: str, build):
        """The layer for KEY, built once per map version."""
        hit = self._cache.get(key)
        if hit is None or hit[0] != self.version:
            hit = (self.version, build())
            self._cache[key] = hit
        return hit[1]

    def _clean(self, raw: np.ndarray) -> np.ndarray:
        """RAW without specks: blobs of at most SPECK_CELLS cells spanning under SPECK_SPAN_M."""
        raw = raw.copy()
        labels, n = ndimage.label(raw, structure=np.ones((3, 3)))
        if n:
            idx = np.arange(1, n + 1)
            cells = ndimage.sum(raw, labels, idx)
            span = ndimage.maximum(self.hit_zmax, labels, idx) - ndimage.minimum(
                self.hit_zmin, labels, idx
            )
            speck = (cells <= SPECK_CELLS) & (span < SPECK_SPAN_M)
            if speck.any():
                raw &= ~np.isin(labels, idx[speck])
        return raw

    def avoid(self) -> np.ndarray:
        """Cells with at least AVOID_SCANS of evidence, specks removed."""
        return self._cached("avoid", lambda: self._clean(self.hits >= AVOID_SCANS))

    def obstacle_slice(self, z_lo: float, z_hi: float) -> np.ndarray:
        """The avoided cells hit between two heights, specks removed."""
        return self._cached(
            f"slice{z_lo:.3f},{z_hi:.3f}",
            lambda: self._clean(
                (self.hits >= AVOID_SCANS) & (self.hit_zmax >= z_lo) & (self.hit_zmin <= z_hi)
            ),
        )

    def walls(self) -> np.ndarray:
        """Cells avoided for at least WALL_AFTER_S."""
        return self.avoid() & (
            (self.now - np.nan_to_num(self.first_hit, nan=np.inf)) >= WALL_AFTER_S
        )

    def recent_hits(self) -> np.ndarray:
        """Cells hit in two scans, the last within RECENT_HIT_S."""
        return (self.hits >= 2) & (self.now - self.last_hit <= RECENT_HIT_S)

    def _free(self, seen: np.ndarray, obstacle: np.ndarray | None = None) -> np.ndarray:
        obstacle = self.avoid() if obstacle is None else obstacle
        free = seen & ~obstacle
        return ndimage.binary_closing(free, structure=np.ones((5, 5)), border_value=1) & ~obstacle

    def interior(self) -> np.ndarray:
        """Unknown cells inside objects: gaps between hits closer than INTERIOR_GAP_M, and
        pockets under INTERIOR_MAX_M2 with less than INTERIOR_RIM_M of free rim."""

        def build():
            obst = self.avoid()
            free = self.known_free() | self.assumed
            unknown = ~(free | obst)
            closed = ndimage.binary_closing(obst, structure=_disk(round(INTERIOR_GAP_M / RES_M)))
            inside = closed & unknown
            labels, n = ndimage.label(unknown & ~inside, structure=np.ones((3, 3)))
            if n == 0:
                return inside
            ring = ndimage.maximum_filter(labels, size=3) * (labels == 0)
            touch_free = np.bincount(ring[free], minlength=n + 1)
            area = np.bincount(labels.ravel(), minlength=n + 1) * RES_M * RES_M
            keep = (touch_free * RES_M < INTERIOR_RIM_M) & (area <= INTERIOR_MAX_M2)
            keep[0] = False
            border = np.zeros(labels.shape, bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            keep[np.unique(labels[border])] = False  # pockets open to the grid's edge
            return inside | keep[labels]

        return self._cached("interior", build)

    def solid(self) -> np.ndarray:
        """The drawn solids: walls, interiors and avoided cells, with unobserved gaps closed."""

        def build():
            raw = self.walls() | self.interior() | self.avoid()
            closed = ndimage.binary_closing(raw, structure=_disk(SOLID_CLOSE_CELLS))
            # the closing fills only cells not free
            drawn = raw | (closed & ~(self.known_free() | self.assumed))
            labels, n = ndimage.label(drawn, structure=np.ones((3, 3)))
            keep = np.bincount(labels.ravel(), minlength=n + 1) >= SOLID_MIN_CELLS
            keep[0] = False
            return keep[labels]

        return self._cached("solid", build)

    def solid_top(self) -> np.ndarray:
        """The drawn height per cell: the nearest highest hit, median-smoothed over
        SOLID_SMOOTH_CELLS and clipped to the view band."""
        hit = np.isfinite(self.hit_zmax) & (self.hit_zmax > -np.inf) & self.ever_hit
        top = np.where(hit, self.hit_zmax, 0.0).astype(np.float32)
        if hit.any() and not hit.all():
            idx = ndimage.distance_transform_edt(~hit, return_indices=True)[1]
            top = top[idx[0], idx[1]]
        top = ndimage.median_filter(top, size=SOLID_SMOOTH_CELLS)
        return np.clip(top, OBSTACLE_BAND_M[0], VIEW_BAND_TOP_M).astype(np.float32)

    def solid_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(vertices, faces) extruding every solid cell from the floor to `solid_top`, with side
        quads where a neighbour is not solid."""
        mask = self.solid()
        v, u = np.nonzero(mask)
        if not len(v):
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
        top = self.solid_top()[v, u]
        x, y = self.xy(u, v)
        h = RES_M / 2.0
        quads = []          # (4, 3) corners, counter-clockwise seen from outside
        corners = np.stack([np.c_[x - h, y - h], np.c_[x + h, y - h],
                            np.c_[x + h, y + h], np.c_[x - h, y + h]], axis=1)   # (n, 4, 2)
        quads.append(np.concatenate([corners, np.repeat(top[:, None, None], 4, axis=1)], -1))
        padded = np.pad(mask, 1)
        for (dv, du), (a, b) in ((( 0,  1), (1, 2)), (( 0, -1), (3, 0)),   # +x, -x faces
                                 ((-1,  0), (2, 3)), (( 1,  0), (0, 1))):  # +y (up a row), -y
            open_side = ~padded[1 + v + dv, 1 + u + du]
            if not open_side.any():
                continue
            p, q = corners[open_side, a], corners[open_side, b]
            z = top[open_side]
            lo = np.zeros_like(z)
            quads.append(np.stack([np.c_[p, lo], np.c_[q, lo], np.c_[q, z], np.c_[p, z]], axis=1))
        allq = np.concatenate(quads, axis=0).astype(np.float32)          # (m, 4, 3)
        verts = allq.reshape(-1, 3)
        base = np.arange(0, len(verts), 4, dtype=np.int32)[:, None]
        faces = np.concatenate([base + np.array([[0, 1, 2]], np.int32),
                                base + np.array([[0, 2, 3]], np.int32)], axis=0)
        return verts, faces

    def wall_points(self, dz: float = 0.15, limit: int = 300_000) -> np.ndarray:
        """Every wall cell as a column of points from its lowest to its highest hit."""
        mask = self.walls()
        v, u = np.nonzero(mask)
        if not len(v):
            return np.zeros((0, 3), np.float32)
        x, y = self.xy(u, v)
        lo = np.nan_to_num(self.hit_zmin[v, u], nan=0.0, posinf=0.0)
        hi = np.nan_to_num(self.hit_zmax[v, u], nan=0.0, neginf=0.0)
        n = np.maximum(1, np.floor((hi - lo) / dz).astype(int) + 1)
        n = np.minimum(n, 20)
        xs = np.repeat(x, n).astype(np.float32)
        ys = np.repeat(y, n).astype(np.float32)
        zs = (np.repeat(lo, n) + dz * (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))).astype(
            np.float32
        )
        pts = np.c_[xs, ys, zs]
        return pts[:: len(pts) // limit + 1] if len(pts) > limit else pts

    def known_free(self) -> np.ndarray:
        return self._cached("known_free", lambda: self._free((self.floor > 0) | self.swept))

    def unknown(self) -> np.ndarray:
        return ~(self.known_free() | self.avoid())

    def blocking_obstacles(self) -> np.ndarray:
        """Walls and recently hit cells."""
        return self.walls() | self.recent_hits()

    def blocked(self, *, include_boundary: bool = True) -> np.ndarray:
        """What routes avoid: cells not known or assumed free, walls, recent hits and, with
        `include_boundary`, cells outside the allowed centre region."""
        solid = self.blocking_obstacles()
        routable = self._free((self.floor > 0) | self.swept | self.assumed, solid)
        blocked = ~routable | solid
        return blocked | ~self.center_allowed if include_boundary else blocked

    def clearance_m(self) -> np.ndarray:
        clearance = ndimage.distance_transform_edt(~self.blocked(include_boundary=False)) * RES_M
        return np.where(self.center_allowed, clearance, 0.)

    def collision_checker(self, radius_m, *, task_scope=True):
        """The exact-footprint collision checker for RADIUS_M."""
        allowed = self.center_allowed if task_scope else self.inside_inset
        if task_scope and not self.scope_ready:
            allowed = np.zeros_like(allowed)
        art = SimpleNamespace(occupancy=self.blocked(include_boundary=False),
                              T_map_px=self.T_map_px(), center_allowed=allowed)
        return occupancy_grid(art, radius_m, exact_footprint=True)[0]

    def geodesic(self, x: float, y: float, radius_m: float, *, checker=None) -> np.ndarray:
        """Four-neighbour walking distance from (x, y) per cell; infinite from an invalid start."""
        from skimage.graph import MCP_Geometric

        if checker is None:
            checker = self.collision_checker(radius_m)
        if not checker.is_safe(x, y):
            return np.full((self.H, self.W), np.inf)
        start = checker.graph_cell((x, y))
        if start is None:
            return np.full((self.H, self.W), np.inf)
        dist, _ = MCP_Geometric(np.where(checker.grid, 1., -1.),
                                fully_connected=False).find_costs([start])
        connector = math.hypot(x - (checker.xmin + start[0] * checker.cell),
                               y - (checker.ymin + start[1] * checker.cell))
        return (dist * checker.cell + connector)[:, ::-1].T

    def rgb(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), 150, np.uint8)
        img[self.assumed & ~self.avoid()] = (196, 206, 201)
        img[self.known_free()] = (232, 240, 236)
        img[self.known_free() & ~self.inside] = (214, 214, 206)  # seen, not to be walked
        img[self.solid()] = (40, 56, 52)   # drawn solids
        return img

    def write_artifact(self, path: str) -> None:
        """Write the raster, transform and occupancy to PATH atomically."""
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".npz")
        os.close(fd)
        np.savez(tmp, raster=self.rgb(), T_map_px=self.T_map_px(),
                 occupancy=self.blocked(include_boundary=False), center_allowed=self.center_allowed)
        os.replace(tmp, path)
