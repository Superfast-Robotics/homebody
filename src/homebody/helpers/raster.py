"""Map pictures and the clicks on them: pixels and metres through a 3x3 homogeneous T_map_px,
the 0..1000 click grid, and the north-up cell grid of the live map.

Two click conventions exist and stay distinct. Exploration's `grid_to_px` puts 0 and GRID on
the first and last pixel centres, `click * (w - 1) / GRID`; SIM's `skills.navigate.map_point`
measures from the image edges, `click * [w, h] / 1000`.
"""
import math

import numpy as np

GRID = 1000.0  # the click grid every exploration clicking action takes


def px_to_world(T_map_px: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """(N,2) map-raster pixels -> (N,2) world metres via the 3x3 homogeneous T_map_px."""
    T = np.asarray(T_map_px, dtype=float).reshape(3, 3)
    p = np.atleast_2d(np.asarray(uv, dtype=float))
    ones = np.ones((p.shape[0], 1))
    out = (np.hstack([p, ones]) @ T.T)
    return out[:, :2] / out[:, 2:3]


def world_to_px(T_map_px: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(N,2) world metres -> (N,2) map-raster pixels; exact inverse of px_to_world."""
    T = np.asarray(T_map_px, dtype=float).reshape(3, 3)
    return px_to_world(np.linalg.inv(T), xy)


def to_metres(T_map_px, u: float, v: float) -> tuple[float, float]:
    """An image pixel as start-frame metres: the exact inverse of `to_px`."""
    (x, y), = px_to_world(np.asarray(T_map_px, float), np.array([[float(u), float(v)]]))
    return float(x), float(y)


def to_px(T_map_px, xy) -> tuple[float, float]:
    """Start-frame metres as an image pixel."""
    (u, v), = world_to_px(np.asarray(T_map_px, float), np.array([[float(xy[0]), float(xy[1])]]))
    return float(u), float(v)


def grid_to_px(grid, shape) -> tuple[float, float]:
    """A 0..1000 click -> image pixel, 0 and GRID on the first and last pixel centres."""
    h, w = shape[:2]
    return float(grid[0]) * (w - 1) / GRID, float(grid[1]) * (h - 1) / GRID


def px_to_grid(u: float, v: float, shape) -> tuple[int, int]:
    h, w = shape[:2]
    return round(u * GRID / max(w - 1, 1)), round(v * GRID / max(h - 1, 1))


def valid_grid(grid):
    return (isinstance(grid, (tuple, list)) and len(grid) == 2 and
            all(not isinstance(v, bool) and isinstance(v, (int, float))
                and math.isfinite(v) and 0 <= v <= GRID
                for v in grid))


def cell_uv(xmin, ymax, resolution, x, y):
    """The (u, v) cell of start-frame points on a north-up grid whose cell (0, 0) is centred at
    (xmin, ymax): u counts east, v counts south."""
    return (
        np.round((np.asarray(x) - xmin) / resolution).astype(int),
        np.round((ymax - np.asarray(y)) / resolution).astype(int),
    )


def cell_xy(xmin, ymax, resolution, u, v):
    """The start-frame centres of cells (u, v) of that grid."""
    return xmin + np.asarray(u) * resolution, ymax - np.asarray(v) * resolution


def cell_transform(xmin, ymax, resolution) -> np.ndarray:
    """That grid's T_map_px: a cell index taken as a pixel lands on its centre."""
    return np.array([[resolution, 0.0, xmin], [0.0, -resolution, ymax], [0.0, 0.0, 1.0]])
