"""Coarse semantic polygons rasterized as centre permissions, never occupied evidence."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy import ndimage


class TaskRegionRejected(ValueError):
    """A task region the map cannot admit; the message is the refusal the model reads."""


def _cross(a, b, c):
    ab, ac = b - a, c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _on_segment(a, b, p):
    return (abs(_cross(a, b, p)) <= 1e-9
            and np.all(p >= np.minimum(a, b) - 1e-9)
            and np.all(p <= np.maximum(a, b) + 1e-9))


def contains(points, xy):
    """Include the polygon's edge; winding and vertex order do not change permission."""
    p, q = np.asarray(points, float), np.atleast_2d(xy).astype(float)
    inside = Path(p).contains_points(q)
    for a, b in zip(p, np.roll(p, -1, axis=0), strict=True):
        ab = b - a
        t = np.clip(((q - a) @ ab) / float(ab @ ab), 0., 1.)
        inside |= np.sum((q - (a + t[:, None] * ab)) ** 2, axis=1) <= 1e-18
    return inside


def validate_polygon(points):
    """Canonical finite vertices with valid polygon geometry, without allocating a raster."""
    try:
        p = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TaskRegionRejected("task region vertices must be finite x/y pairs") from exc
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise TaskRegionRejected("task region vertices must be finite x/y pairs")
    if len(p) > 3 and np.array_equal(p[0], p[-1]):
        p = p[:-1]
    if not 3 <= len(p) <= 64:
        raise TaskRegionRejected("task region needs 3 to 64 vertices")
    if len(np.unique(p, axis=0)) != len(p):
        raise TaskRegionRejected("task region repeats a vertex")
    area = abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                            - p[:, 1] * np.roll(p[:, 0], -1)))) / 2
    if area <= 1e-9:
        raise TaskRegionRejected("task region has no enclosed area")
    for i, a in enumerate(p):
        b = p[(i + 1) % len(p)]
        c = p[(i + 2) % len(p)]
        if _on_segment(a, b, c) or _on_segment(b, c, a):
            raise TaskRegionRejected("task region has an overlapping edge")
        for j in range(i + 2, len(p)):
            if (j + 1) % len(p) == i:
                continue
            c, d = p[j], p[(j + 1) % len(p)]
            crossed = (_cross(a, b, c) * _cross(a, b, d) < 0
                       and _cross(c, d, a) * _cross(c, d, b) < 0)
            if crossed or any((_on_segment(a, b, c), _on_segment(a, b, d),
                               _on_segment(c, d, a), _on_segment(c, d, b))):
                raise TaskRegionRejected("task region crosses or touches itself")
    return tuple(map(tuple, p.tolist()))


def rasterize(points, *, xmin, ymax, shape, resolution):
    """Return canonical vertices and a connected grid mask, or raise TaskRegionRejected.

    Connectivity is checked without occupancy or footprint inflation: furniture may
    divide the room physically. A diagonal-only raster neck is not a traversable
    connection under the route checker's no-corner-cutting convention.
    """
    vertices = validate_polygon(points)
    p = np.asarray(vertices)
    height, width = shape
    lo = np.array([xmin - resolution / 2, ymax - (height - .5) * resolution])
    hi = np.array([xmin + (width - .5) * resolution, ymax + resolution / 2])
    if np.any(p < lo) or np.any(p > hi):
        raise TaskRegionRejected("task region extends outside the map raster")
    vv, uu = np.mgrid[:height, :width]
    xy = np.c_[xmin + uu.ravel() * resolution, ymax - vv.ravel() * resolution]
    mask = contains(p, xy).reshape(shape)
    if ndimage.label(mask)[1] != 1:
        raise TaskRegionRejected("task region is empty or disconnected on the map grid")
    return vertices, mask
