"""Build a conservative static navigation map from a reconstructed MuJoCo scene.

Run after build_kitchen.py. Requires matching named, right-handed, metre/Z-up frames
in semantics.json and task_scene.json, one ground plane, and finite coplanar floor__
visual meshes. Boxes and convex mesh collisions are projected through an explicit
height band above that plane. Holes and unsupported floor remain occupied. The map
describes initial geometry only; the planner applies robot-radius inflation later.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

MAX_CELLS = 4_000_000


def raster(points, bounds, resolution, shape, padding=0.):
    """Cells whose centres lie in a convex polygon, conservatively expanded by padding."""
    points = np.unique(np.asarray(points)[:, :2], axis=0)
    lo, hi = points.min(axis=0) - padding, points.max(axis=0) + padding
    origin = np.array([bounds[0], bounds[3]])
    scale = np.array([resolution, -resolution])
    pixels = (np.array([lo, hi]) - origin) / scale - .5
    first = np.maximum(np.ceil(pixels.min(axis=0) - 1e-9).astype(int), 0)
    last = np.minimum(np.floor(pixels.max(axis=0) + 1e-9).astype(int), shape[::-1] - 1)
    columns, rows = slice(first[0], last[0] + 1), slice(first[1], last[1] + 1)
    if np.any(first > last):
        return rows, columns, np.zeros((0, 0), bool)
    x = origin[0] + (np.arange(first[0], last[0] + 1) + .5) * resolution
    y = origin[1] - (np.arange(first[1], last[1] + 1) + .5) * resolution
    xx, yy = np.meshgrid(x, y)
    if len(points) == 2:
        direction = points[1] - points[0]
        normal = np.array([-direction[1], direction[0]])
        inside = np.abs(normal[0] * (xx - points[0, 0]) +
                        normal[1] * (yy - points[0, 1])) <= padding * np.abs(normal).sum() + 1e-9
    elif len(points) == 1:
        inside = np.ones(xx.shape, bool)
    else:
        hull = ConvexHull(points)
        inside = np.ones(xx.shape, bool)
        for a, b, c in hull.equations:
            inside &= a * xx + b * yy + c <= padding * (abs(a) + abs(b)) + 1e-9
    return rows, columns, inside


def paint(grid, points, bounds, resolution, padding=0., value=True):
    rows, columns, mask = raster(points, bounds, resolution, np.array(grid.shape), padding)
    if mask.size:
        view = grid[rows, columns]
        view[mask] = value


def mesh(model, data, geom):
    identity = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[identity], model.mesh_vertnum[identity]
    vertices = model.mesh_vert[start:start + count].astype(float)
    vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    start, count = model.mesh_faceadr[identity], model.mesh_facenum[identity]
    return vertices, model.mesh_face[start:start + count]


def clipped_footprint(vertices, height, band):
    """Project the convex solid intersected with the vertical height slab."""
    heights = height(vertices)
    if heights.max() < band[0] or heights.min() > band[1]:
        return None
    hull = ConvexHull(vertices)
    edges = np.unique(np.sort(np.concatenate([hull.simplices[:, [0, 1]],
        hull.simplices[:, [1, 2]], hull.simplices[:, [2, 0]]]), axis=1), axis=0)
    points = [vertices[(heights >= band[0]) & (heights <= band[1])]]
    for level in band:
        a, b = edges[:, 0], edges[:, 1]
        cross = (heights[a] < level) & (heights[b] > level) | (
            (heights[b] < level) & (heights[a] > level))
        a, b = a[cross], b[cross]
        fraction = (level - heights[a]) / (heights[b] - heights[a])
        points.append(vertices[a] + fraction[:, None] * (vertices[b] - vertices[a]))
    return np.concatenate(points)[:, :2]


def sources(folder):
    tree = ET.parse(folder / "scene.xml")
    if any(node.tag in ("include", "compiler") for node in tree.iter()):
        raise ValueError("Use build_kitchen.py output without includes or compiler path overrides")
    paths = {"scene.xml", "semantics.json", "task_scene.json"}
    paths.update(node.get("file") for node in tree.iter() if node.get("file"))
    hashes = {}
    for name in sorted(paths):
        path = folder / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"Scene dependency must stay inside the package: {name}")
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def build(folder: Path, resolution: float, height_band, bounds=None, replace=False):
    """Write map.npz atomically, refusing replacement unless explicitly requested."""
    folder = Path(folder)
    output = folder / "map.npz"
    if output.exists() and not replace:
        raise ValueError("map.npz already exists; use --replace to regenerate it")
    band = np.asarray(height_band, float)
    if (not np.isfinite(resolution) or resolution <= 0 or band.shape != (2,) or
            not np.isfinite(band).all() or not 0 < band[0] < band[1]):
        raise ValueError("Resolution must be positive; height band must be finite with 0 < MIN < MAX")
    semantic = json.loads((folder / "semantics.json").read_text())
    task = json.loads((folder / "task_scene.json").read_text())
    frame = semantic["coordinate_frame"]
    if (not frame.get("name") or task.get("frame") != frame["name"] or
            frame.get("units") != "metres" or frame.get("up_axis") != "Z" or
            frame.get("handedness") != "right"):
        raise ValueError("Declare matching frame names, units=metres, up_axis=Z, handedness=right")
    hashes = sources(folder)
    model = mujoco.MjModel.from_xml_path(str((folder / "scene.xml").resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(planes) != 1 or not model.geom(int(planes[0])).name.startswith("floor__"):
        raise ValueError("Scene needs exactly one floor__ ground plane")
    floor = int(planes[0])
    normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
    if normal[2] < np.cos(np.deg2rad(8)):
        raise ValueError("Ground plane must face up and be within 8 degrees of horizontal")

    def height(vertices):
        return (vertices - data.geom_xpos[floor]) @ normal / normal[2]

    floors = [mesh(model, data, g) for g in range(model.ngeom)
              if model.geom(g).name.startswith("floor__") and
              model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    if not floors or any(np.max(np.abs(height(v))) > 1e-5 for v, _ in floors):
        raise ValueError("Finite floor__ meshes must lie on the ground plane; no inferred floor")
    vertices = np.concatenate([v for v, _ in floors])
    if bounds is None:
        bounds = [*vertices[:, :2].min(axis=0), *vertices[:, :2].max(axis=0)]
    bounds = np.asarray(bounds, float)
    if (bounds.shape != (4,) or not np.isfinite(bounds).all() or
            np.any(bounds[2:] <= bounds[:2])):
        raise ValueError("Bounds must be finite XMIN YMIN XMAX YMAX with positive extents")
    dimensions = np.ceil((bounds[2:] - bounds[:2]) / resolution)
    if not np.isfinite(dimensions).all() or np.prod(dimensions) > MAX_CELLS:
        raise ValueError(f"Map exceeds {MAX_CELLS} cells; choose coarser resolution or smaller bounds")
    width, rows = dimensions.astype(int)
    supported = np.zeros((rows, width), bool)
    boundaries = []
    for points, faces in floors:
        points, weld = np.unique(points, axis=0, return_inverse=True)
        faces = weld[faces]
        for face in faces:
            triangle = points[face, :2]
            if abs(np.linalg.det(triangle[1:] - triangle[0])) > 1e-12:
                paint(supported, triangle, bounds, resolution)
        edges, counts = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
            faces[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Floor mesh has non-manifold edges")
        boundaries.extend(points[edge] for edge in edges[counts == 1])
    for edge in boundaries:
        paint(supported, edge, bounds, resolution, resolution / 2, False)
    supported[:, (np.arange(width) + 1) * resolution > bounds[2] - bounds[0] + 1e-9] = False
    supported[(np.arange(rows) + 1) * resolution > bounds[3] - bounds[1] + 1e-9, :] = False
    occupied = ~supported
    collision_count = 0
    for geom in range(model.ngeom):
        if geom == floor or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        name = model.geom(geom).name
        if model.geom_margin[geom] or model.geom_gap[geom]:
            raise ValueError(f"Nonzero collision margin/gap is unsupported: {name}")
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            points, _ = mesh(model, data, geom)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            points = np.array(list(itertools.product([-1, 1], repeat=3))) * model.geom_size[geom]
            points = points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        else:
            raise ValueError(f"Unsupported collision shape for {name}; use built mesh/box collisions")
        try:
            footprint = clipped_footprint(points, height, band)
            if footprint is not None and len(footprint):
                paint(occupied, footprint, bounds, resolution, resolution / 2)
        except QhullError as error:
            raise ValueError(f"Degenerate collision geometry: {name}") from error
        collision_count += 1
    transform = np.array([[resolution, 0, bounds[0] + resolution / 2],
                          [0, -resolution, bounds[3] - resolution / 2], [0, 0, 1]])
    if not np.any(~occupied):
        raise ValueError("No supported free cells remain in the selected bounds/height band")
    metadata = {"frame": frame, "resolution_m": resolution, "bounds_xy_m": bounds.tolist(),
                "height_band_m": band.tolist(), "height_reference": "vertical above floor plane",
                "source_sha256": hashes, "collision_geometries": collision_count,
                "robot_radius_inflated": False, "state": "initial scene configuration",
                "limitations": "Static map; no unobserved floor, dynamic tracking or hardware qualification"}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=folder, prefix=".map-", suffix=".npz",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, occupancy=occupied, T_map_px=transform,
                                metadata_json=json.dumps(metadata, sort_keys=True))
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="Built scene package directory")
    parser.add_argument("--resolution", type=float, required=True, help="Grid cell width in metres")
    parser.add_argument("--height-band", type=float, nargs=2, required=True, metavar=("MIN", "MAX"),
                        help="Robot-relevant obstacle heights above the floor in metres")
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--replace", action="store_true", help="Replace an existing map.npz")
    args = parser.parse_args()
    try:
        result = build(args.scene, args.resolution, args.height_band, args.bounds, args.replace)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Map build refused: {error}\n")
    print(f"Wrote {args.scene / 'map.npz'} in frame {result['frame']['name']}; the planner applies robot-radius inflation")


if __name__ == "__main__":
    main()
