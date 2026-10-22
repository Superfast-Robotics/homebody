"""Arm reach in a camera view: the world point under a click, its horizontal distance from the torso, the pixels
beyond reach, and a click moved within reach on the same labelled surface. Pure math over a frame's depth (metres
along the optical axis), intrinsics, world_camera and world_torso transforms and labels."""
import numpy as np


def pixel(point, shape):
    """(u, v) of point_normalized_1000 POINT on an image of SHAPE (observations.normalized_pixel's rounding)."""
    height, width = shape
    return int(point[0] * (width - 1) / 1000), int(point[1] * (height - 1) / 1000)


def world_of(point, depth, intrinsics, world_camera):
    """The world point under POINT, None where the depth is missing."""
    u, v = pixel(point, depth.shape)
    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0:
        return None
    camera = [(u - intrinsics[0, 2]) * z / intrinsics[0, 0], (v - intrinsics[1, 2]) * z / intrinsics[1, 1], z, 1.0]
    return (world_camera @ camera)[:3]


def reach_of(world, world_torso) -> float:
    """Horizontal distance of WORLD from the torso."""
    return float(np.hypot(*(np.asarray(world)[:2] - world_torso[:3, 3][:2])))


def beyond(depth, intrinsics, world_camera, world_torso, reach_m):
    """The pixels whose point lies beyond REACH_M of the torso, or has no depth."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    camera = np.stack([(u - intrinsics[0, 2]) * depth / intrinsics[0, 0],
                       (v - intrinsics[1, 2]) * depth / intrinsics[1, 1], depth, np.ones_like(depth)], -1)
    world = camera @ world_camera.T
    far = np.hypot(world[..., 0] - world_torso[0, 3], world[..., 1] - world_torso[1, 3]) > reach_m
    return far | ~np.isfinite(depth) | (depth <= 0)


def greyed(rgb, mask):
    """RGB with the MASK pixels greyed out."""
    out = np.array(rgb, dtype=np.uint8, copy=True)
    out[mask] = (out[mask] * 0.25 + 96).astype(np.uint8)
    return out


def pull_within(point, depth, intrinsics, world_camera, world_torso, labels, reach_m, accept_m):
    """POINT moved toward the torso, at its height, until the point the camera sees there is within ACCEPT_M and on
    POINT's label; POINT unchanged when it is already within REACH_M or no such pixel is found."""
    start = world_of(point, depth, intrinsics, world_camera)
    if start is None or reach_of(start, world_torso) <= reach_m:
        return list(point)
    h, w = depth.shape
    u0, v0 = pixel(point, depth.shape)
    label = int(labels[v0, u0])
    torso = world_torso[:3, 3]
    camera_from_world = np.linalg.inv(world_camera)
    for fraction in np.linspace(0.05, 0.6, 12):
        target = np.array(start, dtype=float)
        target[:2] = start[:2] + (torso[:2] - start[:2]) * fraction
        if reach_of(target, world_torso) > reach_m:
            continue
        x, y, z, _ = camera_from_world @ [*target, 1.0]
        if z <= 0:
            continue
        u, v = intrinsics[0, 0] * x / z + intrinsics[0, 2], intrinsics[1, 1] * y / z + intrinsics[1, 2]
        if not (0 <= u < w and 0 <= v < h) or int(labels[int(v), int(u)]) != label:
            continue
        moved = [round(u * 1000 / (w - 1)), round(v * 1000 / (h - 1))]
        seen = world_of(moved, depth, intrinsics, world_camera)
        if seen is not None and reach_of(seen, world_torso) <= accept_m:
            return moved
    return list(point)
