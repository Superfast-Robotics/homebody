"""Place pointing on the bench: where a pointer clicks to set a carried object down, on the reference model's
successful point placements (agent_bench cases with geometry.npz). Each click is scored on the support the reference
placed on (its label), and on reach: the clicked point's horizontal distance from the torso (the arm reaches about
0.42 m ahead).

Variants: "plain" asks as the composite pointer does; "reach" adds that the spot is near the surface's front edge
within reach; "mask" greys out the pixels whose point lies beyond reach before the pointer sees the image; "clamp"
moves a click beyond reach toward the robot until it is within reach, on the same support; "mask+clamp" does both."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from homebody.vlm.composite import POINTER_PROMPTS, Endpoint, to_1000
from homebody.vlm.provider import ProviderError

REACH_M = 0.42  # the palm reaches about this far ahead of the torso (the place skill's prompt line)
SCORE_REACH_M = 0.45
VARIANTS = ("plain", "reach", "mask", "clamp", "mask+clamp")
REACH_NOTE = (" The spot must be a free part of that surface near its front edge, close to the robot: within about "
              "0.4 m of it.")


def geometry(case: Path):
    g = np.load(case / "geometry.npz")
    return g["depth"].astype(np.float32), g["intrinsics"], g["world_camera"], g["world_torso"]


def world_of(point, depth, intrinsics, world_camera):
    """The world point under point_normalized_1000 POINT (None where the depth is missing)."""
    h, w = depth.shape
    u, v = int(point[0] * (w - 1) / 1000), int(point[1] * (h - 1) / 1000)
    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0:
        return None
    camera = [(u - intrinsics[0, 2]) * z / intrinsics[0, 0], (v - intrinsics[1, 2]) * z / intrinsics[1, 1], z, 1.0]
    return (world_camera @ camera)[:3]


def reach_of(world, world_torso) -> float:
    return float(np.hypot(*(world[:2] - world_torso[:3, 3][:2])))


def masked(case: Path, out: Path) -> Path:
    """The head image with every pixel whose point lies beyond reach greyed out."""
    depth, intrinsics, world_camera, world_torso = geometry(case)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    camera = np.stack([(u - intrinsics[0, 2]) * depth / intrinsics[0, 0],
                       (v - intrinsics[1, 2]) * depth / intrinsics[1, 1], depth, np.ones_like(depth)], -1)
    world = camera @ world_camera.T
    far = np.hypot(world[..., 0] - world_torso[0, 3], world[..., 1] - world_torso[1, 3]) > SCORE_REACH_M + 0.05
    far |= ~np.isfinite(depth) | (depth <= 0)
    rgb = np.asarray(Image.open(case / "rgb.png").convert("RGB")).copy()
    rgb[far] = (rgb[far] * 0.25 + 96).astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out / "rgb_masked.png")
    return out / "rgb_masked.png"


def clamped(point, case: Path):
    """POINT moved toward the robot until within reach, on the same support label (unchanged when it is already
    within reach, or when no such pixel is found)."""
    depth, intrinsics, world_camera, world_torso = geometry(case)
    labels = np.load(case / "labels.npz")["labels"]
    start = world_of(point, depth, intrinsics, world_camera)
    if start is None or reach_of(start, world_torso) <= REACH_M:
        return point
    h, w = depth.shape
    label = int(labels[int(point[1] * (h - 1) / 1000), int(point[0] * (w - 1) / 1000)])
    torso = world_torso[:3, 3]
    camera_from_world = np.linalg.inv(world_camera)
    for fraction in np.linspace(0.05, 0.6, 12):  # slide toward the torso, keeping the height
        target = start.copy()
        target[:2] = start[:2] + (torso[:2] - start[:2]) * fraction
        if reach_of(target, world_torso) > REACH_M:
            continue
        x, y, z, _ = camera_from_world @ [*target, 1.0]
        if z <= 0:
            continue
        u, v = intrinsics[0, 0] * x / z + intrinsics[0, 2], intrinsics[1, 1] * y / z + intrinsics[1, 2]
        if 0 <= u < w and 0 <= v < h and int(labels[int(v), int(u)]) == label:
            moved = [round(u * 1000 / (w - 1)), round(v * 1000 / (h - 1))]
            seen = world_of(moved, depth, intrinsics, world_camera)  # what the camera sees there, not the guess
            if seen is not None and reach_of(seen, world_torso) <= SCORE_REACH_M:
                return moved
    return point


def place_target(meta) -> str | None:
    """The destination of a point placement the reference made successfully, in the reference's own words."""
    args = meta["reference"]["arguments"]
    if meta["target"] is None or meta["reference"]["skill"] != "place" or "point_normalized_1000" not in args:
        return None
    return meta["reference"]["text"].rstrip(".")


def ask_place(endpoint: Endpoint, case: Path, out: Path, variant: str, save) -> dict | None:
    meta = json.loads((case / "case.json").read_text())
    if (wanted := place_target(meta)) is None or not (case / "geometry.npz").is_file():
        return None
    started = time.monotonic()
    words = f"the spot to set the carried object down in this robot step: {wanted}"
    prompt = POINTER_PROMPTS[endpoint.format].format(target=words) + (REACH_NOTE if variant == "reach" else "")
    image = masked(case, out / case.name) if variant.startswith("mask") else case / "rgb.png"
    try:
        answer = endpoint.ask(prompt, (image,), attempt=out / case.name / "request", tick=lambda: None,
                              cancelled=lambda: False)
        point = to_1000(answer, endpoint.format, case / "rgb.png")
    except ProviderError as error:
        return save(out, case, {"error": str(error)[:300]}, started)
    final = clamped(point, case) if variant.endswith("clamp") else point
    return save(out, case, {"answer": answer[:300], "raw_point": point, "point": final}, started)


def score_place(rows, cases: Path, tally) -> dict:
    """place.on_support: the click on the reference's support; place.in_reach: within SCORE_REACH_M of the torso;
    place.both: on the support and in reach (what a place needs); also the reference's own reach for scale."""
    for row in rows:
        case = cases / row["case"]
        meta = json.loads((case / "case.json").read_text())
        depth, intrinsics, world_camera, world_torso = geometry(case)
        reference = world_of(meta["reference"]["arguments"]["point_normalized_1000"], depth, intrinsics, world_camera)
        if reference is not None:
            tally.add("place.reference_in_reach", reach_of(reference, world_torso) <= SCORE_REACH_M)
        if "point" not in row:
            tally.add("place.both", False)
            tally.add("place.unparsed", True)
            continue
        labels = np.load(case / "labels.npz")["labels"]
        h, w = labels.shape
        on = int(labels[int(row["point"][1] * (h - 1) / 1000), int(row["point"][0] * (w - 1) / 1000)]) == meta["target"]
        world = world_of(row["point"], depth, intrinsics, world_camera)
        near = world is not None and reach_of(world, world_torso) <= SCORE_REACH_M
        tally.add("place.on_support", on)
        tally.add("place.in_reach", near)
        tally.add("place.both", on and near)
        tally.add("place.unparsed", False)
    return tally.row()
