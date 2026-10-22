"""Place pointing on the bench: where a pointer clicks to set a carried object down, on the reference model's
successful point placements (agent_bench cases with geometry.npz). Each click is scored on the support the reference
placed on (its label), and on reach: the clicked point's horizontal distance from the torso (the arm reaches about
0.42 m ahead).

Variants: "plain" asks as the composite pointer does; "reach" adds that the spot is near the surface's front edge
within reach; "mask" greys out the pixels whose point lies beyond reach before the pointer sees the image; "clamp"
moves a click beyond reach toward the robot until it is within reach, on the same support; "mask+clamp" does both.
The geometry is helpers/reach.py's, as the session's open-model options use it."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from homebody.helpers.reach import beyond, greyed, pull_within, reach_of, world_of
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


def masked(case: Path, out: Path) -> Path:
    """The head image with every pixel whose point lies beyond reach greyed out."""
    depth, intrinsics, world_camera, world_torso = geometry(case)
    rgb = np.asarray(Image.open(case / "rgb.png").convert("RGB"))
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(greyed(rgb, beyond(depth, intrinsics, world_camera, world_torso, SCORE_REACH_M + 0.05))).save(
        out / "rgb_masked.png")
    return out / "rgb_masked.png"


def clamped(point, case: Path):
    """POINT moved within reach on the same support (helpers/reach.pull_within)."""
    depth, intrinsics, world_camera, world_torso = geometry(case)
    labels = np.load(case / "labels.npz")["labels"]
    return pull_within(point, depth, intrinsics, world_camera, world_torso, labels, REACH_M, SCORE_REACH_M)


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
