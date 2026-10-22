import json

import numpy as np
from PIL import Image

from evolve import agent_capabilities as capabilities
from evolve import agent_place as place


def case(tmp_path):
    """A camera at the torso looking straight down the world +x axis at a wall: depth z is the distance ahead, image
    x maps to world -y and image y to world -z. The left half is the support (label 5)."""
    folder = tmp_path / "cases" / "c1"
    folder.mkdir(parents=True)
    h, w = 48, 64
    depth = np.full((h, w), 0.6, np.float32)  # every pixel 0.6 m ahead: beyond reach
    depth[:, :8] = 0.3  # a near strip at the left edge
    labels = np.zeros((h, w), np.int32)
    labels[:, : w // 2] = 5
    intrinsics = np.array([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]])
    world_camera = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.0], [0, 0, 0, 1]], float)
    np.savez(folder / "geometry.npz", depth=depth, intrinsics=intrinsics, world_camera=world_camera,
             world_torso=np.eye(4))
    np.savez(folder / "labels.npz", labels=labels)
    Image.new("RGB", (w, h), (200, 200, 200)).save(folder / "rgb.png")
    reference = {"skill": "place", "text": "Place the can on the left.", "arguments": {"point_normalized_1000": [100, 500]}}
    (folder / "case.json").write_text(json.dumps({"reference": reference, "target": 5}))
    return folder


def test_reach_masking_and_clamping(tmp_path):
    folder = case(tmp_path)
    depth, intrinsics, world_camera, torso = place.geometry(folder)
    far = place.world_of([300, 500], depth, intrinsics, world_camera)
    assert abs(place.reach_of(far, torso) - 0.61) < 0.02  # 0.6 m ahead, slightly to the side
    image = np.asarray(Image.open(place.masked(folder, tmp_path / "out")))
    assert image[24, 40].tolist() != [200, 200, 200] and image[24, 2].tolist() == [200, 200, 200]  # far greyed
    assert place.clamped([50, 500], folder) == [50, 500]  # already within reach (the near strip)
    assert place.clamped([300, 500], folder) == [300, 500]  # no in-reach pixel of label 5 on its way: unchanged


def test_place_scores_support_and_reach(tmp_path):
    folder = case(tmp_path)
    out = tmp_path / "out"
    rows = {"c1": {"case": "c1", "point": [50, 500], "seconds": 1.0}}
    for name, row in rows.items():
        (out / name).mkdir(parents=True)
        (out / name / "bench.json").write_text(json.dumps(row))
    (out / "run.json").write_text(json.dumps({"kind": "place", "model": "m", "cases": str(folder.parent)}))
    stats = capabilities.score_run(out)
    assert stats["place.on_support"] == "1/1" and stats["place.in_reach"] == "1/1" and stats["place.both"] == "1/1"
    assert stats["place.reference_in_reach"] == "1/1"
