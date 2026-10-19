"""warehouse_episodes.py BASE OUT [SETTINGS [MIXED_FROM]]: pick episodes in the warehouse scene packages.

For every warehouse scene with a task sequence (configs/warehouse_*.json, its scene under
assets/real2sim/), one episode per movable object in the front rows (within ROW_DEPTH_M of the
frontmost object): the robot stands in front of it at the sequence's stance distance, picks it with the
right hand, carries it along the sequence's carry path to its destination and walks back. Every third
object (by position) is val, the rest train; each object also gets a second stance shifted by up to 8 cm.
SETTINGS (relative to the candidate root, e.g. configs/dex1.toml) goes into every episode. MIXED_FROM, an
episodes folder, adds every third of its train and every second of its val episodes, so the benchmark
stays mostly warehouse but still covers other scenes. Run on the remote with BASE's .venv (needs mujoco).
"""
import json
import random
import sys
from pathlib import Path

import mujoco

ROW_DEPTH_M = 0.12
SHIFT_M = 0.08


def objects(scene):
    """(name, x, y) of the scene's movable bodies, as compiled."""
    model = mujoco.MjModel.from_xml_path(str(scene / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = [model.jnt_bodyid[j] for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    return [(model.body(b).name, *map(float, data.xpos[b][:2])) for b in bodies]


def scene_episodes(base, sequence, settings):
    steps = json.loads(sequence.read_text())["steps"]
    first = steps[0]
    scene = base / "assets/real2sim" / sequence.stem.split("_three")[0].split("_four")[0]
    standoff = first["facing"][1] - first["goal"][1]
    found = objects(scene)
    front = min(y for _, _, y in found)
    rows = sorted((item for item in found if item[2] <= front + ROW_DEPTH_M), key=lambda item: (item[1], item[2]))
    episodes = []
    for index, (name, x, y) in enumerate(rows):
        split = "val" if index % 3 == 2 else "train"
        step = {key: first[key] for key in ("hand", "carry_goal", "carry_facing", "destination_label",
                                            "release_map_xy_m", "release_delta_z_m", "evaluator_target")}
        step.update(fixture_id=name, label=name, goal=[round(x, 3), round(front - standoff, 3), first["goal"][2]],
                    facing=[round(x, 3), round(y, 3)])
        for variant in range(2):
            rng = random.Random(f"{scene.name}/{name}/{variant}")
            dx, dy = (rng.uniform(-SHIFT_M, SHIFT_M) for _ in range(2)) if variant else (0.0, 0.0)
            episode = {"id": f"wh{split}_{scene.name}_{name}_v{variant}", "skill": "pick", "scene": str(scene),
                       "family": scene.name, "step": step,
                       "stance": [step["goal"][0] + dx, step["goal"][1] + dy, step["goal"][2]], "return": True}
            if settings:
                episode["settings"] = settings
            episodes.append((split, episode))
    return episodes


def main(base, out, settings="", mixed_from=""):
    base, out = Path(base).resolve(), Path(out)
    episodes = [item for sequence in sorted((base / "configs").glob("warehouse_*.json"))
                for item in scene_episodes(base, sequence, settings)]
    if mixed_from:
        for split, every in (("train", 3), ("val", 2)):
            for path in sorted((Path(mixed_from) / split).glob("*.json"))[::every]:
                episode = json.loads(path.read_text())
                if settings:
                    episode["settings"] = settings
                episodes.append((split, episode))
    for split, episode in episodes:
        path = out / split / f"{episode['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(episode, indent=1))
    counts = {split: sum(1 for s, _ in episodes if s == split) for split in ("train", "val")}
    warehouse = {split: sum(1 for s, e in episodes if s == split and e["id"].startswith("wh")) for split in counts}
    print("episodes", counts, "warehouse", warehouse)


if __name__ == "__main__":
    main(*sys.argv[1:5])
