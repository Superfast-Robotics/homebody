"""skill_episodes.py SCENES_DIR OUT_DIR SKILL [OFFSETS]: write one episode file per scored step of every
generated scene under SCENES_DIR (<prefix>train_* -> train, <prefix>val_* -> val, <prefix>test_* -> test), each at the scene's scripted
stance and at OFFSETS-1 more stances shifted by up to 8 cm (seeded per episode, so reproducible)."""
import json
import random
import sys
from pathlib import Path

SHIFT_M = 0.08


def main(scenes_dir, out_dir, skill, offsets="2"):
    scenes_dir, out_dir = Path(scenes_dir), Path(out_dir)
    count = 0
    for scene in sorted(p for p in scenes_dir.iterdir() if (p / "sequence.json").exists()):
        prefix = scene.name.split("_")[0]  # pktrain_*, pltrain_* ... -> train, val, test
        split = next((name for name in ("train", "val", "test") if prefix.endswith(name)), None)
        if split is None:
            continue
        family = scene.name.split("_", 2)[2]
        for index, step in enumerate(json.loads((scene / "sequence.json").read_text())["steps"]):
            for variant in range(int(offsets)):
                rng = random.Random(f"{scene.name}/{index}/{variant}")
                dx, dy = (rng.uniform(-SHIFT_M, SHIFT_M) for _ in range(2)) if variant else (0.0, 0.0)
                episode_id = f"{scene.name}_s{index}_v{variant}"
                episode = {"id": episode_id, "skill": skill, "scene": str(scene.resolve()), "family": family,
                           "step": step, "stance": [step["goal"][0] + dx, step["goal"][1] + dy, step["goal"][2]]}
                path = out_dir / split / f"{episode_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(episode, indent=1))
                count += 1
    print(count, "episodes")


if __name__ == "__main__":
    main(*sys.argv[1:5])
