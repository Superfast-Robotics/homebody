"""A task of the warehouse suite copied with the targets the Dex1 cannot hold dropped (they stay in the scene as
products to keep in place), its wording and scorer steps kept in step. Run on a host that mounts the volume:

    python3 suite_feasible.py SOURCE_TASK DEST_TASK [--drop sweetener_box tray ...] [--widen-staging]

Dropped kinds come from the measured end-to-end results (2026-10-21, every version): trays 0/66 placed, sweetener
boxes 6/132, the picking scene's allulose 0/88 and basil 0/44 (its tiny packed jam jars, 2/88, are not
dropped by kind: drop the picking tasks instead). --widen-staging extends the warehouse_shelf
staging target to the whole empty board it sits on (x from -1.7: the wording "the empty adjoining shelf to the
left" covers the board up to the next rack's stock, the scorer's box ended at -1.5). Standard library only."""
import argparse
import json
import re
import shutil
from pathlib import Path

PHRASES = {"sweetener_box": ("sweetener box",), "tray": ("food tray", "tray"), "allulose": ("allulose",),
           "basil": ("basil",), "bowl": ("bowl",)}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
STAGING_LOW_X = -1.7


def kind(fixture):
    """can_2 -> can, sweetener_box_1 -> sweetener_box, jam_07 -> jam."""
    return re.sub(r"_\d+$", "", fixture)


def prune_text(text, dropped, count):
    """TEXT with the listed products of the DROPPED kinds removed and its count set to COUNT."""
    head, sep, tail = text.partition("first ")
    listing, sep2, rest = tail.partition(". After each pick")
    if not (sep and sep2):
        raise ValueError(f"cannot find the product listing in: {text[:120]}")
    names = listing.split(", then ")
    kept = [n for n in names if not any(p in n for k in dropped for p in PHRASES.get(k, (k.replace("_", " "),)))]
    if len(kept) != count:
        raise ValueError(f"the wording names {len(kept)} kept products but the scorer keeps {count}: {names}")
    rest = re.sub(r"all (\d+|one|two|three|four|five) (kinds|products)",
                  lambda m: f"all {count if m.group(1).isdigit() else WORDS[count]} {m.group(2)}", rest)
    return head + "first " + ", then ".join(kept) + ". After each pick" + rest


def widen(step):
    """A staging-area step's scorer box over the whole empty board (from STAGING_LOW_X), both supports allowed."""
    target = step["evaluator_target"]
    if target.get("support_contacts") != ["staging_area"]:
        return step
    (low, high) = target["bounds"]
    return {**step, "evaluator_target": {**target, "bounds": [[STAGING_LOW_X, *low[1:]], high],
                                         "support_contacts": ["staging_area", "neighbor_rack_left"]}}


def make(source, dest, drop, widen_staging=False):
    """DEST: SOURCE's package (symlinks followed) with DROP kinds out of its targets; the kept fixtures."""
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    shutil.copytree(source, dest, symlinks=False)
    sequence = json.loads((dest / "sequence.json").read_text())
    steps = [s for s in sequence["steps"] if kind(s["fixture_id"]) not in drop]
    if widen_staging:
        steps = [widen(s) for s in steps]
    removed = [s["fixture_id"] for s in sequence["steps"] if kind(s["fixture_id"]) in drop]
    (dest / "sequence.json").write_text(json.dumps({**sequence, "steps": steps}, indent=1))
    task = json.loads((dest / "task_scene.json").read_text())
    if removed:
        task["default_task"] = prune_text(task["default_task"], drop, len(steps))
    (dest / "task_scene.json").write_text(json.dumps(task, indent=1))
    note = {"from": str(source), "dropped": removed, "kept": [s["fixture_id"] for s in steps],
            "widened_staging": bool(widen_staging and any("neighbor_rack_left" in s["evaluator_target"].get(
                "support_contacts", []) for s in steps))}
    (dest / "feasible.json").write_text(json.dumps(note, indent=1))
    return note


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--drop", nargs="*", default=["sweetener_box", "tray", "allulose", "basil", "bowl"])
    parser.add_argument("--widen-staging", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(make(args.source, args.dest, set(args.drop), args.widen_staging)))


if __name__ == "__main__":
    main()
