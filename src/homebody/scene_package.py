"""Export a built scene as a runtime package.

The package holds the self-contained scene.xml with its mesh and texture closure, the
navigation map, the semantic prior, the task and the physics settings. The destination
must not already exist.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image

from homebody.session.settings import Settings

ENTITY_KEYS = ("id", "label", "visual_description", "location_description",
               "reference_anchor_ws_map_m", "portable_candidate", "bounds_ws_map_m")
TASK_KEYS = ("name", "frame", "robot_start", "movable_entities", "default_task")
FRAME_KEYS = ("name", "units", "up_axis", "handedness")


def read(source, name):
    path = source / name
    if not path.resolve().is_relative_to(source):
        raise ValueError(f"Dependency escapes scene: {name}")
    return path


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def package(source, destination, *, validate=None):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination already exists; export to a new directory")
    if destination.resolve().is_relative_to(source):
        raise ValueError("Export outside the source scene")
    destination.parent.mkdir(parents=True, exist_ok=True)
    xml = ET.parse(read(source, "scene.xml"))
    root = xml.getroot()
    if any(n.tag in {"include", "compiler", "plugin", "extension"} for n in root.iter()):
        raise ValueError("Use a self-contained built scene without includes/compiler/plugins")
    for node in list(root):
        if node.tag in {"keyframe", "custom"}:
            root.remove(node)
    # Assets are rewritten as plain OBJ and PNG so the package depends on scene.xml alone.
    with tempfile.TemporaryDirectory(prefix=".sim-export-", dir=destination.parent) as work:
        out = Path(work) / "scene"
        (out / "meshes").mkdir(parents=True)
        copied = {}
        for node in root.iter():
            name = node.get("file")
            if name is None:
                continue
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Scene dependency must be a relative path inside the scene")
            asset = read(source, name)
            if node.tag not in {"mesh", "texture"}:
                raise ValueError(f"Unsupported scene dependency: {node.tag}")
            if name not in copied:
                suffix = asset.suffix.lower()
                if suffix not in {".obj", ".png"}:
                    raise ValueError(f"Unsupported runtime asset format: {suffix}")
                target = f"meshes/asset_{len(copied):04d}{suffix}"
                if suffix == ".obj":
                    # MuJoCo's XML owns materials; OBJ groups/comments/MTL are unnecessary.
                    lines = [line for line in asset.read_text().splitlines()
                             if line.split() and line.split()[0] in {"v", "vt", "vn", "f"}]
                    (out / target).write_text("\n".join(lines) + "\n")
                else:
                    with Image.open(asset) as image:
                        image.load()
                        pixels = image.convert("RGBA") if image.mode == "P" else image
                        clean = Image.fromarray(np.asarray(pixels))
                        clean.save(out / target, format="PNG")
                copied[name] = target
            node.set("file", copied[name])
        xml.write(out / "scene.xml", encoding="unicode")
        with np.load(read(source, "map.npz"), allow_pickle=False) as artifact:
            occupancy = artifact["occupancy"]
            transform = artifact["T_map_px"]
            if occupancy.ndim != 2 or transform.shape != (3, 3) or not np.isfinite(transform).all():
                raise ValueError("Invalid navigation map")
            np.savez_compressed(out / "map.npz", occupancy=occupancy, T_map_px=transform)
        semantics = json.loads(read(source, "semantics.json").read_text())
        json_write(out / "semantics.json", {
            "coordinate_frame": {k: v for k, v in semantics.get("coordinate_frame", {}).items()
                                 if k in FRAME_KEYS},
            "entities": [{k: entity[k] for k in ENTITY_KEYS if k in entity}
                         for entity in semantics["entities"]],
        })
        task = json.loads(read(source, "task_scene.json").read_text())
        json_write(out / "task_scene.json", {k: task[k] for k in TASK_KEYS if k in task})
        physics = json.loads(read(source, "physics.json").read_text())
        json_write(out / "physics.json", {"objects": {name: {} for name in physics["objects"]}})
        if (source / "settings.toml").exists():
            config = tomllib.loads(read(source, "settings.toml").read_text())
            Settings.load(None, scene=source)
            defaults = Settings()
            lines = []
            groups = {f.name for f in fields(defaults)}
            for group, values in config.items():
                if group not in groups:
                    raise ValueError(f"Unknown settings group: {group}")
                allowed = {f.name for f in fields(getattr(defaults, group))}
                lines.append(f"[{group}]")
                for key, value in values.items():
                    if key not in allowed or isinstance(value, dict):
                        raise ValueError(f"Unsupported setting: {group}.{key}")
                    lines.append(f"{key} = {json.dumps(value, allow_nan=False)}")
            (out / "settings.toml").write_text("\n".join(lines) + "\n")
        # Compile the exported scene before moving it into place.
        import mujoco
        mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
        if validate is not None:
            validate(out)
        if destination.exists():
            raise ValueError("Destination appeared during export")
        os.rename(out, destination)
    return sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = package(args.scene, args.output)
    print(f"Exported {len(files)} runtime files to {args.output}")


if __name__ == "__main__":
    main()
