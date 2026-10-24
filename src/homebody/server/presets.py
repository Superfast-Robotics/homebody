"""Scenarios a harness offers its consoles: a name, a scene package and the instruction to run in it.

A presets file (`homebody serve --presets FILE`) is TOML with one `[[preset]]` table per scenario:

    [[preset]]
    name = "Warehouse: one jam jar into the cart"
    scene = "/path/to/warehouse_cart"   # a scene package folder; a relative path is taken from the file's folder
    task = "Pick one jam jar ..."

Without a file, the harness offers its starting scene's default task. A preset whose scene is not a scene package
(scene.xml and task_scene.json) is refused when the file is read.
"""
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

MAX_TASK = 4000  # characters, as a console's submitted task


@dataclass(frozen=True)
class Preset:
    name: str
    scene: Path
    task: str

    def public(self) -> dict:
        """What a console is told: the scene by its folder name, never the host's path."""
        return {"name": self.name, "scene": self.scene.name, "task": self.task}


def is_scene(folder: Path) -> bool:
    return (folder / "scene.xml").is_file() and (folder / "task_scene.json").is_file()


def default(scene: Path) -> tuple[Preset, ...]:
    task_scene = json.loads((scene / "task_scene.json").read_text())
    return (Preset(task_scene.get("name") or scene.name, scene, task_scene.get("default_task", "")),)


def load(path: Path | None, scene: Path) -> tuple[Preset, ...]:
    """The presets in PATH (else SCENE's default task). Raises ValueError naming the first bad entry."""
    if path is None:
        return default(scene)
    entries = tomllib.loads(path.read_text()).get("preset")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} has no [[preset]] tables")
    presets, names = [], set()
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or set(entry) != {"name", "scene", "task"} \
                or not all(isinstance(entry[key], str) and entry[key].strip() for key in entry):
            raise ValueError(f"{path}: preset {index} needs exactly name, scene and task, as text")
        folder = Path(entry["scene"]).expanduser()
        folder = (folder if folder.is_absolute() else path.parent / folder).resolve()
        if not is_scene(folder):
            raise ValueError(f"{path}: preset {entry['name']!r}: {folder} is not a scene package")
        if entry["name"] in names or len(entry["task"]) > MAX_TASK:
            raise ValueError(f"{path}: preset {entry['name']!r} is repeated or its task is too long")
        names.add(entry["name"])
        presets.append(Preset(entry["name"].strip(), folder, entry["task"].strip()))
    return tuple(presets)
