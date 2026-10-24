"""Rollout bundles: one recorded session (or one of its tasks) as a .tar.gz that another machine can import.

The archive holds one folder, named by the rollout id, with the session's files unchanged and `rollout.json`: what
the rollout is (source, scene, tasks, whether it carries monitor logs and physics) and the sha256 of every file. An
import trusts nothing in the archive: every member must be a regular file or folder the manifest lists, inside the
rollout folder, with the listed bytes, and the whole within the size cap; otherwise nothing is kept.
"""
import hashlib
import json
import shutil
import tarfile
import tempfile
import time
import zlib
from pathlib import Path, PurePosixPath

from homebody.rollout.paths import inside

SCHEMA = 1
MANIFEST = "rollout.json"
HEAVY = ("sensors.npz", "depth_preview.png")  # what a light export leaves out
MB = 1024 * 1024
MAX_MANIFEST_MB = 8
MAX_MEMBERS = 200_000
SHAPE = {"schema": int, "id": str, "source": str, "tasks": list, "frames": int, "has_monitor_logs": bool,
         "has_physics": bool, "files": dict}  # what an imported manifest must hold, and as what


class BundleError(ValueError):
    pass


def events(session: Path) -> list[dict]:
    path = session / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def task_events(records: list[dict], index: int) -> list[dict]:
    """The session's opening records and those of its INDEX-th task (from its task_started to the next one)."""
    starts = [n for n, record in enumerate(records) if record.get("kind") == "task_started"]
    if index >= len(starts):
        raise BundleError(f"the session has no task {index}")
    end = starts[index + 1] if index + 1 < len(starts) else len(records)
    return records[:starts[0]] + records[starts[index]:end]


def well_formed(manifest) -> bool:
    if not isinstance(manifest, dict) or any(not isinstance(manifest.get(key), kind) for key, kind in SHAPE.items()):
        return False
    tasks_ok = all(isinstance(t, dict) and isinstance(t.get("id"), str) for t in manifest["tasks"])
    files_ok = all(inside(Path("/r"), name) is not None and isinstance(entry, dict)
                   and isinstance(entry.get("sha256"), str) for name, entry in manifest["files"].items())
    return tasks_ok and files_ok


def describe(rollout: Path) -> dict:
    """The manifest of ROLLOUT: its own rollout.json when imported and well formed, else what its files say
    (without hashes)."""
    if (rollout / MANIFEST).is_file():
        try:
            manifest = json.loads((rollout / MANIFEST).read_text())
        except (OSError, ValueError):
            manifest = None
        if well_formed(manifest):
            return manifest
    return summary(rollout, rollout.name, events(rollout))


def summary(session: Path, rollout_id: str, records: list[dict], task: str | None = None) -> dict:
    provenance = json.loads((session / "provenance.json").read_text()) if (session / "provenance.json").is_file() else {}
    scene = next((r.get("path") for r in records if r.get("kind") == "scene_package"), None)
    folders = [session / "tasks" / task] if task else sorted((session / "tasks").glob("task-*"))
    tasks = []
    for folder in folders:
        spec = json.loads((folder / "task.json").read_text()) if (folder / "task.json").is_file() else {}
        outcome = json.loads((folder / "outcome.json").read_text()) if (folder / "outcome.json").is_file() else {}
        tasks.append({"id": folder.name, "task": spec.get("task", ""), "status": outcome.get("status"),
                      "steps": outcome.get("steps")})
    return {"schema": SCHEMA, "id": rollout_id,
            "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "source": "real" if provenance.get("backend") == "real" else "sim",
            "scene": PurePosixPath(scene).name if scene else None, "tasks": tasks,
            "has_monitor_logs": any((folder / "monitor/readings.jsonl").is_file() for folder in folders),
            "has_physics": any((folder / "physics.jsonl").is_file() for folder in folders),
            "frames": sum(1 for r in records if r.get("kind") == "frame")}


def chosen(session: Path, light: bool, task: str | None, epoch: int | None) -> list[Path]:
    """The session's files an export carries, relative to SESSION."""
    files = []
    for path in sorted(session.rglob("*")):
        relative = path.relative_to(session)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(session) \
                or relative.parts[0] in (MANIFEST, "events.jsonl"):
            continue
        if light and path.name in HEAVY:
            continue
        nested = len(relative.parts) > 1
        if task and relative.parts[0] == "tasks" and (not nested or relative.parts[1] != task):
            continue
        if task and relative.parts[0] == "frames" and epoch is not None \
                and (not nested or not relative.parts[1].startswith(f"{epoch:03d}-")):
            continue
        files.append(relative)
    return files


class Hashing:
    """A file read for the archive, hashed as it is read: the manifest names exactly the bytes stored."""

    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MB), b""):
            digest.update(block)
    return digest.hexdigest()


def export(session: Path, out_dir: Path, *, task: str | None = None, light: bool = True) -> Path:
    """Write SESSION (or only its TASK, e.g. "task-000") to OUT_DIR/<id>.tar.gz and return the archive's path."""
    session = session.resolve()
    records, epoch = events(session), None
    if task:
        folder = session / "tasks" / task
        if not (folder / "task.json").is_file():
            raise BundleError(f"{session.name} has no {task}")
        epoch = json.loads((folder / "task.json").read_text()).get("epoch")
        records = task_events(records, int(task.removeprefix("task-")))
    rollout_id = f"{session.name}-{task}" if task else session.name
    files = chosen(session, light, task, epoch)
    manifest = summary(session, rollout_id, records, task)
    manifest["light"] = light
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "events.jsonl"
        log.write_text("".join(json.dumps(record) + "\n" for record in records))
        sources = {PurePosixPath("events.jsonl"): log, **{PurePosixPath(*f.parts): session / f for f in files}}
        out_dir.mkdir(parents=True, exist_ok=True)
        archive = out_dir / f"{rollout_id}.tar.gz"
        partial = archive.with_suffix(".partial")
        listed = {}
        with tarfile.open(partial, "w:gz") as tar:
            # each file is hashed as it is stored (a running session's files may grow meanwhile); the manifest last
            for name, path in sorted(sources.items()):
                info = tar.gettarinfo(path, f"{rollout_id}/{name}")
                with path.open("rb") as stream:
                    hashing = Hashing(stream)
                    tar.addfile(info, hashing)
                listed[str(name)] = {"sha256": hashing.digest.hexdigest(), "bytes": info.size}
            manifest["files"] = listed
            written = Path(scratch) / MANIFEST
            written.write_text(json.dumps(manifest, indent=1))
            tar.add(written, f"{rollout_id}/{MANIFEST}")
        partial.replace(archive)
    return archive


def checked_members(tar: tarfile.TarFile, max_mb: float) -> tuple[str, dict, list[tarfile.TarInfo]]:
    """The rollout id, its manifest and the archive's members, after every check that needs no extraction."""
    members = []
    for member in tar:
        members.append(member)
        if len(members) > MAX_MEMBERS:
            raise BundleError(f"the bundle holds more than {MAX_MEMBERS} files")
    roots = {PurePosixPath(m.name).parts[0] for m in members if PurePosixPath(m.name).parts}
    if len(roots) != 1:
        raise BundleError("a rollout bundle holds exactly one folder")
    rollout_id = roots.pop()
    if rollout_id in (".", "..") or "/" in rollout_id or "\\" in rollout_id:
        raise BundleError(f"bad rollout folder name {rollout_id!r}")
    entry = next((m for m in members if m.name == f"{rollout_id}/{MANIFEST}" and m.isfile()), None)
    if entry is None:
        raise BundleError(f"the bundle has no {MANIFEST}")
    stream = tar.extractfile(entry) if entry.size <= MAX_MANIFEST_MB * MB else None
    if stream is None:
        raise BundleError(f"the bundle's {MANIFEST} is unreadable or too large")
    manifest = json.loads(stream.read())
    if not well_formed(manifest) or manifest["schema"] != SCHEMA or manifest["id"] != rollout_id:
        raise BundleError(f"the bundle's {MANIFEST} is not a schema {SCHEMA} manifest for {rollout_id}")
    listed = manifest["files"]
    total = 0
    for member in members:
        path = PurePosixPath(member.name)
        relative = PurePosixPath(*path.parts[1:]) if len(path.parts) > 1 else PurePosixPath()
        if path.is_absolute() or ".." in path.parts:
            raise BundleError(f"{member.name} lies outside the rollout")
        if member.isdir():
            continue
        if not member.isfile():
            raise BundleError(f"{member.name} is not a regular file")
        if str(relative) != MANIFEST and str(relative) not in listed:
            raise BundleError(f"{member.name} is not in the manifest")
        total += member.size
    if total > max_mb * MB:
        raise BundleError(f"the bundle unpacks to {total / MB:.1f} MB, larger than the {max_mb:g} MB allowed")
    return rollout_id, manifest, members


def import_(archive: Path, root: Path, *, max_mb: float, replace: bool = False) -> Path:
    """Unpack ARCHIVE into ROOT/<id> after checking it, and return that folder. Raises BundleError when refused."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            rollout_id, manifest, members = checked_members(tar, max_mb)
            target = root / rollout_id
            if target.exists() and not replace:
                raise BundleError(f"{rollout_id} is already imported")
            root.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".incoming-", dir=root))
            try:
                tar.extractall(staging, members=members, filter="data")
                unpacked = staging / rollout_id
                for name, entry in manifest["files"].items():
                    path = unpacked / name
                    if not path.is_file() or sha256(path) != entry["sha256"]:
                        raise BundleError(f"{name} is missing or its hash differs from the manifest")
                if target.exists():
                    shutil.rmtree(target)
                unpacked.replace(target)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
    except (tarfile.TarError, OSError, EOFError, zlib.error, ValueError, KeyError, TypeError, AttributeError) as error:
        if isinstance(error, BundleError):
            raise
        raise BundleError(f"not a readable rollout bundle: {type(error).__name__}: {error}") from error
    return target
