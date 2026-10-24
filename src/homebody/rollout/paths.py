"""Paths read from recorded files, which an imported rollout may have written to point anywhere."""
from pathlib import Path, PurePosixPath


def inside(root: Path, relative) -> Path | None:
    """ROOT/RELATIVE when RELATIVE (from a recorded, possibly imported, file) is a relative path that stays inside
    ROOT; else None."""
    if not isinstance(relative, str) or not relative:
        return None
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        return None
    joined = root / relative
    return joined if joined.resolve().is_relative_to(root.resolve()) else None
