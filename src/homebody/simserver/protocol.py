"""What the sim server and its clients agree on: a request is {method, args}; a reply is {ok, value} or
{ok: false, error, message}, the error naming one of ERRORS so the client raises the same type the in-process backend
would. A scene package is identified by a hash of its files, so a harness refuses a server serving another scene."""
import hashlib
from pathlib import Path

from homebody.primitives.actions import ActionRejected, PhysicsInvalid

ENGINE = "homebody"
SCHEMA = 1
ERRORS = {cls.__name__: cls for cls in (ActionRejected, PhysicsInvalid, ValueError)}
SMALL = 4 * 1024 * 1024  # bytes: files up to this size are hashed whole, larger ones by name and size


def scene_identity(scene: Path) -> str:
    """A hash of SCENE's files (names, sizes and, for files up to SMALL, contents)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(scene).rglob("*") if p.is_file()):
        size = path.stat().st_size
        digest.update(f"{path.relative_to(scene)}\0{size}\0".encode())
        if size <= SMALL:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def failure(error: Exception) -> dict:
    """ERROR as a reply: its type when the client knows it, else a ValueError with its text."""
    name = next((n for n in ("PhysicsInvalid", "ActionRejected", "ValueError") if isinstance(error, ERRORS[n])), None)
    return {"ok": False, "error": name or "ValueError", "message": f"{type(error).__name__}: {error}"[:600]}


def raised(reply: dict) -> Exception:
    """The exception a failed REPLY stands for."""
    return ERRORS.get(str(reply.get("error", "")), ValueError)(str(reply.get("message", "the sim server refused the call")))
