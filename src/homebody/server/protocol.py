"""What the harness server and its clients send each other over the WebSocket.

Text messages are JSON objects with a `type`. Images travel as binary messages: a 4-byte big-endian header length,
the JSON header (with its `type`), then the JPEG bytes. A client's first message is `hello` with the token and its
role; every command after it carries the `epoch` it saw, and the server refuses commands of an older epoch (an
operator stop starts a new one).

Server -> client: state, status, skill, decision, monitor (a call for the client's monitor), refused, and images
`view` (the console's camera) and `shot` (a head camera frame for the monitor).
Client -> server: hello, the operator's commands (COMMANDS), monitor_stop, monitor_verdicts, monitor_event.
"""
import base64
import dataclasses
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

COMMANDS = ("submit", "stop", "pause", "unpause", "stop_skill", "dismiss", "tell", "resume_task", "camera", "preset", "attach_sim")
MONITOR = ("monitor_stop", "monitor_verdicts", "monitor_event")
ROLES = ("commander", "viewer")
HEADER = struct.Struct(">I")
MAX_MESSAGE = 16 * 1024 * 1024  # bytes; a camera frame is far smaller


def jsonable(value):
    """VALUE with what JSON cannot carry made plain: arrays as lists, dataclasses as dicts, paths and anything else
    unknown as text."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 256 else f"<array {value.shape}>"
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable({field.name: getattr(value, field.name) for field in dataclasses.fields(value)})
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def plain(fields: dict) -> dict:
    return {str(key): jsonable(value) for key, value in fields.items()}


def text(kind: str, **fields) -> str:
    return json.dumps({"type": kind, **plain(fields)})


def jpeg(rgb: np.ndarray, quality: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def image(kind: str, rgb: np.ndarray, quality: int, **fields) -> bytes:
    header = json.dumps({"type": kind, **plain(fields)}).encode()
    return HEADER.pack(len(header)) + header + jpeg(rgb, quality)


def parse(message: str | bytes) -> tuple[dict, np.ndarray | None]:
    """A received message as (fields, rgb or None). Raises ValueError on anything malformed."""
    if isinstance(message, str):
        fields = json.loads(message)
        if not isinstance(fields, dict) or not isinstance(fields.get("type"), str):
            raise ValueError("a message is a JSON object with a type")  # noqa: TRY004 -- every bad message is one error
        return fields, None
    if len(message) < HEADER.size:
        raise ValueError("a binary message too short for its header")
    (size,) = HEADER.unpack_from(message)
    fields = json.loads(message[HEADER.size:HEADER.size + size])
    if not isinstance(fields, dict) or not isinstance(fields.get("type"), str):
        raise ValueError("an image header is a JSON object with a type")  # noqa: TRY004 -- every bad message is one error
    with Image.open(io.BytesIO(message[HEADER.size + size:])) as picture:
        return fields, np.asarray(picture.convert("RGB"))


def embedded(path: Path, quality: int) -> str:
    """An image file as base64 JPEG, for a request whose images the client cannot read from disk."""
    with Image.open(path) as picture:
        return base64.b64encode(jpeg(np.asarray(picture.convert("RGB")), quality)).decode()


def unembedded(data: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data, validate=True))
    return path
