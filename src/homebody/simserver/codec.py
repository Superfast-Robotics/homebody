"""The sim server's wire format: msgpack, with numpy arrays as an extension type (dtype, shape, bytes; large ones
zlib-compressed) and the Robot records (primitives/observations) as maps tagged with their name, rebuilt on the
other side through their own constructors, so their checks run again. Nothing else is accepted: no pickle."""
import dataclasses
import zlib
from collections.abc import Mapping

import msgpack
import numpy as np

from homebody.primitives.observations import Frame, HandState, Measurement, NavigationMap, WristView

ARRAY = 1  # msgpack extension code of a numpy array
COMPRESS_FROM = 64 * 1024  # bytes: larger arrays travel zlib-compressed
RECORD = "__record__"
RECORDS = {cls.__name__: cls for cls in (Frame, HandState, Measurement, NavigationMap, WristView)}
DTYPES = frozenset("?bBhHiIlLqQefd")  # numeric and boolean array kinds only


def _default(value):
    if isinstance(value, np.ndarray):
        if value.dtype.char not in DTYPES:
            raise TypeError(f"arrays of {value.dtype} do not travel")
        data = np.ascontiguousarray(value).tobytes()
        packed = zlib.compress(data, 1) if len(data) >= COMPRESS_FROM else data
        header = msgpack.packb([value.dtype.str, list(value.shape), len(packed) != len(data)])
        return msgpack.ExtType(ARRAY, header + packed)
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and type(value).__name__ in RECORDS:
        return {RECORD: type(value).__name__, **{f.name: getattr(value, f.name) for f in dataclasses.fields(value)}}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{type(value).__name__} does not travel")


def _ext(code, data):
    if code != ARRAY:
        raise TypeError(f"unknown extension {code}")
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(data)
    dtype, shape, compressed = unpacker.unpack()
    if dtype[1:] not in {np.dtype(c).str[1:] for c in DTYPES}:
        raise TypeError(f"arrays of {dtype} do not travel")
    body = data[unpacker.tell():]
    return np.frombuffer(zlib.decompress(body) if compressed else body, dtype=np.dtype(dtype)).reshape(shape).copy()


def _record(value: dict):
    name = value.get(RECORD)
    if name is None:
        return value
    if name not in RECORDS:
        raise TypeError(f"unknown record {name}")
    fields = {key: item for key, item in value.items() if key != RECORD}
    return RECORDS[name](**fields)


def pack(value) -> bytes:
    return msgpack.packb(value, default=_default, use_bin_type=True)


def unpack(payload: bytes):
    return msgpack.unpackb(payload, raw=False, ext_hook=_ext, object_hook=_record, strict_map_key=False)
