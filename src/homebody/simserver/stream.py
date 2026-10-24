"""The camera stream's message, byte for byte warehouse-sim's (warehouse_sim/stream.py): two parts, the camera name
then a msgpack map {schema_version, camera, stamp, encoding, shape, data} with the RGB image zlib-compressed. The
monitor app's SimVideo reads both servers alike."""
import zlib

import msgpack
import numpy as np

ENCODING = "rgb8+zlib"
LEVEL = 1  # fast; renders compress well even so
MAX_PIXELS = 1920 * 1080


def encode(camera: str, stamp: dict, rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return msgpack.packb({"schema_version": 1, "camera": camera, "stamp": stamp, "encoding": ENCODING,
                          "shape": list(rgb.shape), "data": zlib.compress(rgb.tobytes(), LEVEL)}, use_bin_type=True)


def decode(payload: bytes) -> dict:
    """A stream message as {camera, stamp, rgb}; ValueError on anything malformed."""
    try:
        message = msgpack.unpackb(payload, raw=False, max_bin_len=MAX_PIXELS * 3, max_str_len=4096)
        height, width, channels = message["shape"]
        if message["encoding"] != ENCODING or channels != 3 or not 0 < height * width <= MAX_PIXELS:
            raise ValueError("unsupported stream frame")
        pixels = zlib.decompressobj().decompress(message["data"], height * width * 3 + 1)
        if len(pixels) != height * width * 3:
            raise ValueError("stream frame size mismatch")
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()
        return {"camera": str(message["camera"]), "stamp": dict(message["stamp"]), "rgb": rgb}
    except (KeyError, TypeError, zlib.error, msgpack.UnpackException) as error:
        raise ValueError("invalid stream frame") from error
