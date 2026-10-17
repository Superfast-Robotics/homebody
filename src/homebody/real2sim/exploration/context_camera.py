"""Optional uncalibrated RGB context; never a source of depth, pose or motion targets."""
from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import zmq

from homebody.primitives.cameras.color_stream import pack


def jpeg_frames(stream, max_bytes: int = 4_000_000):
    """Parse FFmpeg's concatenated JPEG output, refusing unbounded/corrupt frames."""
    buffer = bytearray()
    while chunk := stream.read1(65536):
        buffer.extend(chunk)
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                del buffer[:-1]
                break
            if start:
                del buffer[:start]
            end = buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(buffer) > max_bytes:
                    raise ValueError("context JPEG exceeds size limit")
                break
            if end + 2 > max_bytes:
                raise ValueError("context JPEG exceeds size limit")
            yield bytes(buffer[:end + 2])
            del buffer[:end + 2]


class Recording:
    """One new capture directory holding the encoded video and decode-time records."""

    def __init__(self, directory: Path, transport: str):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.frames = (self.directory / "frames.jsonl").open("x")
        self.state = {"schema": 1, "status": "waiting", "started_realtime_ns": str(time.time_ns()),
                      "input_transport": transport, "timestamp_kind": "receiver_decode_time",
                      "metric_calibration": False, "frames_published": 0,
                      "recording": "video_*.mkv: original encoded video, no audio",
                      "frame_log": "frames.jsonl: sampled context frames, not video frame PTS"}
        self.last_health = 0.0
        self._save()

    def _save(self):
        temp = self.directory / "recording.json.tmp"
        temp.write_text(json.dumps(self.state, indent=2) + "\n")
        temp.replace(self.directory / "recording.json")

    def frame(self, header: dict, jpeg: bytes):
        self.frames.write(json.dumps(header | {
            "capture_realtime_ns": str(header["capture_realtime_ns"])}) + "\n")
        self.state.update(status="receiving", frames_published=header["sequence"],
                          last_decode_realtime_ns=str(header["capture_realtime_ns"]))
        if time.monotonic() - self.last_health >= 1:
            self.frames.flush()
            temp = self.directory / "latest.jpg.tmp"
            temp.write_bytes(jpeg)
            temp.replace(self.directory / "latest.jpg")
            self._save()
            self.last_health = time.monotonic()

    def close(self, returncode: int | None, interrupted: bool, *, failed: bool = False):
        self.frames.close()
        self.state.update(status="failed" if failed else "stopped" if interrupted else "ended",
                          ffmpeg_returncode=returncode, interrupted=interrupted,
                          ended_realtime_ns=str(time.time_ns()))
        self.state["segments"] = [p.name for p in sorted(self.directory.glob("video_*.mkv"))]
        self._save()


def ffmpeg_command(source: str, width: int, fps: float, recording: Recording | None,
                   segment_seconds: float, hwaccel: str = "none"):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n",
               "-probesize", "32768", "-analyzeduration", "100000"]
    if urlsplit(source).scheme in {"rtsp", "rtsps"}:
        command += ["-rtsp_transport", "tcp"]
    filters = f"fps={fps},scale={width}:-2"
    if hwaccel == "cuda":
        command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        # Decode and resize on device; only sampled context frames cross back for JPEG.
        filters = (f"fps={fps},scale_cuda={width}:-2:interp_algo=bicubic,"
                   "hwdownload,format=nv12")
    elif hwaccel != "none":
        raise ValueError(f"unsupported phone decoder: {hwaccel}")
    command += ["-i", source,
               "-map", "0:v:0", "-an", "-vf", filters,
               "-c:v", "mjpeg", "-q:v", "3", "-f", "image2pipe", "pipe:1"]
    if recording is not None:
        command += ["-map", "0:v:0", "-an", "-c:v", "copy", "-f", "segment",
                    "-segment_time", str(segment_seconds), "-reset_timestamps", "1",
                    "-segment_list", str(recording.directory / "segments.csv"),
                    "-segment_list_type", "csv", str(recording.directory / "video_%05d.mkv")]
    return command


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="FFmpeg input URL or local test video")
    parser.add_argument("--publish", default="tcp://127.0.0.1:55570")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--hwaccel", choices=("none", "cuda"), default="none",
                        help="CUDA decodes/resizes 8-bit phone video on the GPU; requires "
                             "FFmpeg CUDA/NVDEC support. Native recording remains stream-copy.")
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument("--record-dir", type=Path,
                           help="new directory for original video + metadata")
    recording.add_argument("--record-root", type=Path,
                           help="parent directory; create a fresh capture for each connection")
    parser.add_argument("--reconnect", action="store_true",
                        help="listen again after an input disconnects")
    parser.add_argument("--segment-seconds", type=float, default=300)
    args = parser.parse_args()
    if not 64 <= args.width <= 1920 or not math.isfinite(args.fps) or not 0 < args.fps <= 30:
        parser.error("width must be 64..1920; fps must be finite and within (0, 30]")
    if not math.isfinite(args.segment_seconds) or args.segment_seconds <= 0:
        parser.error("segment-seconds must be positive and finite")
    if args.reconnect and args.record_dir is not None:
        parser.error("use --record-root with --reconnect to preserve each connection separately")
    return args


def _stop_receiver(_signal, _frame):
    raise KeyboardInterrupt


def receive(args, directory: Path | None) -> int:
    publisher = zmq.Context.instance().socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 1)
    publisher.setsockopt(zmq.LINGER, 0)
    process = None
    recording = None
    interrupted = False

    previous_term = signal.signal(signal.SIGTERM, _stop_receiver)
    try:
        publisher.bind(args.publish)
        if directory is not None:
            recording = Recording(directory, urlsplit(args.input).scheme or "file")
            recording.state["hwaccel"] = args.hwaccel
        command = ffmpeg_command(args.input, args.width, args.fps, recording,
                                 args.segment_seconds, args.hwaccel)
        # FFmpeg runs in its own session; the receiver alone stops it
        process = subprocess.Popen(command, stdout=subprocess.PIPE, start_new_session=True)
        for sequence, jpeg in enumerate(jpeg_frames(process.stdout), 1):
            header = {"sequence": sequence, "capture_realtime_ns": time.time_ns(),
                      "timestamp_kind": "receiver_decode_time", "camera_name": "context_camera",
                      "metric_calibration": False}
            publisher.send(pack(header, jpeg))
            if recording is not None:
                recording.frame(header, jpeg)
        return int(process.wait() != 0)
    except KeyboardInterrupt:
        interrupted = True
        return 130
    finally:
        shutdown_failed = False
        if process is not None:
            requested_stop = process.poll() is None
            if requested_stop:
                process.terminate()
                try:
                    # drain the pipe so FFmpeg can finish its segment
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    shutdown_failed = True
                    print("FFmpeg shutdown timed out; any unfinished native segment is preserved",
                          flush=True)
                    process.kill()
                    process.communicate()
            if interrupted:
                # FFmpeg exits 255 on a handled SIGTERM
                expected = (0, 255) if requested_stop else (0,)
                shutdown_failed |= process.returncode not in expected
            process.stdout.close()
        if recording is not None:
            recording.close(None if process is None else process.returncode, interrupted,
                            failed=shutdown_failed)
        publisher.close()
        signal.signal(signal.SIGTERM, previous_term)
        if shutdown_failed:
            raise RuntimeError("FFmpeg failed during receiver shutdown; recordings kept")


def main() -> int:
    args = arguments()
    root = None if args.record_root is None else args.record_root.resolve()
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    previous_term = signal.signal(signal.SIGTERM, _stop_receiver)
    try:
        while True:
            directory = args.record_dir
            if root is not None:
                directory = root / f"capture_{time.time_ns()}"
                pointer = root / "current.json.tmp"
                pointer.write_text(json.dumps({"capture_directory": str(directory)}) + "\n")
                pointer.replace(root / "current.json")
            result = receive(args, directory)
            if not args.reconnect or result == 130:
                return result
            print("Phone input ended; listening again in 5 seconds", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())
