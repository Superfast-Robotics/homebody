"""Read a recorded capture in place: its manifest, files and stream selection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Selection:
    """Select streams by data kind and, optionally, exact stream identity."""

    kinds: tuple[str, ...] = ("rgb", "depth", "lidar", "imu", "poses", "joints")
    streams: tuple[str, ...] = ()


@dataclass(frozen=True)
class Asset:
    path: str
    selector: str | None = None
    sha256: str | None = None
    time: float | None = None
    epoch: int | None = None


@dataclass(frozen=True)
class Stream:
    id: str
    kind: str
    status: str
    frame_id: str | None
    timestamp_status: str
    timestamp_fields: tuple[dict, ...]
    assets: tuple[Asset, ...]
    reason: str | None = None
    payload_schema: str | dict | None = None


class Capture:
    """A capture's file catalogue; native payloads keep their clocks and encodings."""

    def __init__(self, root, streams, *, clocks=(), frames=(), calibrations=(), transforms=(), alignments=()):
        self.root = Path(root).resolve()
        self.streams = tuple(streams)
        if len({s.id for s in self.streams}) != len(self.streams):
            raise ValueError("Duplicate stream identity")
        self.clocks, self.frames, self.calibrations = clocks, frames, calibrations
        self.transforms, self.alignments = transforms, alignments

    def resolve(self, asset: Asset, *, verify=False) -> Path:
        name = Path(asset.path)
        path = (self.root / name).resolve()
        if name.is_absolute() or ".." in name.parts or not path.is_relative_to(self.root):
            raise ValueError("Path escapes capture")
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify and asset.sha256:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != asset.sha256:
                raise ValueError(f"Capture hash mismatch: {asset.path}")
        return path

    def select(self, selection=None):
        selection = selection or Selection()
        if missing := set(selection.streams) - {s.id for s in self.streams}:
            raise ValueError(f"Unknown streams: {sorted(missing)}")
        return tuple(s for s in self.streams if s.kind in selection.kinds and
                     (not selection.streams or s.id in selection.streams))

    def context(self, selection=None, *, max_files=8):
        """A bounded listing for an agent; unverified or absent streams stay explicit."""
        if max_files < 1:
            raise ValueError("max_files must be positive")
        rows = []
        for stream in self.select(selection):
            row = asdict(stream)
            row["asset_count"] = len(stream.assets)
            row["assets"] = row["assets"][:max_files]
            for asset in stream.assets[:max_files]:
                self.resolve(asset)
            rows.append(row)
        return {"streams": rows, "clocks": self.clocks, "frames": self.frames,
                "calibrations": self.calibrations, "transforms": self.transforms,
                "alignments": self.alignments,
                "notice": "File references are not proof of calibration, synchronization or coverage."}

    @classmethod
    def open(cls, root):
        root = Path(root).resolve()
        manifest_path = root / "capture_manifest.json"
        if manifest_path.is_file():
            cls(root, []).resolve(Asset("capture_manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            files = {f["id"]: f for f in manifest["files"]}
            if len(files) != len(manifest["files"]):
                raise ValueError("Duplicate capture file identity")
            streams = []
            for row in manifest["streams"]:
                assets = []
                for binding in row["bindings"]:
                    file = files[binding["file_id"]]
                    assets.append(Asset(file["path"], binding.get("selector"), file.get("sha256")))
                streams.append(Stream(row["id"], row["kind"], row["status"], row.get("frame_id"),
                                      row["timestamp_status"], tuple(row["timestamp_fields"]),
                                      tuple(assets), row.get("reason"), row.get("payload_schema")))
            return cls(root, streams, clocks=manifest.get("clocks", []),
                       frames=manifest.get("frames", []), calibrations=manifest.get("calibrations", []),
                       transforms=manifest.get("transforms", []), alignments=manifest.get("alignments", []))
        raise ValueError("Capture has no capture_manifest.json")
