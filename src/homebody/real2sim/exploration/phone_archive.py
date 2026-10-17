"""Copy finalized native phone segments for a closed exploration interval without stopping capture.

Segments are selected by receiver clock with one minute of padding, not by exposure time.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def archive(run: Path, root: Path):
    interval = json.loads((run / "meta/record_interval.json").read_text())
    start, end = interval["start_s"], interval["end_s"]
    if not isinstance(end, (int, float)) or end < start:
        raise ValueError("a closed recording interval is required")
    rows, pending = [], []
    destination = run / "phone/native"
    destination.mkdir(parents=True, exist_ok=True)
    for capture in sorted(root.glob("capture_*")):
        meta_path, frame_path = capture / "recording.json", capture / "frames.jsonl"
        if not meta_path.is_file() or not frame_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        with frame_path.open() as frames:
            first = frames.readline()
        if not first.strip():
            continue
        first_s = int(json.loads(first)["capture_realtime_ns"]) / 1e9
        last_s = int(meta.get("last_decode_realtime_ns", 0)) / 1e9
        if last_s < start - 60 or first_s > end + 60:
            continue
        out = destination / capture.name
        out.mkdir(exist_ok=True)
        finalized = set()
        last_finalized_end = 0.0
        listing = capture / "segments.csv"
        if listing.is_file():
            with listing.open(newline="") as stream:
                for entry in csv.reader(stream):
                    if len(entry) != 3:
                        continue  # the writer may be appending its next row
                    name, begin, finish = entry
                    if Path(name).name != name or not name.startswith("video_"):
                        raise ValueError("invalid segment path in capture index")
                    begin, finish = float(begin), float(finish)
                    finalized.add(name)
                    last_finalized_end = max(last_finalized_end, finish)
                    if first_s + finish < start - 60 or first_s + begin > end + 60:
                        continue
                    source, target = capture / name, out / name
                    if source.is_symlink():
                        raise ValueError("refusing symlinked video segment")
                    with source.open("rb") as original:
                        digest = hashlib.file_digest(original, "sha256").hexdigest()
                    if target.exists():
                        with target.open("rb") as existing:
                            if hashlib.file_digest(existing, "sha256").hexdigest() != digest:
                                raise ValueError(f"existing segment differs: {target}")
                    else:
                        temp = target.with_suffix(".copying")
                        shutil.copyfile(source, temp)
                        temp.replace(target)
                    rows.append({"file": str(target.relative_to(run)), "sha256": digest,
                                 "pts_start_s": begin, "pts_end_s": finish,
                                 "first_decode_s": first_s})
        unclosed = [p for p in capture.glob("video_*.mkv") if p.name not in finalized]
        # report unfinalized files unless one segment past the interval is still being written
        if len(unclosed) != 1 or first_s + last_finalized_end <= end + 60:
            pending.extend(str(p) for p in unclosed if p.stat().st_mtime >= start - 60)
        (out / "source_recording_snapshot.json").write_text(json.dumps(meta, indent=2) + "\n")
        for name in ('segments.csv', 'frames.jsonl', 'ffmpeg.log'):
            source = capture / name
            if source.is_file():
                shutil.copyfile(source, out / ('source_' + name))
    report = {"schema": 1, "interval": interval, "segments": rows,
              "coverage_verified": False,
              "pending_unclosed_segments": pending, "selection_padding_s": 60,
              "timestamp_kind": "approximate_receiver_clock_not_exposure_time",
              "complete": bool(rows) and not pending}
    (destination / "archive.json").write_text(json.dumps(report, indent=2) + "\n")
    context_path = run / "meta/phone_context.json"
    if context_path.exists():
        context = json.loads(context_path.read_text())
        context.update(recording_included=bool(rows), native_archive="phone/native/archive.json")
        context_path.write_text(json.dumps(context, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--record-root", required=True, type=Path)
    args = parser.parse_args()
    report = archive(args.run.resolve(), args.record_root.expanduser().resolve())
    print(f"Native phone archive: {len(report['segments'])} closed segments; "
          f"{len(report['pending_unclosed_segments'])} unfinalized. See phone/native/archive.json")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
