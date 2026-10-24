"""Rollout bundles: export a recorded session, import it elsewhere, and refuse what a bundle must not carry."""
import io
import json
import tarfile

import pytest

from homebody.rollout.bundle import BundleError, describe, export, import_


def session(tmp_path, *, monitor=True, physics=True, name="session-20261023T000000Z-abcd1234"):
    """A small recorded session: one task with two frames, and optionally monitor readings and physics."""
    root = tmp_path / name
    (root / "frames/001-00000001").mkdir(parents=True)
    (root / "frames/001-00000002").mkdir()
    for frame in ("001-00000001", "001-00000002"):
        (root / "frames" / frame / "rgb.png").write_bytes(b"png")
        (root / "frames" / frame / "sensors.npz").write_bytes(b"x" * 64)
        (root / "frames" / frame / "depth_preview.png").write_bytes(b"d")
    (root / "provenance.json").write_text(json.dumps({"schema_version": 1, "backend": "mujoco"}))
    events = [{"kind": "scene_package", "path": "/scenes/warehouse_cart"},
              {"kind": "task_started", "task": "pick the jar", "epoch": 1},
              {"kind": "frame", "epoch": 1, "simulation_time": 0.5, "path": "frames/001-00000001"},
              {"kind": "frame", "epoch": 1, "simulation_time": 1.0, "path": "frames/001-00000002"},
              {"kind": "task_finished", "status": "model_done", "steps": 3}]
    (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    task = root / "tasks/task-000"
    task.mkdir(parents=True)
    (task / "task.json").write_text(json.dumps({"task": "pick the jar", "epoch": 1}))
    (task / "outcome.json").write_text(json.dumps({"status": "model_done", "steps": 3}))
    if physics:
        (task / "physics.jsonl").write_text("{}\n")
    if monitor:
        (task / "monitor").mkdir()
        (task / "monitor/readings.jsonl").write_text("{}\n")
    return root


def test_an_exported_rollout_imports_unchanged_with_a_manifest_that_says_what_it_holds(tmp_path):
    source = session(tmp_path)
    archive = export(source, tmp_path / "out", light=False)
    imported = import_(archive, tmp_path / "imported", max_mb=10)
    assert (imported / "events.jsonl").read_bytes() == (source / "events.jsonl").read_bytes()
    assert (imported / "frames/001-00000002/sensors.npz").is_file()
    manifest = describe(imported)
    assert manifest["source"] == "sim" and manifest["scene"] == "warehouse_cart"
    assert manifest["has_monitor_logs"] and manifest["has_physics"] and manifest["frames"] == 2
    assert manifest["tasks"] == [{"id": "task-000", "task": "pick the jar", "status": "model_done", "steps": 3}]


def test_a_light_export_leaves_the_raw_sensors_out_and_a_session_describes_itself_unexported(tmp_path):
    source = session(tmp_path, monitor=False, physics=False)
    assert describe(source)["has_monitor_logs"] is False and describe(source)["has_physics"] is False
    imported = import_(export(source, tmp_path / "out", light=True), tmp_path / "imported", max_mb=10)
    assert not (imported / "frames/001-00000001/sensors.npz").exists()
    assert not (imported / "frames/001-00000001/depth_preview.png").exists()
    assert (imported / "frames/001-00000001/rgb.png").is_file()


def test_an_import_refuses_a_second_copy_unless_asked_to_replace_it(tmp_path):
    archive = export(session(tmp_path), tmp_path / "out")
    import_(archive, tmp_path / "imported", max_mb=10)
    with pytest.raises(BundleError, match="already imported"):
        import_(archive, tmp_path / "imported", max_mb=10)
    assert import_(archive, tmp_path / "imported", max_mb=10, replace=True).is_dir()


def rewrite(archive, target, change):
    """Copy ARCHIVE to TARGET with CHANGE(members) -> [(TarInfo, bytes | None)] applied."""
    with tarfile.open(archive) as source:
        members = [(m, source.extractfile(m).read() if m.isfile() else None)  # type: ignore[union-attr]
                   for m in source.getmembers()]
    with tarfile.open(target, "w:gz") as out:
        for member, data in change(members):
            out.addfile(member, io.BytesIO(data) if data is not None else None)
    return target


def test_an_import_refuses_a_file_whose_bytes_differ_from_the_manifest(tmp_path):
    archive = export(session(tmp_path), tmp_path / "out")

    def tamper(members):
        for member, data in members:
            if member.name.endswith("outcome.json"):
                data = b'{"status": "model_done", "steps": 99}'
                member.size = len(data)
            yield member, data
    with pytest.raises(BundleError, match="hash"):
        import_(rewrite(archive, tmp_path / "bad.tar.gz", tamper), tmp_path / "imported", max_mb=10)
    assert not any((tmp_path / "imported").glob("*/events.jsonl"))


def test_an_import_refuses_paths_outside_the_rollout_and_files_the_manifest_does_not_list(tmp_path):
    archive = export(session(tmp_path), tmp_path / "out")
    root = archive.name.removesuffix(".tar.gz")

    def escape(members):
        yield from members
        extra = tarfile.TarInfo(f"{root}/../../evil.txt")
        extra.size = 1
        yield extra, b"x"

    def unlisted(members):
        yield from members
        extra = tarfile.TarInfo(f"{root}/tasks/task-000/extra.py")
        extra.size = 1
        yield extra, b"x"
    for name, change in (("escape", escape), ("unlisted", unlisted)):
        with pytest.raises(BundleError):
            import_(rewrite(archive, tmp_path / f"{name}.tar.gz", change), tmp_path / "imported", max_mb=10)
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path.parent / "evil.txt").exists()


def test_an_import_refuses_a_bundle_larger_than_its_cap(tmp_path):
    archive = export(session(tmp_path), tmp_path / "out", light=False)
    with pytest.raises(BundleError, match="larger than"):
        import_(archive, tmp_path / "imported", max_mb=0.0001)


def test_a_task_export_keeps_only_that_task_and_its_frames(tmp_path):
    source = session(tmp_path)
    (source / "frames/002-00000003").mkdir()
    (source / "frames/002-00000003/rgb.png").write_bytes(b"png")
    (source / "tasks/task-001").mkdir()
    (source / "tasks/task-001/task.json").write_text(json.dumps({"task": "another", "epoch": 2}))
    with (source / "events.jsonl").open("a") as events:
        events.write(json.dumps({"kind": "task_started", "task": "another", "epoch": 2}) + "\n")
        events.write(json.dumps({"kind": "frame", "epoch": 2, "path": "frames/002-00000003"}) + "\n")
    imported = import_(export(source, tmp_path / "out", task="task-000"), tmp_path / "imported", max_mb=10)
    assert imported.name.endswith("task-000")
    assert not (imported / "tasks/task-001").exists() and not (imported / "frames/002-00000003").exists()
    kinds = [json.loads(line) for line in (imported / "events.jsonl").read_text().splitlines()]
    assert all(event.get("epoch") in (None, 1) for event in kinds) and len(kinds) == 5


def test_a_manifest_missing_what_it_must_say_is_refused_and_never_breaks_describe(tmp_path):
    archive = export(session(tmp_path), tmp_path / "out")

    def gutted(members):
        for member, data in members:
            if member.name.endswith("rollout.json"):
                manifest = json.loads(data)
                del manifest["tasks"]
                data = json.dumps(manifest).encode()
                member.size = len(data)
            yield member, data
    with pytest.raises(BundleError, match="manifest"):
        import_(rewrite(archive, tmp_path / "gutted.tar.gz", gutted), tmp_path / "imported", max_mb=10)
    folder = session(tmp_path / "local")
    (folder / "rollout.json").write_text('{"schema": 1}')
    assert describe(folder)["tasks"][0]["id"] == "task-000"


def test_recorded_paths_that_leave_the_rollout_are_not_followed(tmp_path):
    from homebody.rollout.paths import inside
    root = session(tmp_path)
    assert inside(root, "frames/001-00000001") == root / "frames/001-00000001"
    for bad in ("/etc", "../outside", "frames/../../x", "", None, 3, "a\\..\\b"):
        assert inside(root, bad) is None
