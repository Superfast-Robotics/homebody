"""Reconstruct a scene in a work directory and export the validated package."""
from __future__ import annotations

import json
import math
import runpy
import shlex
import subprocess
import tempfile
from pathlib import Path

from homebody.capture import Capture, Selection
from homebody.capture_package import accepted
from homebody.capture_package import package as package_capture
from homebody.scene_package import package
from homebody.session.settings import Settings

DEFAULT_AGENT = "codex exec --ephemeral --skip-git-repo-check --sandbox workspace-write -"


def reconstruct(capture, output, task, *, root, work=None, agent_command=DEFAULT_AGENT,
                selection=None, resolution=.05, height_band=(.05, 1.6), timeout=3600):
    """Run one agent invocation, then build and validate without a scene fallback."""
    capture, output, root = Path(capture).resolve(), Path(output).resolve(), Path(root).resolve()
    if not task.strip():
        raise ValueError("Reconstruction requires an explicit task scope")
    if output.exists() or output.is_relative_to(capture):
        raise ValueError("Output must be a new directory outside the capture")
    if (not math.isfinite(resolution) or resolution <= 0 or len(height_band) != 2
            or not all(math.isfinite(x) for x in height_band)
            or height_band[0] >= height_band[1] or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Invalid map resolution, height band or timeout")
    if not shlex.split(agent_command):
        raise ValueError("Agent command is empty")
    manifest = capture / "capture_manifest.json"
    if not manifest.is_file():
        raise ValueError("Reconstruction requires a closed capture, not a raw SIM session")
    accepted(capture)
    evidence = Capture.open(capture)
    selected = evidence.select(selection or Selection())
    kinds = {s.kind for s in selected if s.status == "recorded" and s.assets}
    if not kinds.intersection({"lidar", "depth"}):
        raise ValueError("No recorded metric geometry stream was selected")
    for stream in selected:
        if stream.status == "recorded":
            for asset in stream.assets:
                evidence.resolve(asset, verify=True)
    if work is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f".{output.name}-work-", dir=output.parent))
    else:
        work = Path(work).resolve()
        if work.is_relative_to(capture) or capture.is_relative_to(work):
            raise ValueError("Work directory must be separate from the capture")
        work.mkdir(parents=True, exist_ok=False)
    if output.is_relative_to(work) or work.is_relative_to(output):
        raise ValueError("Output must be separate from the work directory")
    capture = package_capture(capture, work / "capture")
    print(f"Work directory: {work}", flush=True)
    evidence = Capture.open(capture)
    (work / "streams.json").write_text(json.dumps(evidence.context(selection), indent=2))
    (work / "scene").mkdir()
    prompt = Path(__file__).with_name("agent.md").read_text()
    prompt += (f"\nCapture: {capture}\nRequested scope: {task}\nWorking directory: {work}\n"
               f"Release root: {root}\nSkill: {root}/.agents/skills/homebody-real2sim/SKILL.md\n"
               f"Workflow guide: {root}/guides/real2sim.md\n")
    (work / "request.txt").write_text(prompt)
    command = shlex.split(agent_command)
    with (work / "agent.log").open("w") as log:
        subprocess.run(command, input=prompt, text=True, cwd=work, stdout=log,
                       stderr=subprocess.STDOUT, timeout=timeout, check=True)
    audit = json.loads((work / "audit.json").read_text())
    if audit.get("ready") is not True or not isinstance(audit.get("checks"), list) or not isinstance(
            audit.get("limitations"), list):
        raise ValueError(f"Reconstruction is incomplete; inspect {work}")
    scene = work / "scene"
    # Reuse the release builders; the agent supplies the source geometry only.
    runpy.run_path(str(root / "tools/build_kitchen.py"))["build"](scene)
    runpy.run_path(str(root / "tools/build_scene_map.py"))["build"](
        scene, resolution, height_band)

    def validate(directory):
        from homebody.backends.scene import load
        load(root, Settings.load(None, scene=directory), directory)

    package(scene, output, validate=validate)
    return work
