"""The `homebody` command line: run, check and provider-check; `homebody real2sim ...` runs REAL2SIM and
`homebody rollout ...` exports, imports, evaluates and views recorded rollouts."""
import argparse
import errno
import json
import math
import os
import shutil
import sys
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

from homebody.session.config import DEFAULT_ENV, MODELS
from homebody.session.settings import Settings
from homebody.vlm.catalog import PROVIDERS, make, model_label, requirement
from homebody.vlm.provider import ProviderError


def parser():
    value = argparse.ArgumentParser(prog="homebody", description="HomeBody SIM: a model drives a simulated G1 "
                                                                 "in a scanned environment")
    value.add_argument("command", nargs="?", choices=("run", "check", "provider-check", "serve", "connect", "monitor",
                                                         "sim-serve"),
                       default="run", help="serve: a harness consoles connect to ([server] settings); connect: a "
                                           "console and monitor here, on a harness at --url; sim-serve: this "
                                           "scene's physics as a simulation server ([sim_server] settings)")
    value.add_argument("--root", type=Path,
                       help="Release root containing assets and configs")
    value.add_argument("--env", help="The environment by name, a folder of assets/real2sim/ "
                                     f"(default: {DEFAULT_ENV})")
    value.add_argument("--scene", type=Path, help="An environment folder by path, instead of --env")
    value.add_argument("--settings", type=Path,
                       help="TOML of only the values to change, over configs/simulation.toml and the scene's")
    value.add_argument("--model", choices=tuple(MODELS), help="The model (default astra)")
    value.add_argument("--task", help="Explicit initial task; ordinary launch waits for input")
    value.add_argument("--place-object", action="append", default=[], metavar="NAME=X,Y,YAW",
                       help="Start a movable object upright at map X, Y (m) and YAW (rad) on the "
                            "surface there instead of its scanned position; repeatable")
    value.add_argument("--headless", action="store_true", help="Run one explicit task without UI")
    value.add_argument("--host", default="127.0.0.1", help="The dashboard's address (default 127.0.0.1)")
    value.add_argument("--port", type=int, default=8080, help="The dashboard's port (default 8080; the 3D scene uses the next one)")
    value.add_argument("--output", type=Path, help="Evidence root (default: ROOT/runs)")
    value.add_argument("--backend", choices=("sim", "real"), default="sim", help="serve: the harness's robot")
    value.add_argument("--url", default="ws://127.0.0.1:8765", help="connect: the harness server")
    value.add_argument("--token-file", help="connect: the server's token (default: server.token_file)")
    value.add_argument("--presets", type=Path, help="serve: TOML of scenarios consoles can run (server/presets.py)")
    value.add_argument("--config", type=Path, help="monitor: the harnesses, simulation servers and scenarios (TOML)")
    value.add_argument("--sim", help="serve: run on a simulation server at this endpoint (tcp://127.0.0.1:PORT)")
    value.add_argument("--sim-engine", choices=("warehouse", "homebody"), default="warehouse",
                       help="serve --sim: a warehouse-sim server (needs --sim-profile) or a homebody sim-serve one")
    value.add_argument("--sim-profile", type=Path, help="serve --sim: the profile.toml that server serves")
    value.add_argument("--sim-root", action="append", default=[], metavar="NAME=PATH",
                       help="serve --sim: a source root the profile names (e.g. HOMEBODY_EFFECTIVE_ROOT=...)")
    return value


def check(root: Path, scene: Path, settings) -> int:
    missing = []
    for package in ("numpy", "scipy", "mujoco", "torch", "PIL", "trimesh", "viser", "imageio_ffmpeg",
                    "holosoma_inference"):
        available = find_spec(package) is not None
        print(f"{'OK' if available else 'MISSING'} package {package}")
        if not available:
            missing.append(package)
    print(f"Scene package: {scene}")
    for directory, names in (
            (scene, ("scene.xml", "task_scene.json", "map.npz", "physics.json",
                     "semantics.json")),
            (settings.robot.directory(root), ("g1_29dof_with_hand.urdf", "calibration.json",
                                              "g1_grasp/g1_closd_grasp_deploy.xml")),
            (root, ("assets/models/amo_jit.pt", "assets/models/adapter_jit.pt",
                    "assets/models/adapter_norm_stats.pt"))):
        for name in names:
            available = (directory / name).is_file()
            print(f"{'OK' if available else 'MISSING'} {name}")
            if not available:
                missing.append(name)
    if settings.agent.provider not in PROVIDERS:
        print(f"MISSING provider {settings.agent.provider!r} for model {settings.agent.model!r} "
              f"in vlm/catalog.py PROVIDERS")
        missing.append(settings.agent.provider)
    executable, variable = requirement(settings.agent.provider) if settings.agent.provider in PROVIDERS else (None, None)
    if executable:
        found = shutil.which(executable)
        print(f"{'OK' if found else 'MISSING'} {executable} executable")
        if not found:
            missing.append(executable)
    if variable:
        found = bool(os.environ.get(variable))
        print(f"{'OK' if found else 'MISSING'} {variable} in the environment")
        if not found:
            missing.append(variable)
    if settings.monitor.enabled:
        from homebody.monitor.decision import KEY as DECISION_KEY
        found = bool(os.environ.get(DECISION_KEY))
        print(f"{'OK' if found else 'MISSING'} {DECISION_KEY} in the environment (the monitor's Decisions API)")
        if not found:
            missing.append(DECISION_KEY)
    print(f"Model: {settings.agent.model}: {model_label(settings.agent)} via {settings.agent.provider}; "
          f"login not tested (homebody provider-check --model {settings.agent.model} sends one request)")
    print("No simulator, web server, provider request, or robot service was started.")
    return int(bool(missing))


def provider_check(settings, output):
    path = output / ("provider-check-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                     + "-" + uuid.uuid4().hex[:6])
    provider = make(settings.agent)
    try:
        decision = provider.generate(
            'Return {"text":"connection checked","skill":"done","arguments_json":"{}"}. '
            'Do not use tools.', images=(), names=("done",), attempt=path,
            tick=lambda: time.sleep(0.02), cancelled=lambda: False)
        print(f"{settings.agent.provider}: {decision.text}")
        return 0
    except ProviderError as exc:
        print(f"{settings.agent.provider} check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"Provider evidence: {path}")


def live_monitor(settings, recorder, ui):
    """The live monitor when [monitor] enabled: its events go to the run's events.jsonl, its view to the console."""
    if not settings.monitor.enabled:
        return None
    from homebody.monitor.decision import OpenAiDecisions
    from homebody.monitor.monitor import Monitor
    from homebody.monitor.vocabulary import VocabularyModel
    return Monitor(settings.monitor, VocabularyModel(settings.monitor), OpenAiDecisions(settings.monitor),
                   sink=recorder.event, on_update=ui.monitor_update if ui is not None else None)


def rollouts(dashboard, settings, output):
    """The dashboard's Rollouts (export, import, evaluate) over OUTPUT's recorded sessions, and its Extract dataset
    (into OUTPUT/datasets)."""
    from homebody.rollout.app import RolloutsApp
    from homebody.rollout.cli import models
    from homebody.ui.dashboard import dataset
    dataset.attach(dashboard, output / "datasets")
    return RolloutsApp(dashboard, runs=output, imported=output / "imported", settings=settings,
                       models=lambda: models(settings))


def robot_profile(root, settings):
    """The arms' kinematics and the robot's calibration."""
    from homebody.helpers.kinematics import Arm
    profile = settings.robot.directory(root)
    arms = {side: Arm(profile / "g1_29dof_with_hand.urdf", side) for side in ("left", "right")}
    return arms, json.loads((profile / "calibration.json").read_text())


def run(args, settings, output):
    from homebody.backends.mujoco import MujocoBackend
    from homebody.session.recorder import Recorder
    from homebody.session.session import Session
    from homebody.ui.controls import Controls
    from homebody.ui.dashboard import Dashboard
    from homebody.ui.dashboard.places import Places
    from homebody.vlm.observations import semantic_prior

    recorder = Recorder(args.root, settings, output)
    recorder.event("scene_package", path=str(args.scene), object_spawns=args.spawns)
    backend = ui = None
    assembled = False
    controls = Controls()
    try:
        if not args.headless:
            task_scene = json.loads((args.scene / "task_scene.json").read_text())
            ui = Dashboard(controls, host=args.host, port=args.port,
                         initial_task=args.task or task_scene.get("default_task", ""),
                         scene_name=task_scene.get("name", "Simulation"),
                         model_name=model_label(settings.agent),
                         topdown_settings=settings.topdown, render_in_background=True,
                         places=Places.from_scene(args.scene, task_scene.get("name", "")),
                         monitor=settings.monitor if settings.monitor.enabled else None, timing=settings.dashboard)
        backend = MujocoBackend(args.root, settings, scene=args.scene, spawns=args.spawns)
        scene_update = None
        if ui is not None:
            from homebody.ui.scene import SceneView
            ui.scene_view = SceneView(ui.server, backend)
            scene_update = ui.scene_view.update
        arms, calibration = robot_profile(args.root, settings)
        if ui is not None:
            ui.arms = arms
        provider = make(settings.agent)
        monitor = live_monitor(settings, recorder, ui)
        session = Session(backend, provider, settings, recorder, controls,
                          arms=arms, jaw_samples=calibration["jaw_samples"], ui=ui,
                          observer_frame=backend.observer_frame if args.headless else None,
                          shoulder_frame=backend.shoulder_frame if args.headless else None,
                          scene_update=scene_update, real_time=ui is not None,
                          semantic_context=semantic_prior(args.scene), monitor=monitor)
        if ui is not None:
            ui.session = session  # read on the session thread only, for the plan, task and hands
            ui.extras["controller"] = lambda: {"state": "in_process", "detail": "The robot's controller runs in this process"}
            rollouts(ui, settings, output)
        assembled = True
        print(f"Evidence: {recorder.path}")
        if args.headless:
            outcome = session.run_task(args.task)
            print(json.dumps(outcome))
            return 0 if outcome["status"] == "model_done" else 1
        ui.update("Ready")
        if args.task:
            controls.submit(args.task)
        print(f"Dashboard: {ui.url} (3D scene: port {ui.scene_port})")
        session.run()
        return 0
    except KeyboardInterrupt:
        controls.cancel.set()
        recorder.event("interrupted", reason="KeyboardInterrupt")
        if args.task and not assembled:
            recorder.setup_failure(args.task, "Interrupted while initializing simulation")
        return 130
    except Exception as exc:  # noqa: BLE001 -- record every failure in the run's evidence
        recorder.event("session_error", error=f"{type(exc).__name__}: {exc}")
        if args.task and not assembled:
            recorder.setup_failure(args.task, f"{type(exc).__name__}: {exc}")
        hint = " (the port is in use; pass --port)" if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE else ""
        print(f"Session failed: {exc}{hint}\nEvidence: {recorder.path}", file=sys.stderr)
        return 1
    finally:
        if backend is not None:
            backend.close()
        if ui is not None:
            ui.close()
        recorder.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["rollout"]:
        from homebody.rollout.cli import main as rollout
        return rollout(argv[1:])
    if argv[:1] == ["real2sim"]:
        from homebody.real2sim.cli import main as real2sim
        return real2sim(argv[1:])
    value = parser()
    args = value.parse_args(argv)
    if args.root is None:
        candidates = (Path.cwd(), Path(__file__).resolve().parents[2])
        args.root = next((path for path in candidates
                          if (path / "assets").is_dir() and
                          (path / "configs/simulation.toml").is_file()), None)
        if args.root is None:
            value.error("Run from the release folder or pass --root PATH containing assets/configs")
    args.root = args.root.resolve()
    environments = sorted(path.name for path in (args.root / "assets/real2sim").iterdir() if path.is_dir())
    if args.env and args.env not in environments:
        value.error(f"--env must be one of {', '.join(environments)}")
    if args.env and args.scene:
        value.error("Pass --env or --scene, not both")
    args.scene = (args.scene or args.root / "assets/real2sim" / (args.env or DEFAULT_ENV)).resolve()
    if args.command == "run" and args.headless and not (args.task and args.task.strip()):
        value.error("--headless requires an explicit --task")
    args.spawns = {}
    for item in args.place_object:
        name, _, numbers = item.partition("=")
        try:
            args.spawns[name] = [float(v) for v in numbers.split(",")]
        except ValueError:
            args.spawns[name] = []
        if not name or len(args.spawns[name]) != 3 or not all(map(math.isfinite, args.spawns[name])):
            value.error(f"--place-object takes NAME=X,Y,YAW in map metres and radians, not {item!r}")
    if not 1 <= args.port <= 65535:
        value.error("--port must be between 1 and 65535")
    if args.spawns and (args.scene / "task_scene.json").is_file():
        movable = json.loads((args.scene / "task_scene.json").read_text()).get("movable_entities", ())
        unknown = sorted(set(args.spawns) - set(movable))
        if unknown:
            value.error(f"--place-object: no movable object {', '.join(unknown)}; "
                        f"this environment's are {', '.join(movable)}")
    try:
        settings = Settings.load(args.root / "configs/simulation.toml", scene=args.scene, override=args.settings)
    except (ValueError, TypeError, KeyError, OSError) as error:
        value.error(f"settings: {error}")
    if args.model:
        settings = replace(settings, agent=replace(settings.agent, model=args.model))
    output = (args.output or args.root / "runs").resolve()
    if args.command == "check":
        return check(args.root, args.scene, settings)
    if args.command == "provider-check":
        return provider_check(settings, output)
    if args.command == "sim-serve":
        from homebody.simserver.serve import sim_serve
        return sim_serve(args, settings)
    if args.command in ("serve", "connect", "monitor"):
        from homebody.server import commands
        return getattr(commands, args.command)(args, settings, output)
    return run(args, settings, output)


if __name__ == "__main__":
    raise SystemExit(main())
