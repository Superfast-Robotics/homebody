"""`homebody serve` (a harness that consoles connect to) and `homebody connect` (a console and monitor on one)."""
import json
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

SWITCH_POLL_S = 0.5  # how often a harness without a working simulation server checks for an attach


def scene_settings(args, scene: Path):
    """The settings of SCENE: configs/simulation.toml, the scene's settings.toml, --settings and --model."""
    from homebody.session.settings import Settings
    settings = Settings.load(args.root / "configs/simulation.toml", scene=scene, override=args.settings)
    return replace(settings, agent=replace(settings.agent, model=args.model)) if args.model else settings


def serve(args, settings, output) -> int:
    """Run the sim harness behind the server until interrupted. Settings: [server] (address, token, disconnect).
    A console's preset in another scene ends this scene's session and builds that scene's, in the same server."""
    if args.backend == "real":
        print("The real-G1 backend (backends/real.py) comes with the REAL release; serve --backend sim for now.",
              file=sys.stderr)
        return 2
    from homebody.server import presets as scenarios
    from homebody.server.links import UiLink
    from homebody.server.serve import HarnessServer, Hub, load_token
    from homebody.ui.controls import Controls

    server_settings = settings.server
    if server_settings.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: server.host {server_settings.host} is reachable from the network and the connection is not "
              "encrypted; the token and camera frames travel in clear. Prefer 127.0.0.1 and an SSH tunnel.",
              file=sys.stderr)
    try:
        offered = scenarios.load(args.presets, args.scene)
    except (ValueError, OSError) as error:
        print(f"presets: {error}", file=sys.stderr)
        return 2
    token = load_token(server_settings.token_file)
    controls, hub = Controls(), Hub()
    ui = UiLink(hub, quality=server_settings.jpeg_quality, view_hz=server_settings.view_hz)
    monitor = monitor_link(hub, settings, server_settings.jpeg_quality, output)
    server = HarnessServer(settings, controls, ui, monitor, session_dir=None, token=token, hub=hub,
                           presets=offered)
    server.start()
    print(f"Harness: ws://{server_settings.host}:{server.port}/stream "
          f"(clients need the token in {Path(server_settings.token_file).expanduser()}); "
          f"{len(offered)} preset{'s' if len(offered) != 1 else ''}", flush=True)
    target, task = initial_target(args), args.task
    signal.signal(signal.SIGTERM, interrupted)  # a stop signal cleans up like Ctrl-C (a simulation server is released)
    try:
        while target is not None:
            try:
                run_scene(args, target, task, output, controls=controls, ui=ui, monitor=monitor, server=server)
            except Exception as error:
                if not is_remote(target):
                    raise
                ui.update("Error", f"The simulation server {target.endpoint} failed: {error}"[:600])
                print(f"Simulation server {target.endpoint} failed: {error}; waiting to be attached to another "
                      "(attach_sim) or stopped", file=sys.stderr, flush=True)
                wait_for_switch(server)
            target, task = server.take_switch()
            controls.closing.clear()
            controls.cancel.clear()
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        controls.closing.set()
        server.stop()


def monitor_link(hub, settings, quality: int, output):
    """The live monitor's link to the commanding client; None on a branch without the live monitor."""
    try:
        from homebody.server.monitor_link import MonitorLink
    except ImportError:
        return None
    return MonitorLink(hub, settings.monitor, quality=quality, root=output)


def is_remote(target) -> bool:
    from homebody.backends.warehouse import SimTarget
    return isinstance(target, SimTarget)


def wait_for_switch(server, poll_s: float = SWITCH_POLL_S):
    """Until a client attaches this harness to another simulation server (or the harness is stopped)."""
    while server.switch is None:
        time.sleep(poll_s)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def connect_sim(target, root, settings):
    """The backend on TARGET's server: warehouse-sim's (backends/warehouse) or HomeBody's own (simserver/client)."""
    if target.engine == "homebody":
        from homebody.simserver.client import connect
    else:
        from homebody.backends.warehouse import connect
    return connect(target, root, settings)


def initial_target(args):
    """The first session's target: a simulation server (--sim; for warehouse-sim also --sim-profile, --sim-root; for
    a HomeBody one --sim-engine homebody) or the local scene."""
    if not args.sim:
        return args.scene
    from homebody.backends.warehouse import SimTarget
    if args.sim_engine == "homebody":
        return SimTarget(args.sim, None, args.scene, engine="homebody")
    if not args.sim_profile:
        raise SystemExit("--sim needs --sim-profile: the profile.toml that server serves (on the shared volume)")
    roots = dict(item.split("=", 1) for item in args.sim_root)
    return SimTarget(args.sim, args.sim_profile.resolve(), args.scene, roots)


def run_scene(args, target, task: str | None, output, *, controls, ui, monitor, server):
    """One session behind SERVER, until a stop of the server or a switch: TARGET is a local scene folder (physics in
    this process) or a SimTarget (physics and the controller on a warehouse-sim server)."""
    from homebody.__main__ import robot_profile
    from homebody.backends.mujoco import MujocoBackend
    from homebody.backends.warehouse import SimTarget
    from homebody.session.recorder import Recorder
    from homebody.session.session import Session
    from homebody.vlm.catalog import make
    from homebody.vlm.observations import semantic_prior
    remote = isinstance(target, SimTarget)
    scene = target.scene if remote else target
    settings = scene_settings(args, scene)
    recorder = Recorder(args.root, settings, output)
    recorder.event("scene_package", path=str(scene),
                   object_spawns=args.spawns if scene == args.scene and not remote else {},
                   **({"simulation_server": target.endpoint, "engine": target.engine,
                       "profile": str(target.profile) if target.profile else None} if remote else {}))
    backend = None
    try:
        if remote:
            ui.update("Loading", f"Attaching to the simulation server at {target.endpoint}…")
            backend = connect_sim(target, args.root, settings)
        else:
            ui.update("Loading", f"Loading the {scene.name} simulation…")
            backend = MujocoBackend(args.root, settings, scene=scene, spawns=args.spawns if scene == args.scene else {})
        # with a simulation server, the monitor app takes its camera images and frames from that server's stream
        ui.send_views = not remote
        if monitor is not None:
            monitor.send_frames = not remote
        arms, calibration = robot_profile(args.root, settings)
        session = Session(backend, make(settings.agent), settings, recorder, controls, arms=arms,
                          jaw_samples=calibration["jaw_samples"], ui=ui, real_time=True,
                          semantic_context=semantic_prior(scene), **({"monitor": monitor} if monitor else {}))
        ui.session = session  # read on the session thread only: its task, plan and hands go with each status
        if monitor is not None:
            monitor.root, monitor.sink = recorder.path, recorder.event
        server.attach(recorder.path, scene, recorder.event, sim=backend if remote else None)
        recorder.event("harness_server", host=settings.server.host, port=server.port)
        task_scene = json.loads((scene / "task_scene.json").read_text())
        ui.update("Ready", task_scene.get("default_task", ""))
        print(f"Evidence: {recorder.path} ({scene.name}{' on ' + target.endpoint if remote else ''})", flush=True)
        if task:
            controls.submit(task)
        session.run()
    except KeyboardInterrupt:
        controls.cancel.set()
        recorder.event("interrupted", reason="KeyboardInterrupt")
        raise
    finally:
        if backend is not None:
            backend.close()
        server.sink = lambda kind, **fields: None  # the session's evidence closes; later clients are not recorded
        recorder.close()


def connect(args, settings, output) -> int:
    """A dashboard (and, with [monitor] enabled, the monitor) on this machine, driving a harness server."""
    from homebody.__main__ import rollouts
    from homebody.monitor.decision import OpenAiDecisions
    from homebody.monitor.monitor import Monitor
    from homebody.monitor.vocabulary import VocabularyModel
    from homebody.server.client import HarnessClient, RemoteControls
    from homebody.ui.dashboard import Dashboard

    token_file = Path(args.token_file or settings.server.token_file).expanduser()
    if not token_file.is_file():
        print(f"No server token at {token_file}; copy the server's token file here (or pass --token-file).",
              file=sys.stderr)
        return 2
    client = HarnessClient(args.url, token_file.read_text().strip(), folder=output / "remote")
    controls = RemoteControls(client)
    console = Dashboard(controls, host=args.host, port=args.port, scene_name="Harness",
                        monitor=settings.monitor if settings.monitor.enabled else None, timing=settings.dashboard)
    console.extras["scenarios"] = lambda: client.presets
    console.extras["robot_link"] = lambda: {"state": "lost" if client.closed.is_set() else "connected",
                                            "detail": f"Harness at {args.url}"}
    console.extras["controller"] = lambda: {"state": "unknown", "checked": None,
                                            "detail": "This dashboard does not see the harness's controller"}
    console.handlers["run_scenario"] = lambda request: run_preset(controls, request)
    rollouts(console, settings, output)
    client.console = console
    if settings.monitor.enabled:
        client.monitor = Monitor(settings.monitor, VocabularyModel(settings.monitor),
                                 OpenAiDecisions(settings.monitor), sink=client.monitor_event,
                                 on_update=client.monitor_updated)
    try:
        client.connect()
        print(f"Dashboard: {console.url} -> {args.url} "
              f"({client.role}{'' if client.monitor else ', no monitor: [monitor] enabled = false'})")
        client.closed.wait()
        print("The harness closed the connection.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    except (OSError, ConnectionError) as error:
        print(f"Could not connect to {args.url}: {error}", file=sys.stderr)
        return 1
    finally:
        client.close()
        if client.monitor is not None:
            client.monitor.end_task()
        console.close()


def run_preset(controls, request: dict) -> dict:
    """The dashboard's Run scenario on a harness's own presets."""
    name, text = request.get("name"), request.get("text", "")
    if not isinstance(name, str) or not name:
        return {"ok": False, "message": "Choose a scenario from the list."}
    controls.preset(name, text if isinstance(text, str) else "")
    return {"ok": True, "message": f"Starting {name}…"}


def monitor(args, settings, output) -> int:
    """The monitor app: a dashboard and the monitor here, over the harnesses and simulation servers --config names."""
    from websockets.exceptions import WebSocketException

    from homebody.__main__ import rollouts
    from homebody.monitor.decision import OpenAiDecisions
    from homebody.monitor.monitor import Monitor
    from homebody.monitor.vocabulary import VocabularyModel
    from homebody.server.monitor_app import MonitorApp, MonitorControls
    from homebody.ui.dashboard import Dashboard

    if not args.config:
        print("homebody monitor needs --config FILE (the harnesses, simulation servers and scenarios)", file=sys.stderr)
        return 2
    controls = MonitorControls(None)
    console = Dashboard(controls, host=args.host, port=args.port, scene_name="Monitor app",
                        monitor=settings.monitor if settings.monitor.enabled else None, timing=settings.dashboard)
    watch = None
    if settings.monitor.enabled:
        watch = Monitor(settings.monitor, VocabularyModel(settings.monitor), OpenAiDecisions(settings.monitor),
                        on_update=console.monitor_update)
    try:
        app = MonitorApp(args.config, console=console, monitor=watch)
    except ValueError as error:
        print(f"config: {error}", file=sys.stderr)
        console.close()
        return 2
    controls.app = app
    app.attach(console)
    rollouts(console, settings, output)
    app.watcher.start()
    app.video.start()
    first = next(iter(app.config.harnesses), None)
    if first is not None:
        try:
            app.use_harness(first)
        except (OSError, ConnectionError, ValueError, WebSocketException) as error:  # a tunnel with no harness behind
            app.say("Ready", f"Harness {first} is not reachable yet: {type(error).__name__}: {error}")
    print(f"Monitor app: {console.url} "
          f"({len(app.config.harnesses)} harnesses, {len(app.config.sims)} simulation servers, "
          f"{len(app.config.scenarios)} scenarios from {args.config})", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        app.watcher.stop()
        app.video.stop()
        if app.client is not None:
            app.client.close()
        if watch is not None:
            watch.end_task()
        console.close()
