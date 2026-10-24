"""homebody sim-serve --scene PACKAGE [--settings TOML]: the scene's physics as a simulation server (server.py) until
interrupted or stopped (SIGTERM). Prints one JSON line once it serves: endpoint, stream, scene and its identity."""
import json
import signal
import sys
import threading


def sim_serve(args, settings) -> int:
    from homebody.backends.mujoco import MujocoBackend
    from homebody.simserver.render import RenderProcess
    from homebody.simserver.server import SimServer

    if not (args.scene / "task_scene.json").is_file():
        print(f"sim-serve needs a scene package (--scene or --env): {args.scene} has no task_scene.json", file=sys.stderr)
        return 2
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    print(f"Loading {args.scene.name}: physics, then the render process…", flush=True)
    backend = MujocoBackend(args.root, settings, scene=args.scene, render=False)
    renderer = RenderProcess(args.root, settings, args.scene)
    server = SimServer(backend, renderer, settings, args.scene).start()
    described = server.rpc_describe()
    print(json.dumps({"endpoint": settings.sim_server.endpoint, "camera_stream": settings.sim_server.stream,
                      "scene": described["scene"], "scene_identity": described["scene_identity"],
                      "profile_id": described["profile"]["id"]}), flush=True)
    try:
        stopped.wait()
    finally:
        server.close()
        backend.close()
    return 0
