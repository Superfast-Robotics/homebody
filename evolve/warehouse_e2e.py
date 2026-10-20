"""End-to-end sessions of the current checkout (its skills and settings) on task scenes, outside any archive:
REPEATS sessions per task on each evaluation host, scored from physics. Run from the repository root:

    python3 -m evolve.warehouse_e2e [--repeats 2] [--settings episode_settings_dex1.toml] [--id demo-e2e-dex1]
        [--task assets/real2sim/warehouse_picking::configs/warehouse_picking_three.json ...]

The default tasks are the two warehouse scenes. SETTINGS is a settings file in evolve/ (episode_settings.toml for
the Dex3 robot); the profile services (the Dex1 planning servers) are started on every host first. Results go to
the evolve work folder's results/<id>/<host>, with summary.json beside them."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import score
from .loop import Loop
from .remote import suite_tasks

TASKS = ("assets/real2sim/warehouse_picking::configs/warehouse_picking_three.json",
         "assets/real2sim/warehouse_shelf::configs/warehouse_shelf_four.json")
SERVICES = (("/lambda/nfs/group-volume/workcell-sim/homebody-baseline/curobo_dex1/curobo_dex1_server.sh start "
             "> /dev/null 2>&1 < /dev/null"),)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=2, help="sessions per task on each host")
    parser.add_argument("--settings", default="episode_settings_dex1.toml", help="settings file in evolve/")
    parser.add_argument("--id", default="demo-e2e-dex1", help="candidate name for this checkout on the remote")
    parser.add_argument("--task", action="append", help="scene::targets (default: the two warehouse scenes)")
    args = parser.parse_args(argv)
    loop = Loop()
    for host in loop.remotes:
        for command in SERVICES if "dex1" in args.settings else ():
            host.ssh(command, timeout=600)
    loop.remotes[0].push_candidate(args.id, Path.cwd())
    tasks = suite_tasks(loop.remotes[0], args.task or list(TASKS))
    model = loop.cfg["agent"]["model"]

    def on(host):
        results = host.run_suite(args.id, tasks, args.repeats, f"e2e_{host.host}", model, args.settings)
        local = loop.work / "results" / args.id / host.host
        host.collect(results, local)
        return score.read_split(local)

    with ThreadPoolExecutor(len(loop.remotes)) as pool:
        runs = [run for host_runs in pool.map(on, loop.remotes) for run in host_runs]
    summary = score.summary(runs)
    loop.log(f"e2e {args.id}: {len(runs)} sessions, score {summary['score']:.3f}, success {summary['success_rate']:.2f}, "
             f"credits per session {summary['credits']}, stages {sorted(r.stage for r in runs)}")
    (loop.work / "results" / args.id / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
