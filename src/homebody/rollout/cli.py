"""`homebody rollout ...`: export, import, evaluate and view recorded rollouts.

    homebody rollout export runs/session-*  [--task task-000] [--full] [--out DIR]
    homebody rollout import bundle.tar.gz   [--into runs/imported] [--replace]
    homebody rollout evaluate ROLLOUT       [--task task-000] [--dry-run] [--settings F.toml]
    homebody rollout view [ROLLOUT]         [--task task-000] [--runs runs] [--port 8090] [--settings F.toml]

`evaluate` runs the monitor over the recorded frames (monitor/offline.py) after printing its cost; it needs the
monitor's models: the vocabulary server and OPENAI_DECISIONS_API_KEY. `view` opens the dashboard on recorded and
imported rollouts without a simulator (More → Rollouts), with ROLLOUT already open.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from homebody.rollout import bundle
from homebody.session.settings import Settings


def parser():
    value = argparse.ArgumentParser(prog="homebody rollout", description="Export, import, evaluate and view "
                                                                         "recorded rollouts")
    value.add_argument("--root", type=Path, help="Release root containing configs (default: found from here)")
    value.add_argument("--settings", type=Path, help="TOML of only the values to change")
    commands = value.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Write a recorded session (or one task) as a .tar.gz bundle")
    export.add_argument("session", type=Path)
    export.add_argument("--task", help="Only this task, e.g. task-000")
    export.add_argument("--full", action="store_true", help="Keep sensors.npz and depth previews")
    export.add_argument("--out", type=Path, default=Path("."), help="Folder for the bundle (default: here)")
    imported = commands.add_parser("import", help="Check and unpack a bundle")
    imported.add_argument("bundle", type=Path)
    imported.add_argument("--into", type=Path, help="Folder of imported rollouts (default: ROOT/runs/imported)")
    imported.add_argument("--replace", action="store_true", help="Replace an earlier import of the same rollout")
    evaluate = commands.add_parser("evaluate", help="Run the monitor over a rollout's recorded frames")
    evaluate.add_argument("rollout", type=Path)
    evaluate.add_argument("--task", help="Only this task, e.g. task-000")
    evaluate.add_argument("--dry-run", action="store_true", help="Print the calls and cost, ask nothing")
    view = commands.add_parser("view", help="The dashboard on recorded and imported rollouts, without a simulator")
    view.add_argument("rollout", type=Path, nargs="?", help="A rollout folder to open first")
    view.add_argument("--task", help="Its task (default: the last)")
    view.add_argument("--runs", type=Path, help="Recorded sessions (default: ROOT/runs)")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8090)
    return value


def release_root(given: Path | None) -> Path | None:
    candidates = (given,) if given else (Path.cwd(), Path(__file__).resolve().parents[3])
    return next((path.resolve() for path in candidates if path and (path / "configs/simulation.toml").is_file()),
                None)


def models(settings):
    """The monitor's vocabulary and decision models, as a live run builds them."""
    from homebody.monitor.decision import OpenAiDecisions
    from homebody.monitor.vocabulary import VocabularyModel
    return VocabularyModel(settings.monitor), OpenAiDecisions(settings.monitor)


def evaluate(args, settings) -> int:
    from homebody.monitor import offline
    planned = offline.estimate(args.rollout, settings, args.task)
    for row in planned["tasks"]:
        print(f"{row['task']}: {row['frames']} frames -> about {row['calls']} decision calls, ${row['usd']:.3f}")
    print(f"total about ${planned['usd']:.3f}; the evaluation stops at ${planned['cap_usd']:.2f} a task "
          "(rollouts.offline_max_cost_usd)")
    if args.dry_run or not planned["calls"]:
        return 0
    vocabulary, decisions = models(settings)
    result = offline.evaluate(args.rollout, settings, vocabulary, decisions, task=args.task,
                              progress=lambda task, fraction: print(f"\r{task} {fraction:4.0%}", end="",
                                                                    flush=True))
    print()
    for report in result["tasks"]:
        print(f"{report['task']}: {report['calls']} calls, ${report['cost_usd']:.3f}, vocabulary "
              f"{', '.join(report['vocabulary_sources']) or 'none'}, frames at {report['frame_hz']} Hz")
        for would in report["would"]:
            print(f"  {would['sim_time']:.1f}s {would['action']} {would['goal_id']}: {would['label']} "
                  f"p={would['p']:.2f} {would['reason'] or ''}")
        for error in report["errors"][:3]:
            print(f"  error: {error}")
    return 0


def view(args, settings, runs: Path) -> int:
    from homebody.monitor.choice import ActivityChoice
    from homebody.rollout.app import RolloutsApp
    from homebody.rollout.reader import Rollout
    from homebody.rollout.replay import replay
    from homebody.ui.controls import Controls
    from homebody.ui.dashboard import Dashboard
    dashboard = Dashboard(Controls(), host=args.host, port=args.port, scene_name="Rollouts")
    dashboard.read_only = True
    dashboard.update("Ready", "Open a rollout: More → Rollouts.")
    RolloutsApp(dashboard, runs=runs, imported=runs / "imported", settings=settings,
                models=lambda: models(settings), replays=True)
    if args.rollout is not None:
        replay(dashboard, Rollout(args.rollout, args.task), settings.monitor.tau,
               choice=ActivityChoice(settings.monitor.choice_window, settings.monitor.choice_votes))
    print(f"Rollouts: {dashboard.url} (More → Rollouts)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        dashboard.close()


def main(argv) -> int:
    value = parser()
    args = value.parse_args(argv)
    root = release_root(args.root)
    try:
        settings = Settings.load(root / "configs/simulation.toml" if root else None, override=args.settings)
    except (ValueError, TypeError, KeyError, OSError) as error:
        value.error(f"settings: {error}")
    runs = (root / "runs") if root else Path("runs")
    try:
        if args.command == "export":
            archive = bundle.export(args.session, args.out, task=args.task, light=not args.full)
            print(f"{archive} ({archive.stat().st_size / bundle.MB:.1f} MB)")
        elif args.command == "import":
            folder = bundle.import_(args.bundle, args.into or runs / "imported",
                                    max_mb=settings.rollouts.max_import_mb, replace=args.replace)
            print(f"{folder}\n{json.dumps({k: v for k, v in bundle.describe(folder).items() if k != 'files'})}")
        elif args.command == "evaluate":
            return evaluate(args, settings)
        else:
            return view(args, settings, args.runs or runs)
    except bundle.BundleError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    return 0
