"""Explore a scene, then reconstruct its minimal HomeBody environment."""
import argparse
import sys
from pathlib import Path

from homebody.capture import Selection
from homebody.capture_package import package

from .exploration.cli import main as explore
from .runner import DEFAULT_AGENT, reconstruct


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "explore":
        return explore(argv[1:])
    parser = argparse.ArgumentParser(prog="homebody real2sim", description=__doc__)
    root_parser = parser
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("explore", help="Explore and record reconstruction inputs")
    exporter = commands.add_parser("export", help="Export the sensor streams of a closed run as a new package")
    exporter.add_argument("--exploration-data", type=Path, required=True, help="The closed exploration run directory")
    exporter.add_argument("--output", type=Path, required=True, help="New sensor package directory")
    parser = commands.add_parser("reconstruct", help="Build a runtime-only SIM environment")
    parser.add_argument("--exploration-data", type=Path, required=True, help="The closed exploration run directory")
    parser.add_argument("--output", type=Path, required=True, help="New runtime-only scene directory")
    parser.add_argument("--task", required=True, help="Reconstruction scope and requirements")
    parser.add_argument("--work", type=Path, help="New directory for intermediates")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3], help="HomeBody release root")
    parser.add_argument("--agent-command", default=DEFAULT_AGENT,
                        help="Agent executable and arguments; receives its task on stdin")
    parser.add_argument("--kind", action="append", dest="kinds", help="Capture kinds handed to the agent (default: all)")
    parser.add_argument("--stream", action="append", dest="streams", help="Exact stream ids handed to the agent")
    parser.add_argument("--resolution", type=float, default=.05, help="Navigation map cell size, metres")
    parser.add_argument("--height-band", type=float, nargs=2, default=(.05, 1.6), help="Heights projected into the map, metres above the floor")
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds allowed for the agent")
    args = root_parser.parse_args(argv)
    if args.command == "export":
        result = package(args.exploration_data, args.output)
        print(f"Sensor package: {result}")
        return 0
    selection = Selection(tuple(args.kinds or Selection().kinds), tuple(args.streams or ()))
    work = reconstruct(args.exploration_data, args.output, args.task, root=args.root, work=args.work,
                       agent_command=args.agent_command, selection=selection,
                       resolution=args.resolution, height_band=args.height_band, timeout=args.timeout)
    print(f"SIM environment: {args.output.resolve()}")
    if work is not None:
        print(f"Work directory: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
