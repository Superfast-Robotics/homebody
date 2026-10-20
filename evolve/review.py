"""One read-only Codex session (Astra by default) that reviews the loop: it reads PROMPT, may read the repository
and the evolve work folder and run read-only commands, and writes its final message to OUT, with the session's
events and credits beside it. Nothing is modified.

    python3 -m evolve.review PROMPT OUT [--model gpt-6-astra] [--effort medium]
"""
import argparse
from pathlib import Path

from . import astra
from .loop import REPO, Loop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default=astra.settings()["model"])
    parser.add_argument("--effort", default=astra.settings()["effort"])
    args = parser.parse_args(argv)
    work = Loop().work
    args.out.parent.mkdir(parents=True, exist_ok=True)
    astra.codex(args.prompt.read_text(), REPO, "read-only", args.out, extra_dirs=(work,), model=args.model,
                effort=args.effort)
    print(f"{args.out}: {astra.credits_in(args.out.parent)} credits in {args.out.parent} so far")


if __name__ == "__main__":
    main()
