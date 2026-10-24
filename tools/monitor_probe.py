"""Probe the live monitor's two models before a run: what they accept, how fast and at what cost.

    python tools/monitor_probe.py decisions [--frames DIR] [--images 5 10] [--sizes 256 384] [--repeats 5]
    python tools/monitor_probe.py vocabulary [--image PNG] [--repeats 2]

`decisions` sends the monitor's full question set to the OpenAI Decisions API (OPENAI_DECISIONS_API_KEY) for each
image count and image size, REPEATS times, and reports p50/p95 latency, input tokens and cost per call, so the
call period (p95 x monitor.p95_margin) can be read off before a run. Frames come from DIR (*.jpg / *.png, e.g. a
task's monitor/frames or a session's frames/*/rgb.png) or are synthetic. `vocabulary` asks the vocabulary model for
one goal's labels with thinking off and on and reports latency and whether the reply parsed. Results go to
--output as JSON too.
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from homebody.monitor.decision import OpenAiDecisions
from homebody.monitor.targets import (
    DecisionRequest,
    Label,
    NodeInfo,
    Phase,
    Shot,
    Vocabulary,
    VocabularyRequest,
)
from homebody.monitor.vocabulary import VocabularyModel
from homebody.session.settings import Settings
from homebody.vlm.provider import ProviderError

ROOT = Path(__file__).resolve().parents[1]
NODE = NodeInfo("g1", "pick up the object on the counter with the right hand", {"name": "holding", "hand": "right"},
                "active", 8, "sequence")
VOCAB = Vocabulary("g1", (Label("approaching", "progress", "the robot walks or turns toward the object"),
                          Label("reaching", "progress", "the arm moves the open hand toward the object"),
                          Label("grasping", "progress", "the fingers close around the object"),
                          Label("lifting", "progress", "the hand raises the held object"),
                          Label("grasp_missed", "failure", "the fingers close beside the object"),
                          Label("object_dropped", "failure", "the object falls out of the hand")))


def frames(folder: Path | None, count: int) -> list[np.ndarray]:
    if folder is None:  # a block sliding across a counter-coloured image
        out = []
        for index in range(count):
            image = np.full((480, 640, 3), (196, 186, 170), np.uint8)
            image[220:300, 260 + 8 * index:340 + 8 * index] = (180, 40, 40)
            out.append(image)
        return out
    paths = sorted([*folder.rglob("*.jpg"), *folder.rglob("rgb.png")])[-count:]
    if len(paths) < count:
        raise SystemExit(f"{folder} holds {len(paths)} images; {count} are needed")
    return [np.asarray(Image.open(path).convert("RGB")) for path in paths]


def summary(latencies, tokens, costs, errors):
    if not latencies:
        return {"calls": 0, "errors": errors}
    ordered = sorted(latencies)
    return {"calls": len(latencies), "errors": errors, "p50_s": round(statistics.median(ordered), 3),
            "p95_s": round(float(np.percentile(ordered, 95)), 3), "max_s": round(ordered[-1], 3),
            "input_tokens": int(statistics.mean(tokens)), "usd_per_call": round(statistics.mean(costs), 6)}


def probe_decisions(args, settings) -> dict:
    results = {}
    for count in args.images:
        shots = tuple(Shot(rgb, index / 10) for index, rgb in enumerate(frames(args.frames, count)))
        for size in args.sizes:
            model = OpenAiDecisions(replace(settings.monitor, image_short_side=size))
            request = DecisionRequest("pick up the object on the counter", NODE,
                                      Phase("tool_call", "pick", "Approaching"), "both hands free", VOCAB, shots)
            latencies, tokens, costs, errors = [], [], [], []
            for _ in range(args.repeats):
                try:
                    reading, _ = model.decide(request)
                except ProviderError as error:
                    errors.append(str(error)[:300])
                    if "credit" in str(error) or "not set" in str(error):
                        break
                    continue
                latencies.append(reading.latency_s)
                tokens.append(reading.input_tokens)
                costs.append(reading.cost_usd)
            key = f"{count} images @ {size}px"
            results[key] = summary(latencies, tokens, costs, errors)
            print(f"{key:>20}: {json.dumps(results[key])}", flush=True)
    return results


def probe_vocabulary(args, settings, output: Path) -> dict:
    image = args.image or next(iter(sorted(ROOT.glob("assets/real2sim/*/preview*.png"))), None)
    request = VocabularyRequest("pick up the object on the counter", "sequence\n  [ACTIVE] g1: pick it up", NODE, (),
                                ("pick: pick the object at a click of the head image with one hand",
                                 "navigate: walk to a map point"), "both hands free",
                                images=(image,) if image else ())
    results = {}
    for thinking in (False, True):
        model = VocabularyModel(replace(settings.monitor, vocab_thinking=thinking))
        runs = []
        for repeat in range(args.repeats):
            started = time.monotonic()
            vocab = model.generate(request, output / f"vocab-thinking{int(thinking)}-{repeat}")
            runs.append({"seconds": round(time.monotonic() - started, 2), "source": vocab.source,
                         "labels": [label.name for label in vocab.labels]})
        results[f"thinking={thinking}"] = runs
        print(f"thinking={thinking}: {json.dumps(runs)}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=("decisions", "vocabulary"))
    parser.add_argument("--settings", type=Path, help="TOML over configs/simulation.toml (its [monitor] table)")
    parser.add_argument("--frames", type=Path, help="Folder of recorded frames (default: synthetic)")
    parser.add_argument("--image", type=Path, help="Head image for the vocabulary probe")
    parser.add_argument("--images", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 384])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "monitor-probe")
    args = parser.parse_args(argv)
    settings = Settings.load(ROOT / "configs/simulation.toml", override=args.settings)
    output = args.output / time.strftime("%Y%m%dT%H%M%S")
    output.mkdir(parents=True)
    results = probe_decisions(args, settings) if args.which == "decisions" else \
        probe_vocabulary(args, settings, output)
    (output / f"{args.which}.json").write_text(json.dumps(results, indent=2))
    print(f"Results: {output / f'{args.which}.json'}")
    return 0 if all(isinstance(r, list) or r.get("calls") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
