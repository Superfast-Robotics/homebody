"""Benchmark the live monitor offline on recordings from tools/monitor_record.py, to choose its configuration and
operating point before it stops real skills.

    python tools/monitor_bench.py runs/checks/manipulation_*/ --config configs/monitor_bench.toml [--dry-run]

For each variant of the config (frames per call, image size, wrist or not) it replays every recording: every
`stride_s` simulation seconds it asks the Decisions API about the newest frames, with a vocabulary per scripted
skill call (from the vocabulary model, or a fixed JSON file), and keeps every reading. It then scores:

- activity: how often the top label is one whose stages include the stage running then (labels with stages only);
- calibration: expected calibration error of "something is wrong" (the largest failure probability, or a standing
  claim not holding) against "after the scripted failure's onset";
- the rules (monitor/decide.py) replayed over the readings for each tau x ticks: false aborts before any failure
  (per minute of healthy run), recall on recordings with a failure and median detection delay;
- p50/p95 latency and cost per minute at the period the monitor would use.

Writes <out>/<variant>/<recording>/readings.jsonl, metrics.json and report.md.
"""
import argparse
import json
import statistics
import sys
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from homebody.monitor.decide import MonitorState, observe, skill_started
from homebody.monitor.decision import OpenAiDecisions, OutOfCredit
from homebody.monitor.targets import (
    Claim,
    DecisionRequest,
    Label,
    NodeInfo,
    Phase,
    Reading,
    Shot,
    Vocabulary,
    VocabularyRequest,
)
from homebody.monitor.vocabulary import VocabularyModel, fallback
from homebody.session.settings import Settings
from homebody.vlm.provider import ProviderError

ROOT = Path(__file__).resolve().parents[1]
BINS = 10


def load(recording: Path) -> dict:
    folder = recording / "monitor" if (recording / "monitor").is_dir() else recording
    lines = lambda name: [json.loads(line) for line in (folder / name).read_text().splitlines() if line.strip()] \
        if (folder / name).is_file() else []
    labels = lines("labels.jsonl")
    if not labels:
        raise SystemExit(f"{folder} holds no labels.jsonl; record it with tools/monitor_record.py")
    disturbances = lines("disturbances.jsonl")
    return {"name": recording.name, "folder": folder, "labels": labels, "requests": lines("requests.jsonl"),
            "onset": disturbances[0]["sim_time"] if disturbances else None}


def calls(data: dict) -> list[dict]:
    """The scripted skill calls with their start and end simulation times."""
    out, open_call = [], None
    for row in data["requests"]:
        if row["event"] == "request":
            open_call = {"skill": row["skill"], "arguments": row.get("arguments") or {}, "start": row["sim_time"],
                         "end": float("inf")}
            out.append(open_call)
        elif open_call is not None:
            open_call["end"] = row["sim_time"]
            open_call["code"] = row.get("code")
    return out


def context_for(data, calls, index) -> tuple[str, tuple]:
    """What the session would tell the vocabulary model at call INDEX: the hands held at its start, and each earlier
    pick that succeeded and still holds as a done goal with a holding check (its claim is checked every reading)."""
    start = calls[index]["start"]
    carried = next((row["carried"] for row in data["labels"] if row["sim_time"] >= start), [])
    hands = "; ".join(f"the {side} hand HOLDS the object from its last pick, not placed yet" for side in carried)
    done = tuple(NodeInfo(f"c{i}", f"pick the object with the {call['arguments'].get('hand', 'right')} hand",
                          {"name": "holding", "hand": call["arguments"].get("hand", "right")}, "done")
                 for i, call in enumerate(calls[:index])
                 if call["skill"] == "pick" and call.get("code") == "OK"
                 and call["arguments"].get("hand", "right") in carried)
    return hands or "both hands are free", done


def vocabulary_for(call, index, task, source, model, out, context=("", ())) -> Vocabulary:
    node = NodeInfo(f"c{index}", f"{call['skill']} {json.dumps(call['arguments'])}", None, "active")
    hands, done = context
    request = VocabularyRequest(task, "", node, done, (f"{call['skill']}",), hands, skill=call["skill"],
                                arguments=call["arguments"])
    if source == "model":
        return model.generate(request, out / f"vocab-{index:03d}")
    if source == "fallback":
        return fallback(request)
    fixed = json.loads(Path(source).read_text())
    return Vocabulary(node.goal_id, tuple(Label(**label) for label in fixed["labels"]))


def saved_vocabulary(record: dict) -> Vocabulary:
    return Vocabulary(record["goal_id"], tuple(Label(**{**label, "stages": tuple(label["stages"])})
                                               for label in record["labels"]),
                      tuple(Claim(**claim) for claim in record.get("standing", ())), record.get("expected_seconds"),
                      record.get("source", "model"))


def replay(data, variant, settings, args, task, out) -> tuple[list[Reading], list[Vocabulary]]:
    """The variant's readings over one recording and the vocabularies they used, saved under OUT and reused."""
    cached, saved = out / "readings.jsonl", out / "vocabularies.json"
    if cached.is_file() and saved.is_file() and not args.refresh:
        rows = [json.loads(line) for line in cached.read_text().splitlines()]
        return ([Reading(**{**row, "phase": Phase(**row["phase"]), "refusals": tuple(row["refusals"])})
                 for row in rows], [saved_vocabulary(v) for v in json.loads(saved.read_text())])
    out.mkdir(parents=True, exist_ok=True)
    decisions = OpenAiDecisions(settings)
    model = VocabularyModel(settings)
    labels, skill_calls = data["labels"], calls(data)
    vocabularies = {}
    readings, next_time = [], labels[0]["sim_time"]
    for position, row in enumerate(labels):
        if row["sim_time"] < next_time:
            continue
        next_time = row["sim_time"] + variant.get("stride_s", args.stride)
        window = labels[max(0, position - settings.window_frames + 1):position + 1]
        index = next((i for i, c in enumerate(skill_calls) if c["start"] <= row["sim_time"] < c["end"]), None)
        if index is None:
            continue
        if index not in vocabularies:
            vocabularies[index] = vocabulary_for(skill_calls[index], index, task, variant.get(
                "vocabulary", args.vocabulary), model, out, context_for(data, skill_calls, index))
        vocabulary = vocabularies[index]
        shots = tuple(Shot(np.asarray(Image.open(data["folder"] / "frames" / w["file"]).convert("RGB")),
                           w["sim_time"]) for w in window)
        phase = Phase("tool_call", row["skill"], row["stage"])
        request = DecisionRequest(task, NodeInfo(vocabulary.goal_id, skill_calls[index]["skill"], None, "active"),
                                  phase, f"holding: {', '.join(row['carried']) or 'nothing'}", vocabulary, shots)
        if args.dry_run:
            readings.append(Reading(vocabulary.goal_id, row["sim_time"], phase=phase))
            continue
        try:
            reading, _ = decisions.decide(request)
        except OutOfCredit:
            raise
        except ProviderError as error:
            print(f"  {data['name']} t={row['sim_time']:.1f}: {error}", file=sys.stderr)
            continue
        readings.append(reading)
        if len(readings) >= args.max_calls:
            break
    if not args.dry_run:
        cached.write_text("".join(json.dumps(asdict(r)) + "\n" for r in readings))
        saved.write_text(json.dumps([asdict(v) for v in vocabularies.values()], indent=1))
    return readings, list(vocabularies.values())


def stage_of(label_stages, stage) -> bool:
    return stage is not None and any(s.lower() == stage.lower() for s in label_stages)


def activity_alignment(readings, data, vocabularies) -> float | None:
    hits = total = 0
    for reading in readings:
        name, _ = reading.top()
        label = next((lab for v in vocabularies for lab in v.labels if lab.name == name and v.goal_id ==
                      reading.goal_id), None)
        if label is None or not label.stages:
            continue
        total += 1
        hits += stage_of(label.stages, reading.phase.stage)
    return hits / total if total else None


def wrong_probability(reading, failures) -> float:
    p = max([reading.activity.get(name, 0.0) for name in failures] or [0.0])
    return max([p, *[1 - holds for holds in reading.still_true.values()]])


def ece(pairs) -> float | None:
    if not pairs:
        return None
    total, error = len(pairs), 0.0
    for low in np.linspace(0, 1, BINS, endpoint=False):
        bucket = [(p, y) for p, y in pairs if low <= p < low + 1 / BINS or (low + 1 / BINS >= 1 and p == 1)]
        if bucket:
            error += len(bucket) / total * abs(statistics.mean(p for p, _ in bucket) - statistics.mean(
                y for _, y in bucket))
    return error


def rules(readings, vocabularies, onset, settings) -> dict:
    """The rules over READINGS: false aborts before ONSET, the first firing after it."""
    by_goal = {v.goal_id: v for v in vocabularies}
    state, false, detected = MonitorState(), 0, None
    current = None
    for reading in readings:
        vocabulary = by_goal.get(reading.goal_id)
        if vocabulary is None:
            continue
        if reading.goal_id != current:
            state, current = skill_started(state, reading.goal_id, reading.sim_time), reading.goal_id
        state, verdict = observe(state, reading, vocabulary, settings)
        if verdict.action in ("abort_skill", "reopen_goal", "flag"):
            if onset is None or reading.sim_time < onset:
                false += verdict.action != "flag"
            elif detected is None:
                detected = reading.sim_time - onset
    return {"false_aborts": false, "detected_after_s": detected}


def score(variant_name, results, settings, sweep) -> dict:
    latencies = [r.latency_s for rs, _, _ in results for r in rs if r.latency_s]
    costs = [r.cost_usd for rs, _, _ in results for r in rs]
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    period = max(settings.min_period_s, p95 * settings.p95_margin)
    pairs, alignment = [], []
    for readings, data, vocabularies in results:
        failures = {name for v in vocabularies for name in v.failures()}
        pairs += [(wrong_probability(r, failures), float(data["onset"] is not None and r.sim_time >= data["onset"]))
                  for r in readings]
        if (value := activity_alignment(readings, data, vocabularies)) is not None:
            alignment.append(value)
    points = []
    for tau in sweep["tau"]:
        for ticks in sweep["ticks"]:
            trial = replace(settings, tau=tau, ticks=ticks)
            outcomes = [(rules(rs, vs, d["onset"], trial), d) for rs, d, vs in results]
            healthy_minutes = sum(((d["onset"] or d["labels"][-1]["sim_time"]) - d["labels"][0]["sim_time"]) / 60
                                  for _, d in outcomes)
            failing = [o for o, d in outcomes if d["onset"] is not None]
            delays = [o["detected_after_s"] for o in failing if o["detected_after_s"] is not None]
            points.append({"tau": tau, "ticks": ticks,
                           "false_aborts_per_min": round(sum(o["false_aborts"] for o, _ in outcomes)
                                                         / max(healthy_minutes, 1e-9), 3),
                           "recall": round(len(delays) / len(failing), 3) if failing else None,
                           "median_delay_s": round(statistics.median(delays), 2) if delays else None})
    return {"variant": variant_name, "readings": sum(len(rs) for rs, _, _ in results),
            "p50_s": round(statistics.median(latencies), 3) if latencies else None, "p95_s": round(p95, 3),
            "period_s": round(period, 3),
            "usd_per_minute": round(statistics.mean(costs) * 60 / period, 4) if costs and period else None,
            "activity_stage_alignment": round(statistics.mean(alignment), 3) if alignment else None,
            "ece_wrong": None if (error := ece(pairs)) is None else round(error, 3), "operating_points": points}


def report(metrics: list[dict]) -> str:
    lines = ["# Monitor benchmark", ""]
    for m in metrics:
        lines += [f"## {m['variant']}", "",
                  (f"{m['readings']} readings; latency p50 {m['p50_s']} s, p95 {m['p95_s']} s; period "
                   f"{m['period_s']} s; ${m['usd_per_minute']} per minute; activity-stage alignment "
                   f"{m['activity_stage_alignment']}; ECE {m['ece_wrong']}"), "",
                  "| tau | ticks | false aborts / min | recall | median delay (s) |", "|---|---|---|---|---|"]
        lines += [f"| {p['tau']} | {p['ticks']} | {p['false_aborts_per_min']} | {p['recall']} | "
                  f"{p['median_delay_s']} |" for p in m["operating_points"]]
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/monitor_bench.toml")
    parser.add_argument("--settings", type=Path, help="TOML over configs/simulation.toml")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/monitor-bench")
    parser.add_argument("--max-calls", type=int, default=400, help="Decision calls per recording and variant")
    parser.add_argument("--dry-run", action="store_true", help="Count the calls without making them")
    parser.add_argument("--refresh", action="store_true", help="Ask again instead of reusing saved readings")
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text())
    bench = config.get("bench", {})
    args.stride, args.vocabulary = bench.get("stride_s", 0.5), bench.get("vocabulary", "model")
    task = bench.get("task", "carry each object to its destination")
    base = Settings.load(ROOT / "configs/simulation.toml", override=args.settings).monitor
    data = [load(path) for path in args.recordings]
    metrics = []
    for variant in config.get("variant", [{"name": "default"}]):
        settings = replace(base, **{k: v for k, v in variant.items() if k in asdict(base)})
        results = []
        for recording in data:
            out = args.out / variant["name"] / recording["name"]
            readings, vocabularies = replay(recording, variant, settings, args, task, out)
            results.append((readings, recording, vocabularies))
            print(f"{variant['name']} / {recording['name']}: {len(readings)} readings"
                  f"{' (dry run)' if args.dry_run else ''}", flush=True)
        if args.dry_run:
            continue
        metrics.append(score(variant["name"], results, settings, config.get("sweep", {"tau": [0.8], "ticks": [3]})))
    if metrics:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (args.out / "report.md").write_text(report(metrics))
        print(f"Report: {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
