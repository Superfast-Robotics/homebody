"""Scores of skill-benchmark episodes (skill_bench.py result.json files).

An episode scores its stage / 2 (0 not lifted, 0.5 lifted but not held, 1 held: success), after any
coder rescue (the dynamic skill); setup failures (stance or view failed before the skill ran) are left
out. Each episode's cost is the coder's credits and its time the skill's seconds, coder included. Versions run the same episodes
(the physics is deterministic), so a candidate is compared with its parent episode by episode, and
the standard error of the mean difference judges it against the spread of those differences.

A profile may judge by completed holds instead (objective "held": 1 when the object is still held after the
whole carry, else 0; stage value is then a diagnostic) and weight episodes by id prefix (weights, e.g.
{"wh": 2.0} for the warehouse episodes), so partial credit and the larger share of other scenes cannot carry
a verdict.
"""
import json
import math
import re
from pathlib import Path

Z_DROP = 2.0  # a split counts as worse only if it falls by more than this many standard errors
Z_RISE = 1.5  # and as better only if it rises by more than this many
TIME_CUT = 0.20  # with unchanged accuracy, credits or time per episode must fall by a fifth
COST_SLACK = 0.10  # a time win may not cost more than this fraction extra
SLOWER = 0.30  # a version this much slower or costlier per episode needs a stronger accuracy gain
Z_RISE_SLOWER = 2.5
CRASH_SHARE = 0.05  # a candidate whose code crashes in more than this share of its episodes is rejected
NONINFERIOR = 0.02  # a time or cost win needs accuracy shown no worse than this (lower bound at Z_RISE)
Z_PROMISING = 1.0  # a first round with no candidate this far up (pooled) or accepted ends the run
# per-episode outcome keys a version can be judged by: stage value, completed holds, placed objects (end to end)
VALUE_KEYS = ("values", "held_values", "placed_values")
METRICS = {"values": "value", "held_values": "held", "placed_values": "placed"}


def read(split_dir):
    """episode id -> result dict for every scored episode under SPLIT_DIR. Setup failures are left out; an
    episode the candidate's code crashed counts as a failure (stage 0), so crashing never looks cheaper (setup
    failures, where the stance or view failed before the skill ran, are left out for every version alike)."""
    results = {}
    for path in sorted(Path(split_dir).glob("*/result.json")):
        result = json.loads(path.read_text())
        if result.get("setup_failed"):
            continue
        if "stage" not in result:  # a result without a score (malformed or crashed) is a failure, never left out
            result = {**result, "stage": 0, "success": False, "crashed": True,
                      "error": result.get("error") or "result without a stage"}
        result.setdefault("episode", path.parent.name)
        results[result["episode"]] = result
    return results


def summary(results):
    """Success rate, mean value, wall time and refusal codes of one split's episodes."""
    if not results:
        return {"episodes": 0, "success_rate": 0.0, "value": 0.0}
    rows = list(results.values())
    by_family, codes = {}, {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(bool(row["success"]))
        code = call(row).get("code", "ERROR" if row.get("error") else "?")
        if row["success"]:
            code = "OK (held)"
        elif code == "OK":  # a carry that stalled with the object still in the hand is not a drop
            code = "OK, carry did not arrive (still held)" if row.get("held") else "OK, then dropped"
        codes[code] = codes.get(code, 0) + 1
    coder = [r["coder"] for r in rows if r.get("coder")]
    return {"episodes": len(rows), "errors": sum(bool(r.get("crashed")) for r in rows),
            "success_rate": round(sum(bool(r["success"]) for r in rows) / len(rows), 3),
            "builtin_success_rate": round(sum(builtin(r) for r in rows) / len(rows), 3),
            "coder_calls": len(coder), "coder_ok": sum(c["code"] == "OK" for c in coder),
            "credits": round(sum(credits(r) for r in rows) / len(rows), 3),
            "value": round(sum(r["stage"] for r in rows) / 2 / len(rows), 3),
            "skill_seconds": round(sum(seconds(r) for r in rows) / len(rows), 1),
            "success_by_family": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_family.items())},
            "outcomes": dict(sorted(codes.items(), key=lambda item: -item[1])),
            "values": {k: r["stage"] / 2 for k, r in results.items()},
            "held_values": {k: float(r["stage"] == 2) for k, r in results.items()},
            "held_by_group": {g: f"{sum(r['stage'] == 2 for k, r in results.items() if group(k) == g)}/"
                                 f"{sum(group(k) == g for k in results)}" for g in sorted({group(k) for k in results})},
            "seconds": {k: seconds(r) for k, r in results.items()},
            "untimed": sorted(k for k, r in results.items() if not timed(r)),
            "credit_values": {k: credits(r) for k, r in results.items()}}


def timed(row):
    """Whether ROW measured the skill's time (a finite, positive wall time), so it may count in a time verdict."""
    timing = row.get("timing") or {}
    value = timing.get("skill_wall", timing.get("pick_wall"))
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def seconds(row):
    """The skill's seconds in an episode, the coder's included."""
    timing = row.get("timing") or {}
    return timing.get("skill_wall", timing.get("pick_wall", 0))


def builtin(row):
    """Success by the skill alone: it returned OK and the episode ended in success, before any coder."""
    return bool(row.get("success")) and call(row).get("code") == "OK"


def call(row):
    """The skill call's result in an episode record (older pick records name it pick)."""
    return row.get("call") or row.get("pick") or {}


def credits(row):
    return (row.get("coder") or {}).get("credits", 0.0)


def group(episode):
    """An episode's group for reporting: the warehouse's (ids starting "wh") or the generated scenes'."""
    return "warehouse" if episode.split("/")[-1].startswith("wh") else "generated"


def weight(episode, weights):
    """EPISODE's weight: the first WEIGHTS prefix its id starts with, else 1."""
    name = episode.split("/")[-1]
    return next((w for prefix, w in (weights or {}).items() if name.startswith(prefix)), 1.0)


def cluster(episode):
    """The independent unit an episode belongs to: its scene step, every stance variant of it together
    (ids end in _v0, _v1 ...), since the variants of one object are not independent samples."""
    return re.sub(r"_v\d+$", "", episode.split("/")[-1])


def paired(parent, child, key, weights=None, clusters=None):
    """(weighted mean difference, standard error) of CHILD over PARENT in per-episode KEY, over shared episodes;
    the standard error is cluster-robust (CR1: residuals summed within each cluster, G / (G - 1)); CLUSTERS maps an
    episode to its independent unit (default cluster())."""
    cluster_of = clusters or cluster
    shared = sorted(set(parent.get(key, {})) & set(child.get(key, {})))
    if len(shared) < 2:
        return 0.0, 0.0
    w = {e: weight(e, weights) for e in shared}
    diffs = {e: child[key][e] - parent[key][e] for e in shared}
    total = sum(w.values())
    mean = sum(w[e] * diffs[e] for e in shared) / total
    sums = {}
    for e in shared:
        sums[cluster_of(e)] = sums.get(cluster_of(e), 0.0) + w[e] * (diffs[e] - mean)
    groups = len(sums)
    if groups < 2:
        return mean, 0.0
    return mean, (groups / (groups - 1) * sum(s * s for s in sums.values()) / total ** 2) ** 0.5


def pooled(side):
    """One split-like summary of train and val together, episodes keyed split/id."""
    return {key: {f"{split}/{episode}": value for split in ("train", "val")
                  for episode, value in side[split].get(key, {}).items()}
            for key in (*VALUE_KEYS, "seconds", "credit_values")}


def judge(parent, child, key="values", weights=None, speed=True, clusters=None, val_margin=None, val_z=Z_RISE):
    """Accuracy first (KEY: "values", stage / 2, or "held_values", completed holds; WEIGHTS by episode prefix):
    no split may fall by more than Z_DROP standard errors; accepted when train and val pooled (every episode
    paired) rise by more than Z_RISE; with accuracy unchanged, also when the skill's time per call falls by
    TIME_CUT or more, significantly. With VAL_MARGIN (preregistered), val must also be shown non-inferior: its
    difference's lower bound (VAL_Z standard errors) above -VAL_MARGIN, not merely no significant drop."""
    child = {split: lost_as_failures(parent[split], child[split]) for split in ("train", "val")}
    out = {"metric": METRICS.get(key, "value")}
    for split in ("train", "val"):
        d, se = paired(parent[split], child[split], key, weights, clusters)
        out[split] = (round(d, 4), round(se, 4))
    d, se = paired(pooled(parent), pooled(child), key, weights, clusters)
    out["pooled"] = (round(d, 4), round(se, 4))
    out["time"] = round(relative(parent, child, "seconds"), 3)
    out["cost"] = round(relative(parent, child, "credit_values"), 3)
    crashed = sum(child[s].get("errors", 0) + len(child[s].get("lost", [])) for s in ("train", "val"))
    untimed = sum(len(side[s].get("untimed", [])) for side in (parent, child) for s in ("train", "val"))
    if crashed > CRASH_SHARE * max(1, sum(child[s].get("episodes", 0) for s in ("train", "val"))):
        return {**out, "ok": False, "basis": f"crashed in {crashed} episodes"}
    if any(d + Z_DROP * se < 0 for d, se in (out["train"], out["val"])):
        return {**out, "ok": False, "basis": "accuracy dropped"}
    if val_margin is not None and out["val"][0] - val_z * out["val"][1] < -val_margin:
        return {**out, "ok": False, "basis": f"val not shown non-inferior (lower bound below -{val_margin:g})"}
    heavier = out["time"] > SLOWER or out["cost"] > SLOWER
    needed = Z_RISE_SLOWER if heavier else Z_RISE
    if out["pooled"][0] - needed * out["pooled"][1] > 0:
        return {**out, "ok": True, "basis": "accuracy (enough to outweigh the extra time or cost)" if heavier else "accuracy"}
    if out["pooled"][0] - Z_RISE * out["pooled"][1] > 0:
        return {**out, "ok": False, "basis": f"accuracy gain too small for {out['time']:+.0%} time, {out['cost']:+.0%} cost"}
    if not speed:
        return {**out, "ok": False, "basis": "no clear gain (time and cost wins are off for this profile)"}
    if crashed or untimed:  # a crashed, lost or untimed episode's time is not a measurement: no time or cost win
        return {**out, "ok": False, "basis": f"no clear gain ({crashed} crashed or lost, {untimed} untimed episodes: "
                                             f"no time or cost acceptance)"}
    if out["pooled"][0] - Z_RISE * out["pooled"][1] < -NONINFERIOR:  # cheaper or faster only when no worse
        return {**out, "ok": False, "basis": "no clear gain (accuracy not shown non-inferior)"}
    d_cost, se_cost = paired(pooled(parent), pooled(child), "credit_values", clusters=clusters)
    if out["cost"] <= -TIME_CUT and d_cost + Z_RISE * se_cost < 0:
        return {**out, "ok": True, "basis": "cost (accuracy unchanged)"}
    d_time, se_time = paired(pooled(parent), pooled(child), "seconds", clusters=clusters)
    if out["time"] <= -TIME_CUT and d_time + Z_RISE * se_time < 0 and out["cost"] <= COST_SLACK:
        return {**out, "ok": True, "basis": "time (accuracy and cost unchanged)"}
    return {**out, "ok": False, "basis": "no clear gain"}


def lost_as_failures(parent, child):
    """CHILD with every episode PARENT scored but CHILD did not (a setup failure or a missing result on its side)
    added as a failure, so a version cannot gain by losing episodes; listed under "lost"."""
    def scored(side):
        return set().union(*(side.get(key, {}) for key in VALUE_KEYS))
    lost = sorted(scored(parent) - scored(child))
    if not lost:
        return child
    out = {**child, "lost": lost}
    for key in VALUE_KEYS:
        out[key] = {**child.get(key, {}), **dict.fromkeys(lost, 0.0)}
    return out


def relative(parent, child, key):
    """Relative change of a per-episode KEY (seconds, credit_values) over the episodes both ran (pooled)."""
    a, b = pooled(parent)[key], pooled(child)[key]
    shared = [e for e in a if e in b]
    base = sum(a[e] for e in shared)
    return (sum(b[e] for e in shared) - base) / base if shared and base > 0 else 0.0


def describe(verdict):
    t, v, p = verdict["train"], verdict["val"], verdict["pooled"]
    extra = f"; time {verdict['time']:+.0%}, cost {verdict.get('cost', 0):+.0%} per episode"
    return (f"{verdict['basis']}: {verdict.get('metric', 'value')} pooled {p[0]:+.3f} (se {p[1]:.3f}); train "
            f"{t[0]:+.3f} (se {t[1]:.3f}), val {v[0]:+.3f} (se {v[1]:.3f}){extra}")


def describe_train(verdict):
    """The verdict without validation numbers, for what the proposer and diagnoser may read."""
    t = verdict["train"]
    return f"{verdict.get('metric', 'value')} on train {t[0]:+.3f} (se {t[1]:.3f})"



def promising(entries):
    """Whether a round produced a reasonable candidate: one accepted, or one whose pooled value rose by more
    than Z_PROMISING standard errors. A first round without one is not worth following with more."""
    for entry in entries:
        d, se = (entry.get("verdict") or {}).get("pooled", (0.0, 0.0))
        if entry.get("status") == "accepted" or d - Z_PROMISING * se > 0:
            return True
    return False

