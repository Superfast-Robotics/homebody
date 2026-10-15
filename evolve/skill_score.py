"""Scores of skill-benchmark episodes (skill_bench.py result.json files).

An episode scores its stage / 2 (0 not lifted, 0.5 lifted but not held, 1 held: success), after any
coder rescue (the dynamic skill); setup failures (stance or view failed before the skill ran) are left
out. Each episode's cost is the coder's credits and its time the skill's seconds, coder included. Versions run the same episodes
(the physics is deterministic), so a candidate is compared with its parent episode by episode, and
the standard error of the mean difference judges it against the spread of those differences.
"""
import json
from pathlib import Path

Z_DROP = 2.0  # a split counts as worse only if it falls by more than this many standard errors
Z_RISE = 1.5  # and as better only if it rises by more than this many
TIME_CUT = 0.20  # with unchanged accuracy, credits or time per episode must fall by a fifth
COST_SLACK = 0.10  # a time win may not cost more than this fraction extra
SLOWER = 0.30  # a version this much slower or costlier per episode needs a stronger accuracy gain
Z_RISE_SLOWER = 2.5


def read(split_dir):
    """episode id -> result dict for every scored episode under SPLIT_DIR (setup failures excluded)."""
    results = {}
    for path in sorted(Path(split_dir).glob("*/result.json")):
        result = json.loads(path.read_text())
        if result.get("setup_failed") or "stage" not in result:
            continue
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
        elif code == "OK":
            code = "OK but dropped"
        codes[code] = codes.get(code, 0) + 1
    coder = [r["coder"] for r in rows if r.get("coder")]
    return {"episodes": len(rows), "success_rate": round(sum(bool(r["success"]) for r in rows) / len(rows), 3),
            "builtin_success_rate": round(sum(builtin(r) for r in rows) / len(rows), 3),
            "coder_calls": len(coder), "coder_ok": sum(c["code"] == "OK" for c in coder),
            "credits": round(sum(credits(r) for r in rows) / len(rows), 3),
            "value": round(sum(r["stage"] for r in rows) / 2 / len(rows), 3),
            "skill_seconds": round(sum(seconds(r) for r in rows) / len(rows), 1),
            "success_by_family": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_family.items())},
            "outcomes": dict(sorted(codes.items(), key=lambda item: -item[1])),
            "values": {k: r["stage"] / 2 for k, r in results.items()},
            "seconds": {k: seconds(r) for k, r in results.items()},
            "credit_values": {k: credits(r) for k, r in results.items()}}


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


def paired(parent, child, key):
    """(mean difference, standard error) of CHILD over PARENT in per-episode KEY, over shared episodes."""
    shared = sorted(set(parent.get(key, {})) & set(child.get(key, {})))
    if len(shared) < 2:
        return 0.0, 0.0
    diffs = [child[key][e] - parent[key][e] for e in shared]
    mean = sum(diffs) / len(diffs)
    spread = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return mean, (spread / len(diffs)) ** 0.5


def pooled(side):
    """One split-like summary of train and val together, episodes keyed split/id."""
    return {key: {f"{split}/{episode}": value for split in ("train", "val")
                  for episode, value in side[split].get(key, {}).items()}
            for key in ("values", "seconds", "credit_values")}


def judge(parent, child):
    """Accuracy first: no split's value may fall by more than Z_DROP standard errors; accepted when the
    value over train and val pooled (every episode paired) rises by more than Z_RISE; with accuracy
    unchanged, also when the skill's time per call falls by TIME_CUT or more, significantly."""
    out = {}
    for split in ("train", "val"):
        d, se = paired(parent[split], child[split], "values")
        out[split] = (round(d, 4), round(se, 4))
    d, se = paired(pooled(parent), pooled(child), "values")
    out["pooled"] = (round(d, 4), round(se, 4))
    out["time"] = round(relative(parent, child, "seconds"), 3)
    out["cost"] = round(relative(parent, child, "credit_values"), 3)
    if any(d + Z_DROP * se < 0 for d, se in (out["train"], out["val"])):
        return {**out, "ok": False, "basis": "accuracy dropped"}
    heavier = out["time"] > SLOWER or out["cost"] > SLOWER
    needed = Z_RISE_SLOWER if heavier else Z_RISE
    if out["pooled"][0] - needed * out["pooled"][1] > 0:
        return {**out, "ok": True, "basis": "accuracy (enough to outweigh the extra time or cost)" if heavier else "accuracy"}
    if out["pooled"][0] - Z_RISE * out["pooled"][1] > 0:
        return {**out, "ok": False, "basis": f"accuracy gain too small for {out['time']:+.0%} time, {out['cost']:+.0%} cost"}
    d_cost, se_cost = paired(pooled(parent), pooled(child), "credit_values")
    if out["cost"] <= -TIME_CUT and d_cost + Z_RISE * se_cost < 0:
        return {**out, "ok": True, "basis": "cost (accuracy unchanged)"}
    d_time, se_time = paired(pooled(parent), pooled(child), "seconds")
    if out["time"] <= -TIME_CUT and d_time + Z_RISE * se_time < 0 and out["cost"] <= COST_SLACK:
        return {**out, "ok": True, "basis": "time (accuracy and cost unchanged)"}
    return {**out, "ok": False, "basis": "no clear gain"}


def relative(parent, child, key):
    """Relative change of a per-episode KEY (seconds, credit_values) over the episodes both ran (pooled)."""
    a, b = pooled(parent)[key], pooled(child)[key]
    shared = [e for e in a if e in b]
    base = sum(a[e] for e in shared)
    return (sum(b[e] for e in shared) - base) / base if shared and base > 0 else 0.0


def describe(verdict):
    t, v, p = verdict["train"], verdict["val"], verdict["pooled"]
    extra = f"; time {verdict['time']:+.0%}, cost {verdict.get('cost', 0):+.0%} per episode"
    return (f"{verdict['basis']}: value pooled {p[0]:+.3f} (se {p[1]:.3f}); train {t[0]:+.3f} (se {t[1]:.3f}), "
            f"val {v[0]:+.3f} (se {v[1]:.3f}){extra}")
