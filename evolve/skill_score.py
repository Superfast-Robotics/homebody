"""Scores of skill-benchmark episodes (skill_bench.py result.json files).

An episode scores its stage / 2 (0 not lifted, 0.5 lifted but not held, 1 held: success); setup
failures (stance or view failed before the skill ran) are left out. Versions run the same episodes
(the physics is deterministic), so a candidate is compared with its parent episode by episode, and
the standard error of the mean difference judges it against the spread of those differences.
"""
import json
from pathlib import Path

Z_DROP = 2.0  # a split counts as worse only if it falls by more than this many standard errors
Z_RISE = 1.5  # and as better only if it rises by more than this many
TIME_CUT = 0.20  # with unchanged accuracy, the skill's wall time per call must fall by a fifth


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
        code = (row.get("pick") or {}).get("code", "ERROR" if row.get("error") else "?")
        if row["success"]:
            code = "OK (held)"
        elif code == "OK":
            code = "OK but dropped"
        codes[code] = codes.get(code, 0) + 1
    return {"episodes": len(rows), "success_rate": round(sum(bool(r["success"]) for r in rows) / len(rows), 3),
            "value": round(sum(r["stage"] for r in rows) / 2 / len(rows), 3),
            "skill_seconds": round(sum((r.get("timing") or {}).get("pick_wall", 0) for r in rows) / len(rows), 1),
            "success_by_family": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_family.items())},
            "outcomes": dict(sorted(codes.items(), key=lambda item: -item[1])),
            "values": {k: r["stage"] / 2 for k, r in results.items()},
            "seconds": {k: (r.get("timing") or {}).get("pick_wall", 0) for k, r in results.items()}}


def paired(parent, child, key):
    """(mean difference, standard error) of CHILD over PARENT in per-episode KEY, over shared episodes."""
    shared = sorted(set(parent.get(key, {})) & set(child.get(key, {})))
    if len(shared) < 2:
        return 0.0, 0.0
    diffs = [child[key][e] - parent[key][e] for e in shared]
    mean = sum(diffs) / len(diffs)
    spread = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return mean, (spread / len(diffs)) ** 0.5


def judge(parent, child):
    """Accuracy first: no split's value may fall by more than Z_DROP standard errors; accepted when one
    rises by more than Z_RISE; with accuracy unchanged, also when the skill's time per call falls by
    TIME_CUT or more, significantly."""
    out = {}
    for split in ("train", "val"):
        d, se = paired(parent[split], child[split], "values")
        out[split] = (round(d, 4), round(se, 4))
    if any(d + Z_DROP * se < 0 for d, se in out.values()):
        return {**out, "ok": False, "basis": "accuracy dropped"}
    if any(d - Z_RISE * se > 0 for d, se in out.values()):
        return {**out, "ok": True, "basis": "accuracy"}
    base = sum(parent[s].get("skill_seconds", 0) for s in ("train", "val")) / 2
    times = [paired(parent[s], child[s], "seconds") for s in ("train", "val")]
    d_time = sum(d for d, _ in times) / 2
    se_time = (sum(se ** 2 for _, se in times) ** 0.5) / 2
    if base > 0 and d_time / base <= -TIME_CUT and d_time + Z_RISE * se_time < 0:
        return {**out, "ok": True, "basis": "time (accuracy unchanged)", "time": round(d_time / base, 3)}
    return {**out, "ok": False, "basis": "no clear gain"}


def describe(verdict):
    t, v = verdict["train"], verdict["val"]
    extra = f", time {verdict['time']:+.0%}" if "time" in verdict else ""
    return f"{verdict['basis']}: value train {t[0]:+.3f} (se {t[1]:.3f}), val {v[0]:+.3f} (se {v[1]:.3f}){extra}"
