"""Compare two parity reports: drop the ignored keys, then list every difference."""
from __future__ import annotations

IGNORED = {"timings", "prompt_sha256", "event_trace_sha256", "receiver_decode_time_s",
           "wall", "wall_s", "saved_at", "context_built_at_s", "checked_at_s", "t0_wall"}


def trimmed(value):
    """Drop the ignored keys at every depth."""
    if isinstance(value, dict):
        return {k: trimmed(v) for k, v in value.items() if k not in IGNORED}
    if isinstance(value, list):
        return [trimmed(v) for v in value]
    return value


def diffs(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                yield path + "/" + key, a.get(key, "<absent>"), b.get(key, "<absent>")
            else:
                yield from diffs(a[key], b[key], path + "/" + key)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path + "#len", len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diffs(x, y, f"{path}[{i}]")
    elif a != b:
        yield path, a, b
