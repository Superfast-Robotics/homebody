"""harvest_fixes.py RESULTS_SPLIT_DIR FIXES_DIR: turn the coder calls that ended OK in an evaluation's train runs
into fix-library entries (src/homebody/harness/fixes/*.json), which later coder calls get for the same skill and
refusal. Only a train split may be harvested, so held-out runs never shape the library.

An entry: skill, code (the refusal it fixed), when (the failure message), repair (the coder's own summary), sketch
(the diff of its edit), validated (how many calls made the same edit), sources.
"""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

SKETCH_LINES = 40


def calls(split_dir):
    """Each coder call folder of the split (task/run/session/tasks/*/coder/call-N)."""
    return sorted(Path(split_dir).glob("*/r*/session/*/tasks/*/coder/call-*"))


def entry(call, split_dir):
    """A fix entry for one coder call that ended OK, or None."""
    answer_path, request_path = call / "last_message.json", call / "REQUEST.md"
    if not answer_path.exists() or not request_path.exists():
        return None
    answer = json.loads(answer_path.read_text())
    if answer.get("status") != "success":
        return None
    request = request_path.read_text()
    skill = re.search(r"# Request: skill `([a-z0-9_]+)`", request)
    failure = re.search(r"^result: ([A-Z_]+): (.*)$", request, flags=re.MULTILINE)
    edited = call / "workspace" / "skills" / f"{skill.group(1)}.py" if skill else None
    started = call / f"start_{skill.group(1)}.py" if skill else None
    if not (skill and failure and edited and edited.exists() and started and started.exists()):
        return None
    diff = [line for line in difflib.unified_diff(started.read_text().splitlines(), edited.read_text().splitlines(),
                                                  lineterm="", n=1) if not line.startswith(("---", "+++"))]
    if not any(line.startswith(("+", "-")) for line in diff):
        return None
    return {"skill": skill.group(1), "code": failure.group(1), "when": failure.group(2)[:240],
            "repair": str(answer.get("summary", ""))[:400], "sketch": "\n".join(diff[:SKETCH_LINES]),
            "validated": 1, "sources": [str(call.relative_to(split_dir))]}


def main(split_dir, fixes_dir):
    if Path(split_dir).name != "train":
        raise SystemExit("harvest only a train split: held-out runs must not shape the library")
    fixes_dir = Path(fixes_dir)
    fixes_dir.mkdir(parents=True, exist_ok=True)
    added = merged = 0
    for call in calls(split_dir):
        fix = entry(call, Path(split_dir))
        if fix is None:
            continue
        name = f"{fix['skill']}-{fix['code'].lower()}-{hashlib.sha1(fix['sketch'].encode()).hexdigest()[:8]}.json"
        path = fixes_dir / name
        if path.exists():  # the same edit for the same refusal again: one more validation
            old = json.loads(path.read_text())
            fix = {**old, "validated": old["validated"] + 1, "sources": [*old["sources"], *fix["sources"]][-8:]}
            merged += 1
        else:
            added += 1
        path.write_text(json.dumps(fix, indent=1) + "\n")
    print(f"{added} new fixes, {merged} repeats, in {fixes_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
