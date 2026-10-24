"""The monitor app's config (homebody monitor --config FILE): the harnesses it can drive, the simulation servers
(warehouse-sim) they run on, and the scenarios that pair them with a task. Read and checked whole; the Refresh button
reads it again.

    [[harness]]  name, url (ws://...), token_file
    [[sim]]      name, endpoint, stream (optional), profile, scene, roots (optional table),
                 local_endpoint, local_stream (optional: this machine's tunnels to the same server)
    [[scenario]] name, harness, sim, task

A sim's endpoint, stream, profile, scene and roots are as the harness host sees them (they are sent to the harness
when a scenario attaches it); local_endpoint and local_stream are how this machine reaches the same server (SSH
tunnels), defaulting to the harness's view.
"""
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

LOOPBACK_TCP = re.compile(r"^tcp://127\.0\.0\.1:\d{1,5}$")  # warehouse-sim serves loopback only
HARNESS_URL = re.compile(r"^ws://(127\.0\.0\.1|localhost|\[::1\]):\d{1,5}/?$")
MAX_TASK = 4000


@dataclass(frozen=True)
class Harness:
    name: str
    url: str
    token_file: str


@dataclass(frozen=True)
class Sim:
    name: str
    endpoint: str
    profile: str
    scene: str
    stream: str | None = None
    roots: dict = field(default_factory=dict)
    local_endpoint: str = ""
    local_stream: str | None = None
    engine: str = "warehouse"  # warehouse-sim's server, or HomeBody's own (homebody sim-serve): no profile

    def target(self) -> dict:
        """What the harness is sent to attach to this server (backends/warehouse.SimTarget)."""
        return {"endpoint": self.endpoint, "profile": self.profile or None, "scene": self.scene,
                "roots": dict(self.roots), "engine": self.engine}


@dataclass(frozen=True)
class Scenario:
    name: str
    harness: str
    sim: str
    task: str


@dataclass(frozen=True)
class MonitorConfig:
    path: Path
    harnesses: dict
    sims: dict
    scenarios: dict


FIELDS = {"harness": ({"name", "url", "token_file"}, set()),
          "sim": ({"name", "endpoint", "scene"}, {"profile", "engine", "stream", "roots", "local_endpoint",
                                                 "local_stream"}),
          "scenario": ({"name", "harness", "sim", "task"}, set())}


def entries(raw: dict, kind: str) -> list[dict]:
    rows = raw.get(kind, [])
    if not isinstance(rows, list):
        raise ValueError(f"[[{kind}]] must be a list of tables")  # noqa: TRY004 -- every bad config is one error
    required, optional = FIELDS[kind]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"each [[{kind}]] is a table")  # noqa: TRY004 -- every bad config is one error
        missing, unknown = required - set(row), set(row) - required - optional
        if missing or unknown:
            raise ValueError(f"[[{kind}]] {row.get('name', '?')!r}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        for key in required | (optional & set(row)) - {"roots"}:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"[[{kind}]] {row.get('name', '?')!r}: {key} must be nonempty text")
    names = [row["name"] for row in rows]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"[[{kind}]] names repeated: {repeated}")
    return rows


def sim(row: dict) -> Sim:
    for key in ("endpoint", "stream", "local_endpoint", "local_stream"):
        if key in row and not LOOPBACK_TCP.match(row[key]):
            raise ValueError(f"sim {row['name']!r}: {key} must be tcp://127.0.0.1:PORT (loopback; tunnel otherwise)")
    engine = row.get("engine", "warehouse")
    if engine not in ("warehouse", "homebody"):
        raise ValueError(f"sim {row['name']!r}: engine is warehouse (warehouse-sim) or homebody (homebody sim-serve)")
    if engine == "warehouse" and "profile" not in row:
        raise ValueError(f"sim {row['name']!r}: a warehouse-sim server needs the profile it serves")
    roots = row.get("roots", {})
    if not isinstance(roots, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in roots.items()):
        raise ValueError(f"sim {row['name']!r}: roots is a table of NAME = \"path\"")
    return Sim(row["name"], row["endpoint"], row.get("profile", ""), row["scene"], row.get("stream"), dict(roots),
               row.get("local_endpoint", row["endpoint"]), row.get("local_stream", row.get("stream")), engine)


def load(path: Path) -> MonitorConfig:
    """PATH's config, or ValueError naming the first problem (nothing of a bad config is used)."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    unknown = set(raw) - set(FIELDS)
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    harnesses = {}
    for row in entries(raw, "harness"):
        if not HARNESS_URL.match(row["url"]):
            raise ValueError(f"harness {row['name']!r}: url must be ws://127.0.0.1:PORT (tunnel to the harness)")
        harnesses[row["name"]] = Harness(row["name"], row["url"].rstrip("/"), row["token_file"])
    sims = {row["name"]: sim(row) for row in entries(raw, "sim")}
    scenarios = {}
    for row in entries(raw, "scenario"):
        if row["harness"] not in harnesses:
            raise ValueError(f"scenario {row['name']!r}: unknown harness {row['harness']!r}")
        if row["sim"] not in sims:
            raise ValueError(f"scenario {row['name']!r}: unknown sim {row['sim']!r}")
        if len(row["task"]) > MAX_TASK:
            raise ValueError(f"scenario {row['name']!r}: task longer than {MAX_TASK} characters")
        scenarios[row["name"]] = Scenario(row["name"], row["harness"], row["sim"], row["task"].strip())
    return MonitorConfig(path, harnesses, sims, scenarios)
