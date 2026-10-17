"""Load the exploration settings once and give every reader the same values.

`load` overlays configs/exploration.toml, an optional `--settings` file and command-line options on
the defaults in `config.py`; the result is written to `<run>/meta/settings.json` and `install`ed
in the console process. Readers call `current()` inside functions; it returns the defaults until
something is installed."""
import json
import os
import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from homebody.session.config import Agent
from homebody.session.settings import check_value

from .config import (
    AGENT,
    Body,
    Drive,
    Exploration,
    Gait,
    Hands,
    Iphone,
    Lidar,
    Map,
    Recording,
    Robot,
    Slam,
)


@dataclass(frozen=True)
class Settings:
    exploration: Exploration = field(default_factory=Exploration)
    agent: Agent = field(default_factory=lambda: AGENT)
    robot: Robot = field(default_factory=Robot)
    body: Body = field(default_factory=Body)
    iphone: Iphone = field(default_factory=Iphone)
    slam: Slam = field(default_factory=Slam)
    recording: Recording = field(default_factory=Recording)
    hands: Hands = field(default_factory=Hands)
    lidar: Lidar = field(default_factory=Lidar)
    map: Map = field(default_factory=Map)
    gait: Gait = field(default_factory=Gait)
    drive: Drive = field(default_factory=Drive)

    def __post_init__(self):
        for group in fields(self):
            section = getattr(self, group.name)
            if type(section) is not group.type:
                raise ValueError(f"{group.name} must be a {group.type.__name__} settings group")
            defaults = _default(group.name)
            for item in fields(section):
                _check(f"{group.name}.{item.name}", getattr(section, item.name),
                       getattr(defaults, item.name))
        if not 1 <= self.exploration.port <= 65535:
            raise ValueError("exploration.port must be between 1 and 65535")
        if self.map.zone_corners < 3:
            raise ValueError("map.zone_corners must be at least 3")
        if self.exploration.done_fraction > 1:
            raise ValueError("exploration.done_fraction must not exceed one")


def _default(name):
    return AGENT if name == "agent" else Settings.__dataclass_fields__[name].type()


def _check(label, value, default):
    """check_value's rules, plus booleans, and empty text or zero where the default is empty or zero."""
    if type(default) is bool:
        if type(value) is not bool:
            raise ValueError(f"{label} must be true or false")
    elif isinstance(default, str) and not default:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be text")
    elif not isinstance(default, str) and default == 0:
        if type(value) not in (int, float) or not 0 <= value <= sys.float_info.max:
            raise ValueError(f"{label} must be a nonnegative finite number")
    else:
        check_value(label, value, default, False)


def _build(tables) -> Settings:
    """Settings from {group: {key: value}} tables over the defaults; unknown groups or keys raise ValueError."""
    groups = {}
    for name, table in tables.items():
        if name not in Settings.__dataclass_fields__:
            raise ValueError(f"Unknown exploration settings group: {name}")
        if not isinstance(table, dict):
            raise ValueError(f"{name} must be a settings table")  # noqa: TRY004 -- settings errors are ValueError
        default = _default(name)
        kinds = {item.name: type(getattr(default, item.name)) for item in fields(default)}
        unknown = table.keys() - kinds.keys()
        if unknown:
            raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
        groups[name] = replace(default, **{
            key: float(value) if kinds[key] is float and type(value) is int else value
            for key, value in table.items()})
    return Settings(**groups)


def load(root, override=None, options=None) -> Settings:
    """Defaults, then ROOT/configs/exploration.toml, then the OVERRIDE file, then OPTIONS, a
    {group: {key: value}} mapping whose None values stay unset; ask_operator only via OPTIONS."""
    files = [Path(root) / "configs/exploration.toml"]
    if override is not None:
        if not Path(override).is_file():
            raise FileNotFoundError(f"Settings override not found: {override}")
        files.append(Path(override))
    documents = [tomllib.loads(path.read_text()) for path in files]
    for document in documents:
        if "ask_operator" in (document.get("exploration") or {}):
            raise ValueError("operator questions require --enable-ask-operator at launch, not a settings file")
    documents.append({name: {key: value for key, value in table.items() if value is not None}
                      for name, table in (options or {}).items()})
    tables = {}
    for document in documents:
        for name, table in document.items():
            if not isinstance(table, dict):
                raise ValueError(f"{name} must be a settings table")  # noqa: TRY004 -- settings errors are ValueError
            tables[name] = {**tables.get(name, {}), **table}
    return _build(tables)


PASSWORD_MASK = "***"  # what a recorded or printed robot.ssh_password reads


def to_dict(settings: Settings) -> dict:
    """Every group and key, as plain values: what a run records. The SSH password is masked."""
    groups = asdict(settings)
    groups["robot"]["ssh_password"] = PASSWORD_MASK
    return groups


def to_json(settings: Settings) -> str:
    return json.dumps(to_dict(settings), indent=2, allow_nan=False)


def from_json(text: str) -> Settings:
    """The settings `to_json` wrote, validated as `load` does; a masked SSH password is replaced
    by the configured one."""
    tables = json.loads(text)
    if not isinstance(tables, dict):
        raise ValueError("exploration settings JSON must be an object of groups")  # noqa: TRY004 -- as load
    robot = tables.get("robot")
    if isinstance(robot, dict) and robot.get("ssh_password") == PASSWORD_MASK:
        tables["robot"] = {**robot, "ssh_password": _configured_password()}
    return _build(tables)


def _configured_password() -> str:
    if os.environ.get("HOMEBODY_G1_SSH_PASSWORD"):
        return os.environ["HOMEBODY_G1_SSH_PASSWORD"]
    root = Path(os.environ.get("HOMEBODY_ROOT") or Path(__file__).resolve().parents[4])
    toml = root / "configs/exploration.toml"
    return (load(root) if toml.is_file() else _DEFAULTS).robot.ssh_password


_installed: dict[str, Settings] = {}


def install(settings: Settings) -> Settings:
    """Make SETTINGS what `current()` returns in this process."""
    if type(settings) is not Settings:
        raise TypeError("install takes exploration Settings")
    _installed["settings"] = settings
    return settings


def current() -> Settings:
    """The installed settings, or the defaults when nothing is installed."""
    return _installed.get("settings") or _DEFAULTS


def ssh_password(settings: Settings | None = None) -> str:
    """The G1's SSH password: HOMEBODY_G1_SSH_PASSWORD when set, else robot.ssh_password of
    SETTINGS or the installed settings."""
    return os.environ.get("HOMEBODY_G1_SSH_PASSWORD") or (settings or current()).robot.ssh_password


_DEFAULTS = Settings()
