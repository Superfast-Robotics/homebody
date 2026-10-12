"""A patch's tier from the files it touches (tiers.toml), and whether it touches frozen files."""
import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Classified:
    tier: int
    frozen: tuple  # frozen paths the patch touches; any makes it invalid
    by_tier: dict  # tier -> paths


def rules(path=HERE / "tiers.toml"):
    spec = tomllib.loads(path.read_text())
    return [("frozen", spec["frozen"])] + [(int(key[4:]), spec[key]) for key in ("tier0", "tier1", "tier2")]


def tier_of(path, table=None):
    for tier, globs in table or rules():
        if any(fnmatchcase(path, glob) for glob in globs):
            return tier
    return 2


def classify(paths, table=None):
    table = table or rules()
    by_tier, frozen = {}, []
    for path in paths:
        tier = tier_of(path, table)
        if tier == "frozen":
            frozen.append(path)
        else:
            by_tier.setdefault(tier, []).append(path)
    return Classified(max(by_tier, default=0), tuple(frozen), by_tier)
