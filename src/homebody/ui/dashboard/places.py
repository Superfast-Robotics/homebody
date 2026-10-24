"""Location labels from a scene's own semantics: the fixed fixture nearest the robot, never a guessed name."""
import json
from dataclasses import dataclass

NEAR = 1.0  # m from a fixture's footprint within which the robot is "near" it
# structure and sub-parts: no place to be near
SKIPPED = {"floor", "opaque_wall", "glass_boundary", "ceiling_fixture", "cable", "electrical_fixture",
           "appliance_handle", "printed_sign"}


@dataclass(frozen=True)
class Fixture:
    label: str
    low: tuple
    high: tuple

    def distance(self, x, y):
        dx = max(self.low[0] - x, 0., x - self.high[0])
        dy = max(self.low[1] - y, 0., y - self.high[1])
        return (dx * dx + dy * dy) ** .5


class Places:
    def __init__(self, fixtures, area=""):
        self.fixtures, self.area = tuple(fixtures), area

    @classmethod
    def from_scene(cls, scene, area=""):
        """The non-portable entities of SCENE/semantics.json with map-frame bounds; None without that file."""
        path = scene / "semantics.json"
        if not path.is_file():
            return None
        fixtures = []
        for entity in json.loads(path.read_text()).get("entities", []):
            bounds = entity.get("bounds_ws_map_m")
            if entity.get("portable_candidate") or not entity.get("label") or entity.get("category") in SKIPPED:
                continue
            low, high = (bounds.get("min"), bounds.get("max")) if isinstance(bounds, dict) else \
                (bounds if isinstance(bounds, list) and len(bounds) == 2 else (None, None))
            if isinstance(low, list) and isinstance(high, list) and len(low) >= 2 and len(high) >= 2:
                fixtures.append(Fixture(entity["label"], tuple(low[:2]), tuple(high[:2])))
        return cls(fixtures, area)

    def near(self, x, y):
        """'AREA · Near LABEL' for the closest fixture within NEAR metres, else None (coordinates only)."""
        if not self.fixtures:
            return None
        nearest = min(self.fixtures, key=lambda fixture: fixture.distance(x, y))
        if nearest.distance(x, y) > NEAR:
            return None
        return f"{self.area} · Near {nearest.label}" if self.area else f"Near {nearest.label}"
