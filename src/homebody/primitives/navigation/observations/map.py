"""map: the scan artifact as raster, px -> world transform (m), occupancy, reference and hash."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from homebody.primitives.navigation.contracts import TermSpec

# The live map artifact the robot drives on.


SPEC = TermSpec(
    "map", (), "struct{raster, T_map_px, occupancy, scan_ref, scan_hash, center_allowed?}",
    "map", "m", staleness_s=float("inf"),
    notes="px<->world round-trip < 1 px through helpers.raster; the scan hash is recorded "
          "with the run",
)


@dataclass(frozen=True)
class MapArtifact:
    raster: np.ndarray          # (H, W, 3) u8 display raster
    T_map_px: np.ndarray        # (3, 3) homogeneous px -> world metres
    occupancy: np.ndarray       # (H, W) bool, True = blocked
    scan_ref: str               # path of the scan artifact
    scan_hash: str              # sha256 of the scan artifact
    center_allowed: np.ndarray | None = None  # (H, W) centre constraint, applied after inflation


def measured(handle):
    """The room's occupancy raster with its own px<->world transform."""
    m = handle.map_artifact()
    return MapArtifact(m["raster"], m["T_map_px"], m["occupancy"], m["scan_ref"], m["scan_hash"],
                       m.get("center_allowed"))
