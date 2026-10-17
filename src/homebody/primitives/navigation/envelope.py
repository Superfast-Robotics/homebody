"""The footprint-radius check shared by navigation and the backend."""
import math
from typing import Any


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")  # noqa: TRY004 -- radius errors are ValueError
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_radius_m(radius_m: float | None) -> float:
    """A footprint radius within 0.20..0.60 m inclusive, as a float."""
    radius = _finite_float(radius_m, "radius_m")
    if not 0.20 <= radius <= 0.60:
        raise ValueError("radius_m must be within the configured 0.20..0.60 m range")
    return radius
