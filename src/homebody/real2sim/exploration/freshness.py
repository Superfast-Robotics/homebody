"""How fresh the observations a decision or a motion reads must be."""

CAMERA_MAP_MAX_AGE_S = 1.0  # oldest map a planning or recovery step reads
DECISION_FRESH_STABLE_S = 0.5  # input stays fresh this long before a recovery decision
