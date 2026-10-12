"""The cuRobo client helpers."""
import numpy as np

from homebody.helpers import curobo


def test_thinned_keeps_endpoints_and_limit():
    rows = np.arange(100)[:, None] * np.ones((1, 7))
    kept = curobo.thinned(rows, 10)
    assert len(kept) == 10 and kept[0, 0] == 0 and kept[-1, 0] == 99
    assert len(curobo.thinned(rows[:5], 10)) == 5
