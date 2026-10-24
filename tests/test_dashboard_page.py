"""The dashboard page's state logic (tests/js), run with node when it is installed."""
import shutil
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).with_name("js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_state_logic():
    files = sorted(str(path) for path in TESTS.glob("*.test.mjs"))
    result = subprocess.run(["node", "--test", *files], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
