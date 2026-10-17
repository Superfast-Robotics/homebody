"""The launcher records its settings with the robot password masked."""
import json
import os
import re
import subprocess
from pathlib import Path

from homebody.real2sim.exploration.settings import load, to_json

ROOT = Path(__file__).resolve().parents[1]
METADATA = (
    "hardware/g1/superodom/g1_mid360.yaml",
    "hardware/g1/superodom/g1_frame_normalizer_ws.yaml",
    "src/homebody/real2sim/exploration/record/topics.txt",
    "src/homebody/real2sim/exploration/record/qos.yaml",
)


def test_launcher_metadata_preserves_resolved_settings_without_raw_credentials(tmp_path):
    release, run = tmp_path / "release", tmp_path / "run"
    config = release / "configs/exploration.toml"
    config.parent.mkdir(parents=True)
    secret = "synthetic-metadata-password-sentinel"
    config.write_text(f"[robot]\nssh_password = {json.dumps(secret)}\n")
    for name in METADATA:
        target = release / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    metadata = run / "meta"
    metadata.mkdir(parents=True)
    resolved = to_json(load(release))
    (metadata / "settings.json").write_text(resolved)

    launcher = (ROOT / "hardware/g1/explore.sh").read_text()
    block = re.search(r"(?ms)^for f in .*?^done$", launcher)
    assert block is not None and 'cp "$f" "$RUN_DIR/meta/"' in block.group()
    # Execute just the production copy loop; none of the launcher's service paths run.
    subprocess.run(["bash", "-euc", block.group()], cwd=release,
                   env={"PATH": os.defpath, "RUN_DIR": str(run)},
                   capture_output=True, text=True, check=True)

    assert (metadata / "settings.json").read_text() == resolved
    assert json.loads(resolved)["robot"]["ssh_password"] == "***"
    for name in METADATA:
        assert (metadata / Path(name).name).read_bytes() == (release / name).read_bytes()
    assert not (metadata / "exploration.toml").exists()
    assert all(secret.encode() not in path.read_bytes() for path in metadata.iterdir())
    assert secret in config.read_text()
