"""Package-owned AMO walking assets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(os.environ.get("HOMEBODY_ROOT") or Path(__file__).resolve().parents[4]).expanduser().resolve()
ROBOT_URDF = REPO / "assets/robot/g1_29dof_with_hand.urdf"
VISUAL_URDF = REPO / "assets/robot/g1_viz/g1_29dof_with_hand_rev_1_0_viz.urdf"
DYNAMICS_URDF = REPO / "assets/robot/g1_body29_hand14.urdf"
DYNAMICS_SHA256 = "8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735"


def validate_assets() -> dict[str, str]:
    """Verify the three walking weights and the exact pinned dynamics model."""
    rows = json.loads((REPO / "assets/models/manifest.json").read_text())["files"]
    wanted = {"amo_jit.pt", "adapter_jit.pt", "adapter_norm_stats.pt"}
    manifest = {row["file"]: row for row in rows if row["file"] in wanted}
    if set(manifest) != wanted:
        raise ValueError("AMO manifest must describe all three walking-policy artifacts")
    for name, row in manifest.items():
        path = REPO / "assets/models" / name
        if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"AMO asset does not match its manifest: {path}")
    if hashlib.sha256(DYNAMICS_URDF.read_bytes()).hexdigest() != DYNAMICS_SHA256:
        raise ValueError(f"G1 dynamics model does not match the pinned artifact: {DYNAMICS_URDF}")
    return {name: row["sha256"] for name, row in manifest.items()}


def load_policy(profile, *, cache_directory: str | Path | None = None):
    """Load the pinned AMO policy, with gravity compensation as the profile asks."""
    from holosoma_inference.g1_control.amo import AmoPolicy, _prepare_amo_runtime
    from holosoma_inference.g1_control.config import AmoPolicyProfile, GravityBehavior, PolicyFamily

    if profile.family is not PolicyFamily.amo or not isinstance(profile.policy, AmoPolicyProfile):
        raise ValueError("G1 exploration supports the pinned AMO profile only")
    hashes = validate_assets()
    _prepare_amo_runtime()  # load order: NumPy, Torch, then optional Pinocchio
    gravity = None
    gravity_error = None
    if profile.policy.gravity_behavior is not GravityBehavior.disabled:
        try:
            from .gravity import PinocchioGravityCompensator
            gravity = PinocchioGravityCompensator(profile.robot, DYNAMICS_URDF)
        except Exception as exc:
            gravity_error = f"{type(exc).__name__}: {exc}"
            if profile.policy.gravity_behavior is GravityBehavior.required:
                raise
    cache = Path(cache_directory) if cache_directory is not None else (
        Path.home() / ".cache" / "homebody/g1-control"
    )
    models = REPO / "assets/models"
    return AmoPolicy(
        profile, models / "amo_jit.pt", models / "adapter_jit.pt", models / "adapter_norm_stats.pt",
        cache_directory=cache,
        policy_sha256=hashes["amo_jit.pt"], adapter_sha256=hashes["adapter_jit.pt"],
        statistics_sha256=hashes["adapter_norm_stats.pt"], gravity=gravity,
        gravity_initialization_error=gravity_error,
    )
