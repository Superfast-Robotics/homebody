"""Robot models fetched from their sources at pinned commits (a sparse git checkout of one folder), cached under
~/.cache/homebody/models: the repository carries no meshes, and every run of a robot uses the same files."""
import subprocess
from dataclasses import dataclass
from pathlib import Path

CACHE = Path.home() / ".cache" / "homebody" / "models"


@dataclass(frozen=True)
class Source:
    repo: str
    commit: str
    folder: str  # the folder holding the model, relative to the repository root
    license: str


SOURCES = {
    "so101": Source("https://github.com/google-deepmind/mujoco_menagerie.git",
                    "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35", "robotstudio_so101", "Apache-2.0"),
    "omx": Source("https://github.com/ROBOTIS-GIT/open_manipulator.git",
                  "d31000d90c679af9c982e73de8b12d777c5ff7dd", "open_manipulator_description", "Apache-2.0"),
    # Koch v1.1 as ManiSkill models it (from jess-moss/koch-v1-1, limits and axes fixed), a URDF
    "koch": Source("https://github.com/mani-skill/ManiSkill.git",
                   "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3", "mani_skill/assets/robots/koch", "Apache-2.0"),
    "openarm": Source("https://github.com/enactic/openarm_mujoco.git",
                      "36d4a443533478a6a1b6e1d6e0d29b35583afc50", "v1", "Apache-2.0"),
    "lekiwi": Source("https://github.com/Ekumen-OS/lekiwi.git",
                     "32cf6a69eb320cc22620cdaa529e35f20fc12b1f", "packages/lekiwi_sim/lekiwi_sim", "Apache-2.0"),
}


def model_dir(name, cache=CACHE):
    """NAME's model folder, checked out on first use."""
    source = SOURCES[name]
    checkout = cache / f"{name}-{source.commit[:12]}"
    folder = checkout / source.folder
    if folder.is_dir():
        return folder
    checkout.parent.mkdir(parents=True, exist_ok=True)
    partial = checkout.with_name(checkout.name + ".partial")
    subprocess.run(["rm", "-rf", str(partial)], check=True)
    git = ["git", "-C", str(partial)]
    subprocess.run(["git", "init", "-q", str(partial)], check=True)
    subprocess.run([*git, "remote", "add", "origin", source.repo], check=True)
    subprocess.run([*git, "sparse-checkout", "set", source.folder], check=True)
    subprocess.run([*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", source.commit], check=True)
    subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)
    partial.rename(checkout)
    return folder
