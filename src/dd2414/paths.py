"""Find the source checkout needed by training scripts and configuration files."""

import os
from pathlib import Path


def repository_root(required=False):
    configured = os.environ.get("DD2414_PROJECT_ROOT")
    if configured:
        candidates = [Path(configured).expanduser().resolve()]
    else:
        candidates = Path(__file__).resolve().parents

    for path in candidates:
        if (
            (path / "nuscenesdataset.py").is_file()
            and (path / "scripts").is_dir()
            and (path / "configs").is_dir()
        ):
            return path

    if required or configured:
        raise FileNotFoundError(
            "Training requires a DD2414 source checkout. "
            "Set DD2414_PROJECT_ROOT to its absolute path."
        )
    return Path(__file__).resolve().parent
