"""Locate source assets explicitly; pure tensor imports also work from wheels."""
import os
from pathlib import Path

def repository_root(required=False):
    configured=os.environ.get('DD2414_PROJECT_ROOT')
    candidates=[Path(configured).expanduser().resolve()] if configured else list(Path(__file__).resolve().parents)
    for path in candidates:
        if (path/'nuscenesdataset.py').is_file() and (path/'scripts').is_dir() and (path/'configs').is_dir():
            return path
    if required or configured:
        raise FileNotFoundError('Historical training needs a DD2414 source checkout; set DD2414_PROJECT_ROOT explicitly')
    return Path(__file__).resolve().parent
