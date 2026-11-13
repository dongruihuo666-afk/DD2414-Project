"""Source-checkout compatibility only; installed callers use dd2414 directly."""
import sys
from pathlib import Path

def enable_source_package():
    source=str(Path(__file__).resolve().parents[1]/'src')
    if source not in sys.path:
        sys.path.insert(0,source)
