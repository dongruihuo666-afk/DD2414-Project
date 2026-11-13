#!/usr/bin/env python3
"""Legacy wrapper for the opt-in real-data/plotting program, not a unit test."""
from pathlib import Path
import runpy
globals().update(runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/visualization/radar_velocity_rotation.py'),run_name=__name__))
