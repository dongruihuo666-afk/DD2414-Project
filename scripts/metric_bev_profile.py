"""Compatibility alias for :mod:`dd2414.experiments.metric_profile`. See docs/integration/COMPATIBILITY.md."""
import sys
from _bootstrap import enable_source_package
enable_source_package()
import dd2414.experiments.metric_profile as _implementation
sys.modules[__name__] = _implementation
