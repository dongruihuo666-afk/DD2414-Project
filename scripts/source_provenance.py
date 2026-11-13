"""Compatibility alias for the content-verified package provenance helpers."""
import sys
from _bootstrap import enable_source_package
enable_source_package()
import dd2414.runtime.provenance as _implementation
sys.modules[__name__] = _implementation
