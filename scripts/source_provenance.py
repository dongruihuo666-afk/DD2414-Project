"""Legacy module alias; branch-local implementation, no cross-branch merge."""
import sys
from _bootstrap import enable_source_package
enable_source_package()
import dd2414.runtime.provenance as _implementation
sys.modules[__name__]=_implementation
