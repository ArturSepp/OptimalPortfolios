"""Compatibility alias for the canonical execution implementation."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("optimalportfolios.execution.v1.improvement")
