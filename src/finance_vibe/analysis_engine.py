"""Compatibility shim — implementation is ``finance_vibe.lab.analysis_engine``."""
from __future__ import annotations

import runpy
import sys

from finance_vibe.lab import analysis_engine as _impl

sys.modules[__name__] = _impl

if __name__ == "__main__":
    runpy.run_module("finance_vibe.lab.analysis_engine", run_name="__main__")
