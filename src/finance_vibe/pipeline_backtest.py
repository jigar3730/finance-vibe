"""Compatibility shim — implementation is ``finance_vibe.lab.pipeline_backtest``."""
from __future__ import annotations

import runpy
import sys

from finance_vibe.lab import pipeline_backtest as _impl

sys.modules[__name__] = _impl

if __name__ == "__main__":
    runpy.run_module("finance_vibe.lab.pipeline_backtest", run_name="__main__")
