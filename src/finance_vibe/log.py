"""One logging setup for every stage and CLI.

Library code only does ``logger = logging.getLogger(__name__)``; each
``__main__`` path calls :func:`setup_logging` once. The level comes from
``FINANCE_VIBE_LOG_LEVEL`` (default ``INFO``). Records go to stderr; the
container sets ``PYTHONUNBUFFERED=1`` so they interleave correctly with printed
tables in the cron logs.
"""

from __future__ import annotations

import logging
import os

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LEVEL_ENV = "FINANCE_VIBE_LOG_LEVEL"


def setup_logging(level: str | int | None = None) -> None:
    """Configure the root logger (idempotent; later calls only change the level)."""
    if level is None:
        level = os.environ.get(LEVEL_ENV, "INFO")
    if isinstance(level, str):
        level = level.strip().upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=LOG_FORMAT)
