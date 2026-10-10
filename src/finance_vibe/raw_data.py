"""The one way stages read raw OHLCV from ``data/raw/{mode}/``.

Raw files are written by the ingestors as ``<TICKER>_<period>_<interval>.csv``
and must satisfy :data:`config.REQUIRED_OHLCV`. :func:`load_raw` applies the
data contract and the as-of cut in one place, so no reader can skip either.

(Not named ``io``: stages run by path put this directory first on
``sys.path``, where an ``io.py`` could shadow the standard library.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from finance_vibe import config


@dataclass(frozen=True)
class RawFile:
    ticker: str
    path: str


def ticker_from_filename(path: str) -> str:
    """Ticker symbol from a raw CSV name (text before the first ``_``)."""
    return os.path.basename(path).split("_")[0].upper()


def raw_files(raw_dir: str) -> list[RawFile]:
    """Raw CSVs in ``raw_dir``, sorted by file name (deterministic scan order)."""
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw data directory does not exist: {raw_dir}")
    return [
        RawFile(ticker_from_filename(name), os.path.join(raw_dir, name))
        for name in sorted(os.listdir(raw_dir))
        if name.lower().endswith(".csv")
    ]


def load_raw(path: str, *, as_of: str | None = None, weekly: bool | None = None) -> pd.DataFrame:
    """Read one raw CSV: contract check, clean dates, then the as-of cut.

    Returns a frame with a naive datetime64 ``Date`` column, sorted ascending
    with one row per date (the last one wins), float64 prices, numeric Volume, and no rows with
    missing Open/High/Low/Close. With ``as_of`` set, only bars complete on that
    date are kept (see :func:`config.cut_to_as_of`); ``weekly`` says how to
    tell when a bar is complete and is required with ``as_of``.

    Raises ``OSError`` if the file can't be read and ``ValueError`` if it is
    empty, unparsable, or missing a :data:`config.REQUIRED_OHLCV` column.
    """
    if as_of is not None and weekly is None:
        raise ValueError("load_raw: `weekly` is required with `as_of`")
    df = config.validate_and_clean_ohlcv(pd.read_csv(path), require_volume=True)
    dates = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None)
    prices = ["Open", "High", "Low", "Close"]
    df = df.assign(Date=dates).astype(dict.fromkeys(prices, "float64")).dropna(subset=["Date"])
    df = df.sort_values("Date", kind="stable").drop_duplicates(subset=["Date"], keep="last")
    df = df.reset_index(drop=True)
    return config.cut_to_as_of(df, as_of, weekly=bool(weekly))
