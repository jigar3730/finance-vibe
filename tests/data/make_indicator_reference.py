"""Regenerate tests/data/indicator_reference.csv from pandas-ta itself.

pandas-ta is *not* a project dependency (Phase 4.4 replaced it with
finance_vibe.indicators). Run this only in a throwaway environment, pinned to
the versions the in-house indicators were matched against:

    docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/w" -w /w -e HOME=/tmp \\
        ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm-slim \\
        uv run --no-project --python 3.12 \\
            --with pandas-ta==0.4.71b0 --with numba==0.61.2 \\
            --with pandas==3.0.5 --with numpy==2.2.6 \\
            python tests/data/make_indicator_reference.py

The series is seeded and deterministic. It includes flat bars (High == Low,
which triggers pandas-ta's epsilon-on-zero-range quirk in ATR, KC and
Bollinger widths) and volumes above 2**32 (OBV and SMA of Volume).
Column naming: single indicators as ``<name>_<length>``, multi-column
indicators as ``<prefix>:<pandas-ta column>``, which
tests/test_indicators_parity.py reads back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

OUT = Path(__file__).with_name("indicator_reference.csv")
N = 600


def make_bars() -> pd.DataFrame:
    rng = np.random.default_rng(44)
    close = 50.0 * np.exp(np.cumsum(0.0003 + 0.018 * rng.standard_normal(N)))
    opn = np.r_[close[0], close[:-1]] * (1 + 0.004 * rng.standard_normal(N))
    wick = np.abs(0.008 * rng.standard_normal(N))
    high = np.maximum(opn, close) * (1 + wick)
    low = np.minimum(opn, close) * (1 - wick)
    flat = rng.choice(np.arange(250, N), 12, replace=False)  # after every warm-up
    opn[flat] = high[flat] = low[flat] = close[flat]
    volume = rng.integers(2_000_000, 40_000_000, N).astype(float)
    volume[rng.choice(N, 40, replace=False)] *= 400.0  # > 2**32
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2022-01-03", periods=N).strftime("%Y-%m-%d"),
            "Open": opn,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def main() -> None:
    df = make_bars()
    c, h, low, v = df["Close"], df["High"], df["Low"], df["Volume"]
    cols: dict[str, pd.Series] = {}
    for n in (10, 20, 50, 100, 200):
        cols[f"ema_{n}"] = ta.ema(c, length=n)
    for n in (20, 50, 200):
        cols[f"sma_{n}"] = ta.sma(c, length=n)
    cols["sma_vol_20"] = ta.sma(v, length=20)
    cols["rsi_14"] = ta.rsi(c, length=14)
    cols["atr_14"] = ta.atr(h, low, c, length=14)
    cols["obv"] = ta.obv(c, v)
    frames = {
        "macd": ta.macd(c),
        "macd_15_30": ta.macd(c, fast=15, slow=30, signal=9),
        "bb": ta.bbands(c, length=20, lower_std=2.0, upper_std=2.0),
        "kc": ta.kc(h, low, c, length=20, scalar=1.5),
    }
    for prefix, frame in frames.items():
        for col in frame.columns:
            cols[f"{prefix}:{col}"] = frame[col]
    out = pd.concat([df, pd.DataFrame(cols)], axis=1)
    out.to_csv(OUT, index=False)  # repr floats: round-trips exactly
    print(f"wrote {OUT} ({len(out)} rows, {len(out.columns)} columns)")


if __name__ == "__main__":
    main()
