"""Statistics for RESEARCH.md §7: date-clustered one-sided t-test and Holm.

Pure numpy (no scipy in the image). ``t_sf`` is the Student-t upper tail via
the regularized incomplete beta function (Numerical Recipes continued
fraction), checked against textbook critical values in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, tiny = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError("betacf did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"betainc: x={x} outside [0, 1]")
    if x in (0.0, 1.0):
        return x
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with ``df`` degrees of freedom."""
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


@dataclass(frozen=True)
class ClusteredTest:
    mean: float  # mean of the per-row excess (what the scorecard reports)
    n_rows: int
    n_dates: int
    date_mean: float  # mean of the daily means (what the test is on)
    t: float
    p: float  # one-sided, H1: mean > 0


def clustered_ttest(excess: pd.Series, dates: pd.Series) -> ClusteredTest:
    """RESEARCH.md §7: average ``excess`` within each date, one-sided t-test on the daily means.

    NaN rows are dropped first. Fewer than 2 dates, or zero variance, gives p = NaN.
    """
    ok = excess.notna().to_numpy()
    x, d = excess[ok].astype("float64"), dates[ok]
    daily = x.groupby(d.to_numpy()).mean()
    n = len(daily)
    mean = float(x.mean()) if len(x) else math.nan
    if n < 2:
        return ClusteredTest(
            mean, len(x), n, float(daily.mean()) if n else math.nan, math.nan, math.nan
        )
    m, sd = float(daily.mean()), float(daily.std(ddof=1))
    if sd == 0.0:
        return ClusteredTest(mean, len(x), n, m, math.nan, math.nan)
    t = m / (sd / math.sqrt(n))
    return ClusteredTest(mean, len(x), n, m, t, t_sf(t, n - 1))


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN stays NaN and is not counted in m)."""
    p = np.asarray(pvalues, dtype="float64")
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    m = len(idx)
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()
