"""RESEARCH.md §13 ranker: labels, LightGBM lambdarank walk-forward, baselines, decision rule.

Input is the ranked feature panel (:func:`research.ranker_features.rank_features`) on
the training grid. Steps, all as frozen in §13:

1. :func:`label_panel` adds ``R20 = Close[t+20] / Open[t+1] - 1`` and the §4 data-quality
   flags via :func:`research.labels.labels_at` (which raises for any date past
   ``RESEARCH_IN_SAMPLE_END``: the holdout stays sealed). Rows with a short window, no
   next open or a session gap are not usable; ``split_suspect`` rows are kept.
2. :func:`walk_forward` trains one model per test fold on the expanding window of
   training rows whose label exit date is before the fold's first score date, then
   scores that fold's score dates.
3. :func:`date_table` puts the three pickers side by side on every score date: model
   top 10%, momentum (``MOM_12_1``) top 10%, the eligible mean (what a random pick
   returns on average) and a seeded random top 10% as a check.
4. :func:`decide` applies the decision rule: A = model − eligible mean and B = model −
   momentum, one-sided t-tests over score dates, p = max, pass at p < alpha, and mean B > 0
   in at least ``RESEARCH_RANKER_MIN_FOLDS`` folds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance_vibe import config
from finance_vibe.research import labels as lb
from finance_vibe.research import ranker_features as rf
from finance_vibe.research import stats

LABEL = "R20"
GRADE = "grade"
SCORE = "score"
PICKERS = ("model", "momentum", "random")


def label_panel(
    ranked: pd.DataFrame, bars: Mapping[str, pd.DataFrame], *, start: str | None = None
) -> pd.DataFrame:
    """``ranked`` plus R20, SPY_R20, entry/exit dates, DQ flags and ``label_ok``."""
    h = config.RESEARCH_RANKER_HORIZON
    lab = lb.labels_at(
        ranked[["Symbol", "Date"]],
        bars,
        bars[config.RESEARCH_BENCHMARK],
        None,
        h,
        start=start or config.RESEARCH_RANKER_IN_SAMPLE_START,
    )
    lab = lab.rename(columns={"R10": LABEL, "SPY_R10": f"SPY_{LABEL}"}).drop(columns="REACH10")
    out = ranked.merge(lab, on=["Symbol", "Date"], how="left", validate="one_to_one")
    out["label_ok"] = (
        out[LABEL].notna() & ~out["no_next_open"] & ~out["short_window"] & ~out["session_gap"]
    )
    return out


def grades(frame: pd.DataFrame, n_grades: int | None = None) -> pd.Series:
    """Training relevance 0 … n_grades−1: the R20 quantile of each row within its date.

    Ties are broken by row order (the frame is sorted by Date, Symbol), so the result is
    deterministic. Only rows with ``label_ok`` should be passed in.
    """
    g = config.RESEARCH_RANKER_GRADES if n_grades is None else n_grades
    rank = frame.groupby("Date", sort=False)[LABEL].rank(method="first")
    n = frame.groupby("Date", sort=False)[LABEL].transform("count")
    return ((rank - 1) * g // n).astype("int64")


def fold_rows(
    frame: pd.DataFrame, score_dates: pd.DatetimeIndex, fold: tuple[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, test rows) for one fold.

    Test rows: usable rows on the fold's score dates. Training rows: usable rows whose
    label exit date is before the fold's first score date (the mandatory gap).
    """
    lo, hi = pd.Timestamp(fold[0]), pd.Timestamp(fold[1])
    dates = score_dates[(score_dates >= lo) & (score_dates <= hi)]
    if dates.empty:
        raise ValueError(f"fold {fold}: no score dates")
    usable = frame[frame["label_ok"]]
    test = usable[usable["Date"].isin(dates)]
    train = usable[usable["exit_date"] < dates.min()]
    return train, test


def train_model(train: pd.DataFrame, params: Mapping[str, object] | None = None):
    """One LightGBM lambdarank booster, one query group per training date (fixed settings)."""
    import lightgbm as lgb  # research-only dependency

    t = train.sort_values(["Date", "Symbol"], kind="stable")
    sizes = t.groupby("Date", sort=True).size().to_numpy()
    data = lgb.Dataset(
        t[rf.FEATURES].to_numpy(dtype="float64"),
        label=grades(t).to_numpy(),
        group=sizes,
        feature_name=list(rf.FEATURES),
        free_raw_data=True,
    )
    p = dict(config.RESEARCH_RANKER_LGB_PARAMS if params is None else params)
    return lgb.train(p, data, num_boost_round=config.RESEARCH_RANKER_ROUNDS)


def walk_forward(
    frame: pd.DataFrame,
    score_dates: pd.DatetimeIndex,
    folds: Sequence[tuple[str, str]] | None = None,
    params: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scored test rows with ``fold`` and ``score``, gain importance per fold)."""
    folds = config.RESEARCH_RANKER_FOLDS if folds is None else folds
    scored, importance = [], []
    for i, fold in enumerate(folds, start=1):
        train, test = fold_rows(frame, score_dates, fold)
        booster = train_model(train, params)
        out = test.copy()
        out[SCORE] = booster.predict(out[rf.FEATURES].to_numpy(dtype="float64"))
        out["fold"] = i
        scored.append(out)
        gain = booster.feature_importance(importance_type="gain")
        importance.append(
            pd.DataFrame(
                {
                    "fold": i,
                    "feature": rf.FEATURES,
                    "gain": gain,
                    "train_rows": len(train),
                    "train_last_exit": train["exit_date"].max(),
                    "test_first_date": test["Date"].min(),
                }
            )
        )
    return pd.concat(scored, ignore_index=True), pd.concat(importance, ignore_index=True)


def top_k(n: int, fraction: float | None = None) -> int:
    """How many stocks the top fraction holds: rounded up, at least 1."""
    f = config.RESEARCH_RANKER_TOP_FRACTION if fraction is None else fraction
    return max(1, math.ceil(f * n))


def _top_mean(values: np.ndarray, order_key: np.ndarray, k: int) -> float:
    """Mean of ``values`` over the k rows with the highest ``order_key`` (stable ties)."""
    idx = np.argsort(-order_key, kind="stable")[:k]
    return float(values[idx].mean())


def date_table(scored: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """One row per score date: each picker's mean R20 and the reported extras.

    Rows within a date are taken in Symbol order before ranking, so ties resolve the
    same way on every run. The random picker draws its scores from a generator seeded
    with ``[seed, date ordinal]``.
    """
    seed = config.RESEARCH_RANKER_SEED if seed is None else seed
    rows = []
    for date, day in scored.sort_values(["Date", "Symbol"], kind="stable").groupby("Date"):
        r = day[LABEL].to_numpy(dtype="float64")
        n, k = len(day), top_k(len(day))
        model = day[SCORE].to_numpy(dtype="float64")
        rng = np.random.default_rng([seed, pd.Timestamp(date).toordinal()])  # type: ignore[arg-type]
        rows.append(
            {
                "Date": date,
                "fold": int(day["fold"].iloc[0]),
                "n": n,
                "k": k,
                "model": _top_mean(r, model, k),
                "momentum": _top_mean(r, day["MOM_12_1"].to_numpy(dtype="float64"), k),
                "random": _top_mean(r, rng.random(n), k),
                "eligible_mean": float(r.mean()),
                "model_top3": _top_mean(r, model, min(3, n)),
                "model_top10": _top_mean(r, model, min(10, n)),
                "spy": float(day[f"SPY_{LABEL}"].iloc[0]),
                "rank_ic": float(pd.Series(model).corr(pd.Series(r), method="spearman")),
            }
        )
    table = pd.DataFrame(rows)
    table["A"] = table["model"] - table["eligible_mean"]
    table["B"] = table["model"] - table["momentum"]
    return table


@dataclass(frozen=True)
class Decision:
    mean_a: float
    p_a: float
    mean_b: float
    p_b: float
    p: float  # the larger of p_a and p_b
    folds_b_positive: int
    n_folds: int
    n_dates: int
    passed: bool


def decide(table: pd.DataFrame) -> Decision:
    """The §13 in-sample rule on a :func:`date_table`."""
    test_a = stats.clustered_ttest(table["A"], table["Date"])
    test_b = stats.clustered_ttest(table["B"], table["Date"])
    # An undefined test (NaN) never passes; plain max() would drop a NaN in second place.
    p = math.nan if math.isnan(test_a.p) or math.isnan(test_b.p) else max(test_a.p, test_b.p)
    by_fold = table.groupby("fold")["B"].mean()
    folds_pos = int((by_fold > 0).sum())
    passed = bool(
        not math.isnan(p)
        and p < config.RESEARCH_RANKER_ALPHA
        and folds_pos >= config.RESEARCH_RANKER_MIN_FOLDS
    )
    return Decision(
        mean_a=test_a.date_mean,
        p_a=test_a.p,
        mean_b=test_b.date_mean,
        p_b=test_b.p,
        p=p,
        folds_b_positive=folds_pos,
        n_folds=len(by_fold),
        n_dates=len(table),
        passed=passed,
    )
