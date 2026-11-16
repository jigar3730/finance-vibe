"""RESEARCH.md §13 in-sample evaluation: run ONCE, then write the result into §9 and §10.

1. Reads the blind feature run (``ranker_features_<stamp>/``), checking its output file
   against its manifest, and the frozen snapshot (``research.snapshot.verify``).
2. Ranks the features within each date, adds R20 labels (``research.ranker.label_panel``;
   the holdout stays sealed), trains one LightGBM lambdarank model per walk-forward fold
   and scores that fold's score dates (``research.ranker.walk_forward``).
3. Compares model, momentum and random pickers on every score date and applies the §13
   decision rule (``research.ranker.decide``). Reported extras: rank IC, top 3 / top 10,
   SPY, a flat cost against SPY, split_suspect rows excluded, and gain importance.
4. Writes ``<out-dir>/ranker_insample_<stamp>/``: ``dates.csv``, ``importance.csv``,
   ``report.md`` and ``manifest.json``, and prints the report.

The container has no git: pass ``--commit`` or ``GIT_COMMIT``. Usage (inside the
container, against the working tree)::

    python scripts/ranker_insample_report.py --features data/research/ranker_features_2026-11-15 \\
        --commit "$(git rev-parse --short HEAD)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from finance_vibe import config
from finance_vibe.data import raw
from finance_vibe.log import setup_logging
from finance_vibe.research import ranker as rk
from finance_vibe.research import ranker_features as rf
from finance_vibe.research import snapshot as sn

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_features(run: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    path = run / "features.csv.gz"
    if _sha256(path) != manifest["output_files"][path.name]:
        raise ValueError(f"{path} changed since the blind feature run")
    panel = pd.read_csv(path, parse_dates=["Date"])
    return panel, manifest


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def by_fold(table: pd.DataFrame) -> pd.DataFrame:
    g = table.groupby("fold")
    return pd.DataFrame(
        {
            "dates": g.size(),
            "mean A": g["A"].mean(),
            "mean B": g["B"].mean(),
            "rank IC": g["rank_ic"].mean(),
        }
    )


def report(
    decision: rk.Decision,
    table: pd.DataFrame,
    clean: rk.Decision,
    importance: pd.DataFrame,
    manifest: dict,
) -> str:
    cost = config.RESEARCH_RANKER_COST
    extras = {
        "model top 10% − eligible mean (A)": table["A"].mean(),
        "model top 10% − momentum top 10% (B)": table["B"].mean(),
        "momentum top 10% − eligible mean": (table["momentum"] - table["eligible_mean"]).mean(),
        "random top 10% − eligible mean (check)": (table["random"] - table["eligible_mean"]).mean(),
        "model top 3 − eligible mean": (table["model_top3"] - table["eligible_mean"]).mean(),
        "model top 10 − eligible mean": (table["model_top10"] - table["eligible_mean"]).mean(),
        "model top 10% − SPY": (table["model"] - table["spy"]).mean(),
        f"model top 10% − {cost:.2%} cost − SPY": (table["model"] - cost - table["spy"]).mean(),
    }
    gain = importance.groupby("feature")["gain"].sum()
    gain = (gain / gain.sum()).sort_values(ascending=False)
    lines = [
        f"# RESEARCH.md §13 in-sample result ({manifest['stamp']})",
        "",
        f"Commit `{manifest['commit']}`, features `{manifest['features_run']}`, "
        f"{decision.n_dates} score dates in {decision.n_folds} folds.",
        "",
        f"**Decision: {'PASS' if decision.passed else 'FAIL'}.** "
        f"A mean {_pct(decision.mean_a)} (p {decision.p_a:.4f}), "
        f"B mean {_pct(decision.mean_b)} (p {decision.p_b:.4f}), p = {decision.p:.4f} "
        f"(pass < {config.RESEARCH_RANKER_ALPHA}); mean B > 0 in {decision.folds_b_positive} of "
        f"{decision.n_folds} folds (pass ≥ {config.RESEARCH_RANKER_MIN_FOLDS}).",
        "",
        "## By fold",
        "",
        by_fold(table).to_markdown(floatfmt=".4f"),
        "",
        "## Reported, never a pass criterion (mean per score date, R20)",
        "",
        pd.Series(extras, name="mean").map(_pct).to_frame().to_markdown(),
        "",
        f"Mean rank IC {table['rank_ic'].mean():.4f}.",
        "",
        f"Without split_suspect rows: A {_pct(clean.mean_a)} (p {clean.p_a:.4f}), "
        f"B {_pct(clean.mean_b)} (p {clean.p_b:.4f}), folds {clean.folds_b_positive} of "
        f"{clean.n_folds}, rule {'met' if clean.passed else 'not met'}.",
        "",
        "## Gain importance (share, all folds; context only)",
        "",
        gain.to_frame("share").to_markdown(floatfmt=".3f"),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stamp = config.RESEARCH_RANKER_SNAPSHOT
    parser.add_argument("--features", required=True, help="the blind feature run folder")
    parser.add_argument("--snapshot", default=os.path.join(config.BACKFILL_SNAPSHOT_DIR, stamp))
    parser.add_argument("--out-dir", default=config.RESEARCH_DIR)
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument(
        "--commit", default=os.environ.get("GIT_COMMIT"), help="code version for the manifest"
    )
    args = parser.parse_args(argv)
    if not args.commit:
        parser.error("--commit (or GIT_COMMIT) is required: the container has no git")
    started = time.perf_counter()
    run_dir = Path(args.out_dir) / f"ranker_insample_{args.stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    features_run = Path(args.features)
    panel, feat_manifest = load_features(features_run)
    snap = Path(args.snapshot)
    sn.verify(snap)
    symbols = sorted(panel["Symbol"].unique())
    bars = {
        s: raw.load_raw(str(snap / "daily" / f"{s}_10y_1d.csv"))
        for s in [*symbols, config.RESEARCH_BENCHMARK]
    }
    ranked = rf.rank_features(panel)
    labelled = rk.label_panel(ranked, bars)
    score_dates = rf.session_grid(
        bars[config.RESEARCH_BENCHMARK]["Date"], config.RESEARCH_RANKER_SCORE_STRIDE
    )

    scored, importance = rk.walk_forward(labelled, score_dates)
    table = rk.date_table(scored)
    decision = rk.decide(table)
    clean_table = rk.date_table(scored[~scored["split_suspect"]])
    clean = rk.decide(clean_table)

    table.to_csv(run_dir / "dates.csv", index=False)
    importance.to_csv(run_dir / "importance.csv", index=False)
    manifest = {
        "stamp": args.stamp,
        "section": "RESEARCH.md §13",
        "commit": args.commit,
        "features_run": str(features_run),
        "features_commit": feat_manifest.get("commit"),
        "snapshot": str(snap),
        "config": {k: v for k, v in vars(config).items() if k.startswith("RESEARCH_RANKER_")},
        "rows": {
            "ranked": len(ranked),
            "label_ok": int(labelled["label_ok"].sum()),
            "split_suspect": int(labelled["split_suspect"].sum()),
            "scored": len(scored),
        },
        "decision": asdict(decision),
        "decision_without_split_suspect": asdict(clean),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    text = report(decision, table, clean, importance, manifest)
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    print(text)
    logger.info("wrote %s in %.1fs", run_dir, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
