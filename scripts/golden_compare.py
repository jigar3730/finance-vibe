#!/usr/bin/env python3
"""Golden-output harness: prove a change leaves pipeline outputs unchanged.

Replays ``run_vibe.py --as-of`` for weekly and daily against a *frozen* copy of
the raw data, in a throwaway workspace, and diffs every CSV the stages write
against a stored baseline. No network, and live ``data/`` is never written.

Subcommands (run inside the container, where the dependencies live; see
``scripts/golden_in_container.sh``):

  snapshot  Copy ``<source-data>/raw/{weekly,daily}`` + ``active_tickers.csv``
            into ``<golden-dir>/fixture`` (once; ``--force`` to replace).
  baseline  Run the pipeline from ``--repo`` on the fixture and store its
            outputs in ``<golden-dir>/baseline`` with a manifest.
  compare   Run the pipeline from ``--repo`` again and diff against the
            baseline. Exit 0 = identical (within ``--atol``), 1 = differences.

The as-of dates are chosen at baseline time from the fixture's QQQ bars (last
completed Friday for weekly, last bar for daily) and stored in the manifest,
so ``compare`` always replays the same dates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ("weekly", "daily")
REPO_DEFAULT = Path(__file__).resolve().parents[1]


def _fixture_dir(golden: Path) -> Path:
    return golden / "fixture"


def _baseline_dir(golden: Path) -> Path:
    return golden / "baseline"


def cmd_snapshot(args: argparse.Namespace) -> int:
    src = Path(args.source_data)
    dst = _fixture_dir(Path(args.golden_dir))
    if dst.exists():
        if not args.force:
            print(f"Fixture already exists at {dst} (use --force to replace).")
            return 1
        shutil.rmtree(dst)
    for mode in MODES:
        raw = src / "raw" / mode
        if not raw.is_dir() or not any(raw.glob("*.csv")):
            print(f"Missing or empty {raw}")
            return 1
        shutil.copytree(raw, dst / "raw" / mode)
    shutil.copy2(src / "active_tickers.csv", dst / "active_tickers.csv")
    counts = {m: len(list((dst / "raw" / m).glob("*.csv"))) for m in MODES}
    (dst / "SNAPSHOT.json").write_text(
        json.dumps(
            {
                "created": datetime.now().isoformat(timespec="seconds"),
                "source": str(src),
                "files": counts,
            },
            indent=2,
        )
    )
    print(f"Fixture written to {dst}: {counts}")
    return 0


def _last_qqq_date(fixture: Path, mode: str) -> date:
    path = next((fixture / "raw" / mode).glob("QQQ_*.csv"), None)
    if path is None:
        raise SystemExit(f"No QQQ file in fixture raw/{mode}; can't pick an as-of date.")
    dates = pd.to_datetime(pd.read_csv(path, usecols=["Date"])["Date"], utc=True)
    return dates.max().tz_localize(None).date()


def _default_as_of(fixture: Path) -> dict[str, str]:
    # Weekly bars are Monday-dated; the week completes on that Monday + 4.
    wk_last = _last_qqq_date(fixture, "weekly")
    wk_friday = wk_last - timedelta(days=wk_last.weekday()) + timedelta(days=4)
    return {"weekly": wk_friday.isoformat(), "daily": _last_qqq_date(fixture, "daily").isoformat()}


def _run_pipeline(
    repo: Path, fixture: Path, as_of: dict[str, str], out: Path, extra: list[str] | None = None
) -> None:
    """Run each mode in a fresh workspace (repo src + fixture data) and copy logs to ``out``."""
    with tempfile.TemporaryDirectory(prefix="golden_") as tmp:
        ws = Path(tmp)
        shutil.copytree(repo / "src", ws / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(fixture / "raw", ws / "data" / "raw")
        shutil.copy2(fixture / "active_tickers.csv", ws / "data" / "active_tickers.csv")
        for mode in MODES:
            cmd = [
                sys.executable,
                str(ws / "src" / "finance_vibe" / "run_vibe.py"),
                "--mode",
                mode,
                "--as-of",
                as_of[mode],
                *(extra or []),
            ]
            print(f"--> {mode} as-of {as_of[mode]}", flush=True)
            proc = subprocess.run(
                cmd,
                cwd=ws,
                capture_output=True,
                text=True,
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "PYTHONPATH": str(ws / "src"),
                    "TZ": "America/New_York",
                    "MPLBACKEND": "Agg",
                },
            )
            (out / f"run_{mode}.log").write_text(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                raise SystemExit(
                    f"{mode} pipeline failed (exit {proc.returncode}); see {out / f'run_{mode}.log'}"
                )
            src_logs = ws / "data" / "logs" / mode
            shutil.copytree(src_logs, out / mode)
            # Full Coiled Cobra scorecard (rejects included) so Score drift is
            # caught for every ticker, not just the few that pass the gates.
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_scorecard",
                    mode,
                    as_of[mode],
                    str(out / mode / "golden_cobra_scorecard.csv"),
                ],
                cwd=ws,
                capture_output=True,
                text=True,
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "PYTHONPATH": str(ws / "src"),
                    "TZ": "America/New_York",
                },
            )
            if proc.returncode != 0:
                (out / f"scorecard_{mode}.log").write_text(proc.stdout + proc.stderr)
                raise SystemExit(f"{mode} scorecard failed; see {out / f'scorecard_{mode}.log'}")


def _scorecard(mode: str, as_of: str, out_csv: str) -> int:
    """Score every raw ticker with include_rejects=True (mirrors run_scanner's loading)."""
    from finance_vibe import coiled_cobra as cc
    from finance_vibe import config

    cc.apply_timeframe(mode)  # calibration constants and RAW_DATA_DIR
    weekly = mode == "weekly"
    bench = {}
    for sym in (cc.BENCHMARK, cc.SPY_BENCHMARK):
        frame = cc.load_benchmark_frame(sym, mode)
        bench[sym] = config.cut_to_as_of(frame, as_of, weekly=weekly) if frame is not None else None
    rows = []
    for path in sorted(Path(cc.RAW_DATA_DIR).glob("*.csv")):
        symbol = path.name.split("_")[0].upper()
        row = {"Symbol": symbol}
        try:
            df = config.validate_and_clean_ohlcv(pd.read_csv(path), require_volume=True)
            df = config.cut_to_as_of(df, as_of, weekly=weekly)
            if len(df) < cc.MIN_BARS_TO_EVALUATE:
                row["Grade"] = "_insufficient_history"
            else:
                df = cc.add_macro_indicators(df)
                res = cc.evaluate_coiled_cobra(
                    df,
                    bench[cc.BENCHMARK],
                    spy_df=bench[cc.SPY_BENCHMARK],
                    qqq_df=bench[cc.BENCHMARK],
                    include_rejects=True,
                )
                if res is None:
                    row["Grade"] = "_not_scored"
                else:
                    parts = res.pop("Parts")
                    row.update(res)
                    row.update({f"part_{k}": v for k, v in parts.items()})
        except ValueError as exc:
            row["Grade"] = f"_error:{exc}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return 0


def _git_rev(repo: Path) -> str | None:
    if os.environ.get("GOLDEN_GIT_REV"):
        return os.environ["GOLDEN_GIT_REV"]
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for name in ("pandas", "numpy", "yfinance"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "?"
    return out


def cmd_baseline(args: argparse.Namespace) -> int:
    golden = Path(args.golden_dir)
    fixture = _fixture_dir(golden)
    if not fixture.exists():
        print(f"No fixture at {fixture}; run 'snapshot' first.")
        return 1
    base = _baseline_dir(golden)
    if base.exists():
        if not args.force:
            print(f"Baseline already exists at {base} (use --force to replace).")
            return 1
        shutil.rmtree(base)
    as_of = _default_as_of(fixture)
    if args.as_of_weekly:
        as_of["weekly"] = args.as_of_weekly
    if args.as_of_daily:
        as_of["daily"] = args.as_of_daily
    base.mkdir(parents=True)
    _run_pipeline(Path(args.repo), fixture, as_of, base)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "as_of": as_of,
        "git": _git_rev(Path(args.repo)),
        "versions": _versions(),
        "files": sorted(str(p.relative_to(base)) for p in base.rglob("*.csv")),
    }
    (base / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Baseline written to {base}: {len(manifest['files'])} CSV(s), as-of {as_of}")
    return 0


def _diff_csv(a: Path, b: Path, atol: float) -> list[str]:
    """Return human-readable differences between two CSVs ([] when equal)."""
    try:
        da, db = pd.read_csv(a), pd.read_csv(b)
    except pd.errors.EmptyDataError:
        return [] if a.read_bytes() == b.read_bytes() else ["one file is empty, the other is not"]
    if list(da.columns) != list(db.columns):
        return [f"columns differ: {list(da.columns)} vs {list(db.columns)}"]
    if len(da) != len(db):
        return [f"row count {len(da)} vs {len(db)}"]
    problems = []
    for col in da.columns:
        x, y = da[col], db[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            bad = ~np.isclose(xv, yv, rtol=0.0, atol=atol, equal_nan=True)
        else:
            both_na = (x.isna() & y.isna()).to_numpy()
            bad = (x.astype(str) != y.astype(str)).to_numpy() & ~both_na
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            problems.append(
                f"{col}: {int(bad.sum())} row(s) differ, first at row {i}: {x.iloc[i]!r} vs {y.iloc[i]!r}"
            )
    return problems


def cmd_compare(args: argparse.Namespace) -> int:
    golden = Path(args.golden_dir)
    base = _baseline_dir(golden)
    manifest_path = base / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"No baseline at {base}; run 'baseline' first.")
        return 1
    manifest = json.loads(manifest_path.read_text())
    with tempfile.TemporaryDirectory(prefix="golden_cmp_") as tmp:
        cur = Path(tmp)
        extra = ["--in-process"] if args.in_process else []
        _run_pipeline(Path(args.repo), _fixture_dir(golden), manifest["as_of"], cur, extra)
        want = set(manifest["files"])
        got = {str(p.relative_to(cur)) for p in cur.rglob("*.csv")}
        failures: list[str] = []
        failures += [f"missing output: {f}" for f in sorted(want - got)]
        failures += [f"new output: {f}" for f in sorted(got - want)]
        for rel in sorted(want & got):
            for msg in _diff_csv(base / rel, cur / rel, args.atol):
                failures.append(f"{rel}: {msg}")
        if args.keep:
            keep = golden / "last_compare"
            shutil.rmtree(keep, ignore_errors=True)
            shutil.copytree(cur, keep)
            print(f"Current outputs kept at {keep}")
    print(f"Baseline: git {manifest.get('git')}, as-of {manifest['as_of']}, {len(want)} CSV(s)")
    if failures:
        print(f"DIFFERENCES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("IDENTICAL: all outputs match the baseline.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_scorecard"]:  # internal: run inside the workspace's PYTHONPATH
        return _scorecard(*argv[1:4])
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--golden-dir",
        default=str(REPO_DEFAULT / "data" / "golden"),
        help="Fixture + baseline location (default: <repo>/data/golden)",
    )
    p.add_argument("--repo", default=str(REPO_DEFAULT), help="Source tree whose src/ is tested")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="Freeze raw data into the fixture")
    s.add_argument(
        "--source-data", default=str(REPO_DEFAULT / "data"), help="Live data dir to copy from"
    )
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_snapshot)

    b = sub.add_parser("baseline", help="Record outputs of --repo on the fixture")
    b.add_argument("--as-of-weekly")
    b.add_argument("--as-of-daily")
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_baseline)

    c = sub.add_parser("compare", help="Diff outputs of --repo against the baseline")
    c.add_argument("--atol", type=float, default=1e-9)
    c.add_argument(
        "--keep", action="store_true", help="Keep current outputs in <golden-dir>/last_compare"
    )
    c.add_argument(
        "--in-process", action="store_true", help="Run the pipeline with run_vibe --in-process"
    )
    c.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
