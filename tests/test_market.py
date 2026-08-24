"""Raw-file selection must follow TIMEFRAME_PROFILES (daily 10y / 1d)."""

from pathlib import Path

from finance_vibe import config
from finance_vibe.market import parse_raw_filename, select_raw_paths


def test_parse_raw_filename_daily_ten_year():
    ticker, period, interval = parse_raw_filename("AAPL_10y_1d.csv")
    assert ticker == "AAPL"
    assert period == "10y"
    assert interval == "1d"


def test_select_raw_paths_prefers_configured_period(tmp_path: Path):
    (tmp_path / "AAPL_5y_1d.csv").write_text("Date,Close\n2020-01-01,1\n", encoding="utf-8")
    (tmp_path / "AAPL_10y_1d.csv").write_text("Date,Close\n2016-01-01,1\n", encoding="utf-8")
    (tmp_path / "MSFT_5y_1d.csv").write_text("Date,Close\n2020-01-01,1\n", encoding="utf-8")

    cfg = {"period": "10y", "interval": "1d", "raw_dir": str(tmp_path)}
    paths = select_raw_paths(str(tmp_path), cfg=cfg)
    names = sorted(Path(p).name for p in paths)
    assert names == ["AAPL_10y_1d.csv", "MSFT_5y_1d.csv"]


def test_select_raw_paths_one_file_per_ticker(tmp_path: Path):
    (tmp_path / "QQQ_2y_1d.csv").write_text("x", encoding="utf-8")
    (tmp_path / "QQQ_5y_1d.csv").write_text("x", encoding="utf-8")
    (tmp_path / "QQQ_10y_1d.csv").write_text("x", encoding="utf-8")
    cfg = config.get_mode_config("daily")
    paths = select_raw_paths(str(tmp_path), cfg=cfg)
    assert len(paths) == 1
    assert Path(paths[0]).name == "QQQ_10y_1d.csv"
