"""Deprecated CLI — ranking lives in ``trade_planner.finalize_trade_plan``.

``run_vibe.py`` no longer invokes this module. Imports remain for tests.
"""
from finance_vibe.trade_planner import (  # noqa: F401
    apply_ingestion_filters,
    process_trade_plan,
    rank_by_expected_value,
    resolve_trade_plan_path,
)


def main(argv=None) -> int:
    from finance_vibe.trade_planner import process_trade_plan as _process
    from finance_vibe import config
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    mode = config.DEFAULT_MODE
    if argv and argv[0].lower() in ("weekly", "daily", "high_beta"):
        mode = argv[0].lower()
    try:
        _process(mode)
    except FileNotFoundError as exc:
        print(f"{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
