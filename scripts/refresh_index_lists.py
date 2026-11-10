#!/usr/bin/env python3
"""Refresh the S&P 500 and S&P 400 constituent lists from Wikipedia.

Run by hand before each quarterly universe rebuild (third Friday of March, June,
September, December). Standard library only, so it runs on the host:

    python3 scripts/refresh_index_lists.py            # fetch, show changes, write
    python3 scripts/refresh_index_lists.py --dry-run  # fetch and show changes only

Writes ``src/finance_vibe/data/sp{500,400}_tickers.csv`` (one ``Ticker`` column,
sorted, Yahoo spelling: ``BRK.B`` -> ``BRK-B``) and ``index_lists.json`` (date
taken, count, source URL per list). A list whose size is far from its index size
is refused, so a changed page layout never writes a short list. Review the printed
joins / leaves, then commit the files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "finance_vibe" / "data"
META_PATH = DATA_DIR / "index_lists.json"
USER_AGENT = "finance-vibe-index-refresh/1.0 (personal research tool)"
# name -> (Wikipedia page, output file, smallest and largest plausible count)
LISTS = {
    "sp500": (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "sp500_tickers.csv",
        (495, 510),
    ),
    "sp400": (
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "sp400_tickers.csv",
        (395, 410),
    ),
}


class _Tables(HTMLParser):
    """Every ``wikitable`` on the page as rows of cell text."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._depth = 0  # nesting inside a wikitable
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self._depth:
                self._depth += 1
            elif "wikitable" in (dict(attrs).get("class") or ""):
                self._depth = 1
                self.tables.append([])
        elif self._depth == 1 and tag == "tr":
            self._row = []
        elif self._depth == 1 and tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._depth:
            self._depth -= 1
        elif self._depth == 1 and tag in ("td", "th") and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif self._depth == 1 and tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def symbols_from_html(html: str) -> list[str]:
    """Symbols from the first wikitable whose header has a ``Symbol`` column."""
    parser = _Tables()
    parser.feed(html)
    for table in parser.tables:
        if not table or "Symbol" not in table[0]:
            continue
        col = table[0].index("Symbol")
        found = {row[col].strip().upper().replace(".", "-") for row in table[1:] if len(row) > col}
        return sorted(s for s in found if s)
    raise ValueError("no wikitable with a Symbol column")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return [row["Ticker"] for row in csv.DictReader(fh)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="show changes, write nothing")
    args = parser.parse_args(argv)

    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    today = date.today().isoformat()
    results = {}
    for name, (url, filename, (lo, hi)) in LISTS.items():
        symbols = symbols_from_html(fetch(url))
        if not lo <= len(symbols) <= hi:
            print(f"{name}: {len(symbols)} symbols, expected {lo}-{hi}; nothing written")
            return 1
        old = read_list(DATA_DIR / filename)
        joined = sorted(set(symbols) - set(old))
        left = sorted(set(old) - set(symbols))
        print(f"{name}: {len(symbols)} symbols (was {len(old)})")
        print(f"  joined ({len(joined)}): {' '.join(joined) or '-'}")
        print(f"  left   ({len(left)}): {' '.join(left) or '-'}")
        results[name] = (filename, symbols, url)

    if args.dry_run:
        return 0
    for name, (filename, symbols, url) in results.items():
        with (DATA_DIR / filename).open("w", newline="") as fh:
            fh.write("Ticker\n" + "".join(f"{s}\n" for s in symbols))
        meta[name] = {"file": filename, "taken": today, "count": len(symbols), "source": url}
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {', '.join(f for f, _, _ in results.values())} and {META_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
