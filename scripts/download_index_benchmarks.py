#!/usr/bin/env python3
"""Snapshot the five published index price series to a committed CSV.

Why a snapshot and not a live download
--------------------------------------
Yahoo revises history silently -- splits, backfills, occasional bad prints -- so
a notebook that downloads at run time is not reproducible: the same code gives
different figures next month. This writes one dated CSV that goes into git, and
the notebooks read only that.

PRICE indices, not total return
-------------------------------
Every ticker below is a price index: it excludes dividends. The benchmark's own
returns come from ``close`` downloaded with ``auto_adjust=True``, so they are
TOTAL returns. The strategies therefore carry the dividend yield (roughly
1.3% for the S&P 100 up to ~3.5% for the FTSE and HSI) that the index line does
not. Over the ~1.8-year test window that is 2-6 percentage points of unearned
advantage, so the index line is a LOWER bound on what an investor could have
had, and must be described as one wherever it is plotted.

A total-return variant is not available on Yahoo for most of these; ^SP500TR
exists but tracks the S&P 500, a different universe from our S&P 100.

The index is also not our universe: it reflects committee changes over time,
while the benchmark uses a static constituent snapshot. That cuts the opposite
way from the dividend gap -- our universe is survivorship-flattered -- so the
two biases do not cancel in a knowable amount and neither should be waved away.

Usage:
    .venv/bin/python scripts/download_index_benchmarks.py
    .venv/bin/python scripts/download_index_benchmarks.py --start 2016-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

# Our market name -> Yahoo ticker. All price series (no dividends).
#
# csi300 is the exception: Yahoo returns no history for the index itself
# (000300.SS and 399300.SZ both give a single day), so this uses the
# Huatai-PineBridge CSI 300 ETF, which tracks it and is CNY-denominated like our
# panel. It is a fund, so it carries tracking error and a management fee, and
# its price return is net of any distribution -- call it a CSI 300 TRACKER in
# any caption, never "the CSI 300". The US-listed ASHR was rejected: it is
# USD-denominated, so it would mix in the CNY/USD move.
TICKERS = {
    "sp100": "^OEX",
    "hsi": "^HSI",
    "csi300": "510300.SS",
    "ftse100": "^FTSE",
    "nikkei225": "^N225",
}
# Printed in the notebooks so a figure never implies more than the data is.
LABELS = {
    "sp100": "S&P 100 (price)",
    "hsi": "Hang Seng (price)",
    "csi300": "CSI 300 ETF tracker",
    "ftse100": "FTSE 100 (price)",
    "nikkei225": "Nikkei 225 (price)",
}

OUT = Path("data/index_benchmarks.csv")


def main(start: str = "2016-01-01", end: str | None = None,
         out: str = str(OUT)) -> None:
    end = end or dt.date.today().isoformat()
    frames = []
    for market, ticker in TICKERS.items():
        # auto_adjust is irrelevant for an index (no dividends or splits in the
        # series) but is passed explicitly so the call is unambiguous.
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False,
                          progress=False)
        if raw.empty:
            print(f"  {market:<10} {ticker:<12} NO DATA -- skipped")
            continue
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):      # yfinance may return a frame
            close = close.iloc[:, 0]
        s = close.dropna()
        frames.append(pd.DataFrame({"market": market, "ticker": ticker,
                                    "label": LABELS[market],
                                    "date": s.index, "close": s.to_numpy()}))
        print(f"  {market:<10} {ticker:<12} {len(s):>5} days  "
              f"{s.index[0].date()} .. {s.index[-1].date()}")

    if not frames:
        raise SystemExit("nothing downloaded")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["market", "date"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows, downloaded {dt.date.today().isoformat()})")
    print("PRICE indices: no dividends. The strategies' returns DO include them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(OUT))
    main(**vars(ap.parse_args()))
