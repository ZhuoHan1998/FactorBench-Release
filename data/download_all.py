"""Download OHLCV data for all index constituents via yfinance.

Date range: 2016-01-01 to 2025-12-31
Produces uniform "yfinance format" parquet files:
  MultiIndex [date, asset]
  Columns: open, high, low, close, volume, vwap, return_1d,
           forward_return_1d, forward_return_5d, forward_return_20d,
           adv, market_cap, tradable
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from index_constituents import SP100, FTSE100, NIKKEI225, HSI, CSI300

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

START = "2016-01-01"
END = "2025-12-31"
BATCH_SIZE = 50


def to_yfinance_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw OHLCV DataFrame (MultiIndex [date, asset]) to yfinance format."""
    close_panel = df["close"].unstack("asset")
    vol_panel = df["volume"].unstack("asset")

    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    df["return_1d"] = close_panel.pct_change(1).stack()
    df["forward_return_1d"] = close_panel.pct_change(1).shift(-1).stack()
    df["forward_return_5d"] = (close_panel.shift(-5) / close_panel - 1).stack()
    df["forward_return_20d"] = (close_panel.shift(-20) / close_panel - 1).stack()
    df["adv"] = vol_panel.rolling(20, min_periods=5).mean().stack()
    df["market_cap"] = df["close"] * df["adv"]
    df["tradable"] = True

    return df.sort_index()


def download_index(tickers: list[str], name: str) -> pd.DataFrame:
    """Download OHLCV for a ticker list, apply yfinance format, save as parquet."""
    out_path = CACHE_DIR / f"{name}.parquet"
    if out_path.exists():
        print(f"  {name}: already cached at {out_path}")
        return pd.read_parquet(out_path)

    all_frames = []
    failed = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        print(f"  {name}: batch {i // BATCH_SIZE + 1}/{-(-len(tickers) // BATCH_SIZE)} ({len(batch)} tickers)...")

        try:
            raw = yf.download(
                batch,
                start=START,
                end=END,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"    batch failed: {e}")
            failed.extend(batch)
            time.sleep(5)
            continue

        for ticker in batch:
            try:
                td = raw if len(batch) == 1 else raw[ticker]
                if td.empty or td["Close"].isna().all():
                    failed.append(ticker)
                    continue

                td = td.dropna(subset=["Close"]).copy()
                td = td.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                td = td[["open", "high", "low", "close", "volume"]]
                td["asset"] = ticker
                td.index.name = "date"
                all_frames.append(td.reset_index())
            except (KeyError, TypeError):
                failed.append(ticker)

        if i + BATCH_SIZE < len(tickers):
            time.sleep(2)

    if not all_frames:
        print(f"  {name}: NO DATA downloaded!")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "asset"]).sort_index()

    # Apply uniform format
    df = to_yfinance_format(df)

    df.to_parquet(out_path)
    n_tickers = df.index.get_level_values("asset").nunique()
    print(f"  {name}: saved {len(df)} rows, {n_tickers} tickers -> {out_path}")

    if failed:
        print(f"  {name}: failed tickers ({len(failed)}): {failed}")

    return df


def main():
    indices = [
        ("sp100", SP100),
        ("ftse100", FTSE100),
        ("nikkei225", NIKKEI225),
        ("hsi", HSI),
        ("csi300", CSI300),
    ]

    for name, tickers in indices:
        print(f"\n{'='*60}")
        print(f"Downloading {name} ({len(tickers)} tickers) | {START} to {END}")
        print(f"{'='*60}")
        download_index(tickers, name)

    print("\nDone. All files in:", CACHE_DIR)


if __name__ == "__main__":
    main()
