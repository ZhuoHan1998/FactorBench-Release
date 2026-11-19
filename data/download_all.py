"""Download OHLCV data for all index constituents via yfinance.

Date range: 2016-01-01 to 2025-12-31
Produces uniform "yfinance format" parquet files:
  MultiIndex [date, asset]
  Columns: open, high, low, close, volume, vwap, return_1d,
           forward_return_1d, forward_return_5d, forward_return_20d,
           adv, market_cap, tradable

Usage:
    .venv/bin/python data/download_all.py                 # current lists -> data/cache
    .venv/bin/python data/download_all.py --universe 20231231 \
        --cache_dir /path/to/panels

``--universe 20231231`` uses the point-in-time constituents as of the test
start (see index_constituents.py) instead of the later snapshots. Without
``--cache_dir`` the target is ``$FACTORBENCH_DATA`` if set, else data/cache.

``--universe pit`` downloads every stock that was a member at any time in
2016-12-31..2025-10-31 (data/index_membership.csv) and sets ``tradable`` to
"in the index on that date", which evaluation uses as a per-date mask. It also
writes ``<cache_dir>/mining/<market>.parquet``, the point-in-time mining
panel: only rows where the stock is an index member that day (qlib's
instrument-span convention), for stocks that joined on or before
MINING_CUTOFF. See ``mining_panel``. Point mining jobs at that directory.
CSI 300 names Yahoo no longer serves are filled from baostock (pip install
baostock), which keeps delisted A-shares.

Membership sources (index_membership.csv; each row's *_basis and notes say
which applies):
- S&P 100: Wikipedia revision history and S&P DJI announcements, checked
  against archived iShares OEF month-end holdings.
- FTSE 100: FTSE Russell "FTSE 100 Historic Additions and Deletions", rolled
  from the 2023-12-31 snapshot and checked against Wikipedia revisions.
- Nikkei 225: archived official component pages (Wayback); change dates are
  the scheduled review date inside each snapshot window, else bounded by it.
- HSI: official Hang Seng Indexes review releases (English or Chinese) and
  archived constituent feeds.
- CSI 300: baostock weekly constituent snapshots.
Members whose price history no free source serves have an empty
yahoo_symbol and are absent from the panel; that residual gap is reported,
not filled.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from index_constituents import (
    CONSTITUENTS_20231231,
    CSI300,
    FTSE100,
    HSI,
    NIKKEI225,
    SP100,
    YAHOO_UNAVAILABLE_20231231,
)

UNIVERSES = {
    "current": {
        "sp100": SP100,
        "ftse100": FTSE100,
        "nikkei225": NIKKEI225,
        "hsi": HSI,
        "csi300": CSI300,
    },
    "20231231": CONSTITUENTS_20231231,
}

MEMBERSHIP_CSV = Path(__file__).parent / "index_membership.csv"
# Last day before the test segment in factor_bench/data/splits.py (checked by a
# test): membership known by then is information available before testing.
MINING_CUTOFF = "2023-12-31"

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


OHLCV = ["open", "high", "low", "close", "volume"]


def backfill_frames(failed: list[str], name: str, backfill_dir: Path) -> list[pd.DataFrame]:
    """Raw OHLCV for ``failed`` tickers taken from an earlier panel of the same market.

    For names Yahoo has since stopped serving. Only raw OHLCV is copied; the
    derived columns are recomputed with everything else. ``auto_adjust`` prices
    from an earlier download can differ from today's by a constant per-ticker
    factor (dividends paid in between), which leaves returns unchanged.
    """
    path = backfill_dir / f"{name}.parquet"
    if not path.exists():
        print(f"  {name}: no backfill panel at {path}")
        return []
    old = pd.read_parquet(path, columns=OHLCV)
    old.columns = pd.Index([str(c) for c in old.columns], name=None)
    have = set(old.index.get_level_values("asset"))
    hits = [t for t in failed if t in have]
    if not hits:
        return []
    rows = old[old.index.get_level_values("asset").isin(hits)].dropna(subset=["close"])
    print(f"  {name}: BACKFILLED from {path}: {hits}")
    return [rows.reset_index()]


def load_membership(market: str) -> pd.DataFrame:
    """Membership spells for one market; ``end`` NaT means still a member."""
    spells = pd.read_csv(MEMBERSHIP_CSV, dtype=str, keep_default_na=False)
    spells = spells[spells["market"] == market].copy()
    if spells.empty:
        raise ValueError(f"no membership rows for {market!r} in {MEMBERSHIP_CSV}")
    spells["start"] = pd.to_datetime(spells["start"])
    spells["end"] = pd.to_datetime(spells["end"].replace("", None))
    return spells


def membership_flags(index: pd.MultiIndex, spells: pd.DataFrame) -> pd.Series:
    """True for (date, asset) rows inside one of that asset's membership spells."""
    dates = index.get_level_values("date")
    assets = index.get_level_values("asset")
    flag = np.zeros(len(index), dtype=bool)
    for row in spells[spells["yahoo_symbol"] != ""].itertuples():
        end = row.end if pd.notna(row.end) else pd.Timestamp.max
        flag |= (assets == row.yahoo_symbol) & (dates >= row.start) & (dates <= end)
    return pd.Series(flag, index=index)


# Must equal factor_bench.data.loader.UNIVERSE_METADATA_KEY;
# load_panel uses it to refuse a panel built for the wrong universe.
UNIVERSE_METADATA_KEY = b"factorbench.universe"


def write_panel(df: pd.DataFrame, path: Path, universe: str | None) -> None:
    """Write a panel to parquet, tagging point-in-time panels with their universe."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df)
    if universe is not None:
        meta = dict(table.schema.metadata or {})
        meta[UNIVERSE_METADATA_KEY] = universe.encode()
        table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)


def mining_panel(df: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """The panel mining methods see: only rows where the stock is an index member.

    This is qlib's instrument-span convention (``inst_calculator`` keeps only
    rows inside each instrument's membership spans), which the official
    AlphaGen-family loaders rely on when they load ``csi300``. Applying it to
    the data means every method mines point-in-time without method changes. As
    in qlib, a stock that has just joined has no pre-join history, so windowed
    features are undefined for its first days as a member.

    Two adjustments: rows of members already in the index at the start of the
    reconstructed history are kept back to the start of the download, so that
    January 2017 has warm-up history (those rows are only lookback; scoring
    starts 2017-01-01); and stocks that first joined after MINING_CUTOFF are
    dropped entirely, so mining never sees test-period membership.
    """
    history_start = spells["start"].min()
    warm = spells.copy()
    warm.loc[warm["start"] <= history_start, "start"] = pd.Timestamp(START)
    keep = membership_flags(df.index, warm)
    joined = set(spells.loc[spells["start"] <= pd.Timestamp(MINING_CUTOFF), "yahoo_symbol"])
    keep &= df.index.get_level_values("asset").isin(joined)
    return df[keep.to_numpy()]


def baostock_frames(tickers: list[str], name: str) -> list[pd.DataFrame]:
    """Raw OHLCV from baostock for A-share tickers Yahoo does not serve.

    Forward-adjusted (adjustflag="2"), the same convention as Yahoo's
    ``auto_adjust``. Halted days (tradestatus 0) are dropped, as Yahoo does not
    print them either.
    """
    try:
        import baostock as bs
    except ImportError:
        print(f"  {name}: baostock not installed; cannot fill {tickers}")
        return []
    frames = []
    bs.login()
    try:
        for ticker in tickers:
            code, exch = ticker.split(".")
            prefix = {"SS": "sh", "SZ": "sz"}.get(exch)
            if prefix is None:
                continue
            rs = bs.query_history_k_data_plus(
                f"{prefix}.{code}", "date,open,high,low,close,volume,tradestatus",
                start_date=START, end_date=END, frequency="d", adjustflag="2")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                continue
            td = pd.DataFrame(rows, columns=rs.fields)
            td = td[td["tradestatus"] == "1"]
            td[OHLCV] = td[OHLCV].apply(pd.to_numeric, errors="coerce")
            td = td.dropna(subset=["close"])
            td["date"] = pd.to_datetime(td["date"])
            td["asset"] = ticker
            frames.append(td[["date", "asset", *OHLCV]])
    finally:
        bs.logout()
    if frames:
        print(f"  {name}: FILLED from baostock: {[f['asset'].iloc[0] for f in frames]}")
    return frames


def download_index(
    tickers: list[str], name: str, cache_dir: Path, backfill_dir: Path | None = None,
    spells: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Download OHLCV for a ticker list, apply yfinance format, save as parquet."""
    out_path = cache_dir / f"{name}.parquet"
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
                # group_by="ticker" yields (ticker, field) columns even for a
                # one-ticker batch in current yfinance; older versions did not.
                td = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
                if td.empty or td["Close"].isna().all():
                    failed.append(ticker)
                    continue

                td = td.dropna(subset=["Close"]).copy()
                td = td.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                td = td[OHLCV]
                td["asset"] = ticker
                td.index.name = "date"
                all_frames.append(td.reset_index())
            except (KeyError, TypeError):
                failed.append(ticker)

        if i + BATCH_SIZE < len(tickers):
            time.sleep(2)

    if failed and backfill_dir is not None:
        all_frames.extend(backfill_frames(failed, name, backfill_dir))
    if failed and spells is not None and name == "csi300":
        have = {f["asset"].iloc[0] for f in all_frames}
        # Keep the market on Yahoo's calendar: baostock has a few dates Yahoo
        # lacks (e.g. 2019-04-29/30), which would otherwise appear as trading
        # days with only the handful of baostock-filled stocks on them.
        yahoo_dates = set(pd.to_datetime(pd.concat([f["date"] for f in all_frames])))
        for frame in baostock_frames([t for t in failed if t not in have], name):
            all_frames.append(frame[frame["date"].isin(yahoo_dates)])

    if not all_frames:
        print(f"  {name}: NO DATA downloaded!")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "asset"]).sort_index()

    # Apply uniform format
    df = to_yfinance_format(df)
    if spells is not None:
        df["tradable"] = membership_flags(df.index, spells)
        mining = mining_panel(df, spells)
        mining_dir = cache_dir / "mining"
        mining_dir.mkdir(parents=True, exist_ok=True)
        write_panel(mining, mining_dir / f"{name}.parquet", "pit-mining")
        n_mining = mining.index.get_level_values("asset").nunique()
        print(f"  {name}: mining panel {n_mining} tickers -> {mining_dir / (name + '.parquet')}")

    write_panel(df, out_path, "pit" if spells is not None else None)
    n_tickers = df.index.get_level_values("asset").nunique()
    print(f"  {name}: saved {len(df)} rows, {n_tickers} tickers -> {out_path}")

    if failed:
        print(f"  {name}: failed Yahoo download ({len(failed)}): {failed}")
    missing = sorted(set(tickers) - set(df.index.get_level_values("asset")))
    if missing:
        print(f"  {name}: NOT in panel ({len(missing)}): {missing}")

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", choices=[*sorted(UNIVERSES), "pit"], default="current")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--backfill_from", default=None,
        help="earlier cache dir whose OHLCV fills tickers Yahoo no longer serves",
    )
    args = parser.parse_args()
    backfill_dir = Path(args.backfill_from) if args.backfill_from else None

    default_dir = os.environ.get("FACTORBENCH_DATA") or Path(__file__).parent / "cache"
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(default_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"universe={args.universe}  cache_dir={cache_dir}")

    if args.universe == "pit":
        markets = ["sp100", "ftse100", "nikkei225", "hsi", "csi300"]
        plan = {}
        for m in markets:
            spells = load_membership(m)
            no_yahoo = sorted(set(spells.loc[spells["yahoo_symbol"] == "", "index_code"]))
            if no_yahoo:
                print(f"  {m}: members with no Yahoo symbol, left out ({len(no_yahoo)}): "
                      f"{no_yahoo}")
            plan[m] = (sorted(set(spells["yahoo_symbol"]) - {""}), spells)
    else:
        plan = {m: (t, None) for m, t in UNIVERSES[args.universe].items()}

    for name, (tickers, spells) in plan.items():
        print(f"\n{'='*60}")
        print(f"Downloading {name} ({len(tickers)} tickers) | {START} to {END}")
        print(f"{'='*60}")
        download_index(tickers, name, cache_dir, backfill_dir, spells)
        if args.universe == "20231231" and YAHOO_UNAVAILABLE_20231231.get(name):
            print(f"  {name}: expected missing (no Yahoo history): "
                  f"{YAHOO_UNAVAILABLE_20231231[name]}")

    print("\nDone. All files in:", cache_dir)


if __name__ == "__main__":
    main()
