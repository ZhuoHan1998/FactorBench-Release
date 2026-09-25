"""Loader for the cached market panels in ``data/cache``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factor_bench.data.schema import AUX_COLUMNS, FIELDS, INDEX_NAMES, MARKETS

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def load_panel(market: str, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Load one market's canonical panel.

    Returns a DataFrame with MultiIndex ``(date, asset)`` (sorted) and the
    canonical field + auxiliary columns. All mining methods and evaluation
    load data through this function only.
    """
    if market not in MARKETS:
        raise ValueError(f"Unknown market {market!r}; expected one of {MARKETS}")
    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    path = cache / f"{market}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached panel {path}. Run data/download_all.py to build the cache."
        )
    panel = pd.read_parquet(path)

    if tuple(panel.index.names) != INDEX_NAMES:
        raise ValueError(f"{path} index names {panel.index.names} != {INDEX_NAMES}")
    # yfinance leaves a named column index ("Price"); normalize it away.
    panel.columns = pd.Index([str(c) for c in panel.columns], name=None)
    missing = [c for c in (*FIELDS, *AUX_COLUMNS) if c not in panel.columns]
    if missing:
        raise ValueError(f"{path} missing canonical columns: {missing}")
    if not panel.index.is_monotonic_increasing:
        panel = panel.sort_index()
    return panel


def trading_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """The market's own trading calendar, taken from the panel's date level."""
    return panel.index.get_level_values("date").unique().sort_values()
