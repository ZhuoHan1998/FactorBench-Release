"""Loader for the cached market panels in ``data/cache`` (or ``$FACTORBENCH_DATA``)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from factor_bench.data.schema import AUX_COLUMNS, FIELDS, INDEX_NAMES, MARKETS
from factor_bench.paths import DATA_DIR

_DEFAULT_CACHE_DIR = DATA_DIR


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
            f"Missing cached panel {path}. Run data/download_all.py to build the cache"
            " (with --cache_dir if FACTORBENCH_DATA is set)."
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
    required = os.environ.get(REQUIRE_UNIVERSE_ENV)
    if required:
        _check_universe(panel, path, required)
    return panel


#: Parquet schema-metadata key naming the universe a panel was built for
#: (written by data/download_all.py: "pit" or "pit-mining"; absent = static).
UNIVERSE_METADATA_KEY = b"factorbench.universe"
#: Set to "pit-mining" (or "pit") to make every load_panel call refuse a panel
#: built for another universe. Subprocesses inherit it, so one export in a
#: mining job covers the vendored agents' own data loading too.
REQUIRE_UNIVERSE_ENV = "FACTORBENCH_REQUIRE_UNIVERSE"


def panel_universe(path: str | Path) -> str:
    """The universe tag stored in a cached panel's parquet metadata ("static" if none)."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    return meta.get(UNIVERSE_METADATA_KEY, b"static").decode()


def _check_universe(panel: pd.DataFrame, path: Path, required: str) -> None:
    """Fail loudly unless the panel is the point-in-time universe the job expects."""
    tag = panel_universe(path)
    if tag != required:
        raise ValueError(
            f"{REQUIRE_UNIVERSE_ENV}={required!r} but {path} is a {tag!r} panel; "
            "point FACTORBENCH_DATA at the right cache (see data/download_all.py)"
        )
    if required == "pit-mining":
        # Rows before the first scored date are warm-up history and may be
        # non-member rows; from then on, every row must be an index member.
        from factor_bench.data.splits import DEFAULT_SPLIT

        scored = panel.index.get_level_values("date") >= pd.Timestamp(DEFAULT_SPLIT.train[0])
        outside = int((scored & ~universe_mask(panel).to_numpy()).sum())
        if outside:
            raise ValueError(f"{path}: {outside} rows from {DEFAULT_SPLIT.train[0]} on are "
                             "not index members; this is not a point-in-time mining panel")


def universe_mask(panel: pd.DataFrame) -> pd.Series:
    """(date, asset) -> True where the asset is in the index universe on that date.

    Read from the ``tradable`` column. Panels built from a static constituent
    list have it True everywhere, so every use of this mask is a no-op on them.
    Point-in-time panels (``data/download_all.py --universe pit``) set it from
    index membership, so evaluation only scores a stock while it is a member.
    Factor inputs are NOT masked: rolling features still see a stock's full
    price history, including the days before it joined.
    """
    return panel["tradable"].fillna(False).astype(bool)


def restrict_to_universe(signal: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Set a signal to NaN wherever the asset is not in the universe on that date."""
    mask = universe_mask(panel).reindex(signal.index, fill_value=False)
    return signal.where(mask)


def universe_key(panel: pd.DataFrame) -> str:
    """Short fingerprint of the universe, for caches keyed on more than factor names.

    ``"static"`` for an all-True mask, which is also what caches written before
    this key existed are taken to mean.
    """
    mask = universe_mask(panel)
    if bool(mask.all()):
        return "static"
    digest = hashlib.sha1()
    digest.update("\n".join(map(str, panel.index.get_level_values("asset").unique())).encode())
    digest.update(np.packbits(mask.to_numpy()).tobytes())
    return f"pit-{digest.hexdigest()[:12]}"


def trading_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """The market's own trading calendar, taken from the panel's date level."""
    return panel.index.get_level_values("date").unique().sort_values()
