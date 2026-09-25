"""Prepare FactorBench panels in the format RD-Agent's factor pipeline expects.

The official pipeline reads HDF5 files from two folders (full + debug), each
described to the LLM automatically from the files' actual schema:

- index: MultiIndex named ``(datetime, instrument)`` — literal names required
  by RD-Agent's evaluators and index normalization;
- HDF key: ``"data"``;
- columns: input fields only. Forward-return columns are deliberately NOT
  exported — the generated code must never see the target (leakage guard).

The debug folder (what the coder loop iterates against) is a slice of the
TRAIN segment only, mirroring the official debug subset, so generated code is
developed without touching valid/test-period data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, load_panel
from factor_bench.data.schema import FIELDS

# Last two years of the train segment, mirroring the official 2-year debug slice.
_DEBUG_START = "2020-01-01"

_README = """# FactorBench market data

`daily_pv.h5` (HDF5 key: `"data"`): daily panel with MultiIndex
`(datetime, instrument)` and columns:

- `open, high, low, close, volume`: daily OHLCV (prices split/dividend adjusted)
- `vwap`: (high + low + close) / 3 proxy
- `return_1d`: close-to-close 1-day return
- `adv`: 20-day average daily volume

Load with `pd.read_hdf("daily_pv.h5", key="data")`. All time-series operations
must stay within each instrument; cross-sectional operations group by datetime.
"""


def prepare_data_folders(
    # Shared per-market cache, deliberately outside the per-run directories
    # (out/mining/rdagent/<market>_<loops>_<seed>) so runs reuse one copy.
    market: str, base_dir: str | Path = "out/mining/rdagent/data"
) -> tuple[Path, Path]:
    """Write full + debug data folders for one market. Returns their paths."""
    panel = load_panel(market)[list(FIELDS)]
    panel.index = panel.index.rename(["datetime", "instrument"])

    base = Path(base_dir) / market
    full_dir, debug_dir = base / "full", base / "debug"
    for d in (full_dir, debug_dir):
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(_README)

    panel.to_hdf(full_dir / "daily_pv.h5", key="data")
    train_end = DEFAULT_SPLIT.train[1]
    debug = panel.loc[pd.Timestamp(_DEBUG_START) : pd.Timestamp(train_end)]
    debug.to_hdf(debug_dir / "daily_pv.h5", key="data")
    return full_dir, debug_dir
