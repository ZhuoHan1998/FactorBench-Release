"""Prepare FactorBench panels in the format AlphaAgent's factor DSL expects.

The generated factor code (template.jinjia2) reads ``./daily_pv.h5``
(key ``"data"``) with MultiIndex named exactly ``(datetime, instrument)`` and
``$``-prefixed columns, and the official Factor Agent prompt restricts
expressions to the variables ``$open $close $high $low $volume $return``.
We export exactly that official variable set — ``$return`` is the 1-day
close-to-close return, the same definition as the official
``factor_data_template/generate.py`` — so prompts and search space stay
official with no prompt edits. Forward returns are never exported (leakage
guard); ``vwap``/``adv`` are not exposed to this method.

The debug folder (used by the coder-evaluator loop) is a slice of the TRAIN
segment only, mirroring the official debug subset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, load_panel

# Official variable set (prompts_alphaagent.yaml): panel column -> $column.
_EXPORT_COLUMNS = {
    "open": "$open",
    "high": "$high",
    "low": "$low",
    "close": "$close",
    "volume": "$volume",
    "return_1d": "$return",
}

# Last two years of the train segment (official debug uses a small slice).
_DEBUG_START = "2020-01-01"

_README = """# FactorBench market data (AlphaAgent DSL format)

`daily_pv.h5` (HDF5 key: `"data"`): daily panel with MultiIndex
`(datetime, instrument)` and columns:

- `$open, $high, $low, $close, $volume`: daily OHLCV (split/dividend adjusted)
- `$return`: 1-day close-to-close return

Load with `pd.read_hdf("daily_pv.h5", key="data")`.
"""


def prepare_data_folders(
    market: str, base_dir: str | Path = "out/mining/alphaagent/data"
) -> tuple[Path, Path]:
    """Write full + debug data folders for one market. Returns absolute paths."""
    panel = load_panel(market)[list(_EXPORT_COLUMNS)].rename(columns=_EXPORT_COLUMNS)
    panel.index = panel.index.rename(["datetime", "instrument"])

    base = Path(base_dir).resolve() / market
    full_dir, debug_dir = base / "full", base / "debug"
    for d in (full_dir, debug_dir):
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(_README)

    panel.to_hdf(full_dir / "daily_pv.h5", key="data")
    train_end = DEFAULT_SPLIT.train[1]
    debug = panel.loc[pd.Timestamp(_DEBUG_START) : pd.Timestamp(train_end)]
    debug.to_hdf(debug_dir / "daily_pv.h5", key="data")
    return full_dir, debug_dir
