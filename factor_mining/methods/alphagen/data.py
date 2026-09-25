"""Qlib-free ``StockData`` built from the FactorBench canonical panel.

The vendored AlphaGen core consumes market data as a ``StockData`` tensor of
shape ``(backtrack + n_days + future, n_features, n_stocks)``. Officially that
tensor is loaded through qlib; here it is built from the shared data contract
(`factor_bench.data`) with the same window semantics as the official loader
(`alphagen_qlib/stock_data.py::StockData._load_exprs`):

- the requested inclusive ``[start_time, end_time]`` window is extended by
  ``max_backtrack_days`` before and ``max_future_days`` after, on the market's
  own trading calendar;
- an ``end_time`` that is not a trading day snaps back to the previous one;
- assets with no data at all inside the extended window are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from factor_mining.vendor import use_vendored

use_vendored("alphagen")

import alphagen_qlib.stock_data as _stock_data_module  # noqa: E402
from alphagen_qlib.stock_data import FeatureType, StockData  # noqa: E402

# The vendored StockData initializes qlib on first construction. FactorBench
# always passes pre-built tensors, so mark qlib as initialized up front; no
# qlib code is ever imported or run.
_stock_data_module._QLIB_INITIALIZED = True


def _build_tensor(
    panel: pd.DataFrame,
    features: list[FeatureType],
    start_time: str,
    end_time: str,
    max_backtrack_days: int,
    max_future_days: int,
    device: torch.device,
) -> tuple[torch.Tensor, pd.DatetimeIndex, pd.Index]:
    dates = panel.index.get_level_values("date").unique().sort_values()
    start_index = int(dates.searchsorted(pd.Timestamp(start_time)))
    end_index = int(dates.searchsorted(pd.Timestamp(end_time)))
    # Official loader convention: a non-trading end_time snaps to the
    # previous trading day.
    if end_index == len(dates) or dates[end_index] != pd.Timestamp(end_time):
        end_index -= 1
    if start_index - max_backtrack_days < 0:
        raise ValueError(
            f"Not enough history before {start_time}: need {max_backtrack_days} "
            f"trading days of warm-up, have {start_index}"
        )
    if end_index + max_future_days >= len(dates):
        raise ValueError(
            f"Not enough data after {end_time}: need {max_future_days} trading "
            f"days of future buffer, have {len(dates) - 1 - end_index}"
        )
    window_dates = dates[start_index - max_backtrack_days : end_index + max_future_days + 1]

    window = panel.loc[window_dates[0] : window_dates[-1]]
    assets = window.index.get_level_values("asset").unique().sort_values()
    columns = [f.name.lower() for f in features]
    wide = window[columns].unstack("asset")
    # (n_dates, n_features, n_assets), features in FeatureType order.
    values = np.stack(
        [wide[col].reindex(index=window_dates, columns=assets).to_numpy(dtype=np.float64)
         for col in columns],
        axis=1,
    )
    # Official subview behavior: drop assets with no observations in-window.
    alive = ~np.isnan(values).all(axis=(0, 1))
    values = values[:, :, alive]
    tensor = torch.tensor(values, dtype=torch.float, device=device)
    return tensor, pd.DatetimeIndex(window_dates), pd.Index(assets[alive])


class PanelStockData(StockData):
    """A ``StockData`` whose tensor comes from the shared contract, not qlib."""

    def __init__(
        self,
        panel: pd.DataFrame,
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: list[FeatureType] | None = None,
        device: torch.device = torch.device("cpu"),
        market: str = "unknown",
    ) -> None:
        features = list(features) if features is not None else list(FeatureType)
        preloaded = _build_tensor(
            panel, features, start_time, end_time, max_backtrack_days, max_future_days, device
        )
        super().__init__(
            instrument=market,
            start_time=start_time,
            end_time=end_time,
            max_backtrack_days=max_backtrack_days,
            max_future_days=max_future_days,
            features=features,
            device=device,
            preloaded_data=preloaded,
        )
