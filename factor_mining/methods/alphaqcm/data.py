"""Qlib-free ``StockData`` for AlphaQCM's embedded AlphaGen fork.

AlphaQCM vendors an older AlphaGen whose ``StockData`` has no
``preloaded_data`` hook, so unlike the AlphaGen adapter we override
``_get_data()`` in a subclass. The tensor construction mirrors the official
qlib loader semantics exactly (same as ``factor_mining/methods/alphagen/data.py``,
duplicated here because the two vendored trees define the same top-level
``alphagen`` package and must never be imported into one process):

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

use_vendored("alphaqcm")

from alphagen_qlib.stock_data import FeatureType, StockData  # noqa: E402


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
    values = np.stack(
        [wide[col].reindex(index=window_dates, columns=assets).to_numpy(dtype=np.float64)
         for col in columns],
        axis=1,
    )
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
        # Consumed by our _get_data() during the official constructor below.
        self._panel = panel
        super().__init__(
            instrument=market,
            start_time=start_time,
            end_time=end_time,
            max_backtrack_days=max_backtrack_days,
            max_future_days=max_future_days,
            features=features,
            device=device,
        )

    @classmethod
    def _init_qlib(cls) -> None:
        # FactorBench feeds pre-built tensors; qlib is never imported or run.
        return

    def _get_data(self) -> tuple[torch.Tensor, pd.DatetimeIndex, pd.Index]:
        return _build_tensor(
            self._panel,
            self._features,
            self._start_time,
            self._end_time,
            self.max_backtrack_days,
            self.max_future_days,
            self.device,
        )
