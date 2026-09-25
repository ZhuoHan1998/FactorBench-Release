"""Qlib-free ``StockData`` for AlphaCFG's embedded AlphaGen-lineage data layer.

AlphaCFG's ``unified_runner`` constructs ``StockData(instrument, start, end,
device=...)`` directly from CLI arguments, so this subclass keeps that exact
signature: the ``instrument`` string is a FactorBench market name whose panel
is loaded (and cached) from the shared contract. Window semantics mirror the
official qlib loader, as in the AlphaGen/AlphaQCM adapters (duplicated per
adapter because each vendored tree defines the same top-level package names
and must never be co-imported).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from factor_bench.data import load_panel
from factor_mining.vendor import use_vendored

use_vendored("alphacfg")

from alphacfg.data.stock_data import FeatureType, StockData  # noqa: E402

_PANEL_CACHE: dict[str, pd.DataFrame] = {}


def _panel(market: str) -> pd.DataFrame:
    if market not in _PANEL_CACHE:
        _PANEL_CACHE[market] = load_panel(market)
    return _PANEL_CACHE[market]


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
    """Signature-compatible drop-in: panels from the shared contract, no qlib."""

    def __init__(
        self,
        instrument: str,
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: list[FeatureType] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # Consumed by our _get_data() during the official constructor below.
        self._panel = _panel(instrument)
        super().__init__(
            instrument=instrument,
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
