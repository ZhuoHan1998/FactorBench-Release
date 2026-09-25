"""Canonical prediction target shared by all mining methods and evaluation.

The target is the 20-trading-day forward close-to-close return, observed at
the close of day t:

    target[t] = close[t + 20] / close[t] - 1

This matches the cached ``forward_return_20d`` column exactly, and matches
AlphaGen's official target ``Ref($close, -20) / $close - 1`` so tree-based
and code-based methods optimize the same quantity. There is no execution
delay baked into the target; cost/turnover-aware evaluation applies delay
explicitly downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    name: str
    column: str
    horizon_days: int
    definition: str

    def from_panel(self, panel: pd.DataFrame) -> pd.Series:
        """Extract the target series ((date, asset) index) from a canonical panel."""
        return panel[self.column]

    def recompute(self, panel: pd.DataFrame) -> pd.Series:
        """Recompute the target from close prices (per asset, time-ordered)."""
        close = panel["close"].unstack("asset")
        fwd = close.shift(-self.horizon_days) / close - 1.0
        return fwd.stack().reindex(panel.index)

    def validate(self, panel: pd.DataFrame, atol: float = 1e-10) -> None:
        """Assert the cached target column matches its definition (leakage guard)."""
        cached = self.from_panel(panel)
        recomputed = self.recompute(panel)
        both = cached.notna() & recomputed.notna()
        if not np.allclose(cached[both], recomputed[both], atol=atol):
            raise ValueError(f"Cached column {self.column!r} does not match {self.definition!r}")


TARGET = TargetSpec(
    name="forward_return_20d",
    column="forward_return_20d",
    horizon_days=20,
    definition="close[t+20] / close[t] - 1",
)
