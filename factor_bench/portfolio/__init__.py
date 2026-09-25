"""Portfolio construction and backtesting (RQ4).

Shared machinery so the strategy notebooks call one implementation rather than
each carrying their own copy of selection, weighting, execution and costs.
"""

from factor_bench.portfolio.selection import SelectionConfig, select_factors
from factor_bench.portfolio.combine import CombineConfig, build_composite, normalise_signal
from factor_bench.portfolio.backtest import (
    BacktestConfig,
    backtest,
    long_only_weights,
    long_short_weights,
    performance_metrics,
)

__all__ = [
    "SelectionConfig", "select_factors",
    "CombineConfig", "build_composite", "normalise_signal",
    "BacktestConfig", "backtest", "long_only_weights", "long_short_weights",
    "performance_metrics",
]
