"""Dependence-aware inference for test-period portfolio returns.

Daily strategy returns are autocorrelated (positions are held for a whole
rebalance interval) and the five markets share one global regime, so neither
an i.i.d. bootstrap nor five independent per-market bootstraps gives honest
intervals. This module uses a **circular block bootstrap on one common
calendar**: each draw resamples the same blocks of dates in every market at
once, so within-market autocorrelation (inside a block) and cross-market
correlation (same dates everywhere) both survive resampling.

Pooling follows the RQ4 tables exactly: a statistic is computed per run
(market x seed), averaged over seeds within a market, and the markets are then
weighted equally. Each bootstrap draw is pooled the same way, so the interval
describes the number that is reported, not a different one.

The p-value tests "statistic = 0" by recentring the bootstrap distribution on
zero (the shift method); ``holm`` then adjusts a family of such p-values, e.g.
the ten methods of one table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_bench.portfolio.backtest import TRADING_DAYS

STATS = ("mean", "sharpe")


def circular_block_indices(n: int, block: int, n_boot: int,
                           rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) row indices: blocks of ``block`` consecutive rows, wrapping."""
    if n < 1 or block < 1:
        raise ValueError(f"need n >= 1 and block >= 1, got n={n}, block={block}")
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def _column_stat(x: np.ndarray, stat: str) -> np.ndarray:
    """Annualised statistic per column over axis -2; NaN rows are skipped.

    NaN marks a date on which that run has no return (another market's holiday
    on the common calendar, or a run that has not started), not a zero return.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(x, axis=-2)
        if stat == "mean":
            return mean * TRADING_DAYS
        if stat == "sharpe":
            sd = np.nanstd(x, axis=-2, ddof=1)
            return np.where(sd > 0, mean / sd * np.sqrt(TRADING_DAYS), np.nan)
    raise ValueError(f"stat must be one of {STATS}, got {stat!r}")


def _pool(col_stats: np.ndarray, markets: np.ndarray) -> np.ndarray:
    """Seeds averaged within a market, then markets averaged equally (last axis)."""
    per_market = [np.nanmean(col_stats[..., markets == m], axis=-1)
                  for m in pd.unique(markets)]
    return np.nanmean(np.stack(per_market, axis=-1), axis=-1)


def pooled_block_bootstrap(panel: pd.DataFrame, stat: str = "mean", block: int = 20,
                           n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                           chunk: int = 250) -> dict:
    """Pooled statistic with a block-bootstrap interval and a two-sided p-value.

    Args:
        panel: daily returns on a common calendar (the union of the markets'
            trading days), one column per run, columns a MultiIndex whose first
            level is the market. NaN where that run has no return that day.
        stat: ``"mean"`` (annualised mean, e.g. excess return over a benchmark)
            or ``"sharpe"`` (annualised Sharpe ratio).
        block: block length in trading days; the rebalance interval is the
            natural choice, since that is how long a position persists.

    Returns:
        estimate, ci_low, ci_high, p_value, plus the settings used.
    """
    if not isinstance(panel.columns, pd.MultiIndex):
        raise ValueError("panel columns must be a (market, run) MultiIndex")
    x = panel.sort_index().to_numpy(dtype=float)
    markets = np.asarray(panel.columns.get_level_values(0))
    estimate = float(_pool(_column_stat(x, stat), markets))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        idx = circular_block_indices(len(x), block, hi - lo, rng)
        draws[lo:hi] = _pool(_column_stat(x[idx], stat), markets)
    draws = draws[np.isfinite(draws)]

    # Shift method: recentre the draws on the null (statistic = 0) and ask how
    # often a draw lands at least as far from it as the estimate did.
    extreme = np.abs(draws - estimate) >= abs(estimate)
    p_value = float((1 + extreme.sum()) / (1 + len(draws)))
    lo_q, hi_q = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"estimate": estimate, "ci_low": float(lo_q), "ci_high": float(hi_q),
            "p_value": p_value, "n_boot": int(len(draws)), "block": block,
            "n_markets": int(len(pd.unique(markets))), "n_runs": int(x.shape[1])}


def holm(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjustment; controls the family-wise error rate."""
    p = p_values.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p.to_numpy())
    out = pd.Series(np.minimum(adj, 1.0), index=p.index)
    return out.reindex(p_values.index)


def run_panel(curves: pd.DataFrame, value: str = "net",
              benchmark: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Daily returns of one method's runs on the common calendar.

    ``curves`` holds one method's rows (date, market, seed, ``value``). With
    ``benchmark`` ({market: daily return}), each run's benchmark return is
    subtracted first, matching ``performance_metrics``: a benchmark gap on a
    trading day counts as a zero benchmark return.
    """
    cols = {}
    for (market, seed), g in curves.groupby(["market", "seed"]):
        r = g.set_index("date")[value].sort_index()
        if benchmark is not None:
            if market not in benchmark:
                raise KeyError(f"no benchmark series for {market!r}")
            r = r - benchmark[market].reindex(r.index).fillna(0.0)
        cols[(market, seed)] = r
    panel = pd.DataFrame(cols)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["market", "seed"])
    return panel
