"""Execution, costs and performance — one engine for every strategy.

Conventions, identical across methods, markets and strategies:

- **Next-day execution.** A signal formed at the close of ``t`` sets target
  weights traded at ``t+1``; returns accrue from ``t+1`` onward. Nothing is
  earned on the day the signal is observed.
- **Daily holdings-based returns.** Between rebalances weights drift with
  prices; they are not silently reset to target.
- **Turnover against drifted holdings.** Measured as the one-way traded
  fraction ``0.5 * sum |w_target - w_drifted|``, so ordinary price drift is not
  billed as a trade.
- **Costs** charged on the dollar traded at each rebalance, in bps.
- **Long-short financing**: a flat annual borrow on the average short notional,
  accrued daily, no rebate on short proceeds. Reported separately from trading
  cost so it can be varied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestConfig:
    rebalance_days: int = 5           # main setting; 1 and 20 tested separately
    cost_bps: float = 10.0            # per dollar traded, one way
    quantile: float = 0.20            # top/bottom 20%
    gross_long: float = 1.0           # long-only: fully invested
    ls_long: float = 0.5              # long-short: +50% / -50% at rebalance
    ls_short: float = 0.5
    borrow_bps_per_year: float = 50.0  # short financing, no rebate
    execution_lag: int = 1            # next-day execution
    min_names: int = 10               # skip a date with too thin a cross-section


# ------------------------------------------------------------------ weights
def long_only_weights(scores: pd.Series, q: float, min_names: int) -> pd.Series:
    """Equal weight the top q of stocks, fully invested."""
    s = scores.dropna()
    n = len(s)
    if n < min_names:
        return pd.Series(dtype=float)
    k = max(int(np.floor(n * q)), 1)
    top = s.nlargest(k).index
    return pd.Series(1.0 / k, index=top)


def long_short_weights(scores: pd.Series, q: float, min_names: int,
                       gross_long: float, gross_short: float) -> pd.Series:
    """Equal weight top q long and bottom q short at the stated exposures."""
    s = scores.dropna()
    n = len(s)
    if n < 2 * min_names:
        return pd.Series(dtype=float)
    k = max(int(np.floor(n * q)), 1)
    top, bot = s.nlargest(k).index, s.nsmallest(k).index
    w = pd.Series(0.0, index=top.union(bot))
    w.loc[top] = gross_long / k
    w.loc[bot] = -gross_short / k
    return w


# ------------------------------------------------------------------ engine
def backtest(scores: pd.Series, returns: pd.DataFrame, config: BacktestConfig,
             kind: str = "long_only") -> pd.DataFrame:
    """Run one strategy.

    Args:
        scores: composite signal, (date, asset) MultiIndex.
        returns: wide daily simple returns, dates x assets.
        kind: ``long_only`` or ``long_short``.

    Returns:
        Daily frame with gross/net returns, turnover and costs. Empty when the
        signal never produced a tradeable book.
    """
    wide = scores.unstack("asset") if isinstance(scores.index, pd.MultiIndex) else scores
    wide = wide.reindex(columns=returns.columns)
    dates = returns.index
    wide = wide.reindex(dates)

    form_dates = dates[::config.rebalance_days]
    daily_borrow = config.borrow_bps_per_year / 1e4 / TRADING_DAYS

    held = pd.Series(dtype=float)
    rows = []
    # Pending trades keyed by the index at which they become *held*. A signal
    # observed at the close of t is traded during t+lag and is only in the book
    # at that close, so the first return it earns is r[t+lag+1]. Applying it one
    # day earlier would credit the strategy with r[t+1] -- same-day execution
    # economics under a next-day label, which is the classic way a backtest
    # invents alpha it could not have captured.
    pending: dict[int, pd.Series] = {}
    for i, d in enumerate(dates):
        turnover = 0.0
        if i in pending:
            target = pending.pop(i)
            union = held.index.union(target.index)
            a = held.reindex(union, fill_value=0.0)
            b = target.reindex(union, fill_value=0.0)
            # turnover against the DRIFTED book, so price drift is not billed
            turnover = 0.5 * float((b - a).abs().sum())
            held = b[b != 0.0]

        r = returns.loc[d]
        gross = float((held * r.reindex(held.index)).sum()) if len(held) else 0.0
        trade_cost = turnover * config.cost_bps / 1e4
        borrow = daily_borrow * float(held[held < 0].abs().sum()) if len(held) else 0.0
        rows.append({"date": d, "gross": gross, "turnover": turnover,
                     "trade_cost": trade_cost, "borrow_cost": borrow,
                     "net": gross - trade_cost - borrow,
                     "n_long": int((held > 0).sum()), "n_short": int((held < 0).sum()),
                     "gross_exposure": float(held.abs().sum()) if len(held) else 0.0,
                     "net_exposure": float(held.sum()) if len(held) else 0.0})

        # weights drift with prices between rebalances
        if len(held):
            grown = held * (1.0 + r.reindex(held.index).fillna(0.0))
            tot = grown.abs().sum()
            held = grown / tot * held.abs().sum() if tot > 0 else grown

        # form a target from today's signal; it enters the book at i+lag+1
        if d in form_dates:
            effective = i + config.execution_lag + 1
            if effective < len(dates):
                s_t = wide.loc[d].dropna()
                if kind == "long_only":
                    tgt = long_only_weights(s_t, config.quantile, config.min_names)
                    if len(tgt):
                        tgt = tgt * config.gross_long
                elif kind == "long_short":
                    tgt = long_short_weights(s_t, config.quantile, config.min_names,
                                             config.ls_long, config.ls_short)
                else:
                    raise ValueError(f"unknown strategy {kind!r}")
                if len(tgt):
                    pending[effective] = tgt
    return pd.DataFrame(rows).set_index("date")


# ------------------------------------------------------------------ metrics
def performance_metrics(daily: pd.DataFrame, benchmark: pd.Series | None = None,
                        rebalance_days: int = 5) -> dict:
    """CAGR, Sharpe, drawdown and turnover, gross and net."""
    if daily.empty:
        return {}
    out: dict[str, float] = {}
    n = len(daily)
    years = n / TRADING_DAYS
    for tag in ("gross", "net"):
        r = daily[tag]
        wealth = (1.0 + r).cumprod()
        total = float(wealth.iloc[-1])
        out[f"{tag}_cagr"] = total ** (1 / years) - 1 if total > 0 and years > 0 else np.nan
        sd = r.std()
        out[f"{tag}_sharpe"] = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan
        dd = wealth / wealth.cummax() - 1.0
        out[f"{tag}_max_dd"] = float(dd.min())
    out["turnover_per_rebalance"] = float(daily["turnover"][daily["turnover"] > 0].mean()) \
        if (daily["turnover"] > 0).any() else 0.0
    out["turnover_annual"] = float(daily["turnover"].sum() / years) if years > 0 else np.nan
    out["avg_gross_exposure"] = float(daily["gross_exposure"].mean())
    out["avg_net_exposure"] = float(daily["net_exposure"].mean())
    out["borrow_cost_annual"] = float(daily["borrow_cost"].sum() / years) if years > 0 else np.nan
    out["days"] = n
    if benchmark is not None:
        b = benchmark.reindex(daily.index).fillna(0.0)
        active = daily["net"] - b
        out["net_active_return"] = float(active.mean() * TRADING_DAYS)
        sd = active.std()
        out["information_ratio"] = float(active.mean() / sd * np.sqrt(TRADING_DAYS)) \
            if sd > 0 else np.nan
    return out
