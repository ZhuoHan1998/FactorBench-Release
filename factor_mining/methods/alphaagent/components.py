"""FactorBench runner injected into the official AlphaAgent loop.

Keeps the official Idea Agent, Factor Agent (+ FactorRegulator originality
gate), DSL parser/executor, and summarizer untouched; replaces only the
qlib/LGBM backtest with FactorBench scoring on the shared contract, via the
official ``QLIB_FACTOR_RUNNER`` injection point.

The official summarizer (``AlphaAgentQlibFactorHypothesisExperiment2Feedback``)
indexes ``exp.result`` with exactly four labels, so this runner emits those
labels with honestly-computed FactorBench quantities (VALID segment only —
feedback never sees test):

- ``IC``                       — mean daily IC of the combined signal vs the shared target
- ``1day.excess_return_without_cost.annualized_return`` — annualized long-short
  (top/bottom quintile) next-day return of the combined signal
- ``1day.excess_return_without_cost.information_ratio`` — its IR
- ``1day.excess_return_without_cost.max_drawdown``      — its max drawdown

This module is imported inside the AlphaAgent virtualenv (pandas 1.5); keep
imports limited to pandas/numpy, alphaagent, and factor_bench.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from alphaagent.log import logger
from alphaagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
from alphaagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel

OFFICIAL_METRIC_LABELS = [
    "1day.excess_return_without_cost.max_drawdown",
    "1day.excess_return_without_cost.information_ratio",
    "1day.excess_return_without_cost.annualized_return",
    "IC",
]

_MARKET_ENV = "FACTORBENCH_MARKET"


def _zero_result() -> pd.Series:
    # name="0" (string) is load-bearing: the official process_results renames
    # column "0" to "Current Result"/"SOTA Result" (qlib_res.csv contract).
    return pd.Series({m: 0.0 for m in OFFICIAL_METRIC_LABELS}, name="0")


def _zscore_by_day(signal: pd.Series) -> pd.Series:
    grouped = signal.groupby(level="date")
    return (signal - grouped.transform("mean")) / grouped.transform("std")


def _daily_ic(signal: pd.Series, target: pd.Series, method: str = "pearson") -> pd.Series:
    df = pd.DataFrame({"s": signal, "t": target}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby(level="date").apply(lambda g: g["s"].corr(g["t"], method=method))


def _long_short_daily_returns(signal: pd.Series, fwd_1d: pd.Series) -> pd.Series:
    """Top-minus-bottom quintile next-day return, per date."""
    df = pd.DataFrame({"s": signal, "r": fwd_1d}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def one_day(g: pd.DataFrame) -> float:
        if len(g) < 10:
            return np.nan
        q = g["s"].rank(pct=True)
        return g.loc[q >= 0.8, "r"].mean() - g.loc[q <= 0.2, "r"].mean()

    return df.groupby(level="date").apply(one_day).dropna()


def _valid_segment(series: pd.Series) -> pd.Series:
    start, end = DEFAULT_SPLIT.valid
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end)]


class FactorBenchFactorRunner(QlibFactorRunner):
    """Execute mined DSL factors on the full panel; score on the shared contract."""

    def develop(self, exp: QlibFactorExperiment, use_local: bool = True) -> QlibFactorExperiment:
        market = os.environ.get(_MARKET_ENV)
        if not market:
            raise RuntimeError(f"{_MARKET_ENV} must be set (run.py does this)")

        # The empty baseline experiment gets zero metrics so round-1 feedback
        # has a SOTA row, as the official summarizer expects.
        for base in getattr(exp, "based_experiments", []):
            if base.result is None and len(base.sub_tasks) == 0:
                base.result = _zero_result()
        if len(exp.sub_tasks) == 0:
            exp.result = _zero_result()
            return exp

        # Official execution path: run every sub-workspace on the full data
        # folder and concatenate the factor value Series.
        factor_df = self.process_factor_data(exp)
        if factor_df is None or factor_df.empty:
            from alphaagent.core.exception import FactorEmptyError

            raise FactorEmptyError("No mined factor produced values on the panel")
        factor_df.index = factor_df.index.rename(["date", "asset"])

        panel = load_panel(market)
        target = TARGET.from_panel(panel)
        # Next-day return for the long-short feedback metrics (no lookahead in
        # signal: signal at t is paired with the t -> t+1 close-to-close return).
        fwd_1d = panel["forward_return_1d"]

        per_factor: dict[str, dict[str, float]] = {}
        signals = []
        for name in factor_df.columns:
            signal = factor_df[name].reindex(panel.index)
            ic_valid = _valid_segment(_daily_ic(signal, target))
            ic_train_series = _daily_ic(signal, target)
            train_start, train_end = DEFAULT_SPLIT.train
            ic_train = ic_train_series.loc[pd.Timestamp(train_start) : pd.Timestamp(train_end)]
            per_factor[str(name)] = {
                "train.IC": float(ic_train.mean()) if len(ic_train) else float("nan"),
                "valid.IC": float(ic_valid.mean()) if len(ic_valid) else float("nan"),
            }
            signals.append(_zscore_by_day(signal))

        combined = pd.concat(signals, axis=1).mean(axis=1)
        ic = _valid_segment(_daily_ic(combined, target)).mean()
        ls = _valid_segment(_long_short_daily_returns(combined, fwd_1d))
        ann = float(ls.mean() * 252) if len(ls) else 0.0
        ir = float(ls.mean() / ls.std() * np.sqrt(252)) if len(ls) > 1 and ls.std() > 0 else 0.0
        cum = ls.cumsum()
        mdd = float((cum - cum.cummax()).min()) if len(ls) else 0.0

        exp.result = pd.Series(
            {
                "1day.excess_return_without_cost.max_drawdown": mdd,
                "1day.excess_return_without_cost.information_ratio": ir,
                "1day.excess_return_without_cost.annualized_return": ann,
                "IC": float(ic) if not np.isnan(ic) else 0.0,
            },
            name="0",  # official qlib_res.csv column-name contract (see _zero_result)
        )
        exp.factorbench_metrics = per_factor  # harvested into factors.json
        logger.info(f"[factorbench-runner] valid-segment result:\n{exp.result}")
        return exp
