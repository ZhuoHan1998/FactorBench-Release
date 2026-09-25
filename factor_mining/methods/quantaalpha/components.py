"""FactorBench runner injected into the official QuantaAlpha loop.

Keeps the official planning, hypothesis/factor agents, quality gates,
mutation/crossover evolution, and summarizer untouched; replaces only the
qlib/LGBM backtest with FactorBench scoring on the shared contract, via the
official ``QLIB_FACTOR_RUNNER`` injection point (this also removes the only
Docker dependency on the factor path).

Metrics are computed on the VALID segment only — feedback and evolutionary
selection never see test. The result Series carries the label set that both
consumers expect: the evolution controller ranks trajectories by "Rank IC"
(alias-mapped to RankIC, success = RankIC > 0), and the feedback prompt reads
"IC" + the three ``1day.excess_return_without_cost.*`` labels.

This module is imported inside the QuantaAlpha virtualenv (pandas<3); keep
imports limited to pandas/numpy, quantaalpha, and factor_bench.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pandas as pd
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.experiment import QlibAlphaAgentScenario, QlibFactorExperiment
from quantaalpha.factors.runner import QlibFactorRunner
from quantaalpha.log import logger

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel

RESULT_LABELS = [
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.information_ratio",
    "1day.excess_return_without_cost.max_drawdown",
]

_MARKET_ENV = "FACTORBENCH_MARKET"


class FactorBenchScenario(QlibAlphaAgentScenario):
    """Official scenario with a conda-independent runtime probe.

    The inherited probe (rdagent 0.8.0's ``get_factor_env``) constructs
    ``CondaConf(conda_env_name=os.environ.get("CONDA_DEFAULT_ENV"))`` — a
    pydantic ValidationError on hosts without conda. Factor code runs under
    ``FACTOR_CoSTEER_PYTHON_BIN``, so probe that interpreter directly (same
    fix as the RD-Agent method)."""

    def get_runtime_environment(self) -> str:
        python_bin = FACTOR_COSTEER_SETTINGS.python_bin
        code = (
            "import json, platform, sys\n"
            "info = {'python_version': sys.version.split()[0],"
            " 'platform': platform.platform()}\n"
            "for pkg in ('pandas', 'numpy', 'scipy', 'tables', 'sklearn'):\n"
            "    try:\n"
            "        info[pkg] = getattr(__import__(pkg), '__version__', 'installed')\n"
            "    except ImportError:\n"
            "        info[pkg] = 'not installed'\n"
            "print(json.dumps(info, indent=2))\n"
        )
        probe = subprocess.run(
            [python_bin, "-c", code], capture_output=True, text=True, timeout=120
        )
        if probe.returncode != 0:
            raise RuntimeError(
                f"Cannot probe factor runtime {python_bin!r}: {probe.stderr[-300:]}"
            )
        return probe.stdout.strip()


def _zero_result() -> pd.Series:
    # name="0": the official process_results renames column "0" to
    # "Current Result"/"SOTA Result" (qlib_res.csv contract).
    return pd.Series({m: 0.0 for m in RESULT_LABELS}, name="0")


def _zscore_by_day(signal: pd.Series) -> pd.Series:
    grouped = signal.groupby(level="date")
    return (signal - grouped.transform("mean")) / grouped.transform("std")


def _daily_ic(signal: pd.Series, target: pd.Series, method: str = "pearson") -> pd.Series:
    df = pd.DataFrame({"s": signal, "t": target}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby(level="date").apply(lambda g: g["s"].corr(g["t"], method=method))


def _long_short_daily_returns(signal: pd.Series, fwd_1d: pd.Series) -> pd.Series:
    df = pd.DataFrame({"s": signal, "r": fwd_1d}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def one_day(g: pd.DataFrame) -> float:
        if len(g) < 10:
            return np.nan
        q = g["s"].rank(pct=True)
        return g.loc[q >= 0.8, "r"].mean() - g.loc[q <= 0.2, "r"].mean()

    return df.groupby(level="date").apply(one_day).dropna()


def _valid(series: pd.Series) -> pd.Series:
    start, end = DEFAULT_SPLIT.valid
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end)]


class FactorBenchFactorRunner(QlibFactorRunner):
    """Execute mined DSL factors on the full panel; score on the shared contract."""

    def develop(self, exp: QlibFactorExperiment, use_local: bool = True) -> QlibFactorExperiment:
        market = os.environ.get(_MARKET_ENV)
        if not market:
            raise RuntimeError(f"{_MARKET_ENV} must be set (run.py does this)")

        for base in getattr(exp, "based_experiments", []):
            if base.result is None and len(base.sub_tasks) == 0:
                base.result = _zero_result()
        if len(exp.sub_tasks) == 0:
            exp.result = _zero_result()
            return exp

        factor_df = self.process_factor_data(exp)
        if factor_df is None or factor_df.empty:
            raise FactorEmptyError("No mined factor produced values on the panel")
        factor_df.index = factor_df.index.rename(["date", "asset"])

        panel = load_panel(market)
        target = TARGET.from_panel(panel)
        fwd_1d = panel["forward_return_1d"]

        per_factor: dict[str, dict[str, float]] = {}
        signals = []
        train_start, train_end = DEFAULT_SPLIT.train
        for name in factor_df.columns:
            signal = factor_df[name].reindex(panel.index)
            ic = _daily_ic(signal, target)
            per_factor[str(name)] = {
                "train.IC": float(
                    ic.loc[pd.Timestamp(train_start): pd.Timestamp(train_end)].mean()
                ),
                "valid.IC": float(_valid(ic).mean()) if len(_valid(ic)) else float("nan"),
            }
            signals.append(_zscore_by_day(signal))

        combined = pd.concat(signals, axis=1).mean(axis=1)
        ic_series = _valid(_daily_ic(combined, target))
        ric_series = _valid(_daily_ic(combined, target, "spearman"))
        ls = _valid(_long_short_daily_returns(combined, fwd_1d))

        def _ir(series: pd.Series) -> float:
            ok = len(series) > 1 and series.std() > 0
            return float(series.mean() / series.std()) if ok else 0.0

        cum = ls.cumsum()
        exp.result = pd.Series(
            {
                "IC": float(ic_series.mean()) if len(ic_series) else 0.0,
                "ICIR": _ir(ic_series),
                "Rank IC": float(ric_series.mean()) if len(ric_series) else 0.0,
                "Rank ICIR": _ir(ric_series),
                "1day.excess_return_without_cost.annualized_return": (
                    float(ls.mean() * 252) if len(ls) else 0.0
                ),
                "1day.excess_return_without_cost.information_ratio": (
                    _ir(ls) * float(np.sqrt(252)) if len(ls) else 0.0
                ),
                "1day.excess_return_without_cost.max_drawdown": (
                    float((cum - cum.cummax()).min()) if len(ls) else 0.0
                ),
            },
            name="0",
        )
        exp.factorbench_metrics = per_factor
        logger.info(f"[factorbench-runner] valid-segment result:\n{exp.result}")
        return exp
