"""FactorBench runner + summarizer injected into RD-Agent's official loop.

The official ``fin_factor`` loop wires its Research→Development→Feedback steps
through injectable classes (``rdagent/app/qlib_rd_loop/conf.py``). We keep the
official scenario, hypothesis generation, and Co-STEER coder untouched, and
replace only the two qlib-coupled evaluation components:

- ``QlibFactorRunner`` (LGBM backtest via qlib/Docker) → :class:`FactorBenchFactorRunner`,
  which executes each generated factor.py on the full panel and scores IC /
  Rank IC against the shared FactorBench target on train and valid segments.
- ``QlibFactorExperiment2Feedback`` → :class:`FactorBenchExperiment2Feedback`,
  the same official feedback flow and prompts, with the metric table built
  from the FactorBench metrics instead of qlib backtest metrics.

Feedback deliberately sees train/valid only — never test.

This module is imported inside the RD-Agent virtualenv by
``rdagent.core.utils.import_class``; keep imports limited to pandas/numpy,
rdagent, and factor_bench.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict

import pandas as pd
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.factor_experiment import (
    QlibFactorExperiment,
    QlibFactorScenario,
)
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel

IMPORTANT_METRICS = ["train.IC", "train.RankIC", "valid.IC", "valid.RankIC"]

_MARKET_ENV = "FACTORBENCH_MARKET"


class FactorBenchScenario(QlibFactorScenario):
    """Official scenario with a conda-independent runtime probe.

    The vendored probe launches ``python runtime_info.py`` inside a
    conda-derived LocalEnv; on hosts without conda the constructed PATH has no
    ``python`` and the probe crashes. Factor code in FactorBench runs under
    ``FACTOR_CoSTEER_PYTHON_BIN``, so probe that interpreter directly — the
    result plays the same role (describing the execution environment to the
    LLM in prompts).
    """

    def get_runtime_environment(self) -> str:
        python_bin = FACTOR_COSTEER_SETTINGS.python_bin
        code = (
            "import json, platform, sys\n"
            "info = {'python_version': sys.version.split()[0],"
            " 'platform': platform.platform()}\n"
            "for pkg in ('pandas', 'numpy', 'scipy', 'tables', 'statsmodels', 'sklearn'):\n"
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


def _daily_ic(signal: pd.Series, target: pd.Series, method: str) -> pd.Series:
    """Per-date cross-sectional correlation between a signal and the target."""
    df = pd.DataFrame({"s": signal, "t": target}).dropna()
    return df.groupby(level="date").apply(lambda g: g["s"].corr(g["t"], method=method))


def _segment_metrics(signal: pd.Series, target: pd.Series) -> dict[str, float]:
    ic = _daily_ic(signal, target, "pearson")
    ric = _daily_ic(signal, target, "spearman")
    out: dict[str, float] = {}
    for seg in ("train", "valid"):
        start, end = DEFAULT_SPLIT.segment(seg)
        for name, series in (("IC", ic), ("RankIC", ric)):
            seg_vals = series.loc[pd.Timestamp(start) : pd.Timestamp(end)]
            out[f"{seg}.{name}"] = float(seg_vals.mean())
    return out


def _zscore_by_day(signal: pd.Series) -> pd.Series:
    grouped = signal.groupby(level="date")
    return (signal - grouped.transform("mean")) / grouped.transform("std")


def _zero_result() -> pd.Series:
    return pd.Series({m: 0.0 for m in IMPORTANT_METRICS}, name="result")


class FactorBenchFactorRunner(Developer[QlibFactorExperiment]):
    """Execute generated factor code on the full panel; score vs shared target."""

    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        market = os.environ.get(_MARKET_ENV)
        if not market:
            raise RuntimeError(f"{_MARKET_ENV} must be set (loop.py does this)")

        # The first based_experiment is the official empty "baseline" run; give
        # it (and an empty exp itself) a zero-metric result so round-1 feedback
        # has a SOTA row to compare against, as the official prompts expect.
        for base in getattr(exp, "based_experiments", []):
            if base.result is None and len(base.sub_tasks) == 0:
                base.result = _zero_result()
        if len(exp.sub_tasks) == 0:
            exp.result = _zero_result()
            return exp

        panel = load_panel(market)
        target = TARGET.from_panel(panel)
        target.index = target.index.rename(["date", "asset"])

        per_factor_metrics: dict[str, dict[str, float]] = {}
        signals: list[pd.Series] = []
        for task, ws in zip(exp.sub_tasks, exp.sub_workspace_list):
            if ws is None or ws.file_dict.get("factor.py") is None:
                continue
            execution_feedback, df = ws.execute(data_type="All")
            if df is None or df.empty:
                logger.info(f"[factorbench-runner] {task.factor_name}: no output "
                            f"({execution_feedback[:200] if execution_feedback else 'n/a'})")
                continue
            signal = df.iloc[:, 0]
            signal.index = signal.index.rename(["date", "asset"])
            signal = signal.reindex(target.index)
            metrics = _segment_metrics(signal, target)
            per_factor_metrics[task.factor_name] = metrics
            signals.append(_zscore_by_day(signal))

        if not signals:
            raise FactorEmptyError("No generated factor produced values on the panel")

        combined = pd.concat(signals, axis=1).mean(axis=1)
        exp.result = pd.Series(_segment_metrics(combined, target), name="result")
        # Stash per-factor metrics for factors.json harvesting (loop.py).
        exp.factorbench_metrics = per_factor_metrics  # type: ignore[attr-defined]
        logger.info(f"[factorbench-runner] combined result:\n{exp.result}")
        return exp


class FactorBenchExperiment2Feedback(Experiment2Feedback):
    """Official factor feedback flow with FactorBench metrics in the table."""

    def generate_feedback(self, exp: QlibFactorExperiment, trace: Trace) -> HypothesisFeedback:
        hypothesis = exp.hypothesis
        tasks_factors = [t.get_task_information_and_implementation_result() for t in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result
        current = exp.result.reindex(IMPORTANT_METRICS)
        sota = (
            sota_result.reindex(IMPORTANT_METRICS)
            if sota_result is not None
            else _zero_result()
        )
        combined_result = "; ".join(
            f"{m} of Current Result is {current[m]:.6f}, of SOTA Result is {sota[m]:.6f}"
            for m in IMPORTANT_METRICS
        )

        sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
            scenario=self.scen.get_scenario_all_desc()
        )
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis.hypothesis,
            task_details=tasks_factors,
            combined_result=combined_result,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )
        response_json = json.loads(response)
        return HypothesisFeedback(
            observations=response_json.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json.get(
                "Feedback for Hypothesis", "No feedback provided"
            ),
            new_hypothesis=response_json.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json.get("Replace Best Result", "no")),
        )
