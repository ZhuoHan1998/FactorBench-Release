"""Evaluate one or more factors.json files through the shared contract.

Every factor — AST or code, from any method — is executed by
factor_bench.eval.executors and scored by factor_bench.eval.metrics against
the same panel, split, and target. This is the comparability test: if two
methods' outputs flow through here without special-casing, their numbers are
directly comparable.

Usage:
    python -m factor_bench.eval.compare run1/factors.json run2/factors.json \
        [--out_csv out/eval/comparison.csv]
"""

from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd

from factor_bench.data import TARGET, load_panel
from factor_bench.eval.executors import FactorExecutionError, compute_signal
from factor_bench.eval.metrics import segment_metrics
from factor_mining.contracts import MiningRun


def evaluate_run(path: str | Path) -> pd.DataFrame:
    """Evaluate every factor in one factors.json. Returns one row per factor."""
    run = MiningRun.load(path)
    panel = load_panel(run.market)
    target = TARGET.from_panel(panel)

    rows: list[dict] = []
    for factor in run.factors:
        row: dict = {
            "method": run.method,
            "market": run.market,
            "seed": run.seed,
            "factor": factor.name,
            "form": factor.type,
        }
        try:
            signal = compute_signal(factor, run.market, panel)
            signal = signal.reindex(panel.index)
            row.update(segment_metrics(signal, target))
            # Fidelity check: the method's own in-search train IC, when it
            # reported one, should be close to ours (definitions differ
            # slightly in NaN handling — see metrics.py).
            reported = factor.train_metrics.get("ic")
            if reported is not None:
                row["train.ic_reported"] = reported
                row["train.ic_delta"] = row["train.ic"] - reported
            row["error"] = ""
        except (FactorExecutionError, NotImplementedError) as e:
            row["error"] = str(e)
        rows.append(row)
    return pd.DataFrame(rows)


def main(*paths: str, out_csv: str | None = None) -> None:
    if not paths:
        raise SystemExit("Pass at least one factors.json path")
    table = pd.concat([evaluate_run(p) for p in paths], ignore_index=True)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(4).to_string(index=False))
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)
        print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    fire.Fire(main)
