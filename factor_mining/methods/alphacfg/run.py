"""Run official AlphaCFG (grammar-guided MCTS) on FactorBench data.

Invokes the official experiment entry (``scripts/run_mcts_experiment.py``,
commit f6be579) with official CLI arguments; only the qlib ``StockData`` is
rebound to the shared-contract bridge before dispatch. Defaults follow the
README's ``paper-pool`` configuration (cfg-sem-k, pool mode, TreeLSTM,
capacity 10, 200 iterations, MCTS-sim 64). Splits and the target horizon are
official CLI arguments, mapped from the shared contract.

Usage:
    python -m factor_mining.methods.alphacfg.run --market csi300 --seed 0
"""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import fire

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alphacfg.data import PanelStockData
from factor_mining.vendor import VENDOR_ROOT, use_vendored

use_vendored("alphacfg")

OFFICIAL_REPO = "https://github.com/HanYang544/AlphaCFG"
OFFICIAL_COMMIT = "f6be57914d54d10e0ccabd5d14e147f756b36fa8"
_ENTRY = VENDOR_ROOT / "alphacfg" / "scripts" / "run_mcts_experiment.py"


def _harvest(out_root: Path, market: str, seed: int, budget: dict, wall: float) -> Path:
    """Export mined expressions from the official run artifacts."""
    factors: list[ExpressionFactor] = []

    def add(name: str, expression: str, weight=None, metrics=None) -> None:
        factors.append(
            ExpressionFactor(
                name=f"alphacfg_{market}_s{seed}_{name}",
                expression=str(expression),
                grammar="alphagen-v1",
                weight=(float(weight) if weight is not None else None),
                train_metrics={k: float(v) for k, v in (metrics or {}).items()},
            )
        )

    pool_files = sorted(out_root.rglob("pool_dicts.json"))
    pool_metrics: dict = {}
    if pool_files:
        # JSONL: one snapshot per iteration; the last line is the final pool.
        lines = [ln for ln in pool_files[-1].read_text().splitlines() if ln.strip()]
        last = json.loads(lines[-1])
        snapshot = last.get("pool_dict", last)
        exprs = snapshot.get("exprs") or []
        weights = snapshot.get("weights") or [None] * len(exprs)
        pool_metrics = {
            k: last[k]
            for k in ("train_ic", "train_rank_ic", "valid_ic", "valid_rank_ic")
            if k in last
        }
        for i, (expr, w) in enumerate(zip(exprs, weights)):
            add(f"f{i:02d}", expr, w)
    else:
        import csv

        for csv_path in sorted(out_root.rglob("single_factors.csv")):
            with open(csv_path) as fh:
                for i, row in enumerate(csv.DictReader(fh)):
                    expr = row.get("expression") or row.get("expr")
                    if expr:
                        ic = row.get("ic") or row.get("IC")
                        add(f"s{i:03d}", expr, None, {"ic": float(ic)} if ic else {})

    run = MiningRun(
        method="alphacfg",
        paradigm="mcts",
        output_form="expression",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget=budget,
        wall_clock_seconds=wall,
        provenance=Provenance(
            repo=OFFICIAL_REPO,
            commit=OFFICIAL_COMMIT,
            entry="scripts/run_mcts_experiment.py",
            notes="Official implementation vendored unmodified; qlib StockData "
            "rebound to the FactorBench shared contract. Official CLI splits "
            "and target horizon mapped from the shared contract. "
            f"Final pool ensemble metrics: {pool_metrics}.",
        ),
        factors=factors,
    )
    return run.save(out_root / "factors.json")


def run_alphacfg(
    market: str = "csi300",
    seed: int = 0,
    variant: str = "cfg-sem-k",
    mode: str = "pool",
    network: str = "treelstm",
    pool_capacity: int = 10,
    max_expression_length: int = 10,
    num_iterations: int = 200,
    inner_batches: int = 2,
    num_games: int = 50,
    num_batches: int = 8,
    batch_size: int = 64,
    mcts_sim: int = 64,
    mcts_parallel: int = 8,
    device: str = "cpu",
    out_dir: str = "out/mining",
    run_tag: str = "factorbench",
) -> Path:
    """Run AlphaCFG on one market and write factors.json. Returns its path."""
    t0 = time.time()
    run_name = f"{market}_{variant}_{mode}_{pool_capacity}_{seed}"
    out_root = (Path(out_dir) / "alphacfg" / run_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    load_panel(market)  # fail fast on unknown market before the heavy import path
    TARGET.validate(load_panel(market))

    # Rebind the qlib data class to the shared-contract bridge BEFORE the
    # official dispatch imports run (module attribute lookup happens at call
    # time inside unified_runner.main).
    import alphacfg.unified_runner as unified_runner

    unified_runner.StockData = PanelStockData

    budget = {
        "variant": variant,
        "mode": mode,
        "network": network,
        "pool_capacity": pool_capacity,
        "max_expression_length": max_expression_length,
        "num_iterations": num_iterations,
        "num_games": num_games,
        "mcts_sim": mcts_sim,
    }
    argv = [
        str(_ENTRY),
        "--variant", variant,
        "--mode", mode,
        "--network", network,
        "--seed", str(seed),
        "--instrument", market,
        "--pool-capacity", str(pool_capacity),
        "--max-expression-length", str(max_expression_length),
        "--num-iterations", str(num_iterations),
        "--inner-batches", str(inner_batches),
        "--num-games", str(num_games),
        "--num-batches", str(num_batches),
        "--batch-size", str(batch_size),
        "--mcts-sim", str(mcts_sim),
        "--mcts-parallel", str(mcts_parallel),
        "--device", device,
        "--run-tag", run_tag,
        "--log-dir", str(out_root),
        "--train-start", DEFAULT_SPLIT.train[0],
        "--train-end", DEFAULT_SPLIT.train[1],
        "--valid-start", DEFAULT_SPLIT.valid[0],
        "--valid-end", DEFAULT_SPLIT.valid[1],
        "--test-start", DEFAULT_SPLIT.test[0],
        "--test-end", DEFAULT_SPLIT.test[1],
        "--target-horizon", str(TARGET.horizon_days),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(_ENTRY), run_name="__main__")
    finally:
        sys.argv = old_argv
        path = _harvest(out_root, market, seed, budget, time.time() - t0)
        print(f"[alphacfg] factors -> {path}")
    return out_root / "factors.json"


if __name__ == "__main__":
    fire.Fire(run_alphacfg)
