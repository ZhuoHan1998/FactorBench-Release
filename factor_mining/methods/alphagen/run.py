"""Run official AlphaGen (MaskablePPO over expression tokens) on FactorBench data.

Mirrors ``scripts/rl.py::run_single_experiment`` from the official repository
(commit 259687e), with qlib initialization replaced by the shared data
contract and the optional LLM branches removed. All search hyperparameters
are the official defaults.

Usage:
    python -m factor_mining.methods.alphagen.run --market sp100 --seed 0
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import fire
import numpy as np
import torch

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alphagen.data import PanelStockData
from factor_mining.vendor import use_vendored

use_vendored("alphagen")

from alphagen.data.expression import Feature, Ref  # noqa: E402
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool  # noqa: E402
from alphagen.rl.env.core import AlphaEnvCore  # noqa: E402
from alphagen.rl.env.wrapper import AlphaEnv  # noqa: E402
from alphagen.rl.policy import LSTMSharedNet  # noqa: E402
from alphagen.utils import reseed_everything  # noqa: E402
from alphagen_qlib.calculator import QLibStockDataCalculator  # noqa: E402
from alphagen_qlib.stock_data import FeatureType  # noqa: E402
from sb3_contrib.ppo_mask import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

OFFICIAL_REPO = "https://github.com/RL-MLDM/alphagen"
OFFICIAL_COMMIT = "259687e8f316994426416c530a94842a2fe6405e"

# Official step budgets by pool capacity (scripts/rl.py::main).
DEFAULT_STEPS = {10: 200_000, 20: 250_000, 50: 300_000, 100: 350_000}


class PoolLoggingCallback(BaseCallback):
    """Official ``CustomCallback`` minus the LLM branches: logs pool state and
    held-out ICs at each rollout end, and checkpoints model + pool."""

    def __init__(
        self,
        save_path: str,
        test_calculators: list[QLibStockDataCalculator],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("pool/size", self.pool.size)
        self.logger.record(
            "pool/significant", (np.abs(self.pool.weights[: self.pool.size]) > 1e-4).sum()
        )
        self.logger.record("pool/best_ic_ret", self.pool.best_ic_ret)
        self.logger.record("pool/eval_cnt", self.pool.eval_cnt)
        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0.0, 0.0
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f"test/ic_{i}", ic_test)
            self.logger.record(f"test/rank_ic_{i}", rank_ic_test)
        self.logger.record("test/ic_mean", ic_test_mean)
        self.logger.record("test/rank_ic_mean", rank_ic_test_mean)
        self.save_checkpoint()

    def save_checkpoint(self) -> None:
        path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
        self.model.save(path)
        with open(f"{path}_pool.json", "w") as f:
            json.dump(self.pool.to_json_dict(), f)

    @property
    def pool(self) -> LinearAlphaPool:
        assert isinstance(self.env_core.pool, LinearAlphaPool)
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore[attr-defined]


def run_alphagen(
    market: str = "csi300",
    seed: int = 0,
    pool_capacity: int = 20,
    steps: int | None = None,
    device: str = "cpu",
    out_dir: str = "out/mining",
    print_expr: bool = False,
    verbose: int = 1,
) -> Path:
    """Train AlphaGen on one market and write factors.json. Returns its path."""
    if steps is None:
        if pool_capacity not in DEFAULT_STEPS:
            raise ValueError(f"No official step budget for capacity {pool_capacity}; pass steps=")
        steps = DEFAULT_STEPS[pool_capacity]
    torch_device = torch.device(device)
    t0 = time.time()

    reseed_everything(seed)
    panel = load_panel(market)
    TARGET.validate(panel)  # leakage guard: cached target matches its definition

    datasets = {
        name: PanelStockData(panel, start, end, device=torch_device, market=market)
        for name, (start, end) in DEFAULT_SPLIT.as_dict().items()
    }
    # Official target: Ref($close, -20) / $close - 1 — identical to the shared
    # contract's forward_return_20d (validated above).
    close = Feature(FeatureType.CLOSE)
    target_expr = Ref(close, -20) / close - 1
    calculators = {name: QLibStockDataCalculator(ds, target_expr) for name, ds in datasets.items()}

    pool = MseAlphaPool(
        capacity=pool_capacity,
        calculator=calculators["train"],
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=torch_device,
    )
    env = AlphaEnv(pool=pool, device=torch_device, print_expr=print_expr)

    save_path = Path(out_dir) / "alphagen" / f"{market}_{pool_capacity}_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)
    callback = PoolLoggingCallback(
        save_path=str(save_path),
        test_calculators=[calculators["valid"], calculators["test"]],
        verbose=verbose,
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=torch_device,
            ),
        ),
        gamma=1.0,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log=str(Path(out_dir) / "alphagen" / "tensorboard"),
        device=torch_device,
        verbose=verbose,
    )
    model.learn(total_timesteps=steps, callback=callback, tb_log_name=save_path.name)

    state = pool.state
    factors = [
        ExpressionFactor(
            name=f"alphagen_{market}_s{seed}_{i:02d}",
            expression=str(expr),
            grammar="alphagen-v1",
            weight=float(weight),
            train_metrics={"ic": float(ic)},
        )
        for i, (expr, weight, ic) in enumerate(
            zip(state["exprs"], state["weights"], state["ics_ret"])
        )
    ]
    run = MiningRun(
        method="alphagen",
        paradigm="reinforcement-learning",
        output_form="expression",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget={"steps": steps, "pool_capacity": pool_capacity},
        wall_clock_seconds=time.time() - t0,
        provenance=Provenance(
            repo=OFFICIAL_REPO,
            commit=OFFICIAL_COMMIT,
            entry="scripts/rl.py::run_single_experiment",
            notes="Official implementation vendored unmodified; qlib data layer "
            "replaced by the FactorBench shared contract. Pool ensemble train "
            f"IC: {state['best_ic_ret']:.4f}.",
        ),
        factors=factors,
    )
    path = run.save(save_path / "factors.json")
    if verbose:
        print(f"[alphagen] {len(factors)} factors -> {path}")
    return path


if __name__ == "__main__":
    fire.Fire(run_alphagen)
