"""Run official AlphaSAGE (structure-aware GFlowNet alpha mining) on FactorBench data.

Mirrors ``train_gfn.py::train`` from the official repository (commit 517467a),
with the hardcoded qlib paths and calendar-year windows replaced by the shared
data contract. All search hyperparameters are the official defaults from
``run.sh`` / the README quick-start.

The official target is ``Ref($close, -20) / $close - 1`` — identical to the
shared contract's ``forward_return_20d``, so unlike AlphaForge no target
substitution is needed here.

Runs inside ``.venv-alphasage``: torchgfn 1.2.1 + torch-geometric 2.6.1 pin
numpy < 2, which the main environment cannot take.

Usage:
    .venv-alphasage/bin/python -m factor_mining.methods.alphasage.run \
        --market sp100 --seed 0
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import fire
import numpy as np
import torch

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alphasage.data import PanelStockData
from factor_mining.methods.alphasage.vendored import use_alphasage

use_alphasage()

from alphagen.data.expression import Feature, Ref  # noqa: E402
from alphagen.utils.correlation import batch_pearsonr  # noqa: E402
from alphagen.utils.pytorch_utils import normalize_by_day  # noqa: E402
from alphagen_qlib.stock_data import FeatureType  # noqa: E402
from gfn.modules import DiscretePolicyEstimator  # noqa: E402
from gfn.samplers import Sampler  # noqa: E402
from gfn.utils.modules import NeuralNet  # noqa: E402
from src.alpha_gfn.alpha_pool import AlphaPoolGFN  # noqa: E402
from src.alpha_gfn.config import (  # noqa: E402
    CONSTANTS,
    DELTA_TIMES,
    FEATURES,
    HIDDEN_DIM,
    LEARNING_RATE,
    OPERATORS,
)
from src.alpha_gfn.env.core import GFNEnvCore  # noqa: E402
from src.alpha_gfn.gflownet import EntropyTBGFlowNet  # noqa: E402
from src.alpha_gfn.modules import SequenceEncoder  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim import Adam  # noqa: E402

OFFICIAL_REPO = "https://github.com/BerkinChen/AlphaSAGE"
OFFICIAL_COMMIT = "517467a34909512a92d6e3139df54891957560de"


class WeightScheduler:
    """Official ``train_gfn.py::WeightScheduler``, reproduced without the dummy
    optimizers it used only to drive a torch LR scheduler.

    The SSL and novelty reward weights decay from their initial value to
    ``final_ratio`` times it over ``total_steps``. Linear/exponential/
    polynomial match ``LinearLR`` / ``ExponentialLR`` / ``PolynomialLR(power=2)``
    with the official arguments.
    """

    def __init__(self, initial_ssl_weight: float, initial_nov_weight: float,
                 final_ratio: float, total_steps: int,
                 scheduler_type: str = "linear") -> None:
        self.ssl0 = initial_ssl_weight
        self.nov0 = initial_nov_weight
        self.final_ratio = final_ratio
        self.total_steps = max(int(total_steps), 1)
        self.scheduler_type = scheduler_type
        self.step_count = 0

    def _factor(self) -> float:
        t = min(self.step_count, self.total_steps)
        progress = t / self.total_steps
        if self.scheduler_type == "linear":
            return 1.0 + (self.final_ratio - 1.0) * progress
        if self.scheduler_type == "exponential":
            gamma = self.final_ratio ** (1.0 / self.total_steps) if self.final_ratio > 0 else 0.0
            return gamma**t
        if self.scheduler_type == "polynomial":
            return self.final_ratio + (1.0 - self.final_ratio) * (1.0 - progress) ** 2.0
        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def step(self) -> None:
        self.step_count += 1

    def get_current_weights(self) -> tuple[float, float]:
        factor = self._factor()
        return self.ssl0 * factor, self.nov0 * factor


def run_alphasage(
    market: str = "csi300",
    seed: int = 0,
    pool_capacity: int = 50,
    n_episodes: int = 10_000,
    update_freq: int = 64,
    log_freq: int = 500,
    encoder_type: str = "gnn",
    entropy_coef: float = 0.01,
    entropy_temperature: float = 1.0,
    mask_dropout_prob: float = 1.0,
    ssl_weight: float = 1.0,
    nov_weight: float = 0.3,
    weight_decay_type: str = "linear",
    final_weight_ratio: float = 0.0,
    device: str = "cpu",
    out_dir: str = "out/mining",
    verbose: int = 1,
) -> Path:
    """Train AlphaSAGE on one market and write factors.json. Returns its path."""
    t0 = time.time()
    # Official reproducibility block.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)

    panel = load_panel(market)
    TARGET.validate(panel)  # leakage guard: cached target matches its definition

    # Official windows are hardcoded calendar years; FactorBench substitutes the
    # shared split. Only `train` feeds the search; `test` is used exclusively
    # for the official logger's progress readout.
    data = PanelStockData(panel, *DEFAULT_SPLIT.train, device=torch_device, market=market)
    data_test = PanelStockData(panel, *DEFAULT_SPLIT.test, device=torch_device, market=market)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1  # == the shared forward_return_20d

    pool = AlphaPoolGFN(capacity=pool_capacity, stock_data=data, target=target)

    n_tokens = len(FEATURES) + len(OPERATORS) + len(DELTA_TIMES) + len(CONSTANTS)
    backbone = SequenceEncoder(n_tokens, encoder_type)
    env = GFNEnvCore(
        pool=pool,
        encoder=backbone,
        device=torch_device,
        mask_dropout_prob=mask_dropout_prob,
        ssl_weight=ssl_weight,
        nov_weight=nov_weight,
    )

    pf_head = NeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions, n_hidden_layers=0)
    # pb does not predict the exit action.
    pb_head = NeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1, n_hidden_layers=0)
    pf_module = nn.Sequential(backbone, pf_head)
    pb_module = nn.Sequential(backbone, pb_head)
    pf = DiscretePolicyEstimator(pf_module, n_actions=env.n_actions,
                                 preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(pb_module, n_actions=env.n_actions,
                                 preprocessor=env.preprocessor, is_backward=True)

    loss_fn = EntropyTBGFlowNet(
        pf=pf, pb=pb, entropy_coef=entropy_coef, entropy_temperature=entropy_temperature
    )
    loss_fn.to(torch_device)
    sampler = Sampler(estimator=pf)
    params = (
        list(backbone.parameters())
        + list(pf_head.parameters())
        + list(pb_head.parameters())
        + [loss_fn.logZ]
    )
    optimizer = Adam(params, lr=LEARNING_RATE)

    save_path = Path(out_dir) / "alphasage" / f"{market}_{pool_capacity}_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    weight_scheduler = WeightScheduler(
        initial_ssl_weight=ssl_weight,
        initial_nov_weight=nov_weight,
        final_ratio=final_weight_ratio,
        total_steps=n_episodes,
        scheduler_type=weight_decay_type,
    )

    target_train = normalize_by_day(target.evaluate(data))

    def signed_train_ic(expr) -> float:
        """Signed mean IC on the training window.

        The pool stores |IC| in ``ics_ret`` (an alpha and its negation are
        equally useful to it), which would make the ``train.ic_delta``
        fidelity check in factor_bench.eval.compare read as a large error on
        every negative-IC alpha. Recomputed here with the same normalization
        and the same batch_pearsonr, just without the absolute value.
        """
        value = normalize_by_day(expr.evaluate(data))
        return float(batch_pearsonr(value, target_train).mean().item())

    def write_artifacts(complete: bool) -> Path:
        """Persist the pool as it stands. Called periodically, not only at the
        end, so a long run that dies keeps what it has already mined."""
        state = pool.state
        factors = [
            ExpressionFactor(
                name=f"alphasage_{market}_s{seed}_{i:02d}",
                expression=str(expr),
                grammar="alphasage-v1",
                weight=float(weight),
                train_metrics={"ic": signed_train_ic(expr), "ic_abs": float(ic)},
            )
            for i, (expr, weight, ic) in enumerate(
                zip(state["exprs"], state["weights"], state["ics_ret"])
            )
        ]
        mining_run = MiningRun(
            method="alphasage",
            paradigm="generative-flow-network",
            output_form="expression",
            market=market,
            seed=seed,
            split=DEFAULT_SPLIT.as_dict(),
            target=TARGET.name,
            search_budget={
                "n_episodes": n_episodes,
                "episodes_run": episodes_run,
                "pool_capacity": pool_capacity,
                "encoder_type": encoder_type,
                "entropy_coef": entropy_coef,
                "entropy_temperature": entropy_temperature,
                "mask_dropout_prob": mask_dropout_prob,
                "ssl_weight": ssl_weight,
                "nov_weight": nov_weight,
                "weight_decay_type": weight_decay_type,
                "final_weight_ratio": final_weight_ratio,
                "update_freq": update_freq,
                "eval_cnt": int(pool.eval_cnt),
                # bools coerce to 0.0 in this dict's value type
                "complete": str(complete).lower(),
            },
            wall_clock_seconds=time.time() - t0,
            provenance=Provenance(
                repo=OFFICIAL_REPO,
                commit=OFFICIAL_COMMIT,
                entry="train_gfn.py::train",
                notes="Official implementation vendored unmodified; qlib data "
                "layer replaced by the FactorBench shared contract and the "
                "hardcoded calendar-year windows by DEFAULT_SPLIT. The official "
                "target Ref($close,-20)/$close - 1 is identical to the shared "
                "forward_return_20d, so no target substitution was needed."
                + ("" if complete else " PARTIAL RUN: training stopped before its "
                   "episode budget; see search_budget.complete."),
            ),
            factors=factors,
        )
        written = mining_run.save(save_path / "factors.json")
        (save_path / "pool.json").write_text(json.dumps(pool.to_dict(), indent=2))
        return written

    episodes_run = 0
    losses: list[float] = []
    minibatch_loss = 0.0
    for episode in range(n_episodes):
        episodes_run = episode + 1
        current_ssl_weight, current_nov_weight = weight_scheduler.get_current_weights()
        env.ssl_weight = current_ssl_weight
        env.nov_weight = current_nov_weight

        save_estimator_outputs = entropy_coef > 0
        trajectories = sampler.sample_trajectories(
            env=env, n_trajectories=1, save_estimator_outputs=save_estimator_outputs
        )
        loss = loss_fn.loss(env=env, trajectories=trajectories)
        if loss is not None and torch.isfinite(loss):
            minibatch_loss = minibatch_loss + loss

        if episode > 0 and (episode + 1) % update_freq == 0 and not isinstance(
            minibatch_loss, float
        ):
            losses.append(float(minibatch_loss.item()))
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            minibatch_loss = 0.0

        if episode > 0 and (episode + 1) % log_freq == 0:
            write_artifacts(complete=False)
            if verbose:
                best = (
                    float(np.max(pool.single_ics[: pool.size])) if pool.size > 0 else float("nan")
                )
                print(f"[alphasage] episode {episode + 1}/{n_episodes}: pool {pool.size}"
                      f"/{pool_capacity}, best single ic {best:.4f}, "
                      f"evals {pool.eval_cnt}, ssl {current_ssl_weight:.3f}, "
                      f"nov {current_nov_weight:.3f}")

        weight_scheduler.step()

    path = write_artifacts(complete=True)
    if verbose:
        print(f"[alphasage] pool {pool.size} alphas ({pool.eval_cnt} evaluations) -> {path}")
    # data_test mirrors the official logger's held-out readout; referenced so
    # the construction is not dead code when verbose logging is off.
    del data_test
    return path


if __name__ == "__main__":
    fire.Fire(run_alphasage)
