"""Run official AlphaQCM (distributional RL over expression tokens) on FactorBench data.

Mirrors ``train_qcm_csi300.py`` from the official repository (commit dffc695),
with the qlib data layer replaced by the shared contract. Hyperparameters come
from the official ``qcm_config/<model>.yaml`` (iqn: 2M steps, batch 128, N=64,
PER, gamma=1); the paper's grid is ``--model {qrdqn,iqn}``,
``--pool {10,20,50,100}``, ``--std_lam {0.5,1.0,2.0}``.

The official agent snapshots the pool whenever validation IC improves
(``valid_best_table.csv``); that valid-best pool is what gets exported to
factors.json (the official model-selection rule — test is never used).

Usage:
    python -m factor_mining.methods.alphaqcm.run --market sp100 --seed 0 --model iqn
"""

from __future__ import annotations

import time
from pathlib import Path

import fire
import pandas as pd
import torch
import yaml

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alphaqcm.data import PanelStockData
from factor_mining.vendor import VENDOR_ROOT, use_vendored

use_vendored("alphaqcm")

from alphagen.data.expression import Feature, FeatureType, Ref  # noqa: E402
from alphagen.models.alpha_pool import AlphaPool  # noqa: E402
from alphagen.rl.env.wrapper import AlphaEnv  # noqa: E402
from alphagen_qlib.calculator import QLibStockDataCalculator  # noqa: E402
from fqf_iqn_qrdqn.agent import FQCMAgent, IQCMAgent, QRQCMAgent  # noqa: E402

OFFICIAL_REPO = "https://github.com/ZhuZhouFan/AlphaQCM"
OFFICIAL_COMMIT = "dffc695f9ea54e6a54e4dac86ee357be314fead4"

# Official mapping from --model to agent class (train_qcm_csi300.py).
AGENTS = {"qrdqn": QRQCMAgent, "iqn": IQCMAgent, "fqf": FQCMAgent}


def _export_factors(save_path: Path, pool: AlphaPool) -> list[ExpressionFactor]:
    """Prefer the official valid-best snapshot; fall back to the final pool."""
    csv = save_path / "valid_best_table.csv"
    factors: list[ExpressionFactor] = []
    if csv.exists():
        table = pd.read_csv(csv, index_col=0)
        rows = table[table["exprs"] != "Ensemble"]
        for i, row in enumerate(rows.itertuples(index=False)):
            factors.append(
                ExpressionFactor(
                    name=f"f{i:02d}",
                    expression=str(row.exprs),
                    grammar="alphagen-v1",
                    weight=float(row.weight),
                    train_metrics={"ic": float(row.ic)},
                )
            )
    else:
        state = pool.state
        for i, (expr, weight, ic) in enumerate(
            zip(state["exprs"], state["weights"], state["ics_ret"])
        ):
            factors.append(
                ExpressionFactor(
                    name=f"f{i:02d}",
                    expression=str(expr),
                    grammar="alphagen-v1",
                    weight=float(weight),
                    train_metrics={"ic": float(ic)},
                )
            )
    return factors


def run_alphaqcm(
    market: str = "csi300",
    seed: int = 0,
    model: str = "qrdqn",
    pool_capacity: int = 20,
    std_lam: float = 1.0,
    device: str = "cpu",
    out_dir: str = "out/mining",
    print_expr: bool = False,
    num_steps: int | None = None,
    start_steps: int | None = None,
    eval_interval: int | None = None,
) -> Path:
    """Train AlphaQCM on one market and write factors.json. Returns its path."""
    if model not in AGENTS:
        raise ValueError(f"model must be one of {sorted(AGENTS)}, got {model!r}")
    config = yaml.safe_load((VENDOR_ROOT / "alphaqcm" / "qcm_config" / f"{model}.yaml").read_text())
    # Smoke-test overrides; None keeps the official budget (iqn/qrdqn: 2M steps).
    for key, value in (("num_steps", num_steps), ("start_steps", start_steps),
                       ("eval_interval", eval_interval)):
        if value is not None:
            config[key] = value

    torch_device = torch.device(device)
    t0 = time.time()

    panel = load_panel(market)
    TARGET.validate(panel)  # leakage guard: cached target matches its definition

    datasets = {
        name: PanelStockData(panel, start, end, device=torch_device, market=market)
        for name, (start, end) in DEFAULT_SPLIT.as_dict().items()
    }
    # Official target (train_qcm_csi300.py): Ref($close, -20) / $close - 1 —
    # identical to the shared contract's forward_return_20d (validated above).
    close = Feature(FeatureType.CLOSE)
    target_expr = Ref(close, -20) / close - 1
    calculators = {name: QLibStockDataCalculator(ds, target_expr) for name, ds in datasets.items()}

    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=calculators["train"],
        ic_lower_bound=None,
        l1_alpha=5e-3,
    )
    env = AlphaEnv(pool=pool, device=torch_device, print_expr=print_expr)
    # The official agents read `self.env.pool` through the wrapper, relying on
    # gym/gymnasium<1.0 implicit attribute forwarding (removed in gymnasium
    # 1.0). Re-expose the pool on the wrapper explicitly.
    env.pool = pool

    save_path = Path(out_dir) / "alphaqcm" / f"{market}_{model}_{pool_capacity}_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)
    agent = AGENTS[model](
        env=env,
        valid_calculator=calculators["valid"],
        test_calculator=calculators["test"],
        log_dir=str(save_path),
        seed=seed,
        std_lam=std_lam,
        cuda=(torch_device.type == "cuda"),
        **config,
    )
    agent.run()

    factors = _export_factors(save_path, pool)
    prefix = f"alphaqcm_{market}_s{seed}_"
    for f in factors:
        f.name = prefix + f.name
    run = MiningRun(
        method="alphaqcm",
        paradigm="distributional-rl",
        output_form="expression",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget={
            "steps": int(config["num_steps"]),
            "pool_capacity": pool_capacity,
            "model": model,
            "std_lam": std_lam,
        },
        wall_clock_seconds=time.time() - t0,
        provenance=Provenance(
            repo=OFFICIAL_REPO,
            commit=OFFICIAL_COMMIT,
            entry="train_qcm_csi300.py",
            notes="Official implementation vendored unmodified; qlib data layer "
            "replaced by the FactorBench shared contract. Exported pool is the "
            "official valid-best snapshot (falls back to final pool if "
            "validation never improved).",
        ),
        factors=factors,
    )
    path = run.save(save_path / "factors.json")
    print(f"[alphaqcm] {len(factors)} factors -> {path}")
    return path


if __name__ == "__main__":
    fire.Fire(run_alphaqcm)
