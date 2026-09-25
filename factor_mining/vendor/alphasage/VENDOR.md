# Vendored: AlphaSAGE (official)

- Upstream: https://github.com/BerkinChen/AlphaSAGE
- Commit: `517467a34909512a92d6e3139df54891957560de` (2026-02-26)
- Vendored on: 2026-08-31
- License: MIT (see LICENSE)
- Paper: "AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets for Robust
  Exploration" ([arXiv:2509.25055](https://arxiv.org/abs/2509.25055), v3
  2026-05-19)

## What was excluded from the snapshot

- `.git/` (history), `src/data_collection/` (baostock/qlib scrapers —
  FactorBench supplies its own data), `img/` (README figures), and the repo's
  own `.gitignore` (nested ignore files hide vendored source from the parent
  repo's commits).

## What FactorBench uses

- `train_gfn.py` — the official stage-1 entry; mirrored in
  `factor_mining/methods/alphasage/run.py` (same pool construction, same
  target, same hyperparameters).
- `src/alpha_gfn/` — the method core: `env/core.py` (`GFNEnvCore`, a torchgfn
  `DiscreteEnv` over expression tokens with the dense multi-signal reward),
  `modules.py` (`SequenceEncoder`; the `gnn` encoder builds a PyG graph from
  the RPN token stream and runs `RGCNConv` — the paper's structure-aware
  encoder), `gflownet.py` (`EntropyTBGFlowNet`, trajectory balance plus an
  entropy bonus), `alpha_pool.py` (`AlphaPoolGFN`), `preprocessors.py`,
  `config.py` (hyperparameters and the action space).
- `src/alphagen/`, `src/alphagen_qlib/`, `src/alphagen_generic/` — AlphaSAGE's
  fork of AlphaGen, inherited via AlphaForge. **Not interchangeable with the
  other forks**: the operators are CamelCased relative to AlphaForge
  (`TsMean` for `ts_mean`, `SLog1p` for `S_log1p`) and renamed relative to
  upstream AlphaGen (`TsMean` for `Mean`, `TsDelta` for `Delta`), and it keeps
  AlphaForge's additions `Inv`, `SLog1p`, `TsDiv`, `TsPctChange`, `TsIr`,
  `TsMinMaxDiff`, `TsMaxDiff`, `TsMinDiff`. Search config
  (`src/alpha_gfn/config.py`): `MAX_EXPR_LENGTH=20`,
  `DELTA_TIMES=[10,20,30,40,50]`, 36 operators, 6 features.
  Exported factors therefore carry the grammar tag `alphasage-v1`.
- `src/alphagen_qlib/stock_data.py` — qlib-backed `StockData` with no
  `preloaded_data` hook, so `factor_mining/methods/alphasage/data.py`
  subclasses it and overrides `_get_data()` (same approach as AlphaQCM and
  AlphaForge). Qlib is never imported or run.
- `pdm.lock` — kept because it records the exact dependency versions the
  authors used; `torchgfn 1.2.1` and `torch-geometric 2.6.1` come from there,
  and the current torchgfn (2.x) has moved `gfn.utils.modules.NeuralNet`, so
  the pin matters.
- `run_adaptive_combination.py` — the official stage-2 combiner (inherited
  from AlphaForge). Not run by FactorBench; see below.
- `train_AFF.py`, `train_GP.py`, `train_ppo.py`, `train_qcm.py`, `src/gan/`,
  `src/gplearn/`, `src/fqf_iqn_qrdqn/`, `exp_*.ipynb` — upstream's baseline
  implementations (AlphaForge, GP, AlphaGen, AlphaQCM) and result notebooks.
  Kept for completeness; not imported by any FactorBench path.

## Environment

Runs in its own venv (`.venv-alphasage`) because the GFlowNet stack pins
`numpy < 2` through `tensordict`, which the main environment (numpy 2.4,
pandas 3.0) cannot take. Installing it into the shared venv downgrades numpy
and breaks cvxpy — don't.

```bash
python3.11 -m venv .venv-alphasage
.venv-alphasage/bin/python -m pip install -e .
.venv-alphasage/bin/python -m pip install \
    torch "torchgfn==1.2.1" "torch-geometric==2.6.1" gymnasium \
    pandas pyarrow tables fire pydantic tensorboard tqdm scipy
```

`factor_bench.eval.executors` shells out to this interpreter to evaluate
`alphasage-v1` expressions.

## Process isolation (important)

The AlphaGen, AlphaQCM, AlphaForge and AlphaSAGE snapshots each provide a
top-level `alphagen` package with **different operator names and semantics**.
Whichever is imported first wins for the whole process.
`factor_mining/methods/alphasage/vendored.py::use_alphasage` raises if a
foreign `alphagen` is already in `sys.modules`, and expression evaluation runs
in a subprocess so one `compare.py` invocation can score all four.

Note this snapshot needs **two** entries on `sys.path`: the snapshot root (its
scripts import `src.alpha_gfn.*`) and `src/` itself (its modules import
`alphagen.*` bare).

## Intentional deviations (documented, semantics-preserving)

- **Library versions**: upstream pins Python 3.11 / torch 2.4.0+cu121 /
  numpy 2.3; we run torch 2.13 CPU with the authors' `torchgfn`/`torch-geometric`
  pins and numpy 1.26 (forced by `tensordict`).
- **Device**: `train_gfn.py` selects `cuda:0` when available; FactorBench
  parameterizes with `--device`.
- **Data window**: the official `train()` hardcodes qlib paths and calendar
  years (`2010-01-01`–`2016-12-31` train, `2018-01-01`–`2020-12-31` test);
  FactorBench substitutes `DEFAULT_SPLIT` through the shared contract. The
  search sees `train` only.
- **`WeightScheduler`**: upstream drives the SSL/novelty decay by constructing
  dummy `nn.Parameter`s and Adam optimizers purely to step a torch LR
  scheduler. `run.py` computes the same schedule arithmetically — identical
  factors for `linear` / `exponential` / `polynomial(power=2)`, without
  allocating optimizers that never optimize anything.
- **Signed train IC**: `AlphaPoolGFN` stores |IC| in `ics_ret`. `run.py`
  additionally records the *signed* train IC under `train_metrics.ic` (keeping
  the pool's value as `ic_abs`) so the `train.ic_delta` fidelity check in
  `factor_bench.eval.compare` is meaningful for negative-IC alphas.
- **Checkpointing**: `factors.json` is rewritten at every `log_freq` interval,
  not only at the end, so a long run that dies keeps what it mined.
- **No code changes** were made inside this directory.

## Target

Official: `Ref($close, -20) / $close - 1`, set in `train_gfn.py::train`.
Identical to the shared contract's `forward_return_20d` — unlike AlphaForge, no
target substitution was needed.

## Stage 2 is not run

`run_adaptive_combination.py` is AlphaForge's dynamic combiner, reused here to
turn a mined pool into a tradeable signal. FactorBench does not run it: the
contract carries one static weight per factor, and FactorBench's own composite
and portfolio layers do the combining. `run.py` exports the pool's linear
weights in the `weight` field. (If a dynamic combination is ever wanted, the
AlphaForge adapter's `combine.py` already implements that algorithm.)
