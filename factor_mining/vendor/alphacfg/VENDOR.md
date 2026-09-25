# Vendored: AlphaCFG (official)

- Upstream: https://github.com/HanYang544/AlphaCFG
- Commit: `f6be57914d54d10e0ccabd5d14e147f756b36fa8` (2026-07-15)
- Vendored on: 2026-08-21
- License: MIT (see LICENSE)
- Paper: "Alpha Discovery via Grammar-Guided Learning and Search"
  (arXiv 2601.22119)

## What was excluded from the snapshot

- `.git/` (history), root `figure/` (README images), root `data/`
  (project-local qlib providers — FactorBench supplies its own data), root
  `outputs/` (run artifacts), and the repo's own `.gitignore` (nested ignore
  files hide vendored source from the parent repo's commits).

## What FactorBench uses

- `alphacfg/unified_runner.py` + `scripts/run_mcts_experiment.py` — the
  official experiment entry, invoked as-is by
  `factor_mining/methods/alphacfg/run.py` with official CLI arguments
  (variant / mode / network / budget / splits — splits are official CLI args,
  so the shared contract maps with no code changes).
- `alphacfg/data/stock_data.py` — qlib-backed `StockData` (same lineage as
  AlphaGen/AlphaQCM, no preload hook). FactorBench subclasses it
  (`factor_mining/methods/alphacfg/data.py`) and rebinds the name inside
  `unified_runner` so panels come from the shared contract; qlib is never
  imported or run.
- Grammar/search core: `alphacfg/grammar/`, `mcts_core.py`, method variants
  under `alphacfg/methods/` (paper config: `cfg-sem-k`, `pool` mode,
  `treelstm` network), network backends (TreeLSTM requires `dgl`).
- Official target `Ref($close, -target_horizon)/$close - 1` with horizon 20 —
  identical to the shared contract's forward_return_20d.
- Expression classes are AlphaGen-lineage with identical operator semantics
  (`Constant(...)` string form parses under the shared `alphagen-v1`
  executor), so mined expressions evaluate through the existing grammar
  executor.
- Outputs per run: `run_config.json`, `training_metrics.csv`,
  `pool_dicts.json` (pool expressions/weights/split ICs — the harvest
  source), `single_factors.csv`, `checkpoint_latest.pt`.

## Intentional deviations (documented, semantics-preserving)

- **Library versions**: upstream verifies Python 3.9 / torch 2.8.0 / qlib
  0.9.7 / dgl 1.1.3; we run the repo environment (Python 3.11 / torch 2.13 /
  dgl 1.1.3, no qlib — data comes from the shared contract bridge).
- **No code changes** were made inside this directory.
