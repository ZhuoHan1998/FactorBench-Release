# Vendored: AlphaGen (official)

- Upstream: https://github.com/RL-MLDM/alphagen
- Commit: `259687e8f316994426416c530a94842a2fe6405e` (2026-06-04)
- Vendored on: 2026-08-12
- License: upstream repository has **no license file** (research code released
  publicly alongside the KDD 2023 paper). Code is included here unmodified for
  benchmarking/research use only; do not redistribute independently.

## What was excluded from the snapshot

- `.git/` (history), `images/` (README figures), `data_collection/`
  (baostock/qlib data scrapers — FactorBench supplies its own data).

## What FactorBench uses

- `alphagen/` — expression grammar, RL env, linear alpha pool (the method core).
- `alphagen_qlib/stock_data.py` — `StockData` tensor container. FactorBench
  subclasses it (`factor_mining/methods/alphagen/data.py`) to feed panels from
  the shared data contract instead of qlib; no qlib code is imported or run.
- `alphagen_qlib/calculator.py` — IC calculator used for reward.
- `scripts/rl.py` — reference for `run_single_experiment`; mirrored (minus
  qlib init and the optional LLM branches) in
  `factor_mining/methods/alphagen/run.py`.
- `gp.py` + `gplearn/` — will back the GP baseline method later.

## Intentional deviations (documented, semantics-preserving)

- **Library versions**: upstream pins torch 2.0.1 / sb3 2.0.0 / gymnasium
  0.28 / pandas 1.2.4, which cannot coexist with this repo's Python 3.11 +
  pandas 3 environment. We run torch 2.13 / stable-baselines3 + sb3-contrib
  2.7 / gymnasium 1.3 — same MaskablePPO algorithm line and same gymnasium
  API the code was written against.
- **Device**: official scripts hardcode `cuda:0`; FactorBench defaults to CPU
  (Apple-silicon host). Hardware only, no semantic change.
- **No code changes** were made inside this directory.

## Note on nested .gitignore

The upstream repository's own `.gitignore` was removed from this snapshot:
inside a vendored tree it makes the parent repo silently drop real source
files from commits (this bit us three times). No other file is affected.

## GP baseline (`gp.py`)

`gp.py` is AlphaGen's own genetic-programming baseline, mirrored by
`factor_mining/methods/gp/run.py`. It runs gplearn's `SymbolicRegressor` over a
one-row design matrix of terminal *names*, so a program's output is expression
*text*; the fitness function evals that text into an `Expression` and returns
its training IC. Official hyperparameters: population 1000, 40 generations,
init_depth (2,6), tournament 600, p_crossover 0.3, p_subtree_mutation 0.1,
p_hoist_mutation 0.01, p_point_mutation 0.1, p_point_replace 0.6,
max_samples 0.9, parsimony_coefficient 0, and a 20-parenthesis cap on program
text. The pool is harvested by `try_pool`: highest-IC programs with mutual
IC <= 0.7. Target `Ref($close,-20)/$close - 1` is identical to the shared
`forward_return_20d`.

Three export gates were added because gp.py's fitness evals program text
directly and therefore scores programs the shared `alphagen-v1` executor
cannot evaluate. All three are measured on the **training window only**, so
nothing is selected using valid/test data:

1. **`is_featured`** — programs built only from constants (`Sum(2.0,40d)`) are
   constant signals. Upstream's own RL env rejects these via `is_valid()`; its
   parser refuses them outright. Scored -1 like an over-long program.
2. **Contract-parser round-trip** — root-level `is_featured` is not enough,
   because the parser validates every node: `Less(...,Min(Max(-0.5,40d),50d))`
   contains a feature but has a bare constant where one is required. Candidates
   are parsed with the exact parser `factor_bench.eval.executors` uses, so an
   exported factor is guaranteed evaluable.
3. **Cross-sectional dispersion** — `Corr($low,0.01,30d)` is identically zero
   yet parses, is `is_featured`, and picks up a handful of distinct values from
   float noise. Counting distinct values would also wrongly reject a legitimate
   binary factor (`Greater($close,$open)` has two), and testing the normalized
   signal fails because `normalize_by_day` divides by a ~0 std and amplifies the
   noise to O(1). The gate is instead the fraction of days with any
   cross-sectional spread, which must be >= 0.95.

Counts for all three are recorded per run in `search_budget`
(`skipped_unparseable`, `skipped_degenerate`, `eval_failures`). gp.py itself
pads the pool to capacity with such programs; dropping them means a run can
emit fewer than `pool_capacity` factors, which is the honest outcome.
