# Vendored: QuantaAlpha (official)

- Upstream: https://github.com/QuantaAlpha/QuantaAlpha
- Commit: `b7ceb27b1001261d7a95b209a963664ae1f8ab23` (2026-06-29)
- Vendored on: 2026-08-28
- License: the README badges MIT, but the repository ships **no LICENSE
  file**. Included unmodified for benchmarking/research use only; do not
  redistribute independently.
- Paper: "QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining"
  (arXiv 2602.07085)

## Lineage

QuantaAlpha is a hard fork of the AlphaAgent fork of RD-Agent (the authors
credit both). The AlphaAgent loop (propose → construct → calculate →
backtest → feedback) survives as `quantaalpha/pipeline/loop.py`; the novel
contribution is the evolution layer (`quantaalpha/pipeline/evolution/`:
trajectory pool, mutation/crossover operators, round controller) plus the
quality gates. The factor DSL is AlphaAgent's, **extended** (WHERE and
comparison operators, TS_KURT/TS_SKEW, index-alignment fixes) — hence it gets
its own grammar tag `quantaalpha-v1` in the shared evaluator, executed by
this repo's own parser + function_lib.

At runtime the package also imports the real `rdagent` PyPI package (pinned
`rdagent==0.8.0` in requirements) for its logger, qlib scenario/experiment
classes, prompt-template loader, and workspace base class.

## What was excluded from the snapshot

- `.git/` (history), `frontend-v2/` (web dashboard, unused by the mining
  loop), `docs/_static/` + `docs/images/` (media), and the repo's own
  `.gitignore` (nested ignore files hide vendored source from the parent
  repo's commits).

## What FactorBench uses

- `quantaalpha/pipeline/factor_mining.py::main` — the official `mine` entry,
  invoked by `factor_mining/methods/quantaalpha/run.py` (minus the CLI
  wrapper's `.env` hard-exit).
- The evolution layer with the official config schema
  (`configs/experiment.yaml`); paper-scale hyperparameters per
  `docs/experiment_hyperparameters.md` (num_directions=10, max_rounds=11,
  crossover_size=2, crossover_n=10).
- `quantaalpha/factors/coder/{expr_parser,function_lib,template.jinjia2}` —
  the DSL and the official factor.py execution contract
  (daily_pv.h5 → result.h5); also FactorBench's executor for the
  `quantaalpha-v1` grammar.
- The qlib/LGBM backtest runner is replaced via the official
  `QLIB_FACTOR_RUNNER` injection point with FactorBench scoring on the shared
  contract (see factor_mining/methods/quantaalpha/); this also removes the
  only Docker dependency on the factor path.
- Harvest sources: `data/factorlib/all_factors_library_<suffix>.json`
  (expressions + lineage metadata, written every loop) and
  `trajectory_pool.json` under the log dir.

## Intentional deviations (documented, semantics-preserving)

- **Environment**: runs in its own venv (`.venv-quantaalpha`) honoring the
  official pins (`numpy<2`, `pandas<3`, `rdagent==0.8.0`).
- **Data variables**: the official variable set
  (`$open $high $low $close $volume $return`) is preserved, exported from the
  shared contract exactly as for the AlphaAgent method.
- **No code changes** were made inside this directory.
