# Vendored: AlphaForge (official)

- Upstream: https://github.com/DulyHao/AlphaForge
- Commit: `d0cfc27df23c60f271bc885fd43027b86b787746` (2024-09-01)
- Vendored on: 2026-08-29
- License: none declared upstream (no LICENSE file in the repository)
- Paper: "AlphaForge: A Framework to Mine and Dynamically Combine Formulaic
  Alpha Factors" (AAAI 2025, arXiv 2406.18394)

## What was excluded from the snapshot

- `.git/` (history), `data_collection/` (baostock/qlib scrapers — FactorBench
  supplies its own data), and the repo's own `.gitignore` (nested ignore files
  hide vendored source from the parent repo's commits).

## What FactorBench uses

- `train_AFF.py` — stage 1, the generative-predictive mining loop; mirrored in
  `factor_mining/methods/alphaforge/run.py`.
- `combine_AFF.py` — stage 2, dynamic factor combination; mirrored in
  `factor_mining/methods/alphaforge/combine.py`.
- `gan/` — the method core: `network/generater.py` (DCGAN-style generator +
  its early-stopped training loop), `network/predictor.py` (CNN score
  predictor), `network/masker.py` (grammar-validity masking), `network/loss.py`
  (prediction + similarity + latent-diversity terms), `dataset/collector.py`,
  and `utils/builder.py` (`Builders`, `filter_valid_blds`, `exprs2tensor`).
  `run.py`/`combine.py` import these directly; only `pre_process_y` and
  `get_metric`, which live in the un-importable `train_AFF.py` script, are
  reproduced in the adapter.
- `alphagen/`, `alphagen_qlib/`, `alphagen_generic/` — AlphaForge's **fork** of
  AlphaGen. This fork is NOT interchangeable with the AlphaGen snapshot:
  - rolling operators are renamed (`ts_mean` vs `Mean`, `ts_delta` vs `Delta`,
    `ts_wma`/`ts_ema` vs `WMA`/`EMA`, `ts_cov`/`ts_corr` vs `Cov`/`Corr`);
  - it adds `Inv`, `S_log1p`, `ts_div`, `ts_pctchange`, `ts_ir`,
    `ts_min_max_diff`, `ts_max_diff`, `ts_min_diff`;
  - `Expression.__str__` emits a Python-evaluable form (infix binary
    operators, bare `close`, bare `10.0`) rather than AlphaGen's parser form
    (`$close`, `Constant(10.0)`, `Add(...)`), and there is no `parser.py`.
  Search config (`alphagen/config.py`): `MAX_EXPR_LENGTH=20`,
  `DELTA_TIMES=[1,5,10,20,30,40,50]`, 23 operators.
  Because of all this, exported factors carry the grammar tag
  `alphaforge-v1`, evaluated by `factor_mining/methods/alphaforge/eval_expr.py`
  in a subprocess (see below).
- `alphagen_qlib/stock_data.py` — qlib-backed `StockData` with no
  `preloaded_data` hook, so `factor_mining/methods/alphaforge/data.py`
  subclasses it and overrides `_get_data()` (same approach as AlphaQCM). Qlib
  is never imported or run.
- `dso/`, `gplearn/`, `train_{GP,DSO,RL}.py`, `exp_*.ipynb` — upstream's
  baseline implementations and result notebooks. Kept for completeness; not
  imported by any FactorBench path.

## Process isolation (important)

The AlphaGen, AlphaQCM, and AlphaForge snapshots each provide a top-level
`alphagen` package with **different operator semantics**. Whichever is imported
first wins for the whole process, so they must never share one. Two guards:

- `factor_mining/methods/alphaforge/vendored.py::use_alphaforge` raises if a
  foreign `alphagen` is already in `sys.modules`;
- `factor_bench.eval.executors` evaluates `alphaforge-v1` expressions in a
  subprocess (same venv, fresh interpreter), so one `compare.py` invocation can
  score AlphaForge and AlphaGen runs together.

## Intentional deviations (documented, semantics-preserving)

- **Library versions**: upstream targets Python 3.8 / torch 1.13 / gym 0.21 /
  qlib 0.9; we run the repo environment (Python 3.11 / torch 2.13 / gymnasium
  1.3, no qlib). `alphagen/rl/env/{core,wrapper}.py` do `import gym`; stage 1
  never builds an environment and imports those modules only for
  `SIZE_ACTION` / `action2token` / token offsets, so `use_alphaforge` aliases
  gymnasium into `sys.modules["gym"]` instead of editing vendored code.
- **Device**: official `cfg` hardcodes `cuda:0`; FactorBench parameterizes.
- **Expression read-back**: `combine_AFF.py` recovers a tree with
  `eval(s.replace('open','open_').replace('$',''))` over the whole
  `alphagen.data.expression` module namespace. `eval_expr.py` does the same
  `eval` but with a namespace restricted to `Expression` subclasses plus the
  six feature singletons (so `open` needs no rewrite).
- **No code changes** were made inside this directory.

## Research-semantics deviations (these change numbers)

- **Target.** Official: `Ref(vwap,-21)/Ref(vwap,-1) - 1` (vwap-to-vwap over 20
  days, one-day execution delay), set in `alphagen_generic/features.py`.
  FactorBench mines against the shared contract target
  `Ref(close,-20)/close - 1` (`forward_return_20d`), matching AlphaGen /
  AlphaQCM / AlphaCFG, so every method optimizes the quantity it is scored on.
- **Round cap.** Official stage 1 loops `while len(zoo_blds) < zoo_size` with
  no upper bound and can spin forever on a market where few factors clear the
  IC/ICIR/correlation filters. `run.py` adds `max_rounds` (default 50), logs
  loudly when it binds, and exports whatever was mined.
- **Non-finite factors in stage 2.** `get_metric` rejects factors that are
  mostly non-finite but inspects only the first day of the training window.
  Across the full combination window some zoo factors do go non-finite — e.g.
  `(-10*low)**-10` underflows to a denormal whose per-day std is ~0, so
  `normalize_by_day` divides by it and yields `inf`. `torch.linalg.lstsq`
  returns an **all-zero** solution on a non-finite design matrix without
  raising, which silently collapses the combined signal to a constant (observed
  on sp100). `combine.py` therefore applies the same finiteness requirement
  across the combination window, prints the excluded factors, and records them
  under `dropped_non_finite` in `combination_log.json`.
- **Zoo dedup in stage 2.** Official `get_blds_list_df` collapses factors
  sharing an identical search score (`groupby('score').first()`), a heuristic
  for merging several run pickles. `combine.py` reads a single already
  deduplicated zoo from `factors.json` and skips that step.
- **Splits.** Official splits are calendar years derived from
  `train_end_year`; FactorBench uses `factor_bench.data.splits.DEFAULT_SPLIT`.
  Stage 1 sees only `train`; stage 2 re-fits on a trailing window that ends
  `shift = 21` trading days before each evaluation day (official value, kept —
  it is what keeps the 20-day forward target realized).

## Stage 2 is a side artifact

`factors.json` carries at most one static weight per factor, so AlphaForge's
per-day weights cannot be represented in the mining contract. `run.py` exports
the zoo with `weight=None`; `combine.py` writes the dynamic combination to
`combined_signal.parquet` + `combination_log.json` next to it. Nothing in
`factor_bench` consumes those today — FactorBench's own composite/portfolio
layer does the combining, which is what keeps methods comparable.
