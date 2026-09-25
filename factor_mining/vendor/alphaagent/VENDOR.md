# Vendored: AlphaAgent (official, KDD 2025)

- Upstream: https://github.com/RndmVariableQ/AlphaAgent
- **Branch: `legacy-main`**, commit `1da96e94a06a925c3997899f1848899440585efe` (2025-06-16)
- Vendored on: 2026-08-20
- License: MIT (see LICENSE)
- Paper: "AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to
  Counteract Alpha Decay" (KDD 2025), arXiv 2502.16789.

## Why the legacy branch

The repository's `main` was rewritten in 2026 into an unrelated Tushare/
AgentScope factor-research framework ("Release new AlphaAgent factor research
framework"). The KDD 2025 paper's implementation — the Idea/Factor/Eval agent
loop with the FactorRegulator originality gate — lives on `legacy-main`, whose
README identifies it as the official KDD 2025 source. (Note: it is a hard fork
of an older microsoft/RD-Agent, renamed `rdagent` → `alphaagent`.)

## What FactorBench uses

- `alphaagent/components/workflow/alphaagent_loop.py::AlphaAgentLoop` — the
  5-step mining loop (propose → construct → calculate → backtest → feedback).
- `alphaagent/scenarios/qlib/proposal/factor_proposal.py` — Idea Agent +
  Factor Agent with the regularization retry loop.
- `alphaagent/scenarios/qlib/regulator/factor_regulator.py` +
  `components/coder/factor_coder/factor_ast.py` — the paper's originality /
  complexity gates (duplicated-subtree ≤ 8 nodes; free-constant and
  unique-variable ratios < 0.5). No factor zoo ships with the repo; the gate
  compares only against factors generated in the same run.
- `components/coder/factor_coder/{expr_parser.py,function_lib.py,template.jinjia2}`
  — the factor DSL (~70 operators) and its execution contract
  (daily_pv.h5 → result.h5). This is also FactorBench's executor for the
  `alphaagent-v1` grammar.
- Backtest runner and summarizer are replaced via the official
  `QLIB_FACTOR_RUNNER` / `QLIB_FACTOR_SUMMARIZER` injection points
  (see factor_mining/methods/alphaagent/).

## Intentional deviations (documented, semantics-preserving)

- **Runtime monkeypatch on macOS**: `core/experiment.py::link_all_files_in_folder_to_workspace`
  only links data files on Linux/Windows, so factor execution fails on Darwin.
  The adapter patches the function at runtime (upstream RD-Agent later fixed
  the same bug by adding Darwin). No file in this directory is modified.
- **Data variables**: the official Factor Agent prompt restricts factors to
  `$open $close $high $low $volume $return`. FactorBench exports exactly those
  columns (`$return` = 1-day close-to-close return, identical to the official
  `generate.py` definition), so prompts and search space stay official;
  `vwap`/`adv` are simply not exposed to this method.
- **Environment**: runs in its own venv (`.venv-alphaagent`) honoring the
  official `numpy==1.23.5` / `pandas==1.5.3` pins (function_lib is written
  against pandas-1.5 semantics).
- **No code changes** were made inside this directory.

## Note on nested .gitignore

The upstream repository's own `.gitignore` was removed from this snapshot:
inside a vendored tree it makes the parent repo silently drop real source
files from commits (this bit us three times). No other file is affected.
