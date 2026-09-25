# CogAlpha — from-paper implementation

- Paper: Liu, F., Huang, Y., Luo, S., Wang, Y., Yang, Y., Li, X., Hu, Z., Feng, J.,
  & Liu, Q. (2026). *Cognitive Alpha Mining via LLM-Driven Code-Based Evolution*.
  Proceedings of the 64th Annual Meeting of the ACL (Volume 1: Long Papers),
  11715–11749. <https://aclanthology.org/2026.acl-long.538/>
- Preprint used: [arXiv:2511.18850](https://arxiv.org/abs/2511.18850) v4 (2026-07-11)
- Implemented on: 2026-08-29

## There is no official implementation

Checked and found nothing from the authors: the arXiv page and full text (v4,
including all appendices), the ACL Anthology page (PDF and checklist only), and
a GitHub search.

Third-party reproductions exist (`thesimbl9/CogAlpha-Repro`,
`GK0421/cogalpha-*`). **None is used here.** The closest one is two commits,
activates 7 of the 21 agents and 10 of 24 generations, and its own author
concludes that the LLM "fails to reliably encode financial concepts into factor
formulas" before pivoting away from the project. It is not a reference.

Everything here is written from the paper. **Nothing in this directory has been
verified against reference code, because there is none.**

## What is implemented, and where the paper says it

| Paper | Here |
|---|---|
| Seven-level, 21-agent hierarchy (§3.1, App. A.1) | `agents.py::AGENTS` |
| Diversified Guidance, 5 paraphrase modes (§3.2, App. A.2) | `agents.py::PARAPHRASE_MODES`, `evolution.py::_guidance` |
| Multi-Agent Quality Checker (§3.3, App. A.3) | `quality.py`, with the deterministic half in `codegen.py` |
| Fitness: IC / ICIR / RankIC / RankICIR / MI (§3.4, App. B.3) | `evaluation.py::compute_fitness` |
| Qualified (p65) / elite (p80) gate + floors (§3.4, App. A.4) | `evaluation.py::classify` |
| Adaptive Generation feedback (§3.5) | `evolution.py::_feedback` |
| Thinking Evolution: mutation / crossover / both (§3.6) | `evolution.py::_one_generation` |
| Agent + checker + evolution prompts (App. C) | `prompts.py` |
| Execution and leakage unit test (App. A.3) | `execution.py` |

Settings taken verbatim from Appendix B.4/B.8: initial pool 80, 13 of 21 agents
selected for it, parent pool 32, children pool 3× parent, sub-cycles of 8
generations, injection every 2 generations, top-2 elite carried forward, >30%
NaN discarded, plateau stop at δ ≤ 0.001.

Factors are emitted as `code` contract entries under the `__main__` script
convention — the same form RD-Agent uses — so `factor_bench.eval.executors`
runs them unchanged.

## The one deliberate scale reduction

Appendix B.4 runs **24 generations per agent** across 21 agents at 96 children
per generation: roughly 48,000 alpha generations per market, which the paper
measures at "approximately 1 hour per generation" (~500 hours per run) on a
**local** `gpt-oss-120b` where they note API cost is zero.

FactorBench keeps the full 21-agent hierarchy — the paper's headline
contribution — and shortens the depth: `generations` defaults to **4** instead
of 24. Everything else structural (parent pool, children multiplier, operator
mix, injection cadence) is the paper's. Raise it with `--generations`.

`--max_llm_calls` caps a run and stops it cleanly at the ceiling rather than
mid-write, because the LLM budget is the binding constraint here.

## Gaps the paper leaves open — and what was chosen

1. **Guidance bodies for 19 of 21 agents.** Appendix C.1 prints the Base Agent
   prompt in full and two agent deltas (BarShape, Composite). For the other 19,
   Appendix A.1 gives a one-line exploration domain only, and the guidance
   block here is written from that line. Which is which is recorded in
   `Agent.guidance_source` (`"paper"` vs `"derived"`).

2. **Mutual-information estimator.** Eq. 6 defines MI as an integral but names
   no estimator. Implemented as a 10×10 histogram over per-day *ranks* of both
   series, pooled across days — scale-free, deterministic, and cheap enough to
   run on every candidate. A different estimator would shift the MI floors.

3. **Fitness floors for three markets.** Appendix A.4 calibrates CSI300
   (MI ≥ 0.02) and S&P500 (MI ≥ 0.012), and says explicitly that thresholds
   "may vary depending on the dataset ... because it is harder to mine alpha
   signals in a more effective stock market". `csi300` takes the A-share
   calibration; `sp100`, `hsi`, `ftse100` and `nikkei225` take the S&P500 one.
   The paper gives no basis for the last three.

4. **One-sided floors.** The floors are written as lower bounds ("IC and RankIC
   are bounded below by 0.005"), so a strongly *negative*-IC alpha is rejected
   rather than sign-flipped. Followed literally.

5. **Repair-round bound.** §3.3 says codes "that cannot be repaired or improved
   after several attempts are discarded" without fixing the number. Capped at 2
   rounds (`quality.MAX_REPAIR_ROUNDS`).

6. **Judge and LLM code audit are switches.** The paper runs the full four-agent
   checker on every candidate. The semantic Judge is the largest single LLM cost
   in the pipeline, and the Code Quality Agent's remit is mostly covered
   deterministically by `codegen.validate` (parsing, undefined names, forbidden
   imports, nested loops, static leakage) at zero cost. Defaults: `judge=True`
   (paper), `llm_code_audit=False` (deviation, for cost). The Judge also runs
   *after* execution rather than in the paper's position before it — same
   admitted set, 82% fewer Judge calls; see the Cost section.

7. **`talib` is unavailable.** Appendix C.1 pre-imports `talib` alongside
   numpy/pandas/scipy. It needs a system C library that is not installed here,
   so it is removed from the allowed list and the prompt says so explicitly.
   This narrows what the LLM can write.

8. **Sub-cycle semantics.** "24 generations and 3 inner sub-cycles ... each
   task-specific agent initiates the evolutionary search 3 times" is read as:
   each sub-cycle restarts from the current parent pool and runs
   `sub_cycle_length` generations.

9. **Plateau window.** Appendix B.4 defines the rule (δ = mean(curr) −
   mean(prev), stop at δ ≤ 0.001) but not `plateau_win`. Set to 3.

10. **Ranking within the elite pool.** §3.4 banks elite alphas but does not say
    how the final `top_k` is chosen from them. Ranked by RankIC.

11. **Non-finite output.** Not discussed. A factor emitting any infinity is
    rejected outright rather than sanitised, so the failure is visible to the
    Code Repair Agent as feedback.

## Leakage testing is stricter here than the paper specifies

Appendix A.3 describes a "Static Safety" scan for "forward-looking shifts (e.g.
shift(-1)), misaligned rolling windows, or implicit temporal violations". That
is implemented (`codegen.leakage_findings`: negative periods on
shift/diff/pct_change, `center=True`, reversed slices).

A static scan only catches what it knows to look for, and this is
LLM-written code being executed against a benchmark whose entire premise is
no lookahead. So there is also an **empirical** test
(`execution.check_no_leakage`): every candidate runs twice on a 300-day window,
once with the final third of prices and volumes multiplied, and the earlier
output must be bit-identical. A factor that reads forward fails regardless of
how the lookahead was written. `tests/test_cogalpha.py` includes a factor whose
lookahead is hidden behind a computed offset — the static scan passes it, the
empirical test catches it.

Cost: one extra short execution per candidate. Disable with
`--leakage_test False` only if you accept unverified factors.

## Deliberate deviations from the paper's experimental setup

- **Target.** The paper predicts 10-day forward returns (Table 4 and most of
  the appendix) on Qlib data. FactorBench mines the shared contract target
  `close[t+20]/close[t] − 1`, as every other method here does.
- **Split.** `DEFAULT_SPLIT`; the search sees `train` only and never touches
  `valid`/`test`. The paper uses rolling training with a 126-day step.
- **Data fields.** §3.1's OHLCV only. The FactorBench panel also carries
  `vwap`, `return_1d` and `adv`; they are withheld so the search space is the
  paper's — note this makes CogAlpha the only method here without `vwap`.
- **Combination and backtest.** The paper feeds the selected alphas to a
  separately trained LightGBM or Ridge model and backtests in Qlib.
  FactorBench's own evaluation and portfolio layers do that instead, so emitted
  factors carry `weight = null`.
- **LLM.** The paper runs a local `gpt-oss-120b`. Here the model comes from
  `CHAT_MODEL`, the same configuration the other LLM methods use.

## Cost

Per candidate: 1 generation call (amortised over `num_per_request` factors),
1 Judge call when enabled, plus repair calls on failure, plus 1–2 subprocess
executions. The paper's cost is GPU time on a **local** `gpt-oss-120b` with, as
they note, zero API cost; ours is latency against a remote endpoint, so the
shape of the bill is completely different.

Measured call budget for one seed at the default config, counted by running the
real search loop against a stub client (21 agents, `generations=4`,
`num_per_request=6`):

| config | LLM calls | of which Judge | candidates |
|---|---|---|---|
| paper-faithful default | 10,129 | 8,310 (82%) | 8,310 |
| `--judge False` | 1,819 | 0 | 8,310 |
| `--judge False --num_per_request 12` | 811 | 0 | 3,522 |
| `--judge False --num_per_request 12 --generations 2` | 412 | 0 | 1,800 |

**The Judge is 82% of the bill.** It runs on every candidate, and its verdict
does not depend on the numbers, so it is the last gate rather than the paper's
third (see the note at the top of `quality.py`) — a candidate that cannot
execute no longer costs a Judge call first. The admitted set is unchanged.

The other structural cost is that every call was sequential. `--concurrency`
(default 8) issues independent requests in parallel: the generation calls within
a generation, the initial pool's agents, and the per-candidate checker work.
Wall clock divides by roughly the number of requests in flight, so at the
observed ~115 s/call the default config goes from ~320 h per seed to ~40 h, and
`--judge False --concurrency 8` to ~7 h.

Reproducibility is preserved: every draw from `self.rng` happens on the main
thread before dispatch, so a given `(seed, concurrency)` reproduces exactly, and
in fact concurrency does not change the result at all —
`test_a_seeded_search_is_reproducible_and_survives_concurrency` pins
`concurrency=1` and `concurrency=8` to the same admitted candidate set.

Local cost is ~3 s per candidate: three subprocess spawns (the factor once, then
twice more for the leakage test), each paying interpreter startup and a pandas
import. That is the floor once the LLM calls are parallel — ~7 h per seed
sequentially at the default candidate count, ~50 min at `concurrency=8`.
`--leakage_test False` removes two of the three.

Use `--generations`, `--judge False` and `--max_llm_calls` to bound a run, and
`--initial_agents` / `--parent_pool` to shrink it for a smoke test. Raising
`--num_per_request` cuts round-trips but not output tokens, so against an
endpoint that is throughput-bound (the one measured here sustained ~16 output
tokens/s) expect less than the naive saving.
