# Alpha Jungle — from-paper implementation

- Paper: Shi, Y., Duan, Y., & Li, J. (2026). *Navigating the Alpha Jungle: An
  LLM-Powered MCTS Framework for Formulaic Alpha Factor Mining*.
  Proceedings of the AAAI Conference on Artificial Intelligence, 40(2), 997–1005.
  <https://doi.org/10.1609/aaai.v40i2.37069>
- Preprint used: [arXiv:2505.11122](https://arxiv.org/abs/2505.11122) v3 (2025-11-12)
- Implemented on: 2026-08-29

## There is no official implementation

Unlike every other method in `factor_mining/`, this one has **no vendored
snapshot**, because no code has been released. Checked and found nothing:
the arXiv abstract page and full text (v2 and v3, including the reproducibility
and limitations appendices), the AAAI proceedings page (PDF / video / poster /
slides only), and a GitHub search for the paper and its authors.

Everything here is written from the paper. **Nothing in this directory has been
verified against reference code, because there is none.** The tests in
`tests/test_alphajungle.py` pin what the paper specifies exactly; they cannot
establish equivalence with the authors' implementation.

## What is implemented, and where the paper says it

| Paper | Here |
|---|---|
| Algorithm 1 (Appendix F) | `run.py::run_alphajungle` + `mcts.py::SearchTree.run` |
| UCT with virtual expansion action `a_e` (Section 3, Eq. 2) | `mcts.py::SearchTree.select` |
| Dimension softmax (Eq. 3) | `evaluation.py::dimension_probabilities` |
| Two-step generation, Eq. 4–5 (Appendix D) | `agent.py::refinement_suggestion` → `refined_portrait` → `compile_portrait` |
| Relative rank / dimension scores (Eq. 6–7) | `evaluation.py::percentile_rank`, `dimension_scores` |
| Aggregate reward (Eq. 8) | `evaluation.py::Evaluation.reward` |
| Backpropagation (Eq. 9–10) | `mcts.py::SearchTree.backpropagate` |
| Frequent Subtree Avoidance (Eq. 11–12) | `fsa.py` |
| Operator table (Appendix E, Table 2) | `grammar.py::OPERATORS` |
| Effective Alpha Check (Appendix G) | `evaluation.py::passes_effective_check` |
| Agent prompts (Appendix K, Figures 15–18) | `prompts.py` |

Hyperparameters taken verbatim from Appendix G: exploration weight `c = 1`,
initial per-tree budget `B = 3`, budget increment `b = 1`, dimension softmax
temperature `T = 1`, few-shot exemplars `k = 1`, FSA top-`k = 3`, correlation
filter `η = 50%`, LLM temperatures 1.0 (generate) / 0.8 (correct) / 0.1 (score),
and the effectiveness thresholds RankIC ≥ 0.015, RankIR ≥ 0.3, percentile
ranks ≤ 0.95, turnover ≤ 1.6, max correlation < 0.8.

## Gaps the paper leaves open — and what was chosen

Each of these is a decision that could differ from the authors' code.

1. **Virtual action Q-value.** Eq. 2 needs `Q(s, a_e)` for the virtual
   expansion action, and the paper defines only its visit count
   (`N = 1 + |C(s)|`). Uses the node's own alpha score `S(f_s)` — expanding `s`
   produces a refinement of `s`, so its prior value is anchored there.
   (`mcts.py::select`)

2. **Empty-repository percentile rank.** Eq. 6 divides by `|F_zoo|`, undefined
   when the repository is empty (which it is for the first tree). Returns 0.5,
   a neutral prior, so early alphas are neither rewarded nor punished for
   arriving first. (`evaluation.py::percentile_rank`)

3. **Turnover and Diversity orientation.** Eq. 7 is written as if higher metric
   is always better, but for turnover and max-correlation lower is better (the
   paper states this in prose only). Both metrics are negated before ranking.
   (`evaluation.py::dimension_scores`)

4. **Turnover construction.** Defined only as "the average daily change in the
   alpha's portfolio holdings". Implemented as a cross-sectionally
   rank-demeaned dollar-neutral book normalised to `sum|w| = 1`, with turnover
   = mean daily `sum|w_t − w_{t−1}|`. This is scale-free, so it is comparable
   across alphas of very different magnitude, and it puts full turnover at 2.0
   — the scale the paper's own 1.6 threshold implies.
   (`evaluation.py::_turnover`)

5. **Correlation between alphas.** "Maximum correlation with the alpha within
   the effective alpha repository", without saying pooled or per-day. Uses mean
   daily cross-sectional Pearson correlation, matching the cross-sectional
   convention of every other metric here.
   (`evaluation.py::daily_cross_sectional_corr`)

6. **Overfitting score scale.** The LLM returns 0–10 (Figure 17) while the
   percentile dimensions are in [0, 1], and Eq. 3/8 assume a shared `e_max`.
   The LLM score is divided by 10. (`evaluation.py::dimension_scores`)

7. **When alphas enter the repository.** Algorithm 1 line 28 adds inside the
   expansion loop on an effectiveness threshold `θ_eff`; Appendix G says the
   check runs "upon the completion of a search tree's expansion" over all nodes
   in the tree, and gives concrete criteria. **The two disagree.** Appendix G is
   followed, because its criteria are specific and testable, and because a
   repository fixed for the duration of a tree makes the relative-ranking
   evaluation stable within that tree. (`run.py`)

8. **Argument-set selection.** Appendix D backtests up to three parameter sets
   and keeps "the configuration yielding the best performance", without saying
   by which measure. Ranked by the mean of the four *numeric* dimension scores;
   the LLM-scored Overfitting dimension is excluded because it does not depend
   on parameter values and would cost three extra LLM calls per expansion.
   (`agent.py::_best_argument_set`)

9. **Correction loop bound.** Algorithm 1 lines 15–18 loop `while not IsValid`
   with no bound. Capped at 3 rounds so one uncooperative response cannot
   consume the whole search budget; a node that fails all 3 is skipped and the
   iteration continues. (`agent.py::MAX_CORRECTION_ROUNDS`)

10. **Lookback window range.** Figure 16 has a `{window_range}` slot whose value
    the paper never states. Defaults to `[2, 60]`, overridable with
    `--window_min` / `--window_max`. (`grammar.py::DEFAULT_WINDOW_RANGE`)

11. **Refinement-suggestion prompt.** Figures 15–18 give the portrait, formula,
    overfitting and refinement prompts, but the prompt that produces the textual
    suggestion `d_{s,i*}` (Eq. 4) is described in prose and never printed.
    `prompts.SUGGESTION_PROMPT` is written from that description — it is the
    one prompt here that is **ours, not the paper's**.

12. **Operator names for four symbols.** Table 2 writes `-x`, `|x|`, `x²`, `1/x`
    without names; they need names to appear in an operator list and in the
    serialised expression. Called `Neg`, `Abs`, `Square`, `Inv`.

13. **`Rank(x,t)` scaling.** "The ranking of x relative to its values over the
    past t days" — returned as a percentile in [0, 1] so the operator is
    dimensionless like the rest of the table.

14. **Non-finite values.** Not discussed in the paper. `Inv`, `Div`, `Log` and
    `Vari` all produce infinities on degenerate inputs; a single infinity
    poisons a whole day's cross-sectional aggregate. All ±inf are mapped to NaN
    at every operator, and NaN pairs are dropped by the metric layer — the same
    treatment missing panel data already gets. (`grammar.py::_finite`)

15. **Missing overfitting judgement.** If the LLM call fails or returns
    unparseable JSON, the dimension scores a neutral 5/10 and the reason field
    records why, rather than aborting the expansion. Visible in
    `repository.json`, not silent. (`agent.py::overfitting_score`)

## Deliberate deviations from the paper's experimental setup

These are FactorBench conventions overriding the paper, not ambiguities.

- **Target.** The paper predicts 10-day and 30-day returns on CSI300/CSI1000
  from Qlib. FactorBench mines against the shared contract target
  `close[t+20]/close[t] − 1` (`forward_return_20d`) on its own cached panels,
  as every other method here does, so all methods optimise the quantity they
  are scored on.
- **Split.** The paper trains on 2011-01-01–2020-12-31 and tests on
  2021-01-01–2024-11-30, with no validation segment. FactorBench uses
  `DEFAULT_SPLIT`; the search sees `train` only and never touches
  `valid`/`test`.
- **Data fields.** Section 2's six raw features (OHLC, volume, VWAP). The
  FactorBench panel also carries `return_1d` and `adv`; they are not exposed,
  to keep the search space the paper's.
- **Pool size.** Appendix G selects the top `k` by RankIR with `k ∈ {10, 50,
  100}`. The default here is `k = 10` — one of the paper's three values, and the
  size the other FactorBench methods emit, so pools are comparable. Override
  with `--top_k`.
- **Backtest and combination.** The paper backtests in Qlib with a
  top-k/drop-n portfolio at 0.15% cost, and combines the selected alphas with a
  separately trained LightGBM or MLP. Neither is reproduced: FactorBench's own
  evaluation and portfolio layers do that, which is what keeps methods
  comparable. Consequently emitted factors carry `weight = null` — the paper's
  combination model is not a linear pool, so there is no per-factor weight to
  record.
- **LLM.** The paper uses GPT-4.1 (with Gemini-2.0-flash-lite and Deepseek-v3
  ablations in Appendix H). Here the model comes from `CHAT_MODEL` against
  `OPENAI_API_BASE`, the same configuration the other LLM methods in this repo
  use, so LLM methods stay comparable to each other.

## Threshold calibration — the paper's gates do not transfer

Appendix G's admission criteria are **absolute**: `RankIC ≥ 0.015`,
`RankIR ≥ 0.3`, turnover ≤ 1.6, max correlation < 0.8. They are calibrated to
the paper's own setting — CSI300/CSI1000, roughly 300 stocks, 10- and 30-day
targets — and they do not carry over to smaller universes.

`RankIR` is the unannualized `mean(daily RankIC) / std(daily RankIC)`. That
definition is confirmed by the paper's own Table 5, where a *random* alpha on
CSI300 scores RankIC 0.0179 / RankIR 0.148 and the method's own alphas average
RankIC 0.0714 / RankIR 0.467. Because the ratio's denominator shrinks with the
size of the cross-section, RankIR is structurally smaller on a 100-stock
universe than on a 300-stock one.

Measured here over random grammar-valid alphas on the FactorBench train
segment:

Null distribution over 200 random grammar-valid alphas per market:

| market | assets | RankIC p95 | RankIR p95 | RankIR p99 | RankIR max | passes the paper's gate |
|---|---|---|---|---|---|---|
| sp100 | 99 | 0.0227 | 0.142 | 0.202 | 0.213 | **0.0%** |
| nikkei225 | 225 | 0.0188 | 0.172 | 0.216 | 0.261 | **0.0%** |
| hsi | 85 | 0.0379 | 0.192 | 0.312 | 0.553 | 2.0% |
| csi300 | 272 | 0.0460 | 0.228 | 0.359 | 2.287 | 4.0% |
| ftse100 | 100 | 0.0452 | 0.252 | 0.619 | 0.972 | 5.0% |

On sp100 the entire random distribution sits below `RankIR = 0.3`, so the gate
admits nothing at all and a run terminates with an empty repository. Note also
that the paper's `RankIC ≥ 0.015` is *below* what a random CSI300 alpha scores
(0.0179), which is a further sign the two numbers are tuned to that market's
signal level rather than being universal.

Consequences for this implementation:

- The thresholds are the paper's by default but are constructor/CLI parameters
  (`evaluation.Thresholds`, `--rank_ic_min` / `--rank_ir_min` /
  `--turnover_max` / `--max_correlation`), and the values actually used are
  recorded in each run's `search_budget`.
- A run that admits nothing **raises** rather than writing an empty
  `factors.json`, and names the blocking criterion. `--allow_empty` overrides.
- Rejection reasons are prefixed with the criterion name so `repository.json`'s
  `rejected` list can be histogrammed to see which gate binds.

Calibrating per market against the random-formula null (for example, admitting
alphas above the 95th percentile of that null) keeps the paper's *intent* —
Section 3 introduces relative ranking precisely to avoid "fixed thresholds that
may be too stringent early on or too lenient later" — while staying reportable.
Whatever is chosen must be stated alongside any result, because it changes what
enters the repository and therefore what the method emits.

## Grammar registration

Emitted factors carry `grammar: "alphajungle-v1"` and are evaluated by
`factor_bench.eval.executors::_alphajungle_signal`, which calls `grammar.py`
directly. Unlike the AlphaGen-lineage grammars this one is pure pandas with no
vendored package, so it runs in-process with nothing to collide with.

## Cost

Each expansion costs 3 LLM calls (suggestion, refined portrait, formula) plus 1
for the overfitting assessment, plus retries — **3.75 per node measured** over a
40-node run on csi300. At the paper's smallest budget of 1000 nodes that is
~3,750 calls per seed, so at the ~115 s/call this benchmark's endpoint sustains,
**~120 h per seed** sequentially, before the 5 markets × 3 seeds of a sweep.

**There is no parallelism inside a tree.** An expansion is a strict chain: the
suggestion feeds the portrait, which feeds the formula, which feeds the
overfitting score. Trees, however, are independent restarts, and with
`INITIAL_BUDGET = 3` a 1000-node run grows *hundreds* of them — so the tree is
the unit that can overlap. `--concurrency` (default 8) grows a batch of trees at
once, bringing a seed to ~15 h.

Two consequences, both deliberate:

- **Trees in a batch share one repository snapshot.** A tree cannot see alphas
  admitted by its batch-mates, so it scores Diversity and picks Frequent
  Subtree Avoidance targets against a slightly staler repository than a
  sequential run would. Admission happens on the main thread after the batch,
  in a deterministic order. This makes `concurrency` part of the run's
  configuration rather than a free knob, so it is recorded in `search_budget`.
- **The node budget can overshoot** by up to one batch, since each tree in a
  batch decides its own depth. The batch size is capped at
  `ceil(remaining / tree_budget)` to bound it.

A run reproduces exactly for a given `(seed, concurrency)`: each tree draws from
its own generator, seeded from the run seed and the tree index, so nothing
depends on thread interleaving. A hard endpoint failure inside a batch is held
until the batch has been admitted and checkpointed, then re-raised — a crash
cannot discard trees that finished. Both are pinned by tests.

### Local compute

The search also does real work per node, and it used to dominate more than the
node count suggests:

| | before | after |
|---|---|---|
| per node, empty repository | 1.61 s | 0.80 s |
| `compile_portrait` at a 48-alpha repository | 16.71 s | 3.33 s |
| per extra repository alpha | 0.31 s | 0.056 s |

Two fixes, neither of which changes a number (verified equal to 2e-16 on
csi300 across six expressions):

1. **The per-date correlations are vectorised.** `_daily_rank_ic` and the
   repository correlation were each a `groupby(level="date").apply(...)` — one
   Python call per date, ~1215 of them per candidate, and the repository one
   ran *once per zoo member*. Both now go through `_row_pearson`, four
   whole-frame reductions on the (date × asset) matrix.
2. **The effective-alpha check no longer re-backtests.** It re-ran the full
   backtest for every node of every tree, recomputing the rank IC and turnover
   the expansion had already paid for. Only Diversity depends on the
   repository, and the signal is a pure function of the expression, so it now
   refreshes `max_correlation` alone via `zoo_max_correlation`.

Cost still grows linearly with the repository, so it is worth watching on long
runs. Use `--num_nodes` for smoke tests.
