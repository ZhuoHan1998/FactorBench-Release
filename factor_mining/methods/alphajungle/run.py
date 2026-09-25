"""Mine alphas with Alpha Jungle on FactorBench data, and write factors.json.

Alpha Jungle has **no official implementation** — the AAAI 2026 paper releases
no code, and none exists publicly. This module implements the paper's
Algorithm 1 (Appendix F) with the hyperparameters of Appendix G and the prompts
of Appendix K. Every place the paper leaves open is listed in METHOD.md next to
this file; nothing here is checked against reference code, because there is
none.

The LLM is configured through the same environment variables the other LLM
methods in this repo use::

    export OPENAI_API_BASE=...   # or OPENAI_BASE_URL
    export OPENAI_API_KEY=...
    export CHAT_MODEL=...

Usage:
    python -m factor_mining.methods.alphajungle.run --market sp100 --seed 0
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import fire
import numpy as np

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alphajungle import grammar
from factor_mining.methods.alphajungle.agent import (
    AlphaAgent,
    Backtester,
    GenerationError,
)
from factor_mining.methods.alphajungle.evaluation import (
    Evaluation,
    Thresholds,
    dimension_scores,
    passes_effective_check,
    zoo_max_correlation,
)
from factor_mining.methods.alphajungle.llm import LLMError, OpenAIChatClient
from factor_mining.methods.alphajungle.mcts import (
    INITIAL_BUDGET,
    Node,
    Repository,
    SearchTree,
    refinement_history,
)

PAPER_URL = "https://arxiv.org/abs/2505.11122"
PAPER_VERSION = "arXiv:2505.11122v3 (AAAI 2026, 40(2):997-1005)"

# Appendix H: "LLM methods evaluated at 1,000, 2,000, or 3,000" node
# evaluations. The smallest reported setting is the default.
DEFAULT_NUM_NODES = 1000

# Trees grown at once. An expansion is a strict LLM chain (suggestion ->
# portrait -> formula -> overfitting score), so nothing inside a tree can
# overlap; independent trees can. 1 restores the fully sequential loop.
DEFAULT_CONCURRENCY = 8


class _Fatal:
    """Wraps an exception raised in a tree worker so the caller can checkpoint
    the rest of the batch before re-raising it."""

    def __init__(self, cause: BaseException) -> None:
        self.cause = cause

    def __str__(self) -> str:
        return str(self.cause)
# Appendix G: "we select the top k alphas with the highest RankIR ... k is set
# to 10, 50, and 100." 10 is chosen so the emitted pool matches the size other
# FactorBench methods emit; see METHOD.md.
DEFAULT_TOP_K = 10


def run_alphajungle(
    market: str = "csi300",
    seed: int = 0,
    num_nodes: int = DEFAULT_NUM_NODES,
    top_k: int = DEFAULT_TOP_K,
    model: str | None = None,
    window_min: int = grammar.DEFAULT_WINDOW_RANGE[0],
    window_max: int = grammar.DEFAULT_WINDOW_RANGE[1],
    tree_budget: int = INITIAL_BUDGET,
    concurrency: int = DEFAULT_CONCURRENCY,
    rank_ic_min: float = Thresholds.rank_ic_min,
    rank_ir_min: float = Thresholds.rank_ir_min,
    turnover_max: float = Thresholds.turnover_max,
    max_correlation: float = Thresholds.max_correlation,
    allow_empty: bool = False,
    out_dir: str = "out/mining",
    verbose: int = 1,
    client: object | None = None,
) -> Path:
    """Run the LLM-guided MCTS search on one market and write factors.json."""
    t0 = time.time()
    # Each tree gets its own generator, derived from this seed and its index,
    # so the run reproduces however the batch's threads interleave.
    np.random.seed(seed)

    panel = load_panel(market)
    TARGET.validate(panel)  # leakage guard: cached target matches its definition
    target = TARGET.from_panel(panel)
    dates = panel.index.get_level_values("date")
    # The search sees the training segment only; valid/test are never touched.
    train_mask = DEFAULT_SPLIT.mask(dates, "train")

    window_range = (window_min, window_max)
    # Paper defaults; overridable because the paper's absolute thresholds are
    # calibrated to ~300-stock A-share universes and admit nothing on smaller
    # ones (see METHOD.md, "Threshold calibration").
    thresholds = Thresholds(
        rank_ic_min=rank_ic_min,
        rank_ir_min=rank_ir_min,
        turnover_max=turnover_max,
        max_correlation=max_correlation,
    )
    client = client or OpenAIChatClient(model=model)
    backtester = Backtester(panel=panel, target=target, train_mask=train_mask)
    agent = AlphaAgent(client, backtester, window_range=window_range)
    repository = Repository()

    save_path = Path(out_dir) / "alphajungle" / f"{market}_{top_k}_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    evaluated = 0
    trees = 0
    rejected: list[str] = []

    def write_artifacts(complete: bool) -> Path:
        """Persist whatever has been mined so far.

        Called after every tree, not only at the end: a search that dies on
        node 277 of 1000 would otherwise throw away every alpha it found.
        """
        chosen = sorted(
            repository.candidates, key=lambda c: c.metrics.rank_ir, reverse=True
        )[:top_k]
        factors = [
            ExpressionFactor(
                name=f"alphajungle_{market}_s{seed}_{i:02d}",
                expression=candidate.formula,
                grammar="alphajungle-v1",
                # The paper combines the selected alphas with a separately
                # trained LightGBM/MLP, not a linear pool, so there is no
                # per-factor weight to record here.
                weight=None,
                train_metrics={
                    "rank_ic": candidate.metrics.rank_ic,
                    "rank_ir": candidate.metrics.rank_ir,
                    "turnover": candidate.metrics.turnover,
                    "max_correlation": candidate.metrics.max_correlation,
                    "coverage": candidate.metrics.coverage,
                },
            )
            for i, candidate in enumerate(chosen)
        ]
        usage = getattr(client, "usage", None)
        mining_run = MiningRun(
            method="alphajungle",
            paradigm="llm-mcts",
            output_form="expression",
            market=market,
            seed=seed,
            split=DEFAULT_SPLIT.as_dict(),
            target=TARGET.name,
            search_budget={
                "num_nodes": num_nodes,
                "nodes_evaluated": evaluated,
                "trees": trees,
                "top_k": top_k,
                "tree_budget": tree_budget,
                "concurrency": concurrency,
                "window_range": f"[{window_min}, {window_max}]",
                "rank_ic_min": rank_ic_min,
                "rank_ir_min": rank_ir_min,
                "turnover_max": turnover_max,
                "max_correlation": max_correlation,
                "model": getattr(client, "model", ""),
                # bools coerce to 0.0 in this dict's value type
                "complete": str(complete).lower(),
                **(usage.as_dict() if usage else {}),
            },
            wall_clock_seconds=time.time() - t0,
            provenance=Provenance(
                repo=PAPER_URL,
                commit=PAPER_VERSION,
                entry="Algorithm 1 (Appendix F)",
                notes="NOT a port of official code - no implementation of Alpha "
                "Jungle has been released. Written from the paper: Algorithm 1, "
                "the hyperparameters of Appendix G, the operator table of "
                "Appendix E, and the prompts of Appendix K. Deviations and gaps "
                "the paper leaves open are documented in "
                "factor_mining/methods/alphajungle/METHOD.md."
                + ("" if complete else " PARTIAL RUN: the search stopped before "
                   "its node budget; see search_budget.complete."),
            ),
            factors=factors,
        )
        written = mining_run.save(save_path / "factors.json")
        (save_path / "repository.json").write_text(json.dumps({
            "market": market,
            "seed": seed,
            "complete": complete,
            "nodes_evaluated": evaluated,
            "repository": [
                {
                    "name": c.name,
                    "formula": c.formula,
                    "description": c.description,
                    "arguments": c.arguments,
                    **c.metrics.as_dict(),
                }
                for c in repository.candidates
            ],
            "forbidden_structures": [
                {"gene": g.text, "support": g.support} for g in repository.forbidden
            ],
            "rejected": rejected[-500:],
        }, indent=2))
        return written

    def grow_one_tree(
        index: int,
    ) -> tuple[int, list[Node], "Exception | _Fatal | None"]:
        """Seed and expand one tree. Returns (index, its nodes, failure).

        Reads the repository but never writes it: admission happens on the main
        thread once the batch is in, so a tree in flight cannot see a
        half-built repository. `agent` is shared, which is safe — the client,
        the backtester and the panel are all read-only or internally locked.

        Failures come back rather than propagating out of the thread pool. An
        `LLMError`/`GenerationError` while seeding means "skip this tree", as it
        did sequentially. Anything else ends the run — but the caller admits and
        checkpoints the batch first, so a hard endpoint failure cannot discard
        the trees that did finish.
        """
        tree: SearchTree | None = None
        try:
            portrait = agent.seed_portrait(repository.forbidden)
            root_candidate = agent.compile_portrait(
                portrait, repository.signals, repository.metrics
            )
            root_history = refinement_history(
                Node(root_candidate, Evaluation(root_candidate.metrics, {}))
            )
            overfitting, reason = agent.overfitting_score(root_candidate, root_history)
            root_scores = dimension_scores(
                root_candidate.metrics, repository.metrics, overfitting
            )
            root = Node(
                candidate=root_candidate,
                evaluation=Evaluation(root_candidate.metrics, root_scores, reason),
            )
            tree = SearchTree(
                root, agent, repository,
                # Its own generator, seeded from the run's seed and the tree
                # index, so a batch reproduces however the threads interleave.
                random.Random(seed * 1_000_003 + index),
                budget=tree_budget,
            )
            return index, tree.run(max_nodes=num_nodes - evaluated), None
        except (LLMError, GenerationError) as e:
            return index, list(tree.nodes) if tree is not None else [], e
        except Exception as e:  # noqa: BLE001 - re-raised by the caller below
            return index, list(tree.nodes) if tree is not None else [], _Fatal(e)

    while evaluated < num_nodes:
        # An MCTS expansion is a strict chain — suggestion, portrait, formula,
        # overfitting score — so there is no parallelism inside a tree. Trees
        # are independent restarts, so they are the unit that can overlap.
        remaining = num_nodes - evaluated
        batch = max(1, min(concurrency, -(-remaining // max(tree_budget, 1))))
        if batch == 1:
            results = [grow_one_tree(trees)]
        else:
            with ThreadPoolExecutor(max_workers=batch) as pool:
                results = list(pool.map(grow_one_tree,
                                        range(trees, trees + batch)))

        nodes: list[Node] = []
        fatal: BaseException | None = None
        for _, tree_nodes, error in results:
            trees += 1
            if isinstance(error, _Fatal):
                fatal = fatal or error.cause
            elif error is not None and verbose:
                print(f"[alphajungle] seed generation failed ({error}); retrying")
            nodes.extend(tree_nodes)
        evaluated += len(nodes)

        # Appendix G, "Effective Alpha Check": run at tree completion over
        # every node in the tree, not inside the expansion loop.
        added = 0
        seen = repository.formulas()
        for node in sorted(nodes, key=lambda n: n.candidate.metrics.rank_ir,
                           reverse=True):
            if node.candidate.formula in seen:
                continue
            candidate = node.candidate
            # Only Diversity depends on the repository, and the signal is a
            # pure function of the expression, so refresh the correlation
            # against the grown repository and keep the rest. Re-running the
            # whole backtest here recomputed the rank IC and turnover the
            # expansion had already paid for, on every node of every tree.
            candidate.metrics = replace(
                candidate.metrics,
                max_correlation=zoo_max_correlation(
                    candidate.signal, repository.signals
                ),
            )
            ok, why = passes_effective_check(
                candidate.metrics, repository.metrics, thresholds
            )
            if ok:
                repository.add(candidate)
                seen.add(candidate.formula)
                added += 1
            else:
                rejected.append(f"{candidate.formula}: {why}")
        if verbose:
            if added == 0 and rejected:
                blocking = Counter(r.split(":")[1].strip() for r in rejected)
                top = ", ".join(f"{k} x{v}" for k, v in blocking.most_common(3))
                print(f"[alphajungle]   nothing admitted; blocking gate: {top}")
            best = max((n.score for n in nodes), default=float("nan"))
            print(
                f"[alphajungle] trees {trees} ({len(results)} in this batch): "
                f"{len(nodes)} nodes (total {evaluated}/{num_nodes}), "
                f"+{added} effective, repository {len(repository)}, "
                f"best node score {best:.3f}"
            )
        write_artifacts(complete=False)
        if fatal is not None:
            # The batch is banked; now let the failure end the run.
            raise fatal
        if not nodes and trees > num_nodes:  # pathological: no seed succeeds
            raise RuntimeError(
                f"no tree seeded successfully in {trees} attempts on {market}"
            )

    if not repository.candidates:
        blocking = Counter(r.split(":")[1].strip() for r in rejected)
        summary = ", ".join(f"{k} (x{v})" for k, v in blocking.most_common())
        message = (
            f"No alpha cleared the effectiveness check on {market} after "
            f"{evaluated} nodes across {trees} trees, so factors.json would be "
            f"empty. Blocking criteria: {summary or 'none recorded'}. The "
            "paper's absolute thresholds are calibrated to ~300-stock A-share "
            "universes; on smaller cross-sections RankIR cannot reach 0.3 (see "
            "METHOD.md). Re-run with calibrated --rank_ir_min / --rank_ic_min, "
            "or pass --allow_empty to write the empty run anyway."
        )
        if not allow_empty:
            raise RuntimeError(message)
        print(f"[alphajungle] WARNING: {message}")

    path = write_artifacts(complete=True)
    if verbose:
        print(f"[alphajungle] {len(repository.candidates)} effective alphas, "
              f"top {min(top_k, len(repository.candidates))} -> {path}")
    return path


if __name__ == "__main__":
    fire.Fire(run_alphajungle)
