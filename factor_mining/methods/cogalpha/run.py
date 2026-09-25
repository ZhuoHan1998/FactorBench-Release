"""Mine alphas with CogAlpha on FactorBench data, and write factors.json.

CogAlpha has **no official implementation** — the ACL 2026 paper releases no
code, and none exists from the authors. This module implements the paper's
Sections 3.1-3.6 with the settings of Appendix B.4/B.8 and the prompts of
Appendix C. Everything the paper leaves open is listed in METHOD.md beside this
file; nothing here is checked against reference code, because there is none.

Factors are emitted as ``code`` contract entries (executable Python under the
``__main__`` script convention), the same form RD-Agent uses, so the existing
code executor evaluates them unchanged.

The LLM is configured through the shared environment variables::

    export OPENAI_API_BASE=...   # or OPENAI_BASE_URL
    export OPENAI_API_KEY=...
    export CHAT_MODEL=...

Usage:
    python -m factor_mining.methods.cogalpha.run --market sp100 --seed 0
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import fire
import numpy as np

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import CodeFactor, MiningRun, Provenance
from factor_mining.llm import OpenAIChatClient
from factor_mining.methods.cogalpha.agents import AGENTS, RAW_FIELDS
from factor_mining.methods.cogalpha.codegen import build_script
from factor_mining.methods.cogalpha.evolution import CogAlphaSearch, SearchConfig
from factor_mining.methods.cogalpha.execution import Sandbox
from factor_mining.methods.cogalpha.quality import QualityChecker

PAPER_URL = "https://arxiv.org/abs/2511.18850"
PAPER_VERSION = "arXiv:2511.18850v4 (ACL 2026, pages 11715-11749)"

# Appendix B.4 runs 24 generations per agent; across 21 agents at 96 children
# that is ~48k alpha generations and ~500 GPU-hours per market on a local
# model. FactorBench keeps the full 21-agent hierarchy and shortens the depth.
DEFAULT_GENERATIONS = 4
DEFAULT_TOP_K = 10


def run_cogalpha(
    market: str = "csi300",
    seed: int = 0,
    generations: int = DEFAULT_GENERATIONS,
    top_k: int = DEFAULT_TOP_K,
    parent_pool: int = SearchConfig.parent_pool,
    initial_pool: int = SearchConfig.initial_pool,
    initial_agents: int = SearchConfig.initial_agents,
    num_per_request: int = SearchConfig.num_per_request,
    judge: bool = True,
    llm_code_audit: bool = False,
    paraphrase: bool = True,
    leakage_test: bool = True,
    max_llm_calls: int = 0,
    concurrency: int = SearchConfig.concurrency,
    model: str | None = None,
    out_dir: str = "out/mining",
    verbose: int = 1,
    client: object | None = None,
) -> Path:
    """Run the CogAlpha evolutionary search on one market and write factors.json."""
    t0 = time.time()
    rng = random.Random(seed)
    np.random.seed(seed)

    panel = load_panel(market)
    TARGET.validate(panel)  # leakage guard: cached target matches its definition
    target = TARGET.from_panel(panel)
    dates = panel.index.get_level_values("date")
    # The search sees the training segment only; valid/test are never touched.
    train_mask = DEFAULT_SPLIT.mask(dates, "train")

    save_path = Path(out_dir) / "cogalpha" / f"{market}_{top_k}_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Section 3.1: OHLCV only — the panel's vwap and derived columns are not
    # exposed, so the search space is the paper's.
    search_panel = panel.loc[train_mask, list(RAW_FIELDS)]

    client = client or OpenAIChatClient(model=model)
    sandbox = Sandbox(
        panel=search_panel, workdir=save_path / "sandbox", leakage_test=leakage_test
    )
    checker = QualityChecker(
        client, sandbox, judge=judge, llm_code_audit=llm_code_audit
    )
    config = SearchConfig(
        generations=generations,
        parent_pool=parent_pool,
        initial_pool=initial_pool,
        initial_agents=initial_agents,
        num_per_request=num_per_request,
        horizon=TARGET.horizon_days,
        max_llm_calls=max_llm_calls,
        paraphrase=paraphrase,
        concurrency=concurrency,
    )
    search = CogAlphaSearch(
        client=client, checker=checker, target=target, train_mask=train_mask,
        market=market, rng=rng, config=config, verbose=verbose,
    )
    elite = search.run()

    # Section 3.4: the elite pool is the final candidate pool; rank by RankIC.
    chosen = sorted(elite, key=lambda c: c.fitness.rank_ic, reverse=True)[:top_k]

    factors = [
        CodeFactor(
            name=f"cogalpha_{market}_s{seed}_{i:02d}",
            source=build_script(candidate.factor),
            entry_point="__main__",
            # The paper combines the selected alphas with a separately trained
            # LightGBM or Ridge model, not a linear pool, so there is no
            # per-factor weight to record.
            weight=None,
            train_metrics={
                **candidate.fitness.as_dict(),
                "generation": float(candidate.generation),
            },
        )
        for i, candidate in enumerate(chosen)
    ]

    if not factors:
        raise RuntimeError(
            f"CogAlpha produced no elite alphas on {market} after "
            f"{search.calls} LLM calls. Checker outcome: "
            f"{checker.stats.as_dict()}. Inspect {save_path / 'search.json'} for "
            "the rejection reasons before re-running."
        )

    usage = getattr(client, "usage", None)
    run = MiningRun(
        method="cogalpha",
        paradigm="evolutionary-llm",
        output_form="code",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget={
            "generations": generations,
            "agents": len(AGENTS),
            "parent_pool": parent_pool,
            "initial_pool": initial_pool,
            "initial_agents": initial_agents,
            "children_multiplier": config.children_multiplier,
            "concurrency": concurrency,
            "top_k": top_k,
            # bools coerce to 0.0 in this dict's value type
            "judge": str(judge).lower(),
            "llm_code_audit": str(llm_code_audit).lower(),
            "paraphrase": str(paraphrase).lower(),
            "leakage_test": str(leakage_test).lower(),
            "search_llm_calls": search.calls,
            "model": getattr(client, "model", ""),
            **(usage.as_dict() if usage else {}),
        },
        wall_clock_seconds=time.time() - t0,
        provenance=Provenance(
            repo=PAPER_URL,
            commit=PAPER_VERSION,
            entry="Sections 3.1-3.6",
            notes="NOT a port of official code — no implementation of CogAlpha "
            "has been released by the authors. Written from the paper: the "
            "seven-level 21-agent hierarchy, the Multi-Agent Quality Checker, "
            "the five-metric fitness gate, and the Thinking Evolution operators, "
            "with the prompts of Appendix C. Evolution depth is shortened from "
            "the paper's 24 generations per agent; deviations and open choices "
            "are documented in factor_mining/methods/cogalpha/METHOD.md.",
        ),
        factors=factors,
    )
    path = run.save(save_path / "factors.json")
    (save_path / "search.json").write_text(json.dumps({
        "market": market,
        "seed": seed,
        "llm_calls": search.calls,
        "checker": checker.stats.as_dict(),
        "elite": [
            {
                "name": c.factor.name,
                "agent": c.agent,
                "origin": c.origin,
                "generation": c.generation,
                "docstring": c.factor.docstring,
                "source": c.factor.source,
                **c.fitness.as_dict(),
            }
            for c in sorted(elite, key=lambda c: c.fitness.rank_ic, reverse=True)
        ],
        "rejections": checker.stats.reasons[-300:],
    }, indent=2))
    if verbose:
        print(f"[cogalpha] {len(factors)} factors from an elite pool of {len(elite)} "
              f"({search.calls} LLM calls) -> {path}")
    return path


if __name__ == "__main__":
    fire.Fire(run_cogalpha)
