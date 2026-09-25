"""Emit the WorldQuant Alpha101 pool as a FactorBench mining run.

Alpha101 is the RQ0 floor: the "have we improved since 2015?" comparator. It is
not a search method — there is no exploration, no reward, and no seed. The whole
"pool" is a fixed, published list of formulas, so one run per market fully
describes it.

It is nevertheless written out as an ordinary ``factors.json`` conforming to
:class:`~factor_mining.contracts.MiningRun`, and executed through the same
registered executor as every mined factor. That is deliberate: if the baseline
needed special-casing anywhere in evaluation, the comparison against it would not
be like-for-like, which is the one thing this benchmark exists to guarantee.

Usage:
    python -m factor_mining.methods.alpha101.run --market csi300
    python -m factor_mining.methods.alpha101.run --market all
"""

from __future__ import annotations

import time
from pathlib import Path

import fire
import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_bench.data.schema import MARKETS
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alpha101.alpha101 import Alpha101, build_alpha101

REFERENCE = "https://arxiv.org/abs/1601.00991"


def run_alpha101(
    market: str = "csi300",
    out_dir: str = "out/mining",
    seed: int = 0,
) -> Path | list[Path]:
    """Compute every Alpha101 factor for one market and write factors.json."""
    if market == "all":
        return [run_alpha101(m, out_dir=out_dir, seed=seed) for m in MARKETS]
    if market not in MARKETS:
        raise ValueError(f"Unknown market {market!r}; expected one of {MARKETS}")

    started = time.time()
    panel = load_panel(market)
    calculator = build_alpha101(panel)

    # strict=True: a formula that cannot be computed is a defect to fix, not a
    # factor to drop. The pool size must be the same on every market, and a
    # silently shorter pool would quietly flatter the baseline's median IC.
    signals = calculator.compute_all(strict=True)

    factors = [
        ExpressionFactor(
            name=f"alpha101_{market}_{name}",
            # The grammar is 'by name': the expression is the alpha's identifier
            # and the executor dispatches to the published formula. Alpha101 is a
            # fixed library, not a generated expression tree, so there is nothing
            # to serialise as an AST.
            expression=name,
            grammar="alpha101-v1",
            train_metrics={},
        )
        for name in sorted(signals)
    ]

    run = MiningRun(
        method="alpha101",
        paradigm="published-formula-library",
        output_form="expression",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget={
            # No search happens. Recorded explicitly so the RQ5 compute-efficiency
            # comparison and the RQ1 budget analysis both read zero rather than
            # a missing value they might impute.
            "candidates_evaluated": 0,
            "formulas_published": len(factors),
            "complete": "true",
        },
        wall_clock_seconds=time.time() - started,
        provenance=Provenance(
            repo=REFERENCE,
            commit="n/a (published paper, not a code release)",
            entry="factor_mining/methods/alpha101/alpha101.py",
            notes=(
                "Kakushadze (2016), '101 Formulaic Alphas'. Ported from the "
                "original FactorBench implementation on the main branch. Alphas "
                "requiring IndNeutralize are not implemented: the panel carries "
                "no industry classification, so roughly a third of the published "
                "101 are absent and the pool is not the full set. Three changes "
                "from that implementation: failures are raised rather than "
                "swallowed, correlation()'s min_periods no longer exceeds short "
                "windows (which had made alpha045 unconditionally fail), and "
                "_safe_corr's NaN-to-zero fill is documented as a deviation — it "
                "is why these factors report ~100% coverage."
            ),
        ),
        factors=factors,
    )

    save_path = Path(out_dir) / "alpha101" / f"{market}_{len(factors)}_{seed}"
    written = run.save(save_path / "factors.json")
    print(f"[alpha101/{market}] {len(factors)} factors -> {written} "
          f"({time.time() - started:.0f}s)")
    return written


if __name__ == "__main__":
    fire.Fire(run_alpha101)
