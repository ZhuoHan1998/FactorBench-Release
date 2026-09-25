"""Factor selection — one procedure applied identically to every method.

The rule is universal; the resulting *count* is not. A method whose pool yields
only three qualifying factors contributes three, and that count is reported
rather than padded.

Procedure (all thresholds fixed before any test evaluation):

1. Drop invalid, constant and poorly covered factors.
2. Rank survivors by **validation** IC, using their training-fixed direction.
3. Optionally keep only factors with positive validation IC
   (``require_positive_ic``, **off by default** -- see the field's note for the
   measured cost of leaving it off).
4. Drop candidates whose correlation is undefined against every other
   candidate -- too little asset overlap to assess -- then walk the ranking,
   accepting a factor only if its absolute correlation with every
   already-accepted factor is below the threshold.
5. Stop at pool exhaustion or the cap.

Nothing here touches the test period: validation IC drives the ranking and the
correlations come from the validation split, so the selected set is frozen
before test evaluation begins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionConfig:
    """Thresholds, fixed in advance. Not tuned on test."""

    max_correlation: float = 0.8      # step 4 admission threshold
    cap: int | None = None            # None = full pool; an int = fixed-size
    min_valid_dates: int = 100        # step 1 coverage floor (validation only)
    # Step 1 also requires the benchmark-wide `admitted` flag when the column is
    # present, so RQ4's pool is strictly nested inside RQ1's. Without it, a
    # factor with three scorable TEST dates was selectable, because selection
    # only ever looks at validation -- 45 such factors in the September table.
    require_admitted: bool = True
    # Step 3, OFF by default. Requiring positive validation IC is a real filter,
    # not a formality: without it the equal-weight composite's mean test rank-IC
    # falls from 0.0222 to 0.0010 across 131 runs, and six of nine mining methods
    # produce a NEGATIVE composite. That is the RQ1 finding in portfolio form --
    # the train-fixed direction persists into validation only ~53% of the time,
    # so averaging every factor with positive weight cancels the signal.
    # It is off by default so the unfiltered pool is the reported baseline and
    # the filter's effect is measured rather than assumed. Turn it on to see the
    # selected-pool result.
    require_positive_ic: bool = False  # step 3
    ic_column: str = "valid.ic"       # ranking key: validation, never test


def select_factors(
    candidates: pd.DataFrame,
    corr: np.ndarray | None = None,
    index_of: dict[str, int] | None = None,
    config: SelectionConfig = SelectionConfig(),
) -> pd.DataFrame:
    """Run the selection procedure over one run's candidate factors.

    Args:
        candidates: rows for a single (method, market, seed) run. Must carry
            ``factor``, the ranking IC column, and ``valid.n_dates``.
        corr: pairwise |correlation| source for step 4 — the validation-split
            matrix from ``factor_bench.eval.redundancy``. When None, step 4 is
            skipped and that is recorded in the result.
        index_of: factor name -> row/column in ``corr``.

    Returns:
        The accepted factors in acceptance order, with a ``rank`` column. Empty
        when nothing qualifies — a real outcome, reported as "no qualifying
        composite" rather than silently backfilled.
    """
    df = candidates.copy()
    ic = config.ic_column

    # step 1 -- validity
    ok = df[ic].notna()
    if "valid.n_dates" in df:
        ok &= df["valid.n_dates"] >= config.min_valid_dates
    if "status" in df:
        ok &= df["status"] == "ok"
    if config.require_admitted and "admitted" in df:
        ok &= df["admitted"].fillna(0).astype(bool)
    df = df[ok]

    # step 3 -- positive validation IC (sign already fixed on train)
    if config.require_positive_ic:
        df = df[df[ic] > 0]
    if df.empty:
        return df.assign(rank=pd.Series(dtype=int))

    # step 2 -- rank by validation IC, best first
    df = df.sort_values(ic, ascending=False)

    # step 4 -- greedy admission under the correlation ceiling
    #
    # Candidates whose correlation is undefined against EVERY other candidate
    # are disqualified first, before any acceptance. Such a factor shares too
    # few assets with the pool for redundancy to be assessed at all -- and
    # without this it would be accepted unchecked whenever it happened to rank
    # first (the ceiling test only applies once something is already accepted),
    # and would then reject every later candidate, since each would compare
    # all-NaN against it. Measured cost of not doing this: 6 of 131 runs came
    # back with a single uncorrelatable factor covering too few stocks to hold
    # a position, so the backtest was flat at exactly 0.00 and averaged into the
    # method's mean as if it were a real result.
    pool_ix = [index_of.get(n) for n in df["factor"]] if index_of is not None else []
    vettable: set[int] = set()
    if corr is not None and index_of is not None:
        known = [j for j in pool_ix if j is not None]
        for pos, j in enumerate(pool_ix):
            if j is None:
                continue
            others = [k for k in known if k != j]
            # With a single candidate there is nothing to compare against, so
            # there is no evidence either way and it is kept.
            if not others or np.isfinite(corr[j, others]).any():
                vettable.add(pos)

    accepted: list[int] = []
    accepted_ix: list[int] = []
    for pos, row in enumerate(df.itertuples()):
        name = row.factor
        if corr is not None and index_of is not None:
            j = index_of.get(name)
            if j is None:
                continue                      # no correlation row: cannot vet it
            if pos not in vettable:
                continue                      # uncorrelatable against the pool
            if accepted_ix:
                c = np.abs(corr[j, accepted_ix])
                # NaN means the pair was never jointly scorable; treat as
                # unvettable and reject rather than assume independence.
                if np.all(np.isnan(c)) or np.nanmax(c) >= config.max_correlation:
                    continue
            accepted_ix.append(j)
        accepted.append(pos)
        if config.cap is not None and len(accepted) >= config.cap:
            break

    out = df.iloc[accepted].copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out
