"""The one admission rule, shared by RQ1-RQ4.

Why this module exists
----------------------
Each research question used to derive its own validity filter, and they did not
agree. Measured on the September tables, on the test split:

    RQ1  >=100 scorable dates in ALL three segments      3831 factors
    RQ2  finite self-correlation, decided PER SPLIT      3870
    RQ3  >=100 style-eligible dates, decided PER SPLIT   3504
    RQ4  valid.n_dates >= 100 only (validation alone)    3876

    RQ1 vs RQ3:  365 factors in one and not the other
    RQ2 vs RQ3:  366

Three of those differences are structural rather than cosmetic:

1. **Per-split vs across-split.** Deciding per split lets a factor be in the
   test sample but not the train sample -- RQ3 had 9 such factors -- so a
   train-vs-test comparison is drawn across two different populations. The
   paired statistics RQ1 is built on assume the same factor on both sides.
2. **Different date universes.** RQ3's window is shorter because ``beta`` needs
   a 252-day warm-up, so its "100 dates" is a stricter bar on less data.
3. **RQ4 checked validation only**, so a factor with three scorable test dates
   was selectable.

The rule
--------
A factor is **admitted** when it executed and produced a RANKING on enough days:
at least ``MIN_SCORABLE_DATES`` dates with a defined cross-sectional IC in
*every* segment. Decided once, across all segments together, in
:mod:`factor_bench.eval.ic_table`, and written to ``ic_table.csv`` as the
``admitted`` column. Everything downstream reads that column instead of
re-deriving a rule.

Two things the rule deliberately does NOT do
--------------------------------------------
- **It is not a quality filter.** A weak-but-scorable factor stays in; that is a
  result, not a defect. Only factors on which the statistics are *undefined* are
  removed -- constant cross-sections above all.
- **It does not hide the exclusions.** A method that emits constant factors has
  a broken generation mechanism, and dropping those quietly would flatter it:
  its survivors look clean while its waste disappears. ``flat_dates`` and the
  per-method admitted rate are reported as RQ0 data-quality results.

Per-RQ application
------------------
- **RQ1, RQ2**: ``admitted`` as-is.
- **RQ3**: ``admitted`` AND style-eligible. The style window is genuinely
  shorter (the 252-day beta warm-up), so RQ3 is a subset by construction -- but
  it is now a *stated* subset of one rule rather than an independent rule that
  happens to differ.
- **RQ4**: ``admitted`` AND the selection procedure's own stricter steps
  (positive validation IC, the correlation ceiling). Strictly nested, so every
  RQ4 factor is an RQ1 factor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from factor_bench.paths import IC_TABLE as DEFAULT_IC_TABLE

# At least this many dates per segment on which the factor's cross-section
# actually varies, so an IC is defined. ~5 months; the segments hold roughly
# 1265 / 501 / 466 dates.
MIN_SCORABLE_DATES = 100

SEGMENTS = ("train", "valid", "test")


def admitted_mask(table: pd.DataFrame) -> pd.Series:
    """Boolean mask over rows of an ``ic_table``-shaped frame.

    Prefers the stored ``admitted`` column. Tables written before that column
    existed are handled by re-deriving it from the same inputs -- identical by
    construction, but the caller is told, because a silent fallback is how two
    rules diverge in the first place.
    """
    if "admitted" in table.columns:
        return table["admitted"].fillna(0).astype(bool)

    print("[admission] ic_table.csv predates the `admitted` column; deriving it. "
          "Rebuild with: python -m factor_bench.eval.ic_table build")
    ok = table["status"] == "ok" if "status" in table else pd.Series(True, index=table.index)
    cols = [f"{s}.n_dates" for s in SEGMENTS]
    return ok & (table[cols] >= MIN_SCORABLE_DATES).all(axis=1)


def admitted_keys(ic_table: str | Path = DEFAULT_IC_TABLE) -> set[tuple[str, str]]:
    """``{(market, factor)}`` for every admitted factor.

    Keyed on (market, factor) rather than factor alone: several methods reuse a
    factor name across markets (``alpha101`` uses ``alpha001`` in all five), so
    a name-only key silently merges different factors.
    """
    t = pd.read_csv(ic_table)
    return set(zip(t.loc[admitted_mask(t), "market"], t.loc[admitted_mask(t), "factor"]))


def admitted_rate(ic_table: str | Path = DEFAULT_IC_TABLE) -> pd.DataFrame:
    """Per-method admission, as an RQ0 data-quality table.

    ``constant`` counts factors that executed but never produced a ranking in
    any segment -- the broken ones, as distinct from the merely weak.
    """
    t = pd.read_csv(ic_table)
    t["_adm"] = admitted_mask(t)
    n_cols = [f"{s}.n_dates" for s in SEGMENTS]
    ok = t["status"] == "ok" if "status" in t else pd.Series(True, index=t.index)
    out = t.groupby("method").agg(
        factors=("factor", "size"),
        executed=("status", lambda s: int((s == "ok").sum())) if "status" in t
        else ("factor", "size"),
        admitted=("_adm", "sum"),
    )
    out["constant"] = (t[ok & (t[n_cols] == 0).all(axis=1)]
                       .groupby("method").size().reindex(out.index).fillna(0).astype(int))
    out["admitted %"] = (out["admitted"] / out["factors"] * 100).round(1)
    return out
