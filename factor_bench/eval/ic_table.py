"""Per-factor IC records: the shared input to RQ0, RQ1, RQ4 and RQ5.

This executes every factor once and stores ~20 numbers describing it — not the
signal itself. The full signal matrix is ~5.4 GB and is only needed for
redundancy work (RQ2); everything that is a statement about a factor's own
predictive power can be answered from this table, which is a few hundred KB.

What is recorded, and why each choice matters
---------------------------------------------
- **Sign fixed on train.** Methods disagree on convention (gp emits 100%
  positive train IC, alphaqcm ~54% negative), so signed averages across a pool
  cancel toward zero and measure convention, not skill. Each factor's sign comes
  from its train IC and is applied unchanged to validation and test. Fixing it
  on test would fabricate the very effect these RQs are testing for.
- **HAC t-statistics.** Daily IC is ~0.95 autocorrelated at lag 1 because the
  20-day target is sampled daily. ICIR is kept as a descriptive ratio, but
  significance comes from :func:`newey_west_tstat`.
- **Scorable-date counts per segment.** A factor whose cross-section is constant
  on many dates has undefined IC there; ``mean()`` skips those silently, so the
  segments would be averaged over different numbers of days without the counts.
- **One admission rule for the whole benchmark.** ``flat_dates`` counts the dates
  where the factor had data but no cross-sectional spread -- every stock the same
  value, so no ranking exists and IC is undefined. ``admitted`` is 1 when the
  factor has at least ``MIN_SCORABLE_DATES`` scorable dates in *every* segment.
  Every RQ should filter on ``admitted`` rather than re-deriving its own rule, so
  the factor set and the denominators agree across RQ1-RQ4.
- **Exclusion is reported, not silent.** A method emitting constant factors has a
  broken generation mechanism, and dropping those factors quietly would flatter
  it: its survivors look clean while its waste disappears. The per-method
  ``admitted`` rate is an RQ0 data-quality result and belongs beside the
  executability numbers.

Usage:
    python -m factor_bench.eval.ic_table build --out_csv out/eval/ic_table.csv
    python -m factor_bench.eval.ic_table build --methods gp,alpha101 --resume
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_bench.eval import executors
from factor_bench.eval.admission import MIN_SCORABLE_DATES
from factor_bench.paths import IC_TABLE as DEFAULT_IC_TABLE
from factor_bench.paths import MINING_DIR as DEFAULT_MINING_DIR
from factor_bench.eval.metrics import newey_west_tstat, train_sign
from factor_mining.contracts import MiningRun

# Lags for the HAC correction. Chosen from the known overlap (a 20-day forward
# return sampled daily), not from a data-driven bandwidth rule, which returns
# ~5 on a two-year window and would leave most of the overlap uncorrected.
HAC_LAGS = 25

SEGMENTS = ("train", "valid", "test")
# The admission rule lives in factor_bench.eval.admission and is imported, not
# restated, so there is exactly one definition in the benchmark. This module is
# where it is APPLIED (the `admitted` column below); that module documents why.
FIELDNAMES = [
    "method", "market", "seed", "run", "factor", "grammar",
    "status", "error", "sign", "admitted",
    *[f"{seg}.{stat}" for seg in SEGMENTS
      for stat in ("ic", "rank_ic", "icir", "t_hac", "t_naive",
                   "n_dates", "flat_dates")],
]


def daily_ic_pair(signal: pd.Series, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Per-date cross-sectional Pearson and Spearman IC, vectorised.

    Equivalent to calling :func:`factor_bench.eval.metrics.daily_ic` twice, but
    without the per-date Python loop — at ~2,900 factors that loop dominates
    runtime. Correlations are computed from within-date centred values (two
    passes) rather than the raw sums-of-squares shortcut, which loses precision
    when signal magnitudes are large relative to returns.

    A date is scored only where at least two pairs are present and both sides
    have non-zero spread; otherwise the correlation is undefined and NaN, never
    zero.
    """
    df = pd.DataFrame({"s": signal, "t": target}).dropna()
    if df.empty:
        empty = pd.Series(dtype="float64")
        return empty, empty
    dates = df.index.get_level_values("date")

    def _corr(x: pd.Series, y: pd.Series) -> pd.Series:
        xc = x - x.groupby(dates).transform("mean")
        yc = y - y.groupby(dates).transform("mean")
        cov = (xc * yc).groupby(dates).sum()
        vx = (xc * xc).groupby(dates).sum()
        vy = (yc * yc).groupby(dates).sum()
        n = x.groupby(dates).size()
        out = cov / np.sqrt(vx * vy)
        out[(n < 2) | (vx <= 0) | (vy <= 0)] = np.nan
        return out

    ic = _corr(df["s"], df["t"])
    rank_ic = _corr(df.groupby(dates)["s"].rank(), df.groupby(dates)["t"].rank())
    return ic, rank_ic


def _as_list(value) -> list[str]:
    """Normalise a CLI filter. fire turns ``--methods a,b`` into a tuple."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def factor_record(factor, run: MiningRun, run_name: str, panel: pd.DataFrame,
                  target: pd.Series) -> dict:
    """Execute one factor and summarise its IC per segment. Never raises."""
    row: dict = {
        "method": run.method, "market": run.market, "seed": run.seed,
        "run": run_name, "factor": factor.name,
        "grammar": getattr(factor, "grammar", "") or getattr(factor, "entry_point", ""),
        "status": "ok", "error": "", "sign": float("nan"),
    }
    try:
        signal = executors.compute_signal(factor, run.market, panel).reindex(panel.index)
        ic, rank_ic = daily_ic_pair(signal, target)

        # Dates where the factor has data but NO cross-sectional spread: every
        # stock gets the same value, so there is no ranking and IC is undefined.
        # This is different from having no data, and the two are counted apart:
        # a constant factor is a broken factor, not a weak one, and the rate is
        # a data-quality result in its own right (RQ0), not a silent exclusion.
        xs = signal.groupby(level="date")
        flat_by_date = (xs.count() >= 2) & (xs.std().fillna(0.0) <= 0.0)

        if ic.dropna().empty:
            raise ValueError("no scorable dates: signal has no cross-sectional spread")

        sign = train_sign(ic, DEFAULT_SPLIT)
        row["sign"] = sign
        ic, rank_ic = ic * sign, rank_ic * sign

        for seg in SEGMENTS:
            start, end = DEFAULT_SPLIT.segment(seg)
            seg_ic = ic.loc[pd.Timestamp(start): pd.Timestamp(end)].dropna()
            seg_ric = rank_ic.loc[pd.Timestamp(start): pd.Timestamp(end)].dropna()
            n = len(seg_ic)
            row[f"{seg}.n_dates"] = n
            row[f"{seg}.flat_dates"] = int(
                flat_by_date.loc[pd.Timestamp(start): pd.Timestamp(end)].sum())
            row[f"{seg}.ic"] = float(seg_ic.mean()) if n else float("nan")
            row[f"{seg}.rank_ic"] = float(seg_ric.mean()) if n else float("nan")
            sd = seg_ic.std()
            icir = float(seg_ic.mean() / sd) if n > 1 and sd and not np.isnan(sd) else float("nan")
            row[f"{seg}.icir"] = icir
            row[f"{seg}.t_hac"] = newey_west_tstat(seg_ic, HAC_LAGS)
            # Reported alongside only to expose the size of the correction.
            # Never use it for inference.
            row[f"{seg}.t_naive"] = float(icir * np.sqrt(n)) if n > 1 and not np.isnan(icir) \
                else float("nan")

        # Admission is decided ONCE, across all three segments together. Deciding
        # it per segment would give each split a different factor set and quietly
        # break the paired train-vs-test statistic RQ1 is built on.
        row["admitted"] = int(all(row[f"{seg}.n_dates"] >= MIN_SCORABLE_DATES
                                  for seg in SEGMENTS))
    except Exception as e:  # noqa: BLE001 - recorded per factor, never swallowed
        row["status"] = "fail"
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}".replace("\n", " / ")
        row["admitted"] = 0
    return row


def build(methods: str | None = None, markets: str | None = None,
          out_csv: str = str(DEFAULT_IC_TABLE),
          mining_dir: str = str(DEFAULT_MINING_DIR),
          limit: int | None = None, resume: bool = False) -> None:
    """Execute every factor and write one IC record per factor."""
    want_m = _as_list(methods)
    want_k = _as_list(markets)

    runs = []
    for path in sorted(Path(mining_dir).glob("*/*/factors.json")):
        try:
            run = MiningRun.load(path)
        except Exception as e:  # noqa: BLE001 - a malformed contract file is a finding
            print(f"[ic] SKIP unreadable {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if (want_m and run.method not in want_m) or (want_k and run.market not in want_k):
            continue
        runs.append((path, run))
    if not runs:
        raise SystemExit(f"No runs matched under {mining_dir}")

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if resume and out.exists():
        prior = pd.read_csv(out)
        # Only successful records count as done. A recorded failure is usually
        # environmental — most often a method venv that was missing on the
        # machine where the row was written — so it must be retried rather than
        # carried forward, otherwise resuming on a machine that CAN run the
        # method would silently keep the failure forever.
        ok = prior[prior["status"] == "ok"]
        done = set(zip(ok["run"].astype(str), ok["factor"].astype(str), strict=False))
        retry = len(prior) - len(ok)
        if retry:
            # Drop the stale failure rows before re-running them, or the retry
            # would append a second record for the same factor and every
            # downstream aggregate would double-count it.
            ok.to_csv(out, index=False)
        print(f"[ic] resume: {len(done)} factors already recorded"
              + (f", {retry} prior failures will be retried" if retry else ""))

    append = bool(resume and out.exists())
    handle = out.open("a" if append else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not append:
        writer.writeheader()

    panels: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    n_done = n_fail = 0
    t0 = time.time()
    try:
        for path, run in runs:
            run_name = path.parent.name
            factors = run.factors[:limit] if limit else run.factors
            pending = [f for f in factors if (run_name, f.name) not in done]
            if not pending:
                continue
            if run.market not in panels:
                panel = load_panel(run.market)
                panels[run.market] = (panel, TARGET.from_panel(panel))
            panel, target = panels[run.market]
            fails = 0
            for factor in pending:
                row = factor_record(factor, run, run_name, panel, target)
                writer.writerow(row)
                handle.flush()
                n_done += 1
                fails += row["status"] == "fail"
            n_fail += fails
            flag = f"  {fails} unusable" if fails else ""
            print(f"[ic] {run.method:12s} {run_name:30s} {len(pending):4d} factors{flag}")
    finally:
        handle.close()
    print(f"\n[ic] {n_done} factors ({n_fail} unusable) in {(time.time()-t0)/60:.1f} min -> {out}")


def compare_to_floor(ic_csv: str = "out/ic_table.csv", floor: str = "alpha101",
                     segment: str = "test") -> None:
    """RQ0: does any method beat the published-formula floor?

    The comparison applies the **same selection rule to the floor**. A mining
    method emits a pool it selected by maximising train IC; Alpha101 ships a
    fixed library. Comparing the method's selected pool against the whole
    library would credit the method for selection it was always going to do.
    So for each method pool of size N in a market, the floor is given its own
    best N by train IC, on the same training data, and the two are compared
    out-of-sample.
    """
    df = pd.read_csv(ic_csv)
    usable = df[df["status"] == "ok"].copy()
    if usable.empty:
        raise SystemExit("no usable factors in the table")
    col = f"{segment}.ic"

    floor_rows = usable[usable["method"] == floor]
    if floor_rows.empty:
        raise SystemExit(f"floor method {floor!r} not present in {ic_csv}")

    out = []
    for (method, market, seed), pool in usable.groupby(["method", "market", "seed"]):
        if method == floor:
            continue
        n = len(pool)
        fl = floor_rows[floor_rows["market"] == market]
        if fl.empty:
            continue
        fl_top = fl.nlargest(min(n, len(fl)), "train.ic")
        out.append({
            "method": method, "market": market, "seed": seed, "n": n,
            f"median_{segment}": pool[col].median(),
            f"floor_median_{segment}": fl_top[col].median(),
            f"best_{segment}": pool[col].max(),
            f"floor_best_{segment}": fl_top[col].max(),
            "frac_positive": (pool[col] > 0).mean(),
            "frac_t_hac_gt2": (pool[f"{segment}.t_hac"].abs() > 2).mean(),
            "floor_frac_t_hac_gt2": (fl_top[f"{segment}.t_hac"].abs() > 2).mean(),
        })
    table = pd.DataFrame(out)
    table["beats_floor"] = table[f"median_{segment}"] > table[f"floor_median_{segment}"]

    print(f"\n=== RQ0: per (method, market, seed), {segment} segment ===")
    print(f"floor = {floor}, matched to each pool's N by train-IC selection\n")
    print(table.round(4).to_string(index=False))

    print("\n=== rolled up by method (seed x market as the replication unit) ===")
    roll = table.groupby("method").agg(
        pools=("n", "size"),
        median_n=("n", "median"),
        median_test=(f"median_{segment}", "median"),
        floor_median_test=(f"floor_median_{segment}", "median"),
        beats_floor=("beats_floor", "mean"),
        frac_t_hac_gt2=("frac_t_hac_gt2", "median"),
        floor_frac_t_hac_gt2=("floor_frac_t_hac_gt2", "median"),
    ).round(4)
    print(roll.to_string())
    print("\nbeats_floor is the share of that method's (market, seed) pools whose median")
    print(f"{segment} IC exceeds the floor's, at matched N. 0.5 is a coin flip.")


if __name__ == "__main__":
    fire.Fire({"build": build, "compare": compare_to_floor})
