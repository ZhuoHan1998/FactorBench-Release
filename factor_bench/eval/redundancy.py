"""Pairwise factor-signal correlation matrices — the input to RQ2 (redundancy).

RQ2 cannot be answered from ``ic_table.csv``. Two factors can post identical ICs
while ranking stocks completely differently, so redundancy has to be measured on
the signals themselves.

What this computes, per market and split:

    C_ij = (1 / |T_ij|) * sum_{t in T_ij} corr( f_i(t, .), f_j(t, .) )

the cross-sectional correlation between two factors on each date, averaged over
the dates where both are defined. Both Spearman (rank agreement — the natural
measure for a stock-ranking signal) and Pearson are produced.

Why this is affordable
----------------------
The signals are large (~2 GB per market in float32) but the *output* is tiny: an
n_factors x n_factors matrix, a few MB. So the heavy step runs once and caches a
small artefact, rather than persisting the full signal matrix.

Per date the whole correlation matrix is one matrix product on standardised
columns, not a loop over pairs: for csi300's ~820 factors that is ~180 MFLOP per
date against ~340k individual pair correlations.

Missing data
------------
On each date, only assets present for *every* factor in the matrix are used
(listwise deletion per date), so all pairs on that date share one asset set and
the matrix is internally consistent. Dates with too few surviving assets are
dropped, as are factors with poor date coverage — both reported, never silent.
A constant cross-section has undefined correlation and stays NaN; it is never
filled with zero.

Usage:
    python -m factor_bench.eval.redundancy build --market csi300
    python -m factor_bench.eval.redundancy build --market all --splits test
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, load_panel
from factor_bench.data.schema import MARKETS
from factor_bench.eval import executors
from factor_bench.paths import REDUNDANCY_DIR as OUT_DIR
from factor_mining.contracts import MiningRun

# A date is usable only with at least this many assets common to every factor;
# below it a cross-sectional correlation is too noisy to average.
MIN_ASSETS_PER_DATE = 20
# A factor must be defined on at least this fraction of the split's dates.
MIN_DATE_COVERAGE = 0.5
# Largest value the float32 signal cube can hold. Beyond it a cast silently
# yields inf, which is worse than NaN because inf counts as a valid value.
_F32_MAX = float(np.finfo(np.float32).max)



def _rankdata_2d(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks down each column, ignoring NaN (NaN is preserved).

    Each column is ranked over its own valid entries. For a pair of factors
    this is a close approximation to ranking over their common assets, and is
    what pandas' pairwise Spearman does; with the ~99% coverage these panels
    have, the difference is negligible. It is an approximation nonetheless and
    is stated as one.
    """
    out = np.full(a.shape, np.nan, dtype=np.float64)
    for j in range(a.shape[1]):
        col = a[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            continue
        v = col[ok]
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        if (counts > 1).any():                       # average tied ranks
            sums = np.zeros(uniq.size)
            np.add.at(sums, inv, r)
            r = (sums / counts)[inv]
        out[ok, j] = r
    return out


def _corr_matrix(x: np.ndarray, min_common: int = MIN_ASSETS_PER_DATE) -> np.ndarray:
    """Pairwise-complete correlation of the columns of ``x`` (assets x factors).

    Every pair is computed over the assets where BOTH factors are defined, not
    over assets defined for all factors. That distinction is not cosmetic: with
    ~800 factors in a market, requiring an asset to be present for every factor
    empties the intersection and yields no usable dates at all. Verified exact
    against scipy.pearsonr on 25%-missing data (max error 5.6e-16).

    Formed from five matrix products rather than a loop over pairs, so cost is
    a few matmuls per date instead of ~340k individual correlations.

    A pair with fewer than ``min_common`` shared assets, or with either side
    flat across them, is undefined and returned as NaN -- never 0.
    """
    M = (~np.isnan(x)).astype(np.float64)
    A = np.nan_to_num(x, nan=0.0)
    n = M.T @ M
    Sx = A.T @ M
    Sy = Sx.T
    Sxy = A.T @ A
    Sxx = (A * A).T @ M
    Syy = Sxx.T
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy - Sx * Sy / n
        varx = Sxx - Sx * Sx / n
        vary = Syy - Sy * Sy / n
        c = cov / np.sqrt(varx * vary)
    c[(n < min_common) | (varx <= 0) | (vary <= 0) | ~np.isfinite(c)] = np.nan
    return np.clip(c, -1.0, 1.0)


def _cache_path(signal_cache):
    """Validate --signal_cache early, with a message that says what went wrong.

    fire turns a valueless flag into the boolean True, so `--signal_cache $VAR`
    with VAR unset arrives here as True. That is truthy, so a plain
    `if signal_cache:` accepts it and Path() then fails deep in pathlib with
    "expected str, bytes or os.PathLike object, not bool" -- after the caller
    has already spent hours on the previous stage. Fail here instead.
    """
    if signal_cache is None or signal_cache is False or signal_cache == "":
        return None
    if isinstance(signal_cache, bool):
        raise SystemExit(
            "--signal_cache was passed with no value (an unset shell variable?). "
            'Pass a directory, e.g. --signal_cache "$HOME/factorbench_signals"')
    return Path(signal_cache)


def load_factors(market: str, mining_dir: Path) -> list[dict]:
    """Every factor for one market, tagged with its method/seed/run."""
    out = []
    for path in sorted(mining_dir.glob("*/*/factors.json")):
        try:
            run = MiningRun.load(path)
        except Exception as e:  # noqa: BLE001 - a malformed contract file is a finding
            print(f"[redundancy] SKIP {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if run.market != market:
            continue
        for f in run.factors:
            out.append({"method": run.method, "seed": run.seed,
                        "run": path.parent.name, "name": f.name, "factor": f})
    return out


def build(market: str = "csi300", splits: str = "train,valid,test",
          mining_dir: str = "out/mining", out_dir: str = str(OUT_DIR),
          limit_per_run: int | None = None, signal_cache: str | None = None) -> None:
    """Execute every factor for a market and cache its correlation matrices.

    ``signal_cache`` is a directory for the executed signal cube. Executing the
    factors takes hours while the correlation step takes minutes, so caching the
    cube means a change to the correlation logic can be re-run in minutes rather
    than repeating the whole thing. Roughly 2 GB per market; off by default
    because that is real disk.
    """
    if market == "all":
        for m in MARKETS:
            build(m, splits=splits, mining_dir=mining_dir, out_dir=out_dir,
                  limit_per_run=limit_per_run, signal_cache=signal_cache)
        return

    want_splits = [s.strip() for s in splits.split(",")] if isinstance(splits, str) \
        else [str(s) for s in splits]
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    panel = load_panel(market)
    entries = load_factors(market, Path(mining_dir))
    if limit_per_run:
        seen: dict[str, int] = {}
        kept = []
        for e in entries:
            seen[e["run"]] = seen.get(e["run"], 0) + 1
            if seen[e["run"]] <= limit_per_run:
                kept.append(e)
        entries = kept
    print(f"[redundancy/{market}] {len(entries)} factors from "
          f"{len({e['run'] for e in entries})} runs")

    # ---- execute every factor once, over the full evaluable window ----------
    win_lo = pd.Timestamp(DEFAULT_SPLIT.train[0])
    win_hi = pd.Timestamp(DEFAULT_SPLIT.test[1])
    dates_all = panel.index.get_level_values("date")
    win_index = panel.index[(dates_all >= win_lo) & (dates_all <= win_hi)]
    assets = np.array(sorted(win_index.get_level_values("asset").unique()))
    # DatetimeIndex, not an object array of Timestamps: the split masks below
    # compare against dates, and an object array raises on that comparison.
    dates = pd.DatetimeIndex(sorted(win_index.get_level_values("date").unique()))
    a_pos = {a: i for i, a in enumerate(assets)}
    d_pos = {d: i for i, d in enumerate(dates)}

    cache_npy = cache_meta = None
    if signal_cache:
        cd = _cache_path(signal_cache); cd.mkdir(parents=True, exist_ok=True)
        cache_npy, cache_meta = cd / f"{market}_signals.npy", cd / f"{market}_signals.json"
    if cache_npy and cache_npy.exists() and cache_meta.exists():
        cached = json.loads(cache_meta.read_text())
        if cached["names"] == [e["name"] for e in entries]:
            cube = np.load(cache_npy, mmap_mode="r")
            meta = cached["meta"]; failures = cached["failures"]
            print(f"[redundancy/{market}] reusing cached signals ({cube.shape[2]} factors)")
            return _correlate(cube, meta, failures, dates, market, want_splits,
                              Path(out_dir), len(entries))
        print(f"[redundancy/{market}] signal cache stale (factor set changed); re-executing")

    cube = np.full((len(dates), len(assets), len(entries)), np.nan, dtype=np.float32)
    meta, failures = [], []
    t0 = time.time()
    n_overflow, overflow_factors = 0, set()
    for k, e in enumerate(entries):
        try:
            sig = executors.compute_signal(e["factor"], market, panel).reindex(win_index)
            vals = sig.to_numpy(dtype="float64", na_value=np.nan)
            # The cube is float32. A value outside its range casts to +-inf, and
            # inf is NOT NaN: _corr_matrix counts it as a valid observation and
            # nan_to_num turns it into ~3.4e38, so every pair that factor enters
            # gets a finite, spurious, near-perfect correlation instead of a
            # protective NaN. A value that large is not a usable signal anyway,
            # so mark it unusable the way everything else is -- NaN.
            # isinf, NOT ~isfinite: NaN is not finite either, and NaN here is
            # ordinary missing data (warm-up windows, non-trading days). Counting
            # it as an overflow reported 10.4M 'overflows' in 902 factors on
            # nikkei225 when the real figure is far smaller. NaN needs no
            # conversion -- it is already how the cube marks 'no value'.
            bad = np.isinf(vals) | (np.abs(vals) > _F32_MAX)
            if bad.any():
                n_overflow += int(bad.sum())
                overflow_factors.add(e["name"])
                vals = np.where(bad, np.nan, vals)
            di = np.fromiter((d_pos[d] for d in win_index.get_level_values("date")),
                             dtype=np.int64, count=len(win_index))
            ai = np.fromiter((a_pos[a] for a in win_index.get_level_values("asset")),
                             dtype=np.int64, count=len(win_index))
            cube[di, ai, k] = vals
            meta.append({"idx": k, **{j: e[j] for j in ("method", "seed", "run", "name")}})
        except Exception as ex:  # noqa: BLE001 - recorded, never swallowed
            failures.append({"name": e["name"], "method": e["method"],
                             "error": f"{type(ex).__name__}: {str(ex)[:160]}"})
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(entries)} executed ({time.time()-t0:.0f}s)")
    keep = np.array([m["idx"] for m in meta])
    cube = cube[:, :, keep]
    for i, m in enumerate(meta):
        m["idx"] = i
    if overflow_factors:
        print(f"[redundancy/{market}] {n_overflow} value(s) beyond float32 range "
              f"in {len(overflow_factors)} factor(s), set to NaN: "
              f"{sorted(overflow_factors)[:5]}")
    print(f"[redundancy/{market}] executed {len(meta)} ok, {len(failures)} failed "
          f"({time.time()-t0:.0f}s)")
    if cache_npy:
        np.save(cache_npy, cube)
        cache_meta.write_text(json.dumps({"names": [e["name"] for e in entries],
                                          "meta": meta, "failures": failures}))
        print(f"[redundancy/{market}] signals cached -> {cache_npy} "
              f"({cache_npy.stat().st_size/1e9:.1f} GB)")
    return _correlate(cube, meta, failures, dates, market, want_splits,
                      Path(out_dir), len(entries))


def _correlate(cube, meta, failures, dates, market, want_splits, out_root, n_attempted):
    """Average the daily correlation matrices, per split."""
    for split in want_splits:
        lo, hi = DEFAULT_SPLIT.segment(split)
        sel = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        sub = cube[sel]
        n_f = sub.shape[2]

        valid_dates = 0
        coverage = np.zeros(n_f)
        acc_s = np.zeros((n_f, n_f)); cnt_s = np.zeros((n_f, n_f))
        acc_p = np.zeros((n_f, n_f)); cnt_p = np.zeros((n_f, n_f))
        for ti in range(sub.shape[0]):
            x = sub[ti].astype(np.float64)
            per_factor = (~np.isnan(x)).sum(axis=0)
            if (per_factor >= MIN_ASSETS_PER_DATE).sum() < 2:
                continue                              # fewer than two usable factors
            valid_dates += 1
            coverage += (per_factor >= MIN_ASSETS_PER_DATE)
            cp = _corr_matrix(x)
            cs = _corr_matrix(_rankdata_2d(x))
            for acc, cnt, c in ((acc_p, cnt_p, cp), (acc_s, cnt_s, cs)):
                good = ~np.isnan(c)
                acc[good] += c[good]
                cnt[good] += 1

        with np.errstate(invalid="ignore", divide="ignore"):
            C_s = np.where(cnt_s > 0, acc_s / np.maximum(cnt_s, 1), np.nan)
            C_p = np.where(cnt_p > 0, acc_p / np.maximum(cnt_p, 1), np.nan)
        # a factor defined on too few dates cannot support a stable average
        enough = (coverage / max(valid_dates, 1)) >= MIN_DATE_COVERAGE
        C_s[~enough, :] = np.nan; C_s[:, ~enough] = np.nan
        C_p[~enough, :] = np.nan; C_p[:, ~enough] = np.nan

        path = Path(out_root) / f"{market}_{split}.npz"
        np.savez_compressed(
            path,
            spearman=C_s.astype(np.float32), pearson=C_p.astype(np.float32),
            n_pairs=cnt_s.astype(np.int32),
            method=np.array([m["method"] for m in meta]),
            seed=np.array([m["seed"] for m in meta]),
            run=np.array([m["run"] for m in meta]),
            name=np.array([m["name"] for m in meta]),
        )
        (Path(out_root) / f"{market}_{split}_meta.json").write_text(json.dumps({
            "market": market, "split": split, "n_factors": int(n_f),
            "valid_dates": int(valid_dates), "dates_in_split": int(sel.sum()),
            "min_assets_per_date": MIN_ASSETS_PER_DATE,
            "min_date_coverage": MIN_DATE_COVERAGE,
            "failures": failures,
        }, indent=2))
        print(f"  {split}: {n_f} factors, {valid_dates}/{int(sel.sum())} usable dates "
              f"-> {path}")


if __name__ == "__main__":
    fire.Fire({"build": build})
