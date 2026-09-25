"""Driver: selection -> combination -> backtest, cached as small result files.

Reads the signal cube written by ``factor_bench.eval.redundancy --signal_cache``
so nothing is re-executed, plus ``ic_table.csv`` for validation IC and the
validation-split correlation matrices for the redundancy filter.

Everything that could leak is frozen before test:

- selection ranks on **validation** IC and vets correlations on the
  **validation** split;
- ridge coefficients are fitted on train with the penalty chosen on validation;
- the backtest then runs on test with those choices held fixed.

Writes one tidy CSV of daily returns per configuration plus a summary row, so
the notebooks plot and tabulate without recomputing anything.

Usage:
    python -m factor_bench.portfolio.run_backtests build --market all \
        --signal_cache /path/to/factorbench_signals
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_bench.data.schema import MARKETS
from factor_bench.eval.metrics import newey_west_tstat
from factor_bench.eval.style import MIN_OBS_PER_DATE, build_style_panel
from factor_bench.paths import REDUNDANCY_DIR as _RED_DIR
from factor_bench.paths import PORTFOLIO_DIR as OUT_DIR
from factor_bench.paths import IC_TABLE as _IC_TABLE
from factor_bench.portfolio.backtest import BacktestConfig, backtest, performance_metrics
from factor_bench.portfolio.combine import CombineConfig, build_composite
from factor_bench.portfolio.selection import SelectionConfig, select_factors

COST_GRID = (0.0, 5.0, 10.0, 20.0, 50.0)
# Daily, weekly, fortnightly, three-weekly, monthly. 10 and 15 fill the gap
# between 5 and 20, which is where turnover and therefore cost drag change
# fastest -- with only {1, 5, 20} the middle of that curve was interpolated.
REBAL_GRID = (1, 5, 10, 15, 20)
# The dense arm of the grid, and the only interval whose curves are exported.
# Must match `MAIN_REBAL` in notebooks/build_rq4_notebooks.py: the pruning
# below keeps the full COST sweep only at this interval, so a notebook that
# reports a different one gets a single-point cost figure and no curves.
MAIN_REBAL = 20
MAIN_COST = 10.0
COMBOS = ("equal", "ic", "ridge")
# top5 was dropped: it answers the same question as top10 -- does the ranking
# survive equal-sized books -- and for small-pool methods (alphaagent selects
# ~4-6) it is barely distinguishable from `full`, so it mostly duplicated top10
# on the large-pool methods. Restore it here if a tighter book is wanted.
CAPS = {"full": None, "top10": 10}
# Step 3 of selection, run BOTH ways in one pass so the gate's effect is a
# column rather than a second full rebuild. "off" is the unfiltered pool (the
# default); "on" keeps only factors with positive validation IC. The gate is
# worth this: without it the equal-weight composite's mean test rank-IC falls
# from 0.0222 to 0.0010 and six of nine mining methods turn negative.
GATES = {"off": False, "on": True}


# Lags for the HAC correction on the paired daily IC differences: set by the
# 20-day overlap, the same choice as ic_table and style.
HAC_LAGS = 25
# A composite needs at least this many style-eligible dates before its
# neutralised IC is worth reporting.
MIN_STYLE_DATES = 100
# Above this share of variance explained, the residual is numerical noise
# rather than a signal, and is not traded. Not 1.0: lstsq leaves ~1e-16 dust,
# and a composite at R^2 = 0.9995 has nothing meaningful left either.
RESIDUAL_R2_CEILING = 0.999
# Run the style-neutralised composite across the full cost/rebalance grid too,
# not just the main cell. It roughly doubles the backtest work (~3.5h -> ~7h on
# five markets), and it is worth it: residualisation ADDS turnover, because
# style loadings move daily, so the cost sensitivity of the residual is a
# different curve from the raw composite's -- and claiming a neutralised signal
# is tradeable without showing its cost curve would be the weaker result.
RESIDUAL_FULL_GRID = True


def style_residual_composite(comp: pd.Series, styles: pd.DataFrame,
                             dates: pd.DatetimeIndex) -> pd.Series:
    """The composite with its five style exposures projected out, per date.

    The same decomposition RQ3 applies to individual factors, applied here to
    the composite a method would actually trade -- which RQ3 structurally cannot
    reach, because selection deliberately picks factors that are mutually
    uncorrelated, so the composite is not a typical member of the pool. Style
    exposures can cancel between its constituents, leaving it cleaner than any
    single factor, or accumulate, leaving it dominated by a style none of them
    showed alone.

    Regressed WITH an intercept, so R^2 is measured against the cross-sectional
    mean rather than against zero. The target is never touched: residualising
    returns as well would answer a different, partial-correlation question.

    Returns a Series on the same index, NaN on dates that could not be fitted.
    """
    out = pd.Series(np.nan, index=comp.index, dtype=float)
    S = styles.reindex(comp.index)
    usable = S.notna().all(axis=1) & comp.notna()
    for d in dates:
        try:
            keep = usable.xs(d, level="date")
        except KeyError:
            continue
        idx = keep[keep].index
        if len(idx) < MIN_OBS_PER_DATE:
            continue
        f = comp.xs(d, level="date").reindex(idx).to_numpy(dtype=float)
        Sm = S.xs(d, level="date").reindex(idx).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(idx)), Sm])
        coef, *_ = np.linalg.lstsq(X, f, rcond=None)
        out.loc[[(d, a) for a in idx]] = f - X @ coef
    return out


def _daily_ic(sig: pd.Series, target: pd.Series, dates: pd.DatetimeIndex):
    """Per-date cross-sectional Pearson and Spearman IC of one signal.

    Returned DATE-INDEXED, not as bare arrays. Dates drop out of this loop for
    reasons that differ between two signals (a flat cross-section above all),
    so anything that differences two of these must align on the index; a
    positional subtraction would silently compare different days.
    """
    ic, ric, keep = [], [], []
    for d in dates:
        try:
            s = sig.xs(d, level="date")
            y = target.xs(d, level="date").reindex(s.index)
        except KeyError:
            continue
        ok = s.notna() & y.notna()
        if int(ok.sum()) < MIN_OBS_PER_DATE:
            continue
        a, b = s[ok].to_numpy(float), y[ok].to_numpy(float)
        if a.std() <= 0 or b.std() <= 0:          # flat: no ranking, IC undefined
            continue
        ic.append(float(np.corrcoef(a, b)[0, 1]))
        ar = pd.Series(a).rank().to_numpy()
        br = pd.Series(b).rank().to_numpy()
        ric.append(float(np.corrcoef(ar, br)[0, 1]))
        keep.append(d)
    idx = pd.DatetimeIndex(keep, name="date")
    return (pd.Series(ic, index=idx, dtype=float),
            pd.Series(ric, index=idx, dtype=float))


def _summarise(ic, ric, tag: str) -> dict:
    """mean IC, mean RankIC and ICIR for one signal.

    ICIR is mean(daily IC) / std(daily IC): steadiness, not strength. It is
    descriptive only -- daily ICs from a 20-day target are heavily
    autocorrelated, so significance comes from the HAC t-statistic instead.
    """
    if ic.size == 0:
        return {f"{tag}_ic": np.nan, f"{tag}_rank_ic": np.nan, f"{tag}_icir": np.nan}
    # ddof=0 explicitly: these used to be numpy arrays, whose .std() is the
    # population sd, while pandas defaults to the sample sd. Stated rather than
    # inherited, so ICIR does not move when the container type changes.
    sd = float(ic.std(ddof=0))
    return {
        f"{tag}_ic": float(ic.mean()),
        f"{tag}_rank_ic": float(ric.mean()) if ric.size else np.nan,
        f"{tag}_icir": float(ic.mean() / sd) if sd > 0 else np.nan,
    }


def composite_style_metrics(comp: pd.Series, resid: pd.Series, target: pd.Series,
                            styles: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Raw vs style-neutralised composite: IC, RankIC, ICIR and the paired drop.

    Both sides are computed on the SAME dates, so the comparison is matched:
    style availability changes which stocks are eligible, and scoring the raw
    composite on its full sample would flatter it.

    The uncertainty in ``delta_ic`` comes from the daily series of
    (raw - residual), never from subtracting two t-statistics -- with a 20-day
    target sampled daily the dependence is severe.
    """
    common = comp.notna() & resid.notna()
    c, r = comp[common], resid[common]

    ic_raw, ric_raw = _daily_ic(c, target, dates)
    ic_res, ric_res = _daily_ic(r, target, dates)

    # The two series iterate the same dates, but either can drop one the other
    # keeps: the residual goes flat on a date where the styles explain the
    # composite almost exactly, while the raw composite still varies. Taking
    # the intersection makes "the same dates" true rather than assumed -- and
    # every number below is then computed on it, so the comparison is matched
    # in dates as well as in stocks.
    m = ic_raw.index.intersection(ic_res.index)
    out = {"comp_style_dates": int(len(m))}
    out.update(_summarise(ic_raw.loc[m], ric_raw.loc[m], "comp_raw"))
    out.update(_summarise(ic_res.loc[m], ric_res.loc[m], "comp_res"))

    if len(m) >= MIN_STYLE_DATES:
        d = ic_raw.loc[m] - ic_res.loc[m]      # aligned on date by construction
        out["comp_delta_ic"] = float(d.mean())
        out["comp_t_hac_delta"] = newey_west_tstat(d, HAC_LAGS)
        denom = out["comp_raw_ic"]
        out["comp_ic_retained"] = (float(out["comp_res_ic"] / denom)
                                   if denom and np.isfinite(denom) else np.nan)
    else:
        out.update({"comp_delta_ic": np.nan, "comp_t_hac_delta": np.nan,
                    "comp_ic_retained": np.nan})

    # How much of the composite the five styles explain, averaged over dates.
    r2 = []
    S = styles.reindex(comp.index)
    usable = S.notna().all(axis=1) & comp.notna()
    for d in dates:
        try:
            keep = usable.xs(d, level="date")
        except KeyError:
            continue
        idx = keep[keep].index
        if len(idx) < MIN_OBS_PER_DATE:
            continue
        f = comp.xs(d, level="date").reindex(idx).to_numpy(float)
        e = resid.xs(d, level="date").reindex(idx).to_numpy(float)
        ss_tot = float(((f - f.mean()) ** 2).sum())
        if ss_tot > 0 and np.isfinite(e).all():
            r2.append(1.0 - float((e ** 2).sum()) / ss_tot)
    out["comp_r2"] = float(np.mean(r2)) if r2 else np.nan
    return out

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


def _load_cache(signal_cache: Path, market: str):
    npy, meta = signal_cache / f"{market}_signals.npy", signal_cache / f"{market}_signals.json"
    if not (npy.exists() and meta.exists()):
        raise SystemExit(f"no signal cache for {market} at {signal_cache}; run "
                         "factor_bench.eval.redundancy build --signal_cache first")
    info = json.loads(meta.read_text())
    return np.load(npy, mmap_mode="r"), pd.DataFrame(info["meta"])


def _corr_lookup(market: str, red_dir: Path):
    f = red_dir / f"{market}_valid.npz"
    if not f.exists():
        return None, None
    z = np.load(f, allow_pickle=False)
    return z["spearman"].astype(float), {n: i for i, n in enumerate(z["name"])}


def build(market: str = "csi300", signal_cache: str = "", mining_dir: str = "out/mining",
          ic_table: str = str(_IC_TABLE), red_dir: str = str(_RED_DIR),
          out_dir: str = str(OUT_DIR), strategies: str = "long_only,long_short") -> None:
    """Backtest every method x market x run x combination x strategy."""
    if market == "all":
        for m in MARKETS:
            build(m, signal_cache=signal_cache, mining_dir=mining_dir, ic_table=ic_table,
                  red_dir=red_dir, out_dir=out_dir, strategies=strategies)
        return
    kinds = [s.strip() for s in strategies.split(",")] if isinstance(strategies, str) \
        else list(strategies)
    out_root = Path(out_dir); out_root.mkdir(parents=True, exist_ok=True)

    panel = load_panel(market)
    target = TARGET.from_panel(panel)
    ret_wide = panel["return_1d"].unstack("asset").sort_index()

    cube, meta = _load_cache(_cache_path(signal_cache), market)
    corr, cix = _corr_lookup(market, Path(red_dir))
    ict = pd.read_csv(ic_table)
    ict = ict[ict["market"] == market]

    win_lo, win_hi = pd.Timestamp(DEFAULT_SPLIT.train[0]), pd.Timestamp(DEFAULT_SPLIT.test[1])
    d_all = panel.index.get_level_values("date")
    win_index = panel.index[(d_all >= win_lo) & (d_all <= win_hi)]
    assets = np.array(sorted(win_index.get_level_values("asset").unique()))
    dates = pd.DatetimeIndex(sorted(win_index.get_level_values("date").unique()))
    full_ix = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

    def signal_of(j: int) -> pd.Series:
        return pd.Series(np.asarray(cube[:, :, j]).ravel(), index=full_ix)

    tr = pd.Series(full_ix.get_level_values("date").isin(
        dates[(dates >= pd.Timestamp(DEFAULT_SPLIT.train[0])) &
              (dates <= pd.Timestamp(DEFAULT_SPLIT.train[1]))]), index=full_ix)
    va = pd.Series(full_ix.get_level_values("date").isin(
        dates[(dates >= pd.Timestamp(DEFAULT_SPLIT.valid[0])) &
              (dates <= pd.Timestamp(DEFAULT_SPLIT.valid[1]))]), index=full_ix)
    test_dates = dates[(dates >= pd.Timestamp(DEFAULT_SPLIT.test[0])) &
                       (dates <= pd.Timestamp(DEFAULT_SPLIT.test[1]))]
    ret_test = ret_wide.reindex(index=test_dates, columns=assets).fillna(0.0)
    bench = ret_test.mean(axis=1)            # equal-weight universe benchmark

    # One style panel per market, identical for every method and composite --
    # the same panel RQ3 uses, so the two are directly comparable.
    styles = build_style_panel(panel)

    name_to_j = {n: i for i, n in enumerate(meta["name"])}
    summaries, curves = [], []

    for (method, run), grp in meta.groupby(["method", "run"], sort=False):
        cand = ict[ict["factor"].isin(grp["name"])]
        if cand.empty:
            continue
        seed = int(grp["seed"].iloc[0])
        for cap_name, cap in CAPS.items():
          for gate_name, gate in GATES.items():
              sel = select_factors(cand, corr, cix,
                                   SelectionConfig(cap=cap, max_correlation=0.8,
                                                   require_positive_ic=gate))
              n_sel = len(sel)
              if n_sel == 0:
                  summaries.append({"method": method, "market": market, "seed": seed,
                                    "run": run, "cap": cap_name,
                                    "positive_ic_gate": gate_name, "combo": None,
                                    "strategy": None, "n_selected": 0,
                                    "status": "no qualifying composite"})
                  continue
              sigs = {n: signal_of(name_to_j[n]) for n in sel["factor"] if n in name_to_j}
              for combo in COMBOS:
                  comp, w = build_composite(sigs, sel, CombineConfig(method=combo),
                                            target=target.reindex(full_ix),
                                            train_mask=tr, valid_mask=va)
                  if comp.empty:
                      continue
                  comp_test = comp[comp.index.get_level_values("date").isin(test_dates)]

                  # Style neutralisation of the TRADED composite, once per
                  # composite: it depends on neither rebalance interval nor cost.
                  # Both the metrics and a tradeable residual signal come out,
                  # so RQ4 can report what survives neutralisation *and* whether
                  # what survives is still worth trading -- different questions.
                  resid_test = style_residual_composite(comp_test, styles, test_dates)
                  decay = composite_style_metrics(comp_test, resid_test, target,
                                                  styles, test_dates)

                  variants = {"raw": comp_test}
                  # A composite the styles explain almost perfectly leaves a
                  # residual that is float noise, not a signal. Its ranking is
                  # arbitrary, but it would still fill the quantiles and be
                  # backtested as though real -- so it is excluded, and the
                  # exclusion is visible as a missing style_neutral row rather
                  # than as a plausible-looking number.
                  r2 = decay.get("comp_r2", np.nan)
                  noise = np.isfinite(r2) and r2 >= RESIDUAL_R2_CEILING
                  if resid_test.notna().any() and not noise:
                      variants["style_neutral"] = resid_test

                  for signal_name, signal in variants.items():
                      for kind in kinds:
                          for rebal in REBAL_GRID:
                              for cost in COST_GRID:
                                  if rebal != MAIN_REBAL and cost != MAIN_COST:
                                      continue      # grid: vary one axis at a time
                                  if (not RESIDUAL_FULL_GRID and signal_name != "raw"
                                          and (rebal != MAIN_REBAL or cost != MAIN_COST)):
                                      continue
                                  cfg = BacktestConfig(rebalance_days=rebal, cost_bps=cost)
                                  daily = backtest(signal, ret_test, cfg, kind)
                                  if daily.empty:
                                      continue
                                  m = performance_metrics(
                                      daily, bench if kind == "long_only" else None, rebal)
                                  # A composite that never holds a position is an
                                  # ABSENT result, not a 0% one. Recording it as
                                  # "ok" with zeros drags the method's mean toward
                                  # zero as though the strategy had been run and
                                  # made nothing. It happens when selection yields
                                  # a single factor too thin to fill the quantiles.
                                  degenerate = float(m.get("avg_gross_exposure", 0.0)) <= 0.0
                                  summaries.append({
                                      "method": method, "market": market, "seed": seed,
                                      "run": run, "cap": cap_name,
                                      "positive_ic_gate": gate_name, "combo": combo,
                                      "signal": signal_name,
                                      "strategy": kind, "rebalance": rebal, "cost_bps": cost,
                                      "n_selected": n_sel,
                                      "status": ("degenerate composite: no positions held"
                                                 if degenerate else "ok"),
                                      "neg_weights": int((w < 0).sum()), **decay, **m})
                                  if (rebal == MAIN_REBAL and cost == MAIN_COST
                                          and not degenerate):
                                      c = daily[["gross", "net"]].copy()
                                      c["method"], c["market"], c["seed"] = method, market, seed
                                      c["cap"], c["combo"], c["strategy"] = cap_name, combo, kind
                                      c["positive_ic_gate"] = gate_name
                                      c["signal"] = signal_name
                                      curves.append(c.reset_index())

    pd.DataFrame(summaries).to_csv(out_root / f"{market}_summary.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            out_root / f"{market}_curves.csv", index=False)
    bench.rename("bench").to_frame().to_csv(out_root / f"{market}_benchmark.csv")
    print(f"[portfolio/{market}] {len(summaries)} rows -> {out_root}/{market}_summary.csv")


if __name__ == "__main__":
    fire.Fire({"build": build})
