"""Style neutralisation — the input to the style-repackaging question (RQ3).

Asks whether a mined factor's predictive power is incremental alpha or a
repackaging of well-known price/volume premia.

Method
------
1. Build ONE style panel per market, identical for every method and factor.
2. Per factor per date, regress the factor cross-sectionally on the styles with
   an intercept, and keep the residual.
3. Recompute raw and residual IC **on exactly the same observations**, against
   the unchanged 20-day forward-return target.

Degenerate vs flat
------------------
``degenerate_rate`` is the share of dates where the styles removed essentially
all of a factor's cross-sectional spread -- the strong form of style overlap.
``flat_rate`` is the share where the factor had no spread to begin with, which
is a broken factor and a data-quality finding, not an RQ3 result. Conflating the
two (as an absolute residual threshold does) reads degenerate factors as
evidence of style repackaging.

Three things that make or break the comparison
----------------------------------------------
- **Matched samples.** Residual IC must never be compared against the IC in
  ``ic_table.csv``: style availability changes which stocks are eligible, so the
  raw baseline is recomputed on the same rows the residual uses.
- **The target is untouched.** Residualising the factor alone asks "does what is
  left still predict returns?". Residualising returns as well would answer a
  different, partial-correlation question.
- **HAC on the paired daily differences.** The uncertainty in ``delta IC`` comes
  from the daily series of (raw - residual), not from subtracting two
  t-statistics. With a 20-day target sampled daily the dependence is severe.

Why this is affordable
----------------------
On a given date the style matrix is the same for every factor, so the projection
is formed once and applied to all factors at once. That turns ~1.8M single
regressions into a few hundred matrix operations.

Usage:
    python -m factor_bench.eval.style build --market all \
        --signal_cache /path/to/factorbench_signals
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_bench.data.schema import MARKETS
from factor_bench.eval import executors
from factor_bench.eval.metrics import newey_west_tstat
from factor_bench.paths import STYLE_DIR as OUT_DIR
from factor_mining.contracts import MiningRun

STYLES = ["beta", "volatility", "momentum", "reversal", "liquidity"]
WINSOR = (0.01, 0.99)
MIN_OBS_PER_DATE = 30          # eligible stocks needed to fit the regression
HAC_LAGS = 25                  # set by the 20-day overlap, not a bandwidth rule
# A residual counts as degenerate when it has no cross-sectional spread LEFT
# RELATIVE TO the factor it came from. The threshold has to be relative: an
# absolute one (the previous 1e-12) flags any factor whose values happen to be
# tiny in magnitude, whatever the styles did to it -- 62 of the 121 factors it
# flagged had a mean R^2 of only 0.07, i.e. the styles explained almost nothing
# and the factor was simply small.
DEGENERATE_REL = 1e-9
# Largest value the float32 signal cube can hold; see redundancy.py.
_F32_MAX = float(np.finfo(np.float32).max)
# A factor that is itself constant across stocks on a date has nothing to
# explain. That is a broken factor, NOT evidence of style overlap, and it is
# counted separately as flat_rate. This test is relative to the factor's own
# magnitude, not absolute: a factor whose values are all near 1e-13 but genuinely
# differ carries a perfectly good ranking, while one whose spread is twelve
# orders below its own level is flat to float64 and its ranking is noise.
FLAT_REL = 1e-12



# ---------------------------------------------------------------- style panel
def build_style_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Five standardised style exposures per (date, asset).

    Every quantity uses only information available at signal formation on date
    ``t``. Rolling windows end at ``t`` inclusive, which is the same information
    set the factor itself sees; nothing is shifted forward.

    Definitions are fixed in advance and applied identically to every method:

    - **beta**       trailing 252d cov(r_i, r_mkt) / var(r_mkt), equal-weight market
    - **volatility** trailing 60d standard deviation of daily returns
    - **momentum**   cumulative return t-252 -> t-21 (skips the most recent month)
    - **reversal**   negative cumulative return over the trailing 21 days
    - **liquidity**  log of trailing 20d mean dollar volume (close x volume)

    Note on liquidity: dollar volume is computed here as ``close * volume``. The
    cached ``adv`` column is average *share* volume, so ``log(adv)`` would be a
    different specification.
    """
    close = panel["close"].unstack("asset").sort_index()
    volume = panel["volume"].unstack("asset").sort_index()
    ret = panel["return_1d"].unstack("asset").sort_index()

    mkt = ret.mean(axis=1)                                   # equal-weight market
    cov = ret.rolling(252, min_periods=200).cov(mkt)
    var = mkt.rolling(252, min_periods=200).var()
    beta = cov.div(var, axis=0)

    vol = ret.rolling(60, min_periods=45).std()
    mom = close.shift(21) / close.shift(252) - 1.0
    rev = -(close / close.shift(21) - 1.0)

    dollar = close * volume
    # Non-positive dollar volume is a non-trading day, not a small number: log()
    # is undefined there and filling it would fabricate a liquidity level.
    dollar = dollar.where(dollar > 0)
    liq = np.log(dollar.rolling(20, min_periods=15).mean())

    raw = {"beta": beta, "volatility": vol, "momentum": mom,
           "reversal": rev, "liquidity": liq}
    out = {}
    for name, wide in raw.items():
        # Plain .stack(): pandas 3 removed the dropna argument, and it is not
        # needed either way -- the reindex onto the panel restores any dropped
        # (date, asset) pair as NaN, so both pandas versions agree.
        s = wide.stack()
        s.index = s.index.rename(["date", "asset"])
        out[name] = s.reindex(panel.index)
    styles = pd.DataFrame(out)

    # Per date: winsorise to the stated percentiles, then standardise. Both are
    # benchmark choices fixed beforehand, not knobs tuned on results.
    def _prep(col: pd.Series) -> pd.Series:
        g = col.groupby(level="date")
        lo = g.transform(lambda x: x.quantile(WINSOR[0]))
        hi = g.transform(lambda x: x.quantile(WINSOR[1]))
        clipped = col.clip(lo, hi)
        g2 = clipped.groupby(level="date")
        mu, sd = g2.transform("mean"), g2.transform("std")
        return (clipped - mu) / sd.where(sd > 0)

    return styles.apply(_prep)


# ---------------------------------------------------------------- regression
def _residualise(F: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regress every column of ``F`` on ``S`` (intercept included by caller).

    Returns (residuals, r2, adj_r2). The projection depends only on ``S``, which
    is shared by all factors on a date, so it is formed once here rather than
    per factor.

    The intercept is explicit: without it R^2 would be measured against zero
    rather than the cross-sectional mean, which changes its interpretation.
    """
    coef, *_ = np.linalg.lstsq(S, F, rcond=None)
    fitted = S @ coef
    resid = F - fitted
    ss_res = (resid * resid).sum(axis=0)
    centred = F - F.mean(axis=0, keepdims=True)
    ss_tot = (centred * centred).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    n, k = S.shape[0], S.shape[1] - 1          # k = regressors excluding intercept
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)
    bad = ~np.isfinite(ss_tot) | (ss_tot <= 0)
    r2[bad] = np.nan
    adj[bad] = np.nan
    return resid, r2, adj


def _xs_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation of each column of ``a`` with vector ``b`` (no NaNs)."""
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean()
    sa = np.sqrt((ac * ac).sum(axis=0))
    sb = np.sqrt((bc * bc).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (ac * bc[:, None]).sum(axis=0) / (sa * sb)
    c[(sa <= 0) | (sb <= 0)] = np.nan
    return c


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, axis=0, kind="mergesort")
    out = np.empty_like(order, dtype=np.float64)
    idx = np.arange(a.shape[0], dtype=np.float64)
    for j in range(a.shape[1]):
        out[order[:, j], j] = idx
    return out


# ---------------------------------------------------------------- driver
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


def _load_cube(market, entries, signal_cache, panel, win_index, dates, assets):
    """Signals for every factor: reuse the redundancy cache when it matches."""
    if signal_cache:
        cache = _cache_path(signal_cache)
        npy = cache / f"{market}_signals.npy"
        meta = cache / f"{market}_signals.json"
        if npy.exists() and meta.exists():
            cached = json.loads(meta.read_text())
            if cached["names"] == [e["name"] for e in entries]:
                print(f"[style/{market}] reusing cached signals from {npy}")
                return np.load(npy, mmap_mode="r"), cached["meta"], cached["failures"]
            print(f"[style/{market}] signal cache stale; re-executing")

    a_pos = {a: i for i, a in enumerate(assets)}
    d_pos = {d: i for i, d in enumerate(dates)}
    cube = np.full((len(dates), len(assets), len(entries)), np.nan, dtype=np.float32)
    meta, failures = [], []
    di = np.fromiter((d_pos[d] for d in win_index.get_level_values("date")),
                     dtype=np.int64, count=len(win_index))
    ai = np.fromiter((a_pos[a] for a in win_index.get_level_values("asset")),
                     dtype=np.int64, count=len(win_index))
    t0 = time.time()
    for k, e in enumerate(entries):
        try:
            sig = executors.compute_signal(e["factor"], market, panel).reindex(win_index)
            vals = sig.to_numpy(dtype="float64", na_value=np.nan)
            # float32 cube: a value beyond its range casts to +-inf, which is not
            # NaN and would be treated as a real observation downstream. Same
            # reasoning as factor_bench.eval.redundancy.
            cube[di, ai, k] = np.where(
                np.isinf(vals) | (np.abs(vals) > _F32_MAX), np.nan, vals)
            meta.append({"idx": k, **{j: e[j] for j in ("method", "seed", "run", "name")}})
        except Exception as ex:  # noqa: BLE001 - recorded, never swallowed
            failures.append({"name": e["name"], "method": e["method"],
                             "error": f"{type(ex).__name__}: {str(ex)[:160]}"})
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(entries)} executed ({time.time()-t0:.0f}s)")
    keep = np.array([m["idx"] for m in meta], dtype=int)
    cube = cube[:, :, keep]
    for i, m in enumerate(meta):
        m["idx"] = i
    return cube, meta, failures


def build(market: str = "csi300", splits: str = "train,valid,test",
          mining_dir: str = "out/mining", out_dir: str = str(OUT_DIR),
          signal_cache: str | None = None, limit_per_run: int | None = None) -> None:
    """Style-neutralise every factor for a market and write per-factor metrics."""
    if market == "all":
        for m in MARKETS:
            build(m, splits=splits, mining_dir=mining_dir, out_dir=out_dir,
                  signal_cache=signal_cache, limit_per_run=limit_per_run)
        return

    want = [s.strip() for s in splits.split(",")] if isinstance(splits, str) else list(splits)
    out_root = Path(out_dir); out_root.mkdir(parents=True, exist_ok=True)

    panel = load_panel(market)
    target_all = TARGET.from_panel(panel)
    entries = []
    for path in sorted(Path(mining_dir).glob("*/*/factors.json")):
        try:
            run = MiningRun.load(path)
        except Exception as e:  # noqa: BLE001
            print(f"[style] SKIP {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if run.market != market:
            continue
        fs = run.factors[:limit_per_run] if limit_per_run else run.factors
        for f in fs:
            entries.append({"method": run.method, "seed": run.seed,
                            "run": path.parent.name, "name": f.name, "factor": f})

    win_lo, win_hi = pd.Timestamp(DEFAULT_SPLIT.train[0]), pd.Timestamp(DEFAULT_SPLIT.test[1])
    d_all = panel.index.get_level_values("date")
    win_index = panel.index[(d_all >= win_lo) & (d_all <= win_hi)]
    assets = np.array(sorted(win_index.get_level_values("asset").unique()))
    dates = pd.DatetimeIndex(sorted(win_index.get_level_values("date").unique()))
    print(f"[style/{market}] {len(entries)} factors, {len(dates)} dates, {len(assets)} assets")

    styles = build_style_panel(panel)
    cov_report = styles.notna().all(axis=1).groupby(level="date").mean()
    print(f"[style/{market}] style coverage: {cov_report.mean():.1%} of rows have all "
          f"5 styles (first usable date {cov_report[cov_report>0].index.min().date()})")

    # style cube aligned to (date, asset)
    a_pos = {a: i for i, a in enumerate(assets)}
    d_pos = {d: i for i, d in enumerate(dates)}
    S_cube = np.full((len(dates), len(assets), len(STYLES)), np.nan, dtype=np.float32)
    sub = styles.reindex(win_index)
    di = np.fromiter((d_pos[d] for d in win_index.get_level_values("date")),
                     dtype=np.int64, count=len(win_index))
    ai = np.fromiter((a_pos[a] for a in win_index.get_level_values("asset")),
                     dtype=np.int64, count=len(win_index))
    for j, nm in enumerate(STYLES):
        S_cube[di, ai, j] = sub[nm].to_numpy(dtype="float64", na_value=np.nan)
    Y = np.full((len(dates), len(assets)), np.nan, dtype=np.float32)
    Y[di, ai] = target_all.reindex(win_index).to_numpy(dtype="float64", na_value=np.nan)

    cube, meta, failures = _load_cube(market, entries, signal_cache, panel,
                                      win_index, dates, assets)
    n_fac = cube.shape[2]
    print(f"[style/{market}] {n_fac} factors executed, {len(failures)} failed")

    # sign fixed on TRAIN, exactly as in ic_table: applied unchanged everywhere
    tr = (dates >= pd.Timestamp(DEFAULT_SPLIT.train[0])) & \
         (dates <= pd.Timestamp(DEFAULT_SPLIT.train[1]))
    acc = np.zeros(n_fac); cnt = np.zeros(n_fac)
    for ti in np.where(tr)[0]:
        F = np.asarray(cube[ti], dtype=np.float64)
        y = Y[ti].astype(np.float64)
        ok = np.isfinite(y)
        if ok.sum() < MIN_OBS_PER_DATE:
            continue
        elig = np.where(ok)[0]
        colok = np.isfinite(F[elig]).all(axis=0)
        if not colok.any():
            continue
        c = _xs_corr(F[elig][:, colok], y[elig])
        j = np.where(colok)[0]
        good = np.isfinite(c)
        acc[j[good]] += c[good]
        cnt[j[good]] += 1
    sign = np.where(cnt > 0, np.sign(np.divide(acc, np.maximum(cnt, 1))), 1.0)
    sign[sign == 0] = 1.0

    for split in want:
        lo, hi = DEFAULT_SPLIT.segment(split)
        sel = np.where((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))[0]
        series = {k: [[] for _ in range(n_fac)] for k in
                  ("ic_raw", "ic_res", "rk_raw", "rk_res", "r2", "adj", "nobs",
                   "degen", "flat", "raw_sd")}
        used_dates = 0
        for ti in sel:
            S = S_cube[ti].astype(np.float64)
            y = Y[ti].astype(np.float64)
            F = np.asarray(cube[ti], dtype=np.float64) * sign
            base = np.isfinite(S).all(axis=1) & np.isfinite(y)
            if base.sum() < MIN_OBS_PER_DATE:
                continue
            # factors complete on the eligible set share one projection
            elig = np.where(base)[0]
            colok = np.isfinite(F[elig]).all(axis=0)
            if not colok.any():
                continue
            used_dates += 1
            X = np.column_stack([np.ones(elig.size), S[elig]])
            Fm = F[elig][:, colok]
            resid, r2, adj = _residualise(Fm, X)
            yy = y[elig]
            ic_raw = _xs_corr(Fm, yy); ic_res = _xs_corr(resid, yy)
            rk_raw = _xs_corr(_ranks(Fm), _ranks(yy[:, None])[:, 0])
            rk_res = _xs_corr(_ranks(resid), _ranks(yy[:, None])[:, 0])
            # Two different things, kept apart:
            #   flat  - the RAW factor has no cross-sectional spread, so there
            #           was never a ranking to explain (a broken factor);
            #   degen - the raw factor DID vary and the styles removed
            #           essentially all of it (genuine style overlap).
            raw_sd = Fm.std(axis=0)
            scale = np.abs(Fm).max(axis=0)
            flat = raw_sd <= FLAT_REL * np.where(scale > 0, scale, 1.0)
            degen = (~flat) & (resid.std(axis=0) < DEGENERATE_REL * raw_sd)
            for n, j in enumerate(np.where(colok)[0]):
                series["ic_raw"][j].append(ic_raw[n]); series["ic_res"][j].append(ic_res[n])
                series["rk_raw"][j].append(rk_raw[n]); series["rk_res"][j].append(rk_res[n])
                series["r2"][j].append(r2[n]); series["adj"][j].append(adj[n])
                series["nobs"][j].append(elig.size)
                series["degen"][j].append(bool(degen[n]))
                series["flat"][j].append(bool(flat[n]))
                series["raw_sd"][j].append(float(raw_sd[n]))

        rows = []
        for j, m in enumerate(meta):
            raw = np.array(series["ic_raw"][j]); res = np.array(series["ic_res"][j])
            keep = np.isfinite(raw) & np.isfinite(res)
            d = raw[keep] - res[keep]
            rows.append({
                "method": m["method"], "market": market, "seed": m["seed"],
                "run": m["run"], "factor": m["name"], "split": split,
                "sign": float(sign[j]),
                "valid_dates": int(keep.sum()),
                "mean_obs": float(np.mean(series["nobs"][j])) if series["nobs"][j] else np.nan,
                # fraction of dates the styles removed the whole signal
                "degenerate_rate": float(np.mean(series["degen"][j])) if series["degen"][j] else np.nan,
                # fraction of dates the factor was constant to begin with
                "flat_rate": float(np.mean(series["flat"][j])) if series["flat"][j] else np.nan,
                "median_raw_sd": float(np.median(series["raw_sd"][j])) if series["raw_sd"][j] else np.nan,
                "r2": float(np.nanmean(series["r2"][j])) if series["r2"][j] else np.nan,
                "adj_r2": float(np.nanmean(series["adj"][j])) if series["adj"][j] else np.nan,
                "ic_raw": float(raw[keep].mean()) if keep.any() else np.nan,
                "ic_res": float(res[keep].mean()) if keep.any() else np.nan,
                "delta_ic": float(d.mean()) if keep.any() else np.nan,
                "rank_ic_raw": float(np.nanmean(series["rk_raw"][j])) if series["rk_raw"][j] else np.nan,
                "rank_ic_res": float(np.nanmean(series["rk_res"][j])) if series["rk_res"][j] else np.nan,
                # HAC on the residual IC series, and on the PAIRED daily
                # differences -- never the difference of two t-statistics.
                "t_hac_ic_res": newey_west_tstat(pd.Series(res[keep]), HAC_LAGS),
                "t_hac_delta": newey_west_tstat(pd.Series(d), HAC_LAGS),
            })
        df = pd.DataFrame(rows)
        path = out_root / f"{market}_{split}_style.csv"
        df.to_csv(path, index=False)
        (out_root / f"{market}_{split}_style_meta.json").write_text(json.dumps({
            "market": market, "split": split, "styles": STYLES,
            "winsor": WINSOR, "min_obs_per_date": MIN_OBS_PER_DATE,
            "hac_lags": HAC_LAGS, "dates_used": int(used_dates),
            "dates_in_split": int(len(sel)), "failures": failures,
        }, indent=2))
        print(f"  {split}: {len(df)} factors, {used_dates}/{len(sel)} dates -> {path}")


if __name__ == "__main__":
    fire.Fire({"build": build, "panel": build_style_panel})
