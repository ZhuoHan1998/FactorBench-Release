"""Cross-sectional signal metrics against the shared target.

IC here is the standard per-date cross-sectional correlation with NaN pairs
dropped (pandas semantics). Note this differs slightly from AlphaGen's
internal calculator, which z-scores by day and fills NaNs with 0 before a
masked Pearson — location/scale invariance makes the two agree closely, but
not bit-exactly; the comparison script reports the delta as a fidelity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, Split


def daily_ic(signal: pd.Series, target: pd.Series, method: str = "pearson") -> pd.Series:
    """Per-date cross-sectional correlation between signal and target."""
    df = pd.DataFrame({"s": signal, "t": target}).dropna()
    if df.empty:
        # All-NaN signal (e.g. a degenerate expression): no dates to score.
        return pd.Series(dtype=float)
    return df.groupby(level="date").apply(lambda g: g["s"].corr(g["t"], method=method))


def segment_metrics(
    signal: pd.Series,
    target: pd.Series,
    split: Split = DEFAULT_SPLIT,
    segments: tuple[str, ...] = ("train", "valid", "test"),
) -> dict[str, float]:
    """Mean IC / Rank IC / ICIR per split segment."""
    ic = daily_ic(signal, target, "pearson")
    ric = daily_ic(signal, target, "spearman")
    out: dict[str, float] = {}
    for seg in segments:
        start, end = split.segment(seg)
        ic_seg = ic.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        ric_seg = ric.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        out[f"{seg}.ic"] = float(ic_seg.mean())
        out[f"{seg}.rank_ic"] = float(ric_seg.mean())
        ic_std = ic_seg.std()
        has_spread = bool(ic_std) and not np.isnan(ic_std)
        out[f"{seg}.icir"] = float(ic_seg.mean() / ic_std) if has_spread else float("nan")
    return out


def long_short_horizon_metrics(
    signal: pd.Series,
    forward_return: pd.Series,
    start: str,
    end: str,
    horizon: int,
    quantile: float = 0.2,
    min_names: int = 10,
) -> dict[str, float]:
    """Top-minus-bottom quantile portfolio held for ``horizon`` trading days.

    ``forward_return`` must be the ``horizon``-day forward return (the shared
    target), so each formation date's spread is what that portfolio earned over
    its holding period. Formation dates inside ``[start, end]`` are taken every
    ``horizon`` trading days, so holding periods do not overlap and each is
    counted once; annualisation uses ``252 / horizon`` periods per year.

    Used by the LLM-agent adapters to fill the official "excess return" feedback
    slots on the same horizon as the IC they are scored on, rather than on
    next-day returns. Returns NaN-free zeros when there is nothing to score, as
    the agents' feedback tables require numbers.
    """
    df = pd.DataFrame({"s": signal, "r": forward_return}).dropna()
    empty = {"annualized_return": 0.0, "information_ratio": 0.0, "max_drawdown": 0.0}
    if df.empty:
        return empty

    def spread(g: pd.DataFrame) -> float:
        if len(g) < min_names:
            return np.nan
        q = g["s"].rank(pct=True)
        return g.loc[q >= 1 - quantile, "r"].mean() - g.loc[q <= quantile, "r"].mean()

    per_date = df.groupby(level="date").apply(spread).dropna()
    per_date = per_date.loc[pd.Timestamp(start): pd.Timestamp(end)]
    held = per_date.iloc[::horizon]
    if held.empty:
        return empty
    periods_per_year = 252.0 / horizon
    sd = held.std()
    cum = held.cumsum()
    return {
        "annualized_return": float(held.mean() * periods_per_year),
        "information_ratio": (float(held.mean() / sd * np.sqrt(periods_per_year))
                              if len(held) > 1 and sd > 0 else 0.0),
        "max_drawdown": float((cum - cum.cummax()).min()),
    }


def newey_west_tstat(series: pd.Series, lags: int) -> float:
    """t-statistic for the mean of ``series`` under HAC (Newey-West) errors.

    The daily IC series is NOT independent across days. The target is a 20-day
    forward return sampled daily, so consecutive observations share 19/20 of
    their horizon; measured lag-1 autocorrelation of daily IC runs ~0.95,
    decaying to zero around lag 20. Treating the days as iid overstates the
    t-statistic roughly fourfold (3.99 -> 0.98 on a real test-period factor),
    which would turn noise into a "significant" result.

    ``lags`` should be chosen from the known overlap length, not from a
    data-driven rule: the usual Newey-West bandwidth formula returns ~5 for a
    two-year test window, far below the 20-day overlap we know is there.
    """
    x = series.dropna().to_numpy(dtype="float64")
    n = x.size
    if n < 3:
        return float("nan")
    centered = x - x.mean()
    variance = float((centered * centered).mean())
    if variance <= 0:
        return float("nan")
    for lag in range(1, min(lags, n - 1) + 1):
        cov = float((centered[lag:] * centered[:-lag]).mean())
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * cov
    if variance <= 0:
        # Bartlett weights keep this PSD in theory; guard anyway rather than
        # returning an imaginary standard error.
        return float("nan")
    return float(x.mean() / np.sqrt(variance / n))


def train_sign(ic: pd.Series, split: Split = DEFAULT_SPLIT) -> float:
    """+1/-1 from the sign of mean IC over the TRAIN segment only.

    Methods disagree on sign convention: gp emits 100% positive train IC while
    alphaqcm and alphaagent emit ~55% negative, because most optimise |IC|.
    Averaging signed IC across such a pool cancels toward zero and measures
    convention rather than skill.

    The sign must be fixed on train and then applied unchanged to validation and
    test. Choosing it on test would manufacture positive test IC out of pure
    noise — the exact artefact that search-overfitting analysis is trying to
    detect.
    """
    start, end = split.segment("train")
    mean = ic.loc[pd.Timestamp(start) : pd.Timestamp(end)].mean()
    if pd.isna(mean) or mean == 0:
        return 1.0
    return float(np.sign(mean))
