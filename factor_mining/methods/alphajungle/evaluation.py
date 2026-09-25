"""Multi-dimensional alpha evaluation (paper Section 3 + Appendices D and G).

Each candidate is backtested on the training window, producing four numeric
dimension metrics (Appendix G, "Evaluation Score"):

    Effectiveness -> RankIC
    Stability     -> RankIR
    Turnover      -> daily turnover rate
    Diversity     -> maximum correlation with the repository

Those are converted to dimension scores by percentile rank against the
repository (Eq. 6-7), a fifth dimension (Overfitting Risk) is supplied by the
LLM, and the mean of all five is the MCTS reward (Eq. 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Appendix G, "Effective Alpha Check" — the concrete criteria an alpha must
# clear to enter the repository.
#
# CALIBRATION WARNING: these are absolute thresholds tuned to the paper's
# setting (CSI300/CSI1000, ~300 stocks, 10- and 30-day targets). RankIR is
# mean/std of daily RankIC, so it scales with the size of the cross-section.
# Measured on FactorBench panels over 100 random grammar-valid alphas, the best
# RankIR achieved is 0.20 on sp100 (99 assets) and 0.37 on csi300 (272), so
# `RANK_IR_MIN = 0.3` admits nothing at all on the smaller universes. Override
# per market with the `Thresholds` dataclass rather than editing these — the
# defaults stay the paper's. See METHOD.md.
RANK_IC_MIN = 0.015
RANK_IR_MIN = 0.3
RANK_IC_PERCENTILE_MAX = 0.95
RANK_IR_PERCENTILE_MAX = 0.95
TURNOVER_MAX = 1.6
MAX_CORRELATION = 0.8


@dataclass(frozen=True)
class Thresholds:
    """Effective-alpha admission criteria; defaults are the paper's."""

    rank_ic_min: float = RANK_IC_MIN
    rank_ir_min: float = RANK_IR_MIN
    turnover_max: float = TURNOVER_MAX
    max_correlation: float = MAX_CORRELATION
    rank_ic_percentile_max: float = RANK_IC_PERCENTILE_MAX
    rank_ir_percentile_max: float = RANK_IR_PERCENTILE_MAX

DIMENSIONS = ("effectiveness", "stability", "turnover", "diversity", "overfitting")
# Numeric dimensions are ranked against the repository; overfitting comes from
# the LLM on a 0-10 scale, rescaled to [0, 1] so every dimension shares
# e_max = 1 (Eq. 3 and Eq. 8 both assume a common scale).
E_MAX = 1.0
LLM_SCORE_MAX = 10.0


@dataclass
class AlphaMetrics:
    """Raw backtest metrics for one alpha on the training window."""

    rank_ic: float
    rank_ir: float
    turnover: float
    max_correlation: float
    n_days: int
    coverage: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rank_ic": self.rank_ic,
            "rank_ir": self.rank_ir,
            "turnover": self.turnover,
            "max_correlation": self.max_correlation,
            "coverage": self.coverage,
        }


@dataclass
class Evaluation:
    """Dimension scores and the aggregate reward for one alpha."""

    metrics: AlphaMetrics
    scores: dict[str, float] = field(default_factory=dict)
    overfitting_reason: str = ""

    @property
    def reward(self) -> float:
        """Eq. 8: S(f) = mean of the dimension scores."""
        if not self.scores:
            return 0.0
        return float(np.mean(list(self.scores.values())))

    def vector(self) -> np.ndarray:
        return np.array([self.scores.get(d, 0.0) for d in DIMENSIONS], dtype=float)


# A date needs at least this many valid pairs to yield a correlation. The
# groupby implementation these helpers replaced spelled it `len(g) > 2`.
MIN_NAMES_PER_DAY = 3


def _row_pearson(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """Per-date cross-sectional Pearson correlation of two wide frames.

    Both frames are (date x asset). Only cells present in both count, matching
    the pairwise `dropna` the per-date `groupby.apply` did — but as four whole-
    frame reductions instead of one Python call per date, which is where this
    search spent most of its local time: a rank IC cost ~1.0 s per candidate
    and every repository-correlation another ~0.16 s, once per zoo member.
    """
    both = x.notna() & y.notna()
    x = x.where(both)
    y = y.where(both)
    xm = x.sub(x.mean(axis=1), axis=0)
    ym = y.sub(y.mean(axis=1), axis=0)
    numerator = (xm * ym).sum(axis=1)
    denominator = np.sqrt(xm.pow(2).sum(axis=1) * ym.pow(2).sum(axis=1))
    corr = numerator / denominator.where(denominator > 0)
    return corr.where(both.sum(axis=1) >= MIN_NAMES_PER_DAY)


def _aligned_wide(a: pd.Series, b: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(date x asset) frames on a common index, as an outer join would give."""
    return a.unstack("asset").align(b.unstack("asset"), join="outer")


def _daily_rank_ic(signal: pd.Series, target: pd.Series) -> pd.Series:
    """Per-date cross-sectional Spearman correlation, NaN pairs dropped."""
    if signal.empty or target.empty:
        return pd.Series(dtype=float)
    wide_signal, wide_target = _aligned_wide(signal, target)
    # Spearman is Pearson on the per-date ranks, and ranking after masking to
    # the valid pairs is what correlating the dropped-NaN subset did.
    both = wide_signal.notna() & wide_target.notna()
    return _row_pearson(
        wide_signal.where(both).rank(axis=1),
        wide_target.where(both).rank(axis=1),
    )


def _weights(signal: pd.Series) -> pd.DataFrame:
    """Cross-sectionally rank-demeaned long-short weights with sum|w| = 1.

    The paper defines turnover as "the average daily change in the alpha's
    portfolio holdings" without pinning the construction. A rank-based
    dollar-neutral book is used here because it is scale-free (so turnover is
    comparable across alphas of wildly different magnitude) and it puts full
    turnover at 2.0, which is the scale the paper's own threshold of 1.6
    implies. See METHOD.md.
    """
    wide = signal.unstack("asset")
    ranks = wide.rank(axis=1, pct=True)
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = centered.abs().sum(axis=1)
    return centered.div(gross.where(gross > 0), axis=0)


def _turnover(signal: pd.Series) -> float:
    w = _weights(signal)
    delta = (w - w.shift(1)).abs().sum(axis=1)
    # The first row and any all-NaN day have no meaningful change.
    live = w.notna().any(axis=1) & w.shift(1).notna().any(axis=1)
    if not live.any():
        return float("inf")
    return float(delta[live].mean())


def daily_cross_sectional_corr(a: pd.Series, b: pd.Series) -> float:
    """Mean daily cross-sectional Pearson correlation between two signals.

    Used for the Diversity dimension and the repository's 0.8 correlation
    ceiling. The paper says "maximum correlation with the alpha within the
    effective alpha repository" without specifying pooled vs. per-day; per-day
    then averaged is used because every other metric here is cross-sectional.
    """
    if a.empty or b.empty:
        return 0.0
    wide_a, wide_b = _aligned_wide(a, b)
    value = _row_pearson(wide_a, wide_b).mean()
    return 0.0 if not np.isfinite(value) else float(value)


def zoo_max_correlation(signal: pd.Series, zoo_signals: list[pd.Series] | None) -> float:
    """Diversity: the largest absolute correlation with anything in the zoo.

    Kept separate from :func:`compute_metrics` because it is the *only* metric
    that depends on the repository, so it is the only one that has to be redone
    when the repository grows. See the effective-alpha check in ``run.py``.
    """
    if not zoo_signals:
        return 0.0
    wide_signal = signal.unstack("asset")
    worst = 0.0
    for other in zoo_signals:
        x, y = wide_signal.align(other.unstack("asset"), join="outer")
        value = _row_pearson(x, y).mean()
        if np.isfinite(value):
            worst = max(worst, abs(float(value)))
    return worst


def compute_metrics(
    signal: pd.Series,
    target: pd.Series,
    zoo_signals: list[pd.Series] | None = None,
) -> AlphaMetrics:
    """Backtest one alpha on the (already segment-sliced) signal and target."""
    ic = _daily_rank_ic(signal, target).dropna()
    n_possible = signal.index.get_level_values("date").nunique()
    if ic.empty:
        return AlphaMetrics(0.0, 0.0, float("inf"), 1.0, 0, 0.0)
    rank_ic = float(ic.mean())
    std = float(ic.std())
    rank_ir = rank_ic / std if std > 0 and np.isfinite(std) else 0.0
    max_corr = zoo_max_correlation(signal, zoo_signals)
    return AlphaMetrics(
        rank_ic=rank_ic,
        rank_ir=float(rank_ir),
        turnover=_turnover(signal),
        max_correlation=float(max_corr),
        n_days=int(len(ic)),
        coverage=float(len(ic) / n_possible) if n_possible else 0.0,
    )


def percentile_rank(value: float, population: list[float]) -> float:
    """Eq. 6: R(f, m, zoo) = fraction of the repository that beats ``value``.

    An empty repository yields 0.5 — a neutral prior, so the very first alphas
    are neither rewarded nor punished for arriving early. The paper does not
    state the empty-repository behavior; see METHOD.md.
    """
    if not population:
        return 0.5
    finite = [p for p in population if np.isfinite(p)]
    if not finite:
        return 0.5
    if not np.isfinite(value):
        return 1.0
    return float(np.mean([value < p for p in finite]))


def dimension_scores(
    metrics: AlphaMetrics,
    zoo_metrics: list[AlphaMetrics],
    overfitting: float | None = None,
) -> dict[str, float]:
    """Eq. 7: e_i = 1 - R(f, m_i, zoo), one per dimension.

    Turnover and Diversity are 'lower is better', so their metrics are negated
    before ranking — the paper states the orientation in prose ("maintain
    turnover within a desirable, low range", "not redundant with those already
    discovered") rather than in the formula.
    """
    scores = {
        "effectiveness": 1.0 - percentile_rank(
            metrics.rank_ic, [m.rank_ic for m in zoo_metrics]
        ),
        "stability": 1.0 - percentile_rank(
            metrics.rank_ir, [m.rank_ir for m in zoo_metrics]
        ),
        "turnover": 1.0 - percentile_rank(
            -metrics.turnover, [-m.turnover for m in zoo_metrics]
        ),
        "diversity": 1.0 - percentile_rank(
            -metrics.max_correlation, [-m.max_correlation for m in zoo_metrics]
        ),
    }
    if overfitting is not None:
        scores["overfitting"] = float(np.clip(overfitting / LLM_SCORE_MAX, 0.0, E_MAX))
    return scores


def dimension_probabilities(
    scores: dict[str, float], temperature: float = 1.0
) -> dict[str, float]:
    """Eq. 3: Softmax((e_max * 1 - E_s) / T) — favours the weakest dimensions."""
    names = [d for d in DIMENSIONS if d in scores]
    gaps = np.array([E_MAX - scores[d] for d in names], dtype=float) / max(temperature, 1e-9)
    gaps -= gaps.max()
    weights = np.exp(gaps)
    weights /= weights.sum()
    return dict(zip(names, (float(w) for w in weights)))


def passes_effective_check(
    metrics: AlphaMetrics,
    zoo_metrics: list[AlphaMetrics],
    thresholds: Thresholds | None = None,
) -> tuple[bool, str]:
    """Appendix G, "Effective Alpha Check". Returns (passed, reason-if-not).

    The reason string starts with the criterion name so callers can histogram
    rejections and see which gate is actually binding.
    """
    t = thresholds or Thresholds()
    if metrics.rank_ic < t.rank_ic_min:
        return False, f"rank_ic: {metrics.rank_ic:.4f} < {t.rank_ic_min}"
    if metrics.rank_ir < t.rank_ir_min:
        return False, f"rank_ir: {metrics.rank_ir:.4f} < {t.rank_ir_min}"
    if metrics.turnover > t.turnover_max:
        return False, f"turnover: {metrics.turnover:.3f} > {t.turnover_max}"
    if metrics.max_correlation >= t.max_correlation:
        return False, (
            f"correlation: {metrics.max_correlation:.3f} >= {t.max_correlation}"
        )
    r_ic = percentile_rank(metrics.rank_ic, [m.rank_ic for m in zoo_metrics])
    if r_ic > t.rank_ic_percentile_max:
        return False, f"rank_ic_pct: {r_ic:.3f} > {t.rank_ic_percentile_max}"
    r_ir = percentile_rank(metrics.rank_ir, [m.rank_ir for m in zoo_metrics])
    if r_ir > t.rank_ir_percentile_max:
        return False, f"rank_ir_pct: {r_ir:.3f} > {t.rank_ir_percentile_max}"
    return True, ""
