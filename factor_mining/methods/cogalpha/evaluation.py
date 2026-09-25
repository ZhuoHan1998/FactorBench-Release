"""Fitness evaluation (paper Section 3.4, Appendices A.4 and B.3).

Five predictive-power metrics per alpha — IC, ICIR, RankIC, RankICIR and
Mutual Information — then a two-tier gate:

- **qualified**: every metric above the 65th percentile of the same generation,
  and above an absolute floor;
- **elite**: every metric above the 80th percentile, and above a higher floor.

Qualified alphas form the next parent pool; elite alphas are also carried into
the final candidate pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METRICS: tuple[str, ...] = ("ic", "icir", "rank_ic", "rank_icir", "mi")

# Section 3.4: qualified > 65th percentile, elite > 80th.
QUALIFIED_PERCENTILE = 65.0
ELITE_PERCENTILE = 80.0

# Mutual-information estimator settings. The paper defines MI by its integral
# (Eq. 6) but names no estimator; a fixed-bin histogram over per-day ranks is
# used here — scale-free, deterministic, and cheap enough to run on every
# candidate. See METHOD.md.
MI_BINS = 10


@dataclass(frozen=True)
class Floors:
    """Absolute minimum bounds, "to prevent the dominance of outliers"."""

    ic: float
    icir: float
    mi: float

    def as_dict(self) -> dict[str, float]:
        return {
            "ic": self.ic, "rank_ic": self.ic,
            "icir": self.icir, "rank_icir": self.icir,
            "mi": self.mi,
        }


# Appendix A.4 gives two calibrations and says explicitly that thresholds "may
# vary depending on the dataset ... because it is harder to mine alpha signals
# in a more effective stock market". CSI300 uses MI >= 0.02; S&P500 relaxes the
# MI floor to 0.012 while keeping the rest.
FLOORS_CSI = (Floors(ic=0.005, icir=0.05, mi=0.02), Floors(ic=0.01, icir=0.1, mi=0.02))
FLOORS_SPX = (Floors(ic=0.005, icir=0.05, mi=0.012), Floors(ic=0.01, icir=0.1, mi=0.012))

# The paper calibrates only CSI300 and S&P500. csi300 takes the A-share
# calibration; the four developed markets take the S&P500 one — see METHOD.md.
MARKET_FLOORS: dict[str, tuple[Floors, Floors]] = {
    "csi300": FLOORS_CSI,
    "sp100": FLOORS_SPX,
    "hsi": FLOORS_SPX,
    "ftse100": FLOORS_SPX,
    "nikkei225": FLOORS_SPX,
}


def floors_for(market: str) -> tuple[Floors, Floors]:
    """(qualified floors, elite floors) for a market."""
    return MARKET_FLOORS.get(market, FLOORS_SPX)


@dataclass
class Fitness:
    """The five metrics for one alpha on the training window."""

    ic: float
    icir: float
    rank_ic: float
    rank_icir: float
    mi: float
    n_days: int
    coverage: float

    def as_dict(self) -> dict[str, float]:
        return {
            "ic": self.ic, "icir": self.icir,
            "rank_ic": self.rank_ic, "rank_icir": self.rank_icir,
            "mi": self.mi, "coverage": self.coverage,
        }

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, m) for m in METRICS], dtype=float)

    @property
    def score(self) -> float:
        """Single summary used for ranking and for plateau detection."""
        return float(np.mean([self.rank_ic, self.rank_icir / 10.0, self.mi]))


def _daily_corr(signal: pd.Series, target: pd.Series, method: str) -> pd.Series:
    frame = pd.DataFrame({"s": signal, "t": target}).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.groupby(level="date").apply(
        lambda g: g["s"].corr(g["t"], method=method) if len(g) > 2 else np.nan
    )


def mutual_information(signal: pd.Series, target: pd.Series, bins: int = MI_BINS) -> float:
    """MI between factor and forward return (Eq. 6), histogram estimator.

    Both series are rank-transformed within each day before pooling, so the
    estimate is invariant to each alpha's arbitrary scale and is not dominated
    by whichever days happen to have the widest dispersion.
    """
    frame = pd.DataFrame({"s": signal, "t": target}).dropna()
    if len(frame) < bins * bins:
        return 0.0
    ranked = frame.groupby(level="date").rank(pct=True)
    x = ranked["s"].to_numpy()
    y = ranked["t"].to_numpy()
    joint, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py)
        terms = np.where(joint > 0, joint * np.log(ratio), 0.0)
    value = float(np.nansum(terms))
    return max(value, 0.0)


def compute_fitness(signal: pd.Series, target: pd.Series) -> Fitness:
    """All five metrics on an already segment-sliced signal and target."""
    ic_series = _daily_corr(signal, target, "pearson").dropna()
    ric_series = _daily_corr(signal, target, "spearman").dropna()
    n_possible = signal.index.get_level_values("date").nunique()
    if ic_series.empty:
        return Fitness(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    def _ir(series: pd.Series) -> float:
        std = float(series.std())
        if std <= 0 or not np.isfinite(std):
            return 0.0
        return float(series.mean() / std)

    return Fitness(
        ic=float(ic_series.mean()),
        icir=_ir(ic_series),
        rank_ic=float(ric_series.mean()) if not ric_series.empty else 0.0,
        rank_icir=_ir(ric_series) if not ric_series.empty else 0.0,
        mi=mutual_information(signal, target),
        n_days=int(len(ic_series)),
        coverage=float(len(ic_series) / n_possible) if n_possible else 0.0,
    )


def _percentiles(population: list[Fitness], q: float) -> dict[str, float]:
    if not population:
        return {m: -np.inf for m in METRICS}
    stacked = np.vstack([f.vector() for f in population])
    return {m: float(np.percentile(stacked[:, i], q)) for i, m in enumerate(METRICS)}


def classify(
    fitness: Fitness,
    generation: list[Fitness],
    market: str,
) -> tuple[bool, bool, str]:
    """Return (qualified, elite, reason-if-neither) for one alpha.

    Section 3.4: a metric must clear BOTH the generation percentile and the
    absolute floor. The floors are one-sided as the paper writes them ("bounded
    below"), so a strongly negative-IC alpha is rejected rather than sign
    flipped — see METHOD.md.
    """
    qualified_floors, elite_floors = floors_for(market)
    values = {m: getattr(fitness, m) for m in METRICS}

    def _passes(percentile: float, floors: Floors) -> tuple[bool, str]:
        cuts = _percentiles(generation, percentile)
        bounds = floors.as_dict()
        for metric in METRICS:
            value = values[metric]
            if not np.isfinite(value):
                return False, f"{metric} is not finite"
            if value < bounds[metric]:
                return False, f"{metric} {value:.4f} below floor {bounds[metric]}"
            if value < cuts[metric]:
                return False, (
                    f"{metric} {value:.4f} below the p{percentile:.0f} of this "
                    f"generation ({cuts[metric]:.4f})"
                )
        return True, ""

    is_qualified, why = _passes(QUALIFIED_PERCENTILE, qualified_floors)
    if not is_qualified:
        return False, False, why
    is_elite, _ = _passes(ELITE_PERCENTILE, elite_floors)
    return True, is_elite, ""
