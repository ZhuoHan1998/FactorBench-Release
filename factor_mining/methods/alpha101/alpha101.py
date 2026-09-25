"""WorldQuant Alpha101 - the "have we improved since 2015?" floor for RQ0.

Ported from the original FactorBench implementation on the main branch
(factor_bench/factors/alpha101.py) with three changes, all noted inline:
failures are surfaced instead of swallowed, correlation no longer rejects
short windows (recovering alpha045), and the NaN-to-zero policy in
_safe_corr is documented rather than implicit.

Reference: Kakushadze (2016) "101 Formulaic Alphas" https://arxiv.org/abs/1601.00991

Data format: all inputs/outputs are pd.DataFrames with shape (n_days, n_assets).
This matches the FactorBench panel convention.

Alphas requiring IndNeutralize (industry classification) are excluded as that data
is not universally available. ~65 alphas are implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================
# Operators
# ============================================================================

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (per row)."""
    return df.rank(axis=1, pct=True)


def delay(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return df.shift(period)


def delta(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return df.diff(period)


def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=1).sum()


def sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=1).mean()


def stddev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=2).std()


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    # min_periods must not exceed the window: alpha045 legitimately asks for
    # correlation(close, volume, 2), which a hardcoded min_periods=3 rejects.
    return x.rolling(window, min_periods=min(3, window)).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return x.rolling(window, min_periods=3).cov(y)


def ts_rank(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=1).rank(pct=True)


def ts_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=1).min()


def ts_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=1).max()


def ts_argmax(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=1).apply(np.argmax, raw=True) + 1


def ts_argmin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=1).apply(np.argmin, raw=True) + 1


def product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window, min_periods=1).apply(np.prod, raw=True)


def decay_linear(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Linearly-weighted moving average: weights = [1, 2, ..., period] / sum."""
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()

    def _wma(x):
        n = len(x)
        if n < period:
            w = np.arange(1, n + 1, dtype=float)
            w /= w.sum()
            return np.dot(x, w)
        return np.dot(x[-period:], weights)

    return df.rolling(period, min_periods=1).apply(_wma, raw=True)


def scale(df: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    """Scale so sum(abs(x)) = k per row."""
    abs_sum = df.abs().sum(axis=1)
    return df.mul(k).div(abs_sum.where(abs_sum > 1e-10, np.nan), axis=0)


def sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df)


def log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df.abs().clip(lower=1e-10))


# ============================================================================
# Alpha101 class
# ============================================================================

class Alpha101:
    """Compute WorldQuant Alpha101 factors on panel data.

    Args:
        open_: (n_days, n_assets) DataFrame
        high: (n_days, n_assets) DataFrame
        low: (n_days, n_assets) DataFrame
        close: (n_days, n_assets) DataFrame
        volume: (n_days, n_assets) DataFrame
        vwap: (n_days, n_assets) DataFrame — if None, approximated as (high+low+close)/3
        returns: (n_days, n_assets) DataFrame — if None, computed as close.pct_change()
    """

    def __init__(
        self,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
    ):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap if vwap is not None else (high + low + close) / 3
        self.returns = returns if returns is not None else close.pct_change()

    def _safe_corr(self, x, y, window):
        """Rolling correlation with undefined values forced to 0.

        DEVIATION, kept deliberately: where the correlation is undefined (a
        constant series over the window, or insufficient history) this returns 0
        rather than NaN. Kakushadze's formulas do not specify a fill, so this is
        an implementation choice inherited from the original FactorBench
        implementation on the `main` branch, preserved so the baseline
        reproduces it.

        Consequence to keep in mind when reading results: Alpha101 factors report
        ~100% coverage by construction, and some of those values are fabricated
        zero-correlations rather than real observations. This is the one place in
        the benchmark where NaN is silently converted to 0.
        """
        return correlation(x, y, window).replace([np.inf, -np.inf], 0).fillna(0)

    # ------------------------------------------------------------------
    def alpha001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5))

    def alpha002(self):
        return -1 * self._safe_corr(
            rank(delta(log(self.volume), 2)),
            rank((self.close - self.open) / self.open), 6
        )

    def alpha003(self):
        return -1 * self._safe_corr(rank(self.open), rank(self.volume), 10)

    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self):
        return rank(self.open - sma(self.vwap, 10)) * (-1 * rank(self.close - self.vwap).abs())

    def alpha006(self):
        return -1 * self._safe_corr(self.open, self.volume, 10)

    def alpha007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(delta(self.close, 7).abs(), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha

    def alpha008(self):
        return -1 * rank(
            ts_sum(self.open, 5) * ts_sum(self.returns, 5)
            - delay(ts_sum(self.open, 5) * ts_sum(self.returns, 5), 10)
        )

    def alpha009(self):
        d = delta(self.close, 1)
        cond1 = ts_min(d, 5) > 0
        cond2 = ts_max(d, 5) < 0
        alpha = -1 * d
        alpha[cond1 | cond2] = d
        return alpha

    def alpha010(self):
        d = delta(self.close, 1)
        cond1 = ts_min(d, 4) > 0
        cond2 = ts_max(d, 4) < 0
        alpha = -1 * d
        alpha[cond1 | cond2] = d
        return rank(alpha)

    def alpha011(self):
        return (
            (rank(ts_max(self.vwap - self.close, 3)) + rank(ts_min(self.vwap - self.close, 3)))
            * rank(delta(self.volume, 3))
        )

    def alpha012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        return -1 * rank(delta(self.returns, 3)) * self._safe_corr(self.open, self.volume, 10)

    def alpha015(self):
        return -1 * ts_sum(rank(self._safe_corr(rank(self.high), rank(self.volume), 3)), 3)

    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (
            rank(ts_rank(self.close, 10))
            * rank(delta(delta(self.close, 1), 1))
            * rank(ts_rank(self.volume / adv20, 5))
        )

    def alpha018(self):
        return -1 * rank(
            stddev((self.close - self.open).abs(), 5)
            + (self.close - self.open)
            + self._safe_corr(self.close, self.open, 10)
        )

    def alpha019(self):
        return (
            -1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))
            * (1 + rank(1 + ts_sum(self.returns, 250)))
        )

    def alpha020(self):
        return -1 * (
            rank(self.open - delay(self.high, 1))
            * rank(self.open - delay(self.close, 1))
            * rank(self.open - delay(self.low, 1))
        )

    def alpha021(self):
        cond1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close.values), index=self.close.index,
                             columns=self.close.columns)
        alpha[cond1 | cond2] = -1
        return alpha

    def alpha022(self):
        corr = self._safe_corr(self.high, self.volume, 5)
        return -1 * delta(corr, 5) * rank(stddev(self.close, 20))

    def alpha023(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close.values), index=self.close.index,
                             columns=self.close.columns)
        alpha[cond] = -1 * delta(self.high, 2)
        return alpha

    def alpha024(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha

    def alpha025(self):
        adv20 = sma(self.volume, 20)
        return rank(-1 * self.returns * adv20 * self.vwap * (self.high - self.close))

    def alpha026(self):
        return -1 * ts_max(self._safe_corr(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)

    def alpha028(self):
        adv20 = sma(self.volume, 20)
        return scale(self._safe_corr(adv20, self.low, 5) + (self.high + self.low) / 2 - self.close)

    def alpha029(self):
        inner = rank(rank(-1 * rank(delta(self.close - 1, 5))))
        return (
            ts_min(rank(rank(scale(log(ts_sum(inner, 2))))), 5)
            + ts_rank(delay(-1 * self.returns, 6), 5)
        )

    def alpha030(self):
        d = delta(self.close, 1)
        inner = sign(d) + sign(delay(d, 1)) + sign(delay(d, 2))
        return (1.0 - rank(inner)) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)

    def alpha033(self):
        return rank(-1 + self.open / self.close)

    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([np.inf, -np.inf], 1).fillna(1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    def alpha035(self):
        return (
            ts_rank(self.volume, 32)
            * (1 - ts_rank(self.close + self.high - self.low, 16))
            * (1 - ts_rank(self.returns, 32))
        )

    def alpha037(self):
        return (
            rank(self._safe_corr(delay(self.open - self.close, 1), self.close, 200))
            + rank(self.open - self.close)
        )

    def alpha038(self):
        inner = (self.close / self.open).replace([np.inf, -np.inf], 1).fillna(1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)

    def alpha039(self):
        adv20 = sma(self.volume, 20)
        return (
            -1 * rank(delta(self.close, 7) * (1 - rank(decay_linear(self.volume / adv20, 9))))
            * (1 + rank(sma(self.returns, 250)))
        )

    def alpha040(self):
        return -1 * rank(stddev(self.high, 10)) * self._safe_corr(self.high, self.volume, 10)

    def alpha041(self):
        return (self.high * self.low).pow(0.5) - self.vwap

    def alpha042(self):
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank(-1 * delta(self.close, 7), 8)

    def alpha044(self):
        return -1 * self._safe_corr(self.high, rank(self.volume), 5)

    def alpha045(self):
        return -1 * (
            rank(sma(delay(self.close, 5), 20))
            * self._safe_corr(self.close, self.volume, 2)
            * rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        )

    def alpha046(self):
        inner = (
            (delay(self.close, 20) - delay(self.close, 10)) / 10
            - (delay(self.close, 10) - self.close) / 10
        )
        alpha = -1 * delta(self.close, 1)
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    def alpha047(self):
        adv20 = sma(self.volume, 20)
        return (
            ((rank(1 / self.close) * self.volume / adv20)
             * (self.high * rank(self.high - self.close) / (sma(self.high, 5) / 5)))
            - rank(self.vwap - delay(self.vwap, 5))
        )

    def alpha049(self):
        inner = (
            (delay(self.close, 20) - delay(self.close, 10)) / 10
            - (delay(self.close, 10) - self.close) / 10
        )
        alpha = -1 * delta(self.close, 1)
        alpha[inner < -0.1] = 1
        return alpha

    def alpha050(self):
        return -1 * ts_max(rank(self._safe_corr(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha051(self):
        inner = (
            (delay(self.close, 20) - delay(self.close, 10)) / 10
            - (delay(self.close, 10) - self.close) / 10
        )
        alpha = -1 * delta(self.close, 1)
        alpha[inner < -0.05] = 1
        return alpha

    def alpha052(self):
        return (
            -1 * delta(ts_min(self.low, 5), 5)
            * rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
            * ts_rank(self.volume, 5)
        )

    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta(
            ((self.close - self.low) - (self.high - self.close)) / inner, 9
        )

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * self.open ** 5 / (inner * self.close ** 5)

    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / divisor
        return -1 * self._safe_corr(rank(inner), rank(self.volume), 6)

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return -(2 * scale(rank(inner)) - scale(rank(ts_argmax(self.close, 10))))

    def alpha061(self):
        adv180 = sma(self.volume, 180)
        return (rank(self.vwap - ts_min(self.vwap, 16)) < rank(
            self._safe_corr(self.vwap, adv180, 18)
        )).astype(float)

    def alpha062(self):
        adv20 = sma(self.volume, 20)
        return (
            (rank(self._safe_corr(self.vwap, sma(adv20, 22), 10))
             < rank((rank(self.open) + rank(self.open))
                    < (rank((self.high + self.low) / 2) + rank(self.high))))
            * -1
        ).astype(float)

    def alpha064(self):
        adv120 = sma(self.volume, 120)
        return (
            (rank(self._safe_corr(
                sma(self.open * 0.178404 + self.low * 0.821596, 13),
                sma(adv120, 13), 17))
             < rank(delta((self.high + self.low) / 2 * 0.178404 + self.vwap * 0.821596, 4)))
            * -1
        ).astype(float)

    def alpha065(self):
        adv60 = sma(self.volume, 60)
        return (
            (rank(self._safe_corr(self.open * 0.00817 + self.vwap * 0.99183, sma(adv60, 9), 6))
             < rank(self.open - ts_min(self.open, 14)))
            * -1
        ).astype(float)

    def alpha066(self):
        denom = (self.open - (self.high + self.low) / 2).replace(0, 0.0001)
        return -1 * (
            rank(decay_linear(delta(self.vwap, 4), 7))
            + ts_rank(decay_linear((self.low - self.vwap) / denom, 11), 7)
        )

    def alpha068(self):
        adv15 = sma(self.volume, 15)
        return (
            (ts_rank(self._safe_corr(rank(self.high), rank(adv15), 9), 14)
             < rank(delta(self.close * 0.518371 + self.low * 0.481629, 1)))
            * -1
        ).astype(float)

    def alpha072(self):
        adv40 = sma(self.volume, 40)
        num = rank(decay_linear(self._safe_corr((self.high + self.low) / 2, adv40, 9), 10))
        denom = rank(decay_linear(
            self._safe_corr(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3
        ))
        return num / denom.replace(0, np.nan)

    def alpha074(self):
        adv30 = sma(self.volume, 30)
        return (
            (rank(self._safe_corr(self.close, sma(adv30, 37), 15))
             < rank(self._safe_corr(
                 rank(self.high * 0.0262 + self.vwap * 0.9738), rank(self.volume), 11)))
            * -1
        ).astype(float)

    def alpha075(self):
        adv50 = sma(self.volume, 50)
        return (
            rank(self._safe_corr(self.vwap, self.volume, 4))
            < rank(self._safe_corr(rank(self.low), rank(adv50), 12))
        ).astype(float)

    def alpha077(self):
        adv40 = sma(self.volume, 40)
        p1 = rank(decay_linear((self.high + self.low) / 2 + self.high - self.vwap - self.high, 20))
        p2 = rank(decay_linear(self._safe_corr((self.high + self.low) / 2, adv40, 3), 6))
        return pd.DataFrame(
            np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns
        )

    def alpha078(self):
        adv40 = sma(self.volume, 40)
        return rank(self._safe_corr(
            ts_sum(self.low * 0.352233 + self.vwap * 0.647767, 20),
            ts_sum(adv40, 20), 7
        )) ** rank(self._safe_corr(rank(self.vwap), rank(self.volume), 6))

    def alpha081(self):
        adv10 = sma(self.volume, 10)
        corr = self._safe_corr(self.vwap, ts_sum(adv10, 50), 8)
        return (
            (rank(log(product(rank(rank(corr) ** 4), 15)))
             < rank(self._safe_corr(rank(self.vwap), rank(self.volume), 5)))
            * -1
        ).astype(float)

    def alpha083(self):
        mid = (self.high - self.low) / (ts_sum(self.close, 5) / 5)
        denom = mid / (self.vwap - self.close).replace(0, np.nan)
        return rank(delay(mid, 2)) * rank(rank(self.volume)) / denom

    def alpha084(self):
        return ts_rank(self.vwap - ts_max(self.vwap, 15), 21).pow(delta(self.close, 5))

    def alpha085(self):
        adv30 = sma(self.volume, 30)
        return rank(self._safe_corr(
            self.high * 0.876703 + self.close * 0.123297, adv30, 10
        )) ** rank(self._safe_corr(
            ts_rank((self.high + self.low) / 2, 4), ts_rank(self.volume, 10), 7
        ))

    def alpha086(self):
        adv20 = sma(self.volume, 20)
        return (
            (ts_rank(self._safe_corr(self.close, sma(adv20, 15), 6), 20)
             < rank(self.open + self.close - self.vwap - self.open))
            * -1
        ).astype(float)

    def alpha094(self):
        adv60 = sma(self.volume, 60)
        return -1 * rank(self.vwap - ts_min(self.vwap, 12)).pow(
            ts_rank(self._safe_corr(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)
        )

    def alpha095(self):
        adv40 = sma(self.volume, 40)
        return (
            rank(self.open - ts_min(self.open, 12))
            < ts_rank(rank(self._safe_corr(
                sma((self.high + self.low) / 2, 19), sma(adv40, 19), 13
            )) ** 5, 12)
        ).astype(float)

    def alpha099(self):
        adv60 = sma(self.volume, 60)
        return (
            (rank(self._safe_corr(ts_sum((self.high + self.low) / 2, 20), ts_sum(adv60, 20), 9))
             < rank(self._safe_corr(self.low, self.volume, 6)))
            * -1
        ).astype(float)

    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)

    # ------------------------------------------------------------------
    # Compute all available alphas
    # ------------------------------------------------------------------
    def alpha_names(self) -> list[str]:
        """Names of every implemented alpha, in stable order."""
        return sorted(n for n in dir(self) if n.startswith("alpha") and n[5:].isdigit())

    def compute(self, name: str) -> pd.DataFrame:
        """Compute one alpha by name. Raises if it is not implemented."""
        if name not in self.alpha_names():
            raise KeyError(f"{name!r} is not an implemented Alpha101 factor")
        return getattr(self, name)()

    def compute_all(self, strict: bool = True) -> dict[str, pd.DataFrame]:
        """Compute every implemented alpha.

        The original implementation wrapped each call in a bare
        except Exception: pass, which silently dropped broken alphas from the
        result -- a genuine bug in alpha045 (window 2 against min_periods 3) was
        hidden that way, and the pool quietly shipped 70 factors instead of 71.
        Failures are surfaced here: strict raises, otherwise they are
        returned so the caller can record them.
        """
        results: dict[str, pd.DataFrame] = {}
        failures: dict[str, str] = {}
        for name in self.alpha_names():
            try:
                results[name] = getattr(self, name)()
            except Exception as e:  # noqa: BLE001 - reported, never swallowed
                if strict:
                    raise RuntimeError(f"{name} failed: {type(e).__name__}: {e}") from e
                failures[name] = f"{type(e).__name__}: {e}"
        self.last_failures = failures
        return results


def build_alpha101(panel: pd.DataFrame) -> Alpha101:
    """Construct the calculator from the canonical panel.

    The panel is long ``(date, asset)``; Alpha101 is defined on wide
    ``(n_days, n_assets)`` frames, so each field is unstacked once here and
    shared across all alphas.

    ``vwap`` and ``returns`` are passed explicitly rather than left to the
    class defaults — the defaults approximate vwap as ``(high+low+close)/3``
    and returns as ``close.pct_change()``, and the panel carries both fields
    for real. Using the panel's own values keeps the baseline on exactly the
    inputs every other method sees.
    """
    wide = {field: panel[field].unstack("asset") for field in
            ("open", "high", "low", "close", "volume", "vwap", "return_1d")}
    return Alpha101(
        open_=wide["open"], high=wide["high"], low=wide["low"],
        close=wide["close"], volume=wide["volume"],
        vwap=wide["vwap"], returns=wide["return_1d"],
    )
