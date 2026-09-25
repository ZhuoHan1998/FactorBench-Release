"""Factor combination — put every factor on one scale, then weight it.

Normalisation is fixed for all methods: per date, convert to percentile rank
across stocks and subtract 0.5, so every factor contributes a bounded,
zero-centred, scale-free score regardless of its raw units. The training-fixed
direction is applied first.

Three weightings, applied to an identical selected pool:

- **equal**  w_k = 1/K. The main baseline: no fitted parameters at all.
- **ic**     w proportional to positive validation IC, summing to 1. Simplest
             challenger; its weakness is that validation IC is noisy and says
             nothing about what a factor adds *alongside* the others.
- **ridge**  coefficients fitted on train, penalty chosen on validation. The
             flexible challenger, and the only one that can assign negative
             weights -- allowed deliberately, and reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CombineConfig:
    method: str = "equal"                       # equal | ic | ridge
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def normalise_signal(sig: pd.Series) -> pd.Series:
    """Per-date percentile rank minus 0.5. Bounded in (-0.5, 0.5]."""
    r = sig.groupby(level="date").rank(pct=True)
    return r - 0.5


def _ridge_weights(Z: pd.DataFrame, y: pd.Series, split_mask_train: pd.Series,
                   split_mask_valid: pd.Series, alphas) -> tuple[np.ndarray, float]:
    """Ridge fitted on train, penalty chosen by validation IC.

    Weights are NOT constrained to be positive: a factor can legitimately earn a
    negative weight as a hedge against the others, and forbidding that would
    quietly change what the model is allowed to learn.
    """
    tr = split_mask_train & Z.notna().all(axis=1) & y.notna()
    va = split_mask_valid & Z.notna().all(axis=1) & y.notna()
    Xtr, ytr = Z[tr].to_numpy(float), y[tr].to_numpy(float)
    if Xtr.shape[0] < 50 or Xtr.shape[1] == 0:
        return np.full(Z.shape[1], 1.0 / max(Z.shape[1], 1)), float("nan")
    Xc = Xtr - Xtr.mean(0)
    yc = ytr - ytr.mean()
    G = Xc.T @ Xc
    b = Xc.T @ yc
    best, best_ic = None, -np.inf
    Xva, yva = Z[va].to_numpy(float), y[va].to_numpy(float)
    dates_va = Z[va].index.get_level_values("date")
    for a in alphas:
        w = np.linalg.solve(G + a * np.eye(G.shape[0]), b)
        if Xva.shape[0] < 50:
            ic = -np.inf
        else:
            pred = pd.Series(Xva @ w, index=dates_va)
            act = pd.Series(yva, index=dates_va)
            df = pd.DataFrame({"p": pred.to_numpy(), "a": act.to_numpy()}, index=dates_va)
            ic = df.groupby(level=0).apply(
                lambda g: g["p"].corr(g["a"]) if g["p"].std() > 0 else np.nan).mean()
        if np.isfinite(ic) and ic > best_ic:
            best, best_ic = w, ic
    if best is None:
        return np.full(Z.shape[1], 1.0 / Z.shape[1]), float("nan")
    scale = np.abs(best).sum()
    return (best / scale if scale > 0 else best), float(best_ic)


def build_composite(signals: dict[str, pd.Series], selected: pd.DataFrame,
                    config: CombineConfig = CombineConfig(),
                    target: pd.Series | None = None,
                    train_mask: pd.Series | None = None,
                    valid_mask: pd.Series | None = None,
                    ic_column: str = "valid.ic") -> tuple[pd.Series, pd.Series]:
    """Combine the selected factors into one composite signal.

    Returns (composite, weights). The composite is a plain (date, asset) Series
    on the union index of the normalised inputs.
    """
    names = selected["factor"].tolist()
    if not names:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    Z = pd.DataFrame({n: normalise_signal(signals[n]) for n in names if n in signals})
    if Z.empty or Z.shape[1] == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    names = list(Z.columns)

    if config.method == "equal":
        w = np.full(len(names), 1.0 / len(names))
    elif config.method == "ic":
        ic = selected.set_index("factor").loc[names, ic_column].clip(lower=0).to_numpy(float)
        w = ic / ic.sum() if ic.sum() > 0 else np.full(len(names), 1.0 / len(names))
    elif config.method == "ridge":
        if target is None or train_mask is None or valid_mask is None:
            raise ValueError("ridge needs target, train_mask and valid_mask")
        y = target.reindex(Z.index)
        w, _ = _ridge_weights(Z, y, train_mask.reindex(Z.index, fill_value=False),
                              valid_mask.reindex(Z.index, fill_value=False),
                              config.ridge_alphas)
    else:
        raise ValueError(f"unknown combination method {config.method!r}")

    composite = (Z * w).sum(axis=1, min_count=1)
    return composite, pd.Series(w, index=names)
