"""Run official AlphaForge stage 2 (dynamic factor combination) on a mined zoo.

Mirrors ``combine_AFF.py::main`` from the official repository (commit
d0cfc27). Every trading day from the first validation date onward, the zoo is
re-scored on a trailing window, the best ``n_factors`` are selected, an OLS
regression is re-fit on that window, and the next day's signal is predicted.
The window ends ``shift = 21`` trading days before the evaluation day so the
20-day forward target used to fit is fully realized — this is the official
lookahead guard and must not be tightened.

This is a **side artifact**: per-day weights cannot be represented in
``factors.json`` (which carries at most one static weight per factor), so
nothing in ``factor_bench`` consumes the output today. It is written next to
the mining run so the paper's headline result stays reproducible.

Outputs, in the same directory as ``factors.json``:

- ``combined_signal.parquet`` — the daily combined signal, indexed
  ``(date, asset)``, with a ``segment`` column (``valid`` / ``test``).
- ``combination_log.json`` — per date, the selected factor names and the
  fitted OLS coefficients (last entry is the intercept).

Usage:
    python -m factor_mining.methods.alphaforge.combine \
        --run out/mining/alphaforge/sp100_100_0/factors.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import MiningRun
from factor_mining.methods.alphaforge.data import PanelStockData
from factor_mining.methods.alphaforge.eval_expr import parse
from factor_mining.methods.alphaforge.vendored import use_alphaforge

use_alphaforge()

from alphagen.data.expression import Feature, Ref  # noqa: E402
from alphagen.utils.correlation import (  # noqa: E402
    batch_pearsonr,
    batch_ret,
    batch_spearmanr,
)
from alphagen_qlib.stock_data import FeatureType  # noqa: E402
from gan.utils.builder import exprs2tensor  # noqa: E402

# Official ``combine_AFF.py`` constant: the trailing window used to score and
# fit must end 21 trading days back so its 20-day forward target is realized.
SHIFT = 21


def chunk_batch_spearmanr(x, y, chunk_size=100):
    """Official ``combine_AFF.py::chunk_batch_spearmanr``."""
    spearmanr_list = []
    for i in range(0, len(x), chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i : i + chunk_size], y[i : i + chunk_size]))
    return torch.cat(spearmanr_list, dim=0)


def get_tensor_metrics_raw(x, y):
    """Official ``combine_AFF.py::get_tensor_metrics_raw`` — daily IC/RIC/ret."""
    ic_s = torch.nan_to_num(batch_pearsonr(x, y), nan=0)
    ric_s = torch.nan_to_num(chunk_batch_spearmanr(x, y, chunk_size=400), nan=0)
    ret_s = torch.nan_to_num(batch_ret(x, y), nan=0)
    return ic_s, ric_s, ret_s


def combine_alphaforge(
    run: str,
    n_factors: int = 10,
    window: float | int | str = "inf",
    device: str = "cpu",
    verbose: int = 1,
) -> Path:
    """Dynamically combine one mined zoo. Returns the combined-signal path."""
    t0 = time.time()
    run_path = Path(run)
    mining = MiningRun.load(run_path)
    if mining.method != "alphaforge":
        raise ValueError(f"{run_path} is a {mining.method!r} run, not alphaforge")
    if isinstance(window, str):
        if window != "inf":
            raise ValueError(f"window must be an int or 'inf', got {window!r}")
        window = float("inf")
    if n_factors > len(mining.factors):
        raise ValueError(
            f"n_factors={n_factors} exceeds the mined zoo size {len(mining.factors)}"
        )

    torch_device = torch.device(device)
    panel = load_panel(mining.market)
    TARGET.validate(panel)

    # One window spanning train..test, as in the official ``data_all``.
    data_all = PanelStockData(
        panel, DEFAULT_SPLIT.train[0], DEFAULT_SPLIT.test[1],
        device=torch_device, market=mining.market,
    )
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    # Official ordering: the zoo is sorted by |search score| before selection.
    factors = sorted(
        mining.factors,
        key=lambda f: abs(f.train_metrics.get("ic_abs", f.train_metrics.get("ic", 0.0))),
        reverse=True,
    )
    names = [f.name for f in factors]
    exprs = [parse(f.expression) for f in factors]

    fct_tensor = exprs2tensor(exprs, data_all, normalize=True)   # (n_days, n_stocks, n_exprs)
    tgt_tensor = exprs2tensor([target], data_all, normalize=False)

    # AlphaForge's own validity filter (train_AFF.py::get_metric) rejects
    # factors that are mostly non-finite, but only inspects the first day of
    # the training window. Over the full combination window some zoo factors do
    # go non-finite — e.g. ``(-10*low)**-10`` underflows to a denormal, whose
    # per-day std is ~0, so normalize_by_day divides by it and yields inf.
    # torch.linalg.lstsq silently returns an all-zero solution when the design
    # matrix holds inf/nan, which would turn the whole combined signal into a
    # constant. Apply the same finiteness requirement across the window instead
    # of letting the regression fail quietly.
    finite = torch.isfinite(fct_tensor).all(dim=1).all(dim=0)
    dropped = [names[i] for i in range(len(names)) if not bool(finite[i])]
    if dropped:
        print(f"[alphaforge/combine] dropping {len(dropped)} non-finite factor(s) "
              f"from the candidate pool: {', '.join(dropped)}")
        keep = [i for i in range(len(names)) if bool(finite[i])]
        if not keep:
            raise ValueError("every mined factor is non-finite over the combination window")
        names = [names[i] for i in keep]
        fct_tensor = fct_tensor[..., keep]
    n_factors = min(n_factors, len(names))

    ic_list, ric_list, ret_list = [], [], []
    for cur in range(fct_tensor.shape[-1]):
        ic_s, ric_s, ret_s = get_tensor_metrics_raw(fct_tensor[..., cur], tgt_tensor[..., 0])
        ic_list.append(ic_s)
        ric_list.append(ric_s)
        ret_list.append(ret_s)
    ic_s = torch.stack(ic_list, dim=-1)
    ric_s = torch.stack(ric_list, dim=-1)
    ret_s = torch.stack(ret_list, dim=-1)

    # Same slicing as StockData.make_dataframe: drop the warm-up head and the
    # forward-target tail so `dates` lines up row-for-row with the tensors.
    dates = data_all._dates[data_all.max_backtrack_days :]
    if data_all.max_future_days:
        dates = dates[: -data_all.max_future_days]
    assets = data_all._stock_ids
    n_valid = int(DEFAULT_SPLIT.mask(dates, "valid").sum())
    n_test = int(DEFAULT_SPLIT.mask(dates, "test").sum())

    pred_list: list[torch.Tensor] = []
    log: list[dict] = []
    ics, rics = [], []

    # Evaluate from the first day of the valid set until the last test day.
    for cur in range(len(fct_tensor) - n_test - n_valid, len(fct_tensor)):
        begin = 0 if not np.isfinite(window) else max(0, int(cur - window - SHIFT))

        cur_ic = ic_s[begin : cur - SHIFT]
        cur_ric = ric_s[begin : cur - SHIFT]
        cur_ret = ret_s[begin : cur - SHIFT]

        metrics = {
            "ic": cur_ic.mean(dim=0),
            "ic_std": cur_ic.std(dim=0),
            "ric": cur_ric.mean(dim=0),
            "ric_std": cur_ric.std(dim=0),
            "ret": cur_ret.mean(dim=0),
            "ret_std": cur_ret.std(dim=0),
        }
        metrics["icir"] = metrics["ic"] / metrics["ic_std"]
        metrics["ricir"] = metrics["ric"] / metrics["ric_std"]
        metrics["retir"] = metrics["ret"] / metrics["ret_std"]
        table = pd.DataFrame({k: v.detach().cpu().numpy() for k, v in metrics.items()})
        table = table.sort_values("ricir", ascending=False, key=lambda x: abs(x))

        # Official filter, then top-n by |RankICIR|; fall back to the single
        # best factor when nothing clears the bar.
        selected = table[(table["ric"] > 0.02) & (table["ricir"] > 0.2)]
        if len(selected) < 1:
            selected = table.iloc[:1]
        good_idx = selected.iloc[:n_factors].index.to_list()

        x = fct_tensor[begin : cur - SHIFT, :, good_idx]
        y = tgt_tensor[begin : cur - SHIFT]
        to_pred = fct_tensor[cur, :, good_idx]
        y_true = tgt_tensor[cur]
        y = y.reshape(-1, y.shape[-1])
        x = x.reshape(-1, x.shape[-1])

        to_select = torch.isfinite(y)[:, 0]
        y = y[to_select]
        x = x[to_select]
        to_pred = torch.nan_to_num(to_pred, nan=0)

        # add the constant term
        x = torch.cat([x, torch.ones_like(x[..., 0:1])], dim=-1)
        to_pred = torch.cat([to_pred, torch.ones_like(to_pred[..., 0:1])], dim=-1)

        coef = torch.linalg.lstsq(x, y).solution
        pred = to_pred @ coef

        ics.append(batch_pearsonr(pred.T, y_true.T)[0].item())
        rics.append(batch_spearmanr(pred.T, y_true.T)[0].item())
        pred_list.append(pred[:, 0])
        log.append({
            "date": str(dates[cur].date()),
            "factors": [names[i] for i in good_idx],
            "coef": [float(c) for c in coef[:, 0].detach().cpu().numpy()],
        })
        if verbose and len(pred_list) % 100 == 0:
            print(f"[alphaforge/combine] {len(pred_list)} days, "
                  f"ic:{np.nanmean(ics):.4f} ric:{np.nanmean(rics):.4f}")

    signal = torch.stack(pred_list, dim=0).detach().cpu().numpy()
    index = pd.MultiIndex.from_product(
        [dates[len(dates) - n_test - n_valid :], assets], names=["date", "asset"]
    )
    frame = pd.DataFrame({"signal": signal.reshape(-1)}, index=index)
    segment = np.where(
        DEFAULT_SPLIT.mask(frame.index.get_level_values("date"), "valid"), "valid", "test"
    )
    frame["segment"] = segment

    out_dir = run_path.parent
    signal_path = out_dir / "combined_signal.parquet"
    frame.to_parquet(signal_path)
    (out_dir / "combination_log.json").write_text(json.dumps({
        "run": str(run_path),
        "market": mining.market,
        "seed": mining.seed,
        "n_factors": n_factors,
        "dropped_non_finite": dropped,
        "window": "inf" if not np.isfinite(window) else int(window),
        "shift": SHIFT,
        "wall_clock_seconds": time.time() - t0,
        "daily": log,
    }, indent=2))

    if verbose:
        valid_mask = frame["segment"] == "valid"
        print(f"[alphaforge/combine] {len(pred_list)} days "
              f"({int(valid_mask.sum() / len(assets))} valid, "
              f"{len(pred_list) - int(valid_mask.sum() / len(assets))} test), "
              f"ic:{np.nanmean(ics):.4f} ric:{np.nanmean(rics):.4f} -> {signal_path}")
    return signal_path


if __name__ == "__main__":
    fire.Fire(combine_alphaforge)
