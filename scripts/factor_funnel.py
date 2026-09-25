#!/usr/bin/env python3
"""How many factors each method still has after each RQ's filter.

One table, so the funnel from harvested to deployed is visible in one place and
the denominators used by RQ1-RQ4 can be checked against each other.

The stages are nested by construction (see factor_bench.eval.admission):

    harvested            every factor in out/mining
    computes             produced a value (RQ0 gate 1)
    admitted             >=100 scorable dates in ALL three segments == RQ1
    RQ2                  admitted AND non-degenerate on the split
    RQ3                  admitted AND style-eligible
    RQ4                  median factors actually selected into a composite

RQ4 is deliberately a different kind of number. Selection runs per (method,
market, seed) run and picks a handful of mutually uncorrelated factors, so there
is no single "surviving set" -- the median pool size per run is what the
composites are actually built from.

Usage:
    .venv/bin/python scripts/factor_funnel.py
    .venv/bin/python scripts/factor_funnel.py --split test
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run as `python scripts/factor_funnel.py`, so sys.path[0] is scripts/ and the
# package is not importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_bench.paths import EVAL_DIR  # noqa: E402

SEGMENTS = ("train", "valid", "test")
ORDER = ["alpha101", "gp", "alphagen", "alphacfg", "alphaqcm", "alphaforge",
         "alphasage", "rdagent", "alphaagent", "quantaalpha"]


def main(split: str = "test") -> None:
    ic_path = EVAL_DIR / "ic_table.csv"
    if not ic_path.exists():
        raise SystemExit(f"no ic_table at {ic_path} -- is FACTORBENCH_OUT set?")
    t = pd.read_csv(ic_path)
    print(f"reading {EVAL_DIR}\n")

    nd = [f"{s}.n_dates" for s in SEGMENTS]
    if "admitted" in t.columns:
        t["_adm"] = t["admitted"].fillna(0).astype(bool)
    else:
        print("NOTE: ic_table predates the `admitted` column; deriving it\n")
        t["_adm"] = (t.status == "ok") & (t[nd] >= 100).all(axis=1)

    err = t["error"].astype(str)
    computes = ~((t.status != "ok") & (
        err.str.startswith("FactorExecutionError")
        | err.str.contains("could not parse|SyntaxError|IndentationError", regex=True)))

    out = pd.DataFrame({
        "harvested": t.groupby("method").size(),
        "computes": computes.groupby(t.method).sum(),
        "RQ1 admitted": t.groupby("method")["_adm"].sum(),
    })

    # ---- RQ2: in the correlation matrix, non-degenerate on this split -------
    adm = set(zip(t.loc[t._adm, "market"], t.loc[t._adm, "factor"]))
    rq2: dict[str, int] = {}
    for npz in sorted((EVAL_DIR / "redundancy").glob(f"*_{split}.npz")):
        mkt = npz.name.rsplit("_", 1)[0]
        z = np.load(npz, allow_pickle=False)
        ok = np.isfinite(np.diag(z["spearman"].astype(float)))
        for meth, name, good in zip(z["method"], z["name"], ok):
            if good and (mkt, name) in adm:
                rq2[meth] = rq2.get(meth, 0) + 1
    out["RQ2"] = pd.Series(rq2)

    # ---- RQ3: admitted AND style-eligible ----------------------------------
    sfiles = sorted(glob.glob(str(EVAL_DIR / "style" / f"*_{split}_style.csv")))
    if sfiles:
        S = pd.concat([pd.read_csv(p) for p in sfiles], ignore_index=True)
        elig = (S.valid_dates >= 100) & S.ic_res.notna()
        keyed = pd.Series(list(zip(S.market, S.factor)), index=S.index).isin(adm)
        out["RQ3"] = (elig & keyed).groupby(S.method).sum()

    # ---- RQ4: median factors per composite ---------------------------------
    pfiles = sorted(glob.glob(str(EVAL_DIR / "portfolio" / "*_summary.csv")))
    if pfiles:
        P = pd.concat([pd.read_csv(p) for p in pfiles], ignore_index=True)
        sel = P[(P.status == "ok") & (P.cap == "full") & (P.combo == "equal")]
        if "positive_ic_gate" in sel.columns:
            sel = sel[sel.positive_ic_gate == "off"]
        out["RQ4 median/run"] = sel.groupby("method")["n_selected"].median()
        out["RQ4 runs"] = sel.groupby("method")["run"].nunique()

    out = out.reindex([m for m in ORDER if m in out.index])
    out = out.fillna(0).astype(int)
    out["kept %"] = (out["RQ1 admitted"] / out["harvested"] * 100).round(1)
    total = out.sum(numeric_only=True)
    total["kept %"] = round(out["RQ1 admitted"].sum() / out["harvested"].sum() * 100, 1)
    out.loc["TOTAL"] = total

    print(f"factors surviving each stage  (split = {split})\n")
    print(out.to_string())
    print("\nharvested >= computes >= RQ1 >= RQ2, RQ3.  RQ4 is per-run, not a subset count.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    main(**vars(ap.parse_args()))
