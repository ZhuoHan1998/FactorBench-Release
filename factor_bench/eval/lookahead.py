"""Look-ahead test: does any mined factor use data from after the date it is dated?

A factor's value on date ``t`` may depend only on data up to ``t``. So if every
row after a cut date ``T`` is replaced with noise, the factor's values on dates
``<= T`` must not move at all. Any change means the factor reads the future --
a forward shift, a whole-history normalisation, a centred window, a back-fill.
No pipeline check downstream can catch that, because the leaked signal looks
like a real one.

Procedure, per market and cut date ``T``:

1. Write a tampered copy of the panel: open, high, low, close and volume after
   ``T`` are each multiplied by independent noise in [0.5, 1.5); derived columns
   are recomputed from them. Rows up to ``T`` are verified bit-identical.
2. Execute every factor twice, in separate processes -- once on the real panel,
   once with ``FACTORBENCH_DATA`` pointing at the tampered copy. Separate
   processes are required: executors cache data per market in memory, and
   several methods load the panel themselves inside their own interpreters.
3. Compare the two signals on the last ``window`` trading days up to ``T``. A
   forward reference of ``k`` days changes the last ``k`` of them, and a
   whole-history statistic changes all of them, so this window catches both
   without storing full-history signals for thousands of factors.

Usage::

    python -m factor_bench.eval.lookahead run                     # every run
    python -m factor_bench.eval.lookahead run --methods rdagent --resume
    python -m factor_bench.eval.lookahead run --cuts 2022-12-30   # one cut

Results go to ``$FACTORBENCH_OUT/eval/lookahead.csv``: one row per (factor, cut)
with status ``clean``, ``lookahead``, ``failed`` (did not execute) or
``no_signal`` (nothing to compare in the window). The panel read is whatever
``FACTORBENCH_DATA`` names, so the same command checks static or point-in-time
data.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from factor_bench.data import load_panel
from factor_bench.data.loader import UNIVERSE_METADATA_KEY
from factor_bench.paths import DATA_DIR, EVAL_DIR, OUT_ROOT

#: One cut inside each segment (train / valid / test of the purged split).
DEFAULT_CUTS = ("2019-06-28", "2022-12-30", "2024-12-31")
#: Trading days compared before each cut. A forward reference of k <= WINDOW
#: days is caught on the last k of them; any longer one also changes them.
WINDOW = 60
NOISE_COLUMNS = ("open", "high", "low", "close", "volume")
# Relative + absolute tolerance. Unchanged inputs give bit-identical outputs in
# practice; the tolerance only absorbs float noise from a different summation
# order, which is orders of magnitude below any real dependence on noise of
# +-50%.
RTOL, ATOL = 1e-7, 1e-12

FIELDNAMES = ["method", "market", "seed", "run", "factor", "cut", "status",
              "n_dates", "n_changed_dates", "first_changed_date", "n_changed_values",
              "max_abs_diff", "nan_mismatch", "error"]


# ------------------------------------------------------------------ tampering
def recompute_derived(panel: pd.DataFrame) -> pd.DataFrame:
    """Derived columns from OHLCV, with the formulas data/download_all.py uses.

    Used only for rows after the cut, so the tampered future stays internally
    consistent (vwap, returns and adv follow the noisy prices).
    """
    out = panel.copy()
    close = out["close"].unstack("asset")
    volume = out["volume"].unstack("asset")

    def back(wide: pd.DataFrame) -> pd.Series:
        s = wide.stack()
        s.index = s.index.rename(["date", "asset"])
        return s.reindex(out.index)

    out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3
    out["return_1d"] = back(close.pct_change(1))
    out["forward_return_1d"] = back(close.pct_change(1).shift(-1))
    out["forward_return_5d"] = back(close.shift(-5) / close - 1)
    out["forward_return_20d"] = back(close.shift(-20) / close - 1)
    out["adv"] = back(volume.rolling(20, min_periods=5).mean())
    out["market_cap"] = out["close"] * out["adv"]
    return out


def tamper_panel(panel: pd.DataFrame, cut: str, seed: int = 0) -> pd.DataFrame:
    """The panel with every row after ``cut`` replaced by noise, past untouched.

    Each of open/high/low/close/volume gets its own noise, so a factor that
    peeks at a future *ratio* (close/open, high/low) is caught as surely as one
    that peeks at a level.
    """
    rng = np.random.default_rng(seed)
    after = (panel.index.get_level_values("date") > pd.Timestamp(cut))
    noisy = panel.copy()
    for col in NOISE_COLUMNS:
        noise = rng.uniform(0.5, 1.5, int(after.sum()))
        noisy.loc[after, col] = noisy.loc[after, col].to_numpy() * noise
    derived = recompute_derived(noisy)

    # Rows up to the cut keep the cached values exactly, whatever formula (or
    # pandas version) produced them; only rows after the cut are replaced.
    tampered = panel.copy()
    for col in panel.columns:
        if col in derived.columns and col != "tradable":
            tampered.loc[after, col] = derived.loc[after, col].to_numpy()

    from factor_bench.data.schema import FIELDS
    for col in FIELDS:
        a = panel.loc[~after, col].to_numpy(dtype="float64")
        b = tampered.loc[~after, col].to_numpy(dtype="float64")
        if not np.array_equal(a, b, equal_nan=True):
            raise AssertionError(f"tampering changed {col!r} on or before {cut}")
    return tampered


def write_tampered(src: Path, dst: Path, cut: str, seed: int) -> None:
    """Write a tampered copy of one panel file, keeping its universe tag."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    market = src.stem
    panel = load_panel(market, src.parent)
    tampered = tamper_panel(panel, cut, seed)
    table = pa.Table.from_pandas(tampered)
    tag = (pq.read_schema(src).metadata or {}).get(UNIVERSE_METADATA_KEY)
    if tag is not None:
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), UNIVERSE_METADATA_KEY: tag})
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)


# ------------------------------------------------------------------ comparison
def compare_window(real: pd.DataFrame, tampered: pd.DataFrame) -> dict:
    """Compare two (date x asset) signal windows. Pure; used by run and tests."""
    tampered = tampered.reindex(index=real.index, columns=real.columns)
    a = real.to_numpy(dtype="float64")
    b = tampered.to_numpy(dtype="float64")
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    nan_mismatch = nan_a != nan_b
    both = ~nan_a & ~nan_b
    diff = np.zeros_like(a)
    diff[both] = np.abs(a[both] - b[both])
    changed = nan_mismatch | (both & (diff > ATOL + RTOL * np.abs(np.where(both, a, 0))))
    by_date = changed.any(axis=1)
    if not both.any() and not nan_mismatch.any():
        status = "no_signal"
    else:
        status = "lookahead" if changed.any() else "clean"
    return {
        "status": status,
        "n_dates": int(len(real.index)),
        "n_changed_dates": int(by_date.sum()),
        "first_changed_date": str(real.index[by_date][0].date()) if by_date.any() else "",
        "n_changed_values": int(changed.sum()),
        "max_abs_diff": float(diff.max()) if both.any() else float("nan"),
        "nan_mismatch": int(nan_mismatch.sum()),
    }


# ------------------------------------------------------------------ child job
def signals(spec: str) -> None:
    """Child process: execute one run's factors, save each cut's window.

    Reads the panel from ``FACTORBENCH_DATA`` as inherited from the parent, so
    the parent decides real vs tampered by environment alone.
    """
    from factor_bench.eval import executors
    from factor_mining.contracts import MiningRun

    job = json.loads(Path(spec).read_text())
    run = MiningRun.load(job["run_path"])
    panel = load_panel(run.market)
    calendar = panel.index.get_level_values("date").unique().sort_values()
    windows = {}
    for cut in job["cuts"]:
        upto = calendar[calendar <= pd.Timestamp(cut)]
        windows[cut] = upto[-job["window"]:]
    wanted = set(job["factors"])
    out: dict[str, object] = {}
    for factor in run.factors:
        if factor.name not in wanted:
            continue
        try:
            sig = executors.compute_signal(factor, run.market, panel)
            if isinstance(sig, pd.DataFrame):
                sig = sig.iloc[:, 0]
            wide = sig.unstack("asset")
            out[factor.name] = {cut: wide.reindex(index=dates) for cut, dates in windows.items()}
        except Exception as e:  # noqa: BLE001 - recorded per factor, reported as failed
            out[factor.name] = f"{type(e).__name__}: {str(e)[:200]}".replace("\n", " / ")
    pd.to_pickle(out, job["out"])


def _run_child(spec: dict, data_dir: Path | None, work: Path, tag: str) -> dict:
    """Run one signals job; ``data_dir`` None = the parent's own panels."""
    spec_path = work / f"{tag}.json"
    spec["out"] = str(work / f"{tag}.pkl")
    spec_path.write_text(json.dumps(spec))
    env = dict(os.environ)
    if data_dir is not None:
        env["FACTORBENCH_DATA"] = str(data_dir)
    proc = subprocess.run(
        [sys.executable, "-m", "factor_bench.eval.lookahead", "signals", str(spec_path)],
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{tag} child failed: {proc.stderr[-800:]}")
    result = pd.read_pickle(spec["out"])
    Path(spec["out"]).unlink()
    spec_path.unlink()
    return result


# ------------------------------------------------------------------ CLI
def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def run(methods: str | None = None, markets: str | None = None,
        cuts: str | tuple = ",".join(DEFAULT_CUTS), window: int = WINDOW,
        mining_dir: str = "out/mining", out_csv: str | None = None,
        work_dir: str | None = None, limit: int | None = None,
        resume: bool = False, seed: int = 0) -> None:
    """Check every mined factor for look-ahead. See the module docstring."""
    from factor_bench.eval.audit import _discover

    cut_list = _as_list(cuts)
    runs = _discover(_as_list(methods), _as_list(markets), None, Path(mining_dir))
    if not runs:
        raise SystemExit(f"No runs matched under {mining_dir}")
    out_path = Path(out_csv) if out_csv else EVAL_DIR / "lookahead.csv"
    work = Path(work_dir) if work_dir else OUT_ROOT / "lookahead_work"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str, str]] = set()
    if resume and out_path.exists():
        prior = pd.read_csv(out_path, dtype=str)
        done = set(zip(prior["run"], prior["factor"], prior["cut"], strict=False))
    handle = out_path.open("a" if resume and out_path.exists() else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    if handle.tell() == 0:
        writer.writeheader()

    # Fresh tampered panels for every market involved: a reused directory could
    # hold derived h5 copies from an earlier noise draw.
    print(f"[lookahead] panels from {DATA_DIR}; cuts {cut_list}; window {window} days")
    tampered_dirs: dict[str, Path] = {}
    for cut in cut_list:
        d = work / f"tampered_{cut}"
        if d.exists():
            shutil.rmtree(d)
        tampered_dirs[cut] = d
        for m in sorted({r.market for _, r in runs}):
            write_tampered(DATA_DIR / f"{m}.parquet", d / f"{m}.parquet", cut, seed)
    jobs_dir = work / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    started, counts = time.perf_counter(), {}
    try:
        for path, mrun in runs:
            run_name = path.parent.name
            factors = mrun.factors[:limit] if limit else mrun.factors
            names = [f.name for f in factors
                     if any((run_name, f.name, c) not in done for c in cut_list)]
            if not names:
                continue
            spec = {"run_path": str(path), "factors": names, "cuts": cut_list,
                    "window": window}
            real = _run_child(dict(spec), None, jobs_dir, f"{run_name}_real")
            fake = {c: _run_child(dict(spec, cuts=[c]), tampered_dirs[c], jobs_dir,
                                  f"{run_name}_tampered_{c}") for c in cut_list}
            for name in names:
                for c in cut_list:
                    if (run_name, name, c) in done:
                        continue
                    row = {"method": mrun.method, "market": mrun.market, "seed": mrun.seed,
                           "run": run_name, "factor": name, "cut": c, "error": ""}
                    r, t = real.get(name), fake[c].get(name)
                    if isinstance(r, str) or isinstance(t, str) or r is None or t is None:
                        row.update(status="failed", error=r if isinstance(r, str) else str(t))
                    else:
                        row.update(compare_window(r[c], t[c]))
                    writer.writerow(row)
                    counts[(mrun.method, row["status"])] = \
                        counts.get((mrun.method, row["status"]), 0) + 1
            handle.flush()
            flagged = counts.get((mrun.method, "lookahead"), 0)
            print(f"[lookahead] {mrun.method:12s} {run_name:28s} {len(names):4d} factors"
                  f"{'  ' + str(flagged) + ' LOOKAHEAD so far' if flagged else ''}")
    finally:
        handle.close()

    if counts:
        summary = (pd.Series(counts).unstack(fill_value=0)
                   .reindex(columns=["clean", "lookahead", "failed", "no_signal"],
                            fill_value=0))
        print(f"\n[lookahead] {time.perf_counter() - started:.0f}s -> {out_path}")
        print(summary.to_string())


if __name__ == "__main__":
    fire.Fire({"run": run, "signals": signals})
