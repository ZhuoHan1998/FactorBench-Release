"""Audit factor *executability* — does every mined factor produce a usable signal?

This is deliberately separate from :mod:`factor_bench.eval.compare`, which
scores factors. Nothing here computes IC. The question is narrower and comes
first: can the factor be executed at all, and is what comes back a signal that
metrics could be computed on?

Why this exists as its own tool
-------------------------------
Methods are executed by different interpreters on purpose — each method's
factors run under the pandas/numpy pins their own code targets (see
:mod:`factor_bench.eval.executors`). That makes version skew a *silent*
correctness risk, not just a crash risk: RD-Agent's generated code uses
``pd.DataFrame(series, columns=[name])``, which builds the column under its own
pandas pin but returns an empty frame under pandas 3. Run under the wrong
interpreter, 18% of its factors crashed and the rest could not be trusted
either. So this audit reports, alongside per-factor status, *which interpreter
and which pandas version* executed each method. A benchmark that cannot say
that cannot defend its numbers.

Beyond pass/fail, a factor can "succeed" and still be unscorable: an all-NaN
signal, a signal that is constant across the cross-section on most dates
(cross-sectional IC is undefined when there is no spread), or one whose index
does not align with the market panel. Those are reported as columns rather
than folded into a single boolean.

Usage
-----
    # everything
    python -m factor_bench.eval.audit

    # one method, quick smoke over 20 factors
    python -m factor_bench.eval.audit --methods rdagent --limit 20

    # HPC: one method per job, resumable
    python -m factor_bench.eval.audit --methods rdagent \
        --out_csv out/audit_rdagent.csv --resume
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, load_panel
from factor_bench.eval import executors
from factor_mining.contracts import MiningRun

# Columns written per factor, in order.
FIELDNAMES = [
    "method", "market", "seed", "run", "factor", "form", "grammar",
    "status", "error_type", "error",
    "seconds",
    "assets_unknown", "dates_unknown",
    "window_rows", "n_obs", "coverage",
    "train_dates", "train_scorable", "frac_train_scorable",
    "valid_dates", "valid_scorable", "frac_valid_scorable",
    "test_dates", "test_scorable", "frac_test_scorable",
    "n_inf", "all_nan", "constant",
]


@dataclass
class FactorAudit:
    """One row of the audit: the outcome of executing a single factor."""

    method: str
    market: str
    seed: int
    run: str
    factor: str
    form: str
    grammar: str = ""
    status: str = "ok"
    error_type: str = ""
    error: str = ""
    seconds: float = float("nan")
    assets_unknown: int = 0
    dates_unknown: int = 0
    window_rows: int = 0
    n_obs: int = 0
    coverage: float = float("nan")
    train_dates: int = 0
    train_scorable: int = 0
    frac_train_scorable: float = float("nan")
    valid_dates: int = 0
    valid_scorable: int = 0
    frac_valid_scorable: float = float("nan")
    test_dates: int = 0
    test_scorable: int = 0
    frac_test_scorable: float = float("nan")
    n_inf: int = 0
    all_nan: bool = False
    constant: bool = False


# Map each method to the interpreter its factors execute under. Mirrors the
# dispatch in executors.compute_signal; kept here so the audit can report the
# environment without importing a method's internals.
def _interpreter_for(method: str) -> tuple[Path | None, str]:
    """Return (interpreter path, note). None means this process."""
    table: dict[str, tuple[Path | None, str]] = {
        "rdagent": (executors._RDAGENT_PYTHON, "code factors, RD-Agent script contract"),
        "alphaagent": (executors._ALPHAAGENT_PYTHON, "official DSL parser + function_lib"),
        "quantaalpha": (executors._QUANTAALPHA_PYTHON, "official DSL parser"),
        "alphasage": (executors._ALPHASAGE_PYTHON, "GFlowNet stack pins numpy 1.26"),
    }
    if method in table:
        return table[method]
    return None, "in-process (this interpreter)"


def _pandas_version(python_bin: Path | None) -> tuple[str, bool]:
    """pandas version reported by an interpreter, and whether it is usable."""
    if python_bin is None:
        return pd.__version__, True
    if not python_bin.exists():
        return "", False
    try:
        out = subprocess.run(
            [str(python_bin), "-c", "import pandas; print(pandas.__version__)"],
            capture_output=True, text=True, timeout=120, check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        return f"<error: {type(e).__name__}>", False
    return out.stdout.strip(), True


def executor_environments(methods: list[str]) -> pd.DataFrame:
    """Report the interpreter + pandas version behind each method's factors."""
    rows = []
    for m in sorted(methods):
        python_bin, note = _interpreter_for(m)
        version, available = _pandas_version(python_bin)
        rows.append({
            "method": m,
            "interpreter": str(python_bin) if python_bin else sys.executable,
            "pandas": version or "<missing>",
            "available": available,
            "note": note,
        })
    return pd.DataFrame(rows)


def _signal_diagnostics(signal: pd.Series, panel: pd.DataFrame, row: FactorAudit) -> None:
    """Fill the signal-quality fields of ``row`` in place.

    Everything is measured over the *evaluable window* only — the split's train
    start through test end. Executors legitimately return nothing outside it:
    the panel carries a warm-up buffer before train and a tail buffer after
    test (see :mod:`factor_bench.data.splits`), so counting those as missing
    data would flag correct behaviour as a defect.

    Misalignment is measured as unknown *labels* — assets or dates the market
    panel does not contain — rather than as row-count mismatch. Executors may
    return a dense (date x asset) grid while the panel is sparse, which is
    benign: reindexing drops the untraded pairs.
    """
    win_start = pd.Timestamp(DEFAULT_SPLIT.train[0])
    win_end = pd.Timestamp(DEFAULT_SPLIT.test[1])

    panel_dates = panel.index.get_level_values("date")
    window_index = panel.index[(panel_dates >= win_start) & (panel_dates <= win_end)]
    row.window_rows = len(window_index)

    # Real misalignment: labels the panel has never heard of.
    row.assets_unknown = int(
        len(signal.index.get_level_values("asset").unique().difference(
            panel.index.get_level_values("asset").unique()))
    )
    sig_dates = signal.index.get_level_values("date").unique()
    in_window = sig_dates[(sig_dates >= win_start) & (sig_dates <= win_end)]
    row.dates_unknown = int(len(in_window.difference(panel_dates.unique())))

    values = pd.to_numeric(signal.reindex(window_index), errors="coerce")
    numeric = values.to_numpy(dtype="float64", na_value=np.nan)
    row.n_inf = int(np.isinf(numeric).sum())
    row.n_obs = int(np.isfinite(numeric).sum())
    row.coverage = row.n_obs / row.window_rows if row.window_rows else float("nan")
    row.all_nan = row.n_obs == 0
    if row.all_nan:
        return

    # Cross-sectional IC needs, per date, at least two observations AND some
    # spread — a constant cross-section gives an undefined correlation.
    stats = values.groupby(level="date").agg(["count", "std"])
    scorable = (stats["count"] >= 2) & (stats["std"] > 0)
    total_scorable = int(scorable.sum())
    row.constant = total_scorable == 0

    for seg in ("train", "valid", "test"):
        start, end = DEFAULT_SPLIT.segment(seg)
        mask = (stats.index >= pd.Timestamp(start)) & (stats.index <= pd.Timestamp(end))
        n_dates = int(mask.sum())
        n_ok = int((scorable & mask).sum())
        setattr(row, f"{seg}_dates", n_dates)
        setattr(row, f"{seg}_scorable", n_ok)
        setattr(row, f"frac_{seg}_scorable", n_ok / n_dates if n_dates else float("nan"))


def audit_factor(factor, run: MiningRun, run_name: str, panel: pd.DataFrame) -> FactorAudit:
    """Execute one factor and describe the result. Never raises."""
    row = FactorAudit(
        method=run.method, market=run.market, seed=run.seed, run=run_name,
        factor=factor.name, form=factor.type,
        grammar=getattr(factor, "grammar", "") or getattr(factor, "entry_point", ""),
    )
    started = time.perf_counter()
    try:
        signal = executors.compute_signal(factor, run.market, panel)
        if not isinstance(signal, pd.Series):
            raise TypeError(f"executor returned {type(signal).__name__}, expected Series")
        _signal_diagnostics(signal, panel, row)
    except Exception as e:  # noqa: BLE001
        # Recording every failure mode IS the purpose of this tool, so the broad
        # catch is deliberate. Nothing is hidden: the type, the message and the
        # originating frame all land in the output, and the summary counts them.
        row.status = "fail"
        row.error_type = type(e).__name__
        tb = traceback.extract_tb(e.__traceback__)
        where = f" @ {tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}" if tb else ""
        row.error = f"{str(e)[:300]}{where}".replace("\n", " ⏎ ")
    row.seconds = round(time.perf_counter() - started, 3)
    return row


def _discover(methods, markets, seeds, mining_dir: Path) -> list[tuple[Path, MiningRun]]:
    runs = []
    for path in sorted(mining_dir.glob("*/*/factors.json")):
        try:
            run = MiningRun.load(path)
        except Exception as e:  # noqa: BLE001 - a malformed contract file is a finding
            print(f"[audit] SKIP unreadable {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if methods and run.method not in methods:
            continue
        if markets and run.market not in markets:
            continue
        if seeds is not None and run.seed not in seeds:
            continue
        runs.append((path, run))
    return runs


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def main(
    methods: str | None = None,
    markets: str | None = None,
    seeds: str | None = None,
    limit: int | None = None,
    out_csv: str = "out/factor_audit.csv",
    mining_dir: str = "out/mining",
    resume: bool = False,
) -> None:
    """Execute every mined factor and report whether it yields a usable signal.

    Args:
        methods: comma-separated method filter, e.g. ``"rdagent,alphagen"``.
        markets: comma-separated market filter.
        seeds: comma-separated seed filter.
        limit: audit at most this many factors per run (smoke testing).
        out_csv: rows are appended here as they complete, so a killed job keeps
            its partial results.
        resume: skip factors already present in ``out_csv``.
    """
    method_filter = _as_list(methods)
    market_filter = _as_list(markets)
    seed_filter = {int(s) for s in _as_list(seeds)} if seeds is not None else None

    runs = _discover(method_filter, market_filter, seed_filter, Path(mining_dir))
    if not runs:
        raise SystemExit(f"No runs matched under {mining_dir}")

    present = {m for _, r in runs for m in [r.method]}
    env = executor_environments(sorted(present))
    print("[audit] executor environments")
    print(env.to_string(index=False))
    missing = env.loc[~env["available"], "method"].tolist()
    if missing:
        print(
            f"[audit] WARNING: no usable interpreter for {missing}; "
            "those factors will be reported as failures, not skipped.",
            file=sys.stderr,
        )
    print()

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if resume and out_path.exists():
        prior = pd.read_csv(out_path)
        done = set(zip(prior["run"].astype(str), prior["factor"].astype(str), strict=False))
        print(f"[audit] resume: {len(done)} factors already audited in {out_path}")

    write_header = not (resume and out_path.exists())
    handle = out_path.open("a" if resume else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    panels: dict[str, pd.DataFrame] = {}
    rows: list[FactorAudit] = []
    started = time.perf_counter()
    try:
        for path, run in runs:
            run_name = path.parent.name
            factors = run.factors[:limit] if limit else run.factors
            pending = [f for f in factors if (run_name, f.name) not in done]
            if not pending:
                continue
            if run.market not in panels:
                panels[run.market] = load_panel(run.market)
            panel = panels[run.market]
            for factor in pending:
                row = audit_factor(factor, run, run_name, panel)
                rows.append(row)
                writer.writerow({k: v for k, v in asdict(row).items() if k in FIELDNAMES})
                handle.flush()
            n_fail = sum(1 for r in rows if r.run == run_name and r.status == "fail")
            flag = f"  {n_fail} FAILED" if n_fail else ""
            print(f"[audit] {run.method:12s} {run_name:28s} {len(pending):4d} factors{flag}")
    finally:
        handle.close()

    if not rows:
        print("[audit] nothing to do (all factors already audited)")
        return

    table = pd.DataFrame([asdict(r) for r in rows])[FIELDNAMES]
    elapsed = time.perf_counter() - started
    print(f"\n[audit] {len(table)} factors in {elapsed / 60:.1f} min -> {out_path}")
    print_summary(table)


def print_summary(table: pd.DataFrame) -> None:
    """Per-method rollup: failures first, then signals that ran but are unusable."""
    grouped = table.groupby("method")
    summary = pd.DataFrame({
        "factors": grouped.size(),
        "failed": grouped["status"].apply(lambda s: (s == "fail").sum()),
        "all_nan": grouped["all_nan"].sum(),
        "constant": grouped["constant"].sum(),
        "misaligned": grouped.apply(
            lambda g: int(((g["assets_unknown"] > 0) | (g["dates_unknown"] > 0)).sum()),
            include_groups=False,
        ),
        "median_coverage": grouped["coverage"].median().round(3),
        "med_train_scorable": grouped["frac_train_scorable"].median().round(3),
        "med_valid_scorable": grouped["frac_valid_scorable"].median().round(3),
        "med_test_scorable": grouped["frac_test_scorable"].median().round(3),
        "sec_per_factor": grouped["seconds"].median().round(2),
    })
    summary["fail_rate"] = (summary["failed"] / summary["factors"]).round(3)
    summary["unusable"] = summary["failed"] + summary["all_nan"] + summary["constant"]
    print("\n[audit] summary by method")
    print(summary.to_string())

    failures = table[table["status"] == "fail"]
    if not failures.empty:
        print(f"\n[audit] {len(failures)} failures by (method, error type)")
        modes = failures.groupby(["method", "error_type"]).size().sort_values(ascending=False)
        print(modes.to_string())
        print("\n[audit] failures concentrated in these runs")
        by_run = failures.groupby(["method", "run"]).size().sort_values(ascending=False)
        print(by_run.head(15).to_string())
        print("\n[audit] example messages")
        for (m, et), grp in failures.groupby(["method", "error_type"]):
            print(f"  {m}/{et}: {grp['error'].iloc[0][:160]}")


if __name__ == "__main__":
    fire.Fire(main)
