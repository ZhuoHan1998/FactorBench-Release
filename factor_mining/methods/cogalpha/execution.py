"""Sandboxed execution and the numerical-stability / leakage checks.

Appendix A.3: after the logical audits, "the code is executed in a restricted
sandbox. We evaluate numerical stability by detecting runtime errors, NaN
propagation, overflow/underflow, invalid logarithms, and unstable
normalizations", and then "a domain-specific leakage test to ensure that the
factor does not use future information".

Execution goes through a subprocess running the same ``__main__`` contract
``factor_bench.eval.executors`` uses for code factors, so an alpha that runs
here runs identically at evaluation time.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from factor_mining.methods.cogalpha.codegen import FactorCode, build_script

DEFAULT_TIMEOUT = 120
# Section 3.4 / B.4: "All alphas containing more than 30% NaN values ... are
# discarded."
MAX_NAN_FRACTION = 0.30
# Appendix A.3, "distinct values per day": a factor that is near-constant
# across the cross-section carries no rankable information.
MIN_DISTINCT_RATIO = 0.10
# Leakage is checked on a short tail of the panel to keep the extra execution
# cheap; a lookahead that only shows up outside this window would be exotic.
LEAKAGE_WINDOW = 300


class ExecutionError(RuntimeError):
    """The factor failed to run or failed a stability/leakage check."""


@dataclass
class ExecutionResult:
    signal: pd.Series
    nan_fraction: float
    distinct_ratio: float


def _write_panel(panel: pd.DataFrame, path: Path) -> None:
    frame = panel.copy()
    frame.index = frame.index.rename(["datetime", "instrument"])
    frame.to_hdf(path, key="data")


def run_script(script: str, panel_h5: Path, timeout: int = DEFAULT_TIMEOUT) -> pd.Series:
    """Execute one factor script against a prepared panel file."""
    with tempfile.TemporaryDirectory(prefix="cogalpha_exec_") as tmp:
        workspace = Path(tmp)
        (workspace / "factor.py").write_text(script)
        (workspace / "daily_pv.h5").symlink_to(panel_h5.resolve())
        try:
            proc = subprocess.run(
                [sys.executable, "factor.py"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise ExecutionError(f"execution timed out after {timeout}s") from e
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()
            raise ExecutionError(
                "the factor raised at runtime: " + (tail[-1] if tail else "unknown error")
            )
        result = workspace / "result.h5"
        if not result.exists():
            raise ExecutionError("the factor produced no output")
        signal = pd.read_hdf(result)
    if isinstance(signal, pd.DataFrame):
        signal = signal.iloc[:, 0]
    signal.index = signal.index.rename(["date", "asset"])
    return signal


def check_stability(signal: pd.Series) -> tuple[float, float]:
    """Numerical-stability audit. Raises with LLM-facing feedback."""
    values = pd.to_numeric(signal, errors="coerce")
    if np.isinf(values.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise ExecutionError(
            "the factor produces infinities; guard divisions and logarithms"
        )
    nan_fraction = float(values.isna().mean())
    if nan_fraction > MAX_NAN_FRACTION:
        raise ExecutionError(
            f"{nan_fraction:.0%} of the output is NaN (limit {MAX_NAN_FRACTION:.0%}); "
            "shorten the lookback windows or handle missing data"
        )
    wide = values.unstack("asset")
    per_day = wide.nunique(axis=1, dropna=True)
    live = wide.notna().sum(axis=1)
    ratio = float((per_day / live.where(live > 0)).mean(skipna=True))
    # Two conditions, because the ratio alone is universe-size dependent: on a
    # small cross-section a fully constant factor still scores 1/n.
    flat = float(per_day[live > 1].mean()) if (live > 1).any() else 0.0
    if flat <= 1.0:
        raise ExecutionError(
            "the factor takes a single value across all stocks on a typical day, "
            "so it cannot rank them"
        )
    if not np.isfinite(ratio) or ratio < MIN_DISTINCT_RATIO:
        raise ExecutionError(
            "the factor is near-constant across stocks on a typical day, so it "
            "cannot rank them; avoid transforms that collapse the cross-section"
        )
    return nan_fraction, ratio


@dataclass(frozen=True)
class LeakageFixture:
    """The two panels the leakage test compares, written once per run.

    Only the candidate's script changes between candidates, so the untampered
    and tampered panels are built and written to disk once and reused. Doing it
    per candidate cost two HDF5 writes each, and used fixed filenames in a
    shared directory, which made the test unsafe to run from several threads.
    """

    base_h5: Path
    tampered_h5: Path
    cut: pd.Timestamp


def build_leakage_fixture(panel: pd.DataFrame, scratch: Path) -> LeakageFixture:
    """Write the untampered/tampered panel pair the leakage test runs against."""
    dates = panel.index.get_level_values("date").unique().sort_values()
    window = dates[-LEAKAGE_WINDOW:] if len(dates) > LEAKAGE_WINDOW else dates
    subset = panel.loc[window[0]: window[-1]].copy()
    cut = window[int(len(window) * 2 / 3)]

    tampered = subset.copy()
    future = tampered.index.get_level_values("date") >= cut
    for column in ("open", "high", "low", "close"):
        if column in tampered.columns:
            tampered.loc[future, column] = tampered.loc[future, column] * 3.0
    if "volume" in tampered.columns:
        tampered.loc[future, "volume"] = tampered.loc[future, "volume"] * 7.0

    scratch.mkdir(parents=True, exist_ok=True)
    base_h5 = scratch / "leak_base.h5"
    tampered_h5 = scratch / "leak_tampered.h5"
    _write_panel(subset, base_h5)
    _write_panel(tampered, tampered_h5)
    return LeakageFixture(base_h5=base_h5, tampered_h5=tampered_h5, cut=cut)


def check_no_leakage(
    script: str,
    fixture: LeakageFixture,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    """Empirical leakage test: perturbing the future must not move the past.

    The static scan in ``codegen.py`` catches the constructs the paper names,
    but only the ones it knows to look for. This runs the factor twice on a
    short window — once as-is, once with the final third of the prices
    multiplied — and requires the earlier output to be bit-identical. A factor
    that reads forward fails regardless of how the lookahead was written.
    """
    cut = fixture.cut
    base = run_script(script, fixture.base_h5, timeout)
    after = run_script(script, fixture.tampered_h5, timeout)

    past = base.index.get_level_values("date") < cut
    a = base[past].to_numpy(dtype="float64", na_value=np.nan)
    b = after.reindex(base.index)[past].to_numpy(dtype="float64", na_value=np.nan)
    if not np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True):
        differing = int((~np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)).sum())
        raise ExecutionError(
            f"temporal leakage: changing future prices altered {differing} past "
            "factor values, so the factor reads information it could not have had"
        )


class Sandbox:
    """Prepares the panel once and runs candidates against it."""

    def __init__(
        self,
        panel: pd.DataFrame,
        workdir: Path,
        timeout: int = DEFAULT_TIMEOUT,
        leakage_test: bool = True,
    ) -> None:
        self.panel = panel
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.leakage_test = leakage_test
        self.panel_h5 = self.workdir / "daily_pv.h5"
        if not self.panel_h5.exists():
            _write_panel(panel, self.panel_h5)
        # Built once, not per candidate: the pair depends only on the panel.
        self.leakage_fixture = (
            build_leakage_fixture(panel, self.workdir / "leakage")
            if leakage_test
            else None
        )

    def evaluate(self, factor: FactorCode) -> ExecutionResult:
        """Run, audit stability, then leakage. Raises :class:`ExecutionError`."""
        script = build_script(factor)
        signal = run_script(script, self.panel_h5, self.timeout)
        nan_fraction, distinct_ratio = check_stability(signal)
        if self.leakage_fixture is not None:
            check_no_leakage(script, self.leakage_fixture, self.timeout)
        return ExecutionResult(
            signal=signal, nan_fraction=nan_fraction, distinct_ratio=distinct_ratio
        )
