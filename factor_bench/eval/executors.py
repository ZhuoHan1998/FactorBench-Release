"""Executors: turn factors.json entries into (date, asset) signal Series.

Both factor forms flow through here into the same metric path — this is the
shared evaluation contract that makes tree-based and code-based methods
directly comparable:

- ``expression`` factors in the ``alphagen-v1`` grammar are compiled and
  evaluated by the vendored official AlphaGen compiler (guaranteeing identical
  operator semantics to the method that produced them);
- ``expression`` factors in the ``alphajungle-v1`` grammar are evaluated
  in-process by that method's own pandas grammar (it is a from-paper
  implementation with no vendored package, so nothing collides);
- ``expression`` factors in the ``alphasage-v1`` grammar run in a subprocess
  inside ``.venv-alphasage`` (its GFlowNet stack pins numpy 1.26, and its
  ``alphagen`` fork renames the operators again);
- ``expression`` factors in the ``alphaforge-v1`` grammar run in a subprocess
  in *this* venv through AlphaForge's own vendored expression classes (its
  fork renames and extends AlphaGen's operators, so the two forks cannot share
  a process);
- ``expression`` factors in the ``alphaagent-v1`` DSL run in a subprocess
  inside ``.venv-alphaagent`` through the official parser + function_lib
  (their pandas-1.5 pins make in-process evaluation unsafe here);
- ``code`` factors run in a subprocess inside ``.venv-rdagent`` under the
  official RD-Agent script contract (read ``daily_pv.h5``, write
  ``result.h5``); their generated code targets that venv's pandas pin.

Evaluation never imports a method's search internals — only its expression
compiler / execution contract.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, load_panel
from factor_bench.data.schema import FIELDS

_CODE_TIMEOUT_SECONDS = 600

# Reused across factors of one run: expression-side tensor data per market,
# and the code-side daily_pv.h5 input file per market.
_EXPR_DATA_CACHE: dict[str, object] = {}
_H5_CACHE: dict[str, Path] = {}


class FactorExecutionError(RuntimeError):
    """A factor could not be evaluated; message carries the reason."""


def _alphagen_signal(expression: str, market: str, panel: pd.DataFrame) -> pd.Series:
    # Lazy imports: torch + the vendored official compiler are only needed
    # when expression factors are present.
    from factor_mining.methods.alphagen.data import PanelStockData
    from factor_mining.vendor import use_vendored

    use_vendored("alphagen")
    from alphagen.data.expression import (
        CSRank,
        Greater,
        Kurt,
        Less,
        Operators,
        Pow,
        Rank,
        Sign,
        Skew,
        Sub,
    )
    from alphagen.data.parser import ExpressionParser, ExpressionParsingError

    if market not in _EXPR_DATA_CACHE:
        # One window spanning all segments; buffers guaranteed by DEFAULT_SPLIT.
        _EXPR_DATA_CACHE[market] = PanelStockData(
            panel, DEFAULT_SPLIT.train[0], DEFAULT_SPLIT.test[1], market=market
        )
    data = _EXPR_DATA_CACHE[market]
    # Same alias mapping as the official scripts/rl.py build_parser. The extra
    # operators are defined upstream with identical semantics but excluded
    # from AlphaGen's own search list; AlphaCFG's grammar emits them.
    extra_operators = [Sign, CSRank, Pow, Skew, Kurt, Rank]
    parser = ExpressionParser(
        [*Operators, *extra_operators],
        ignore_case=True,
        additional_operator_mapping={"Max": [Greater], "Min": [Less], "Delta": [Sub]},
    )
    try:
        expr = parser.parse(expression)
    except ExpressionParsingError as e:
        # Otherwise one unparseable factor aborts an entire compare.py run
        # instead of being recorded in that factor's `error` column.
        raise FactorExecutionError(f"could not parse expression: {e}") from e
    frame = data.make_dataframe(expr.evaluate(data), columns=["signal"])
    signal = frame["signal"]
    signal.index = signal.index.rename(["date", "asset"])
    return signal


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RDAGENT_PYTHON = _REPO_ROOT / ".venv-rdagent" / "bin" / "python"
_ALPHAAGENT_PYTHON = _REPO_ROOT / ".venv-alphaagent" / "bin" / "python"
_QUANTAALPHA_PYTHON = _REPO_ROOT / ".venv-quantaalpha" / "bin" / "python"
_ALPHASAGE_PYTHON = _REPO_ROOT / ".venv-alphasage" / "bin" / "python"


def _alphajungle_signal(expression: str, panel: pd.DataFrame) -> pd.Series:
    """Evaluate an Alpha Jungle expression with its own operator table.

    Pure pandas over the canonical panel, so unlike the vendored-fork grammars
    this one is safe to run in-process.
    """
    from factor_mining.methods.alphajungle import grammar as aj

    try:
        expr = aj.parse(expression)
    except aj.InvalidExpressionError as e:
        raise FactorExecutionError(str(e)) from e
    return aj.signal(expr, panel)


def _alpha101_signal(name: str, market: str, panel: pd.DataFrame) -> pd.Series:
    """Evaluate one WorldQuant Alpha101 factor by name.

    Runs in-process: this is our own port, on this venv's pandas, with no
    third-party pins to honour. The calculator unstacks the whole panel once,
    so it is cached per market — rebuilding it per factor would repeat that
    unstack 71 times.

    Alpha101 is defined on wide (n_days x n_assets) frames; the result is
    stacked back and reindexed onto the panel's own (date, asset) index, which
    drops the untraded pairs the dense grid introduces.
    """
    # Import the library module only: factor_mining.methods.alpha101.run pulls in
    # the pydantic contract layer and fire, which evaluation has no need of.
    from factor_mining.methods.alpha101.alpha101 import Alpha101, build_alpha101

    key = f"alpha101:{market}"
    calculator = _EXPR_DATA_CACHE.get(key)
    if calculator is None:
        calculator = build_alpha101(panel)
        _EXPR_DATA_CACHE[key] = calculator
    assert isinstance(calculator, Alpha101)

    try:
        wide = calculator.compute(name)
    except KeyError as e:
        raise FactorExecutionError(str(e)) from e
    signal = wide.stack()
    signal.index = signal.index.rename(["date", "asset"])
    return signal.reindex(panel.index)


def _alphasage_signal(expression: str, market: str) -> pd.Series:
    """Evaluate an AlphaSAGE expression with AlphaSAGE's own operator set.

    Its venv is separate because the GFlowNet stack (torchgfn 1.2.1 +
    torch-geometric 2.6.1) pins numpy < 2, which the main environment cannot
    take; the fork collision on ``alphagen`` would force a subprocess anyway.
    """
    if not _ALPHASAGE_PYTHON.exists():
        raise FactorExecutionError(
            f"missing {_ALPHASAGE_PYTHON}; create the .venv-alphasage environment first"
        )
    return _subprocess_expression_signal(
        str(_ALPHASAGE_PYTHON),
        "factor_mining.methods.alphasage.eval_expr",
        expression,
        market,
    )


def _subprocess_expression_signal(
    python_bin: str, module: str, expression: str, market: str
) -> pd.Series:
    """Run one expression through a method's own evaluator, in its own process."""
    with tempfile.TemporaryDirectory(prefix="expr_eval_") as tmp:
        out_path = Path(tmp) / "result.h5"
        try:
            proc = subprocess.run(
                [python_bin, "-m", module, expression, market, str(out_path)],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=_CODE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as e:
            raise FactorExecutionError(f"timed out after {_CODE_TIMEOUT_SECONDS}s") from e
        if proc.returncode != 0:
            raise FactorExecutionError(f"exit {proc.returncode}: {proc.stderr[-500:]}")
        if not out_path.exists():
            raise FactorExecutionError("expression evaluation produced no result.h5")
        signal = pd.read_hdf(out_path)
    if isinstance(signal, pd.DataFrame):
        signal = signal.iloc[:, 0]
    signal.index = signal.index.rename(["date", "asset"])
    return signal


def _alphaforge_signal(expression: str, market: str) -> pd.Series:
    """Evaluate an AlphaForge expression with AlphaForge's own operator set.

    Runs in this repo's interpreter but in a fresh process: the vendored
    AlphaForge tree and the vendored AlphaGen tree both provide a top-level
    ``alphagen`` package, and whichever is imported first would silently
    supply its classes to the other.
    """
    with tempfile.TemporaryDirectory(prefix="alphaforge_eval_") as tmp:
        out_path = Path(tmp) / "result.h5"
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "factor_mining.methods.alphaforge.eval_expr",
                    expression,
                    market,
                    str(out_path),
                ],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=_CODE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as e:
            raise FactorExecutionError(f"timed out after {_CODE_TIMEOUT_SECONDS}s") from e
        if proc.returncode != 0:
            raise FactorExecutionError(f"exit {proc.returncode}: {proc.stderr[-500:]}")
        if not out_path.exists():
            raise FactorExecutionError("expression evaluation produced no result.h5")
        signal = pd.read_hdf(out_path)
    if isinstance(signal, pd.DataFrame):
        signal = signal.iloc[:, 0]
    signal.index = signal.index.rename(["date", "asset"])
    return signal


def _dsl_input_h5(method: str, market: str, python_path: Path) -> Path:
    """Panel in the method's official h5 format, shared with its mining runs.

    The file must be WRITTEN by the method's own venv: their older pandas pins
    cannot read HDF5 datetime indexes written by this repo's pandas 3."""
    key = f"{method}:{market}"
    if key not in _H5_CACHE:
        data_root = _REPO_ROOT / "out" / "mining" / method / "data"
        path = data_root / market / "full" / "daily_pv.h5"
        if not path.exists():
            prep = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    f"from factor_mining.methods.{method}.data_prep import "
                    f"prepare_data_folders; prepare_data_folders({market!r})",
                ],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=_CODE_TIMEOUT_SECONDS,
            )
            if prep.returncode != 0 or not path.exists():
                raise FactorExecutionError(
                    f"could not prepare {method} data for {market}: {prep.stderr[-300:]}"
                )
        _H5_CACHE[key] = path
    return _H5_CACHE[key]


def _dsl_subprocess_signal(method: str, expression: str, market: str,
                           python_path: Path) -> pd.Series:
    """Evaluate a method-DSL expression via its official function_lib inside
    its own venv (each method's eval_expr harness mirrors its official
    template execution exactly)."""
    if not python_path.exists():
        raise FactorExecutionError(f"missing {python_path}; install that venv first")
    h5 = _dsl_input_h5(method, market, python_path)
    with tempfile.TemporaryDirectory(prefix=f"{method}_eval_") as tmp:
        out_path = Path(tmp) / "result.h5"
        try:
            proc = subprocess.run(
                [
                    str(python_path),
                    "-m",
                    f"factor_mining.methods.{method}.eval_expr",
                    expression,
                    str(h5.resolve()),
                    str(out_path),
                ],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=_CODE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as e:
            raise FactorExecutionError(f"timed out after {_CODE_TIMEOUT_SECONDS}s") from e
        if proc.returncode != 0:
            raise FactorExecutionError(f"exit {proc.returncode}: {proc.stderr[-500:]}")
        if not out_path.exists():
            raise FactorExecutionError("expression evaluation produced no result.h5")
        signal = pd.read_hdf(out_path)
    if isinstance(signal, pd.DataFrame):
        signal = signal.iloc[:, 0]
    signal.index = signal.index.rename(["date", "asset"])
    return signal


def _code_input_h5(market: str, panel: pd.DataFrame, cache_dir: Path) -> Path:
    if market not in _H5_CACHE:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{market}_daily_pv.h5"
        if not path.exists():
            inputs = panel[list(FIELDS)].copy()
            inputs.index = inputs.index.rename(["datetime", "instrument"])
            inputs.to_hdf(path, key="data")
        _H5_CACHE[market] = path
    return _H5_CACHE[market]


def _code_signal(
    source: str,
    entry_point: str,
    market: str,
    panel: pd.DataFrame,
    python_bin: str | None = None,
    cache_dir: Path = Path("out/eval_cache"),
) -> pd.Series:
    if entry_point != "__main__":
        raise NotImplementedError(
            f"Only the '__main__' script contract is implemented; got {entry_point!r}"
        )
    # RD-Agent's generated factors are written against its own pandas pin. Running
    # them under a newer pandas does not just crash — some constructions change
    # meaning silently (e.g. pd.DataFrame(series, columns=[name]) yields an empty
    # frame when the name does not match, rather than building the column), which
    # would corrupt results rather than fail them. Never fall back to this
    # interpreter: a missing venv is an error, not a reason to use the wrong pandas.
    if python_bin is None:
        if not _RDAGENT_PYTHON.exists():
            raise FactorExecutionError(
                f"missing {_RDAGENT_PYTHON}; code factors must run under RD-Agent's "
                "own pandas pin — create that venv first"
            )
        python_bin = str(_RDAGENT_PYTHON)
    h5 = _code_input_h5(market, panel, cache_dir)
    with tempfile.TemporaryDirectory(prefix="factor_eval_") as tmp:
        workspace = Path(tmp)
        (workspace / "factor.py").write_text(source)
        (workspace / "daily_pv.h5").symlink_to(h5.resolve())
        try:
            subprocess.run(
                [python_bin, "factor.py"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=_CODE_TIMEOUT_SECONDS,
                check=True,
            )
        except subprocess.TimeoutExpired as e:
            raise FactorExecutionError(f"timed out after {_CODE_TIMEOUT_SECONDS}s") from e
        except subprocess.CalledProcessError as e:
            raise FactorExecutionError(f"exit {e.returncode}: {e.stderr[-500:]}") from e
        result_path = workspace / "result.h5"
        if not result_path.exists():
            raise FactorExecutionError("factor code produced no result.h5")
        frame = pd.read_hdf(result_path)
    # Validate the official contract explicitly. The previous code took
    # frame.iloc[:, 0] and renamed the index unconditionally, which turns a
    # contract violation into a *silently wrong signal*: a wide (date x
    # instrument) frame would yield one instrument's series, scored as if it
    # were a cross-sectional factor. Fail loudly with the observed shape
    # instead — a malformed result is a finding about the factor, not
    # something to guess our way past.
    if isinstance(frame, pd.Series):
        signal = frame
    elif frame.shape[1] == 1:
        signal = frame.iloc[:, 0]
    else:
        raise FactorExecutionError(
            f"result.h5 has {frame.shape[1]} columns {list(frame.columns)[:6]} "
            f"(shape {frame.shape}); the contract is a single signal column"
        )
    if signal.index.nlevels != 2:
        raise FactorExecutionError(
            f"result.h5 index has {signal.index.nlevels} level(s) "
            f"{list(signal.index.names)} (shape {signal.shape}); expected a "
            "2-level (datetime, instrument) MultiIndex"
        )
    signal.index = signal.index.rename(["date", "asset"])
    return signal


def compute_signal(factor, market: str, panel: pd.DataFrame | None = None) -> pd.Series:
    """Evaluate one contract factor (ExpressionFactor | CodeFactor) to a signal."""
    if panel is None:
        panel = load_panel(market)
    if factor.type == "expression":
        if factor.grammar == "alphagen-v1":
            return _alphagen_signal(factor.expression, market, panel)
        if factor.grammar == "alphajungle-v1":
            return _alphajungle_signal(factor.expression, panel)
        if factor.grammar == "alpha101-v1":
            return _alpha101_signal(factor.expression, market, panel)
        if factor.grammar == "alphasage-v1":
            return _alphasage_signal(factor.expression, market)
        if factor.grammar == "alphaforge-v1":
            return _alphaforge_signal(factor.expression, market)
        if factor.grammar == "alphaagent-v1":
            return _dsl_subprocess_signal("alphaagent", factor.expression, market,
                                          _ALPHAAGENT_PYTHON)
        if factor.grammar == "quantaalpha-v1":
            return _dsl_subprocess_signal("quantaalpha", factor.expression, market,
                                          _QUANTAALPHA_PYTHON)
        raise NotImplementedError(f"No executor registered for grammar {factor.grammar!r}")
    if factor.type == "code":
        return _code_signal(factor.source, factor.entry_point, market, panel)
    raise NotImplementedError(f"Unknown factor type {factor.type!r}")
