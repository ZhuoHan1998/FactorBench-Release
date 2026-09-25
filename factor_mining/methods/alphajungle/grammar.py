"""The ``alphajungle-v1`` expression grammar (paper Appendix E, Table 2).

Alpha Jungle has no official implementation, so this is written from the paper
specification. Table 2 defines the complete operator set; every operator there
is *temporal* — it reads the current day or a trailing window for one asset.
There are no cross-sectional operators, so the whole grammar evaluates as
per-asset pandas operations along the date axis of a (date x asset) frame,
which vectorizes across assets.

Two representations are used:

- an :class:`Expr` tree, which the MCTS/FSA code manipulates;
- a functional string such as ``Div(Ma(close,5),Std(close,20))``, which is what
  ``factors.json`` carries and :func:`parse` reads back.

Naming note: Table 2 writes four unary operators as symbols (``-x``, ``|x|``,
``x^2``, ``1/x``). They need names to appear in an LLM-facing operator list and
in the string form, so they are called ``Neg``, ``Abs``, ``Square`` and ``Inv``
here. Every other name is the paper's.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

# Paper Section 2: "Raw features include daily open, high, low, close prices
# (OHLC), trading volume, and Volume-Weighted Average Price (VWAP)."
FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "vwap")

# Appendix G: "The parameter value that indicates the length of the lookback
# window (if applicable) must be within the {window_range} range." The range
# itself is a prompt slot the paper never fills in; see METHOD.md.
DEFAULT_WINDOW_RANGE: tuple[int, int] = (2, 60)


class InvalidExpressionError(ValueError):
    """An expression violates the grammar; the message is LLM-facing feedback."""


# --- operator table --------------------------------------------------------


@dataclass(frozen=True)
class OpSpec:
    """One row of Table 2."""

    name: str
    n_inputs: int
    n_params: int
    description: str
    # kind drives validation: 'window' params are lookback lengths and must sit
    # inside the allowed window range; 'plain' operators take no parameters.
    kind: Literal["plain", "window", "window_lag"]
    fn: Callable[..., pd.DataFrame]


def _roll(x: pd.DataFrame, t: int):
    return x.rolling(window=t, min_periods=t)


def _vari(x: pd.DataFrame, t: int) -> pd.DataFrame:
    # Table 2: Vari(x,t) = Std(x,t) / Ma(x,t)
    return _roll(x, t).std() / _roll(x, t).mean()


def _zscore(x: pd.DataFrame, t: int) -> pd.DataFrame:
    return (x - _roll(x, t).mean()) / _roll(x, t).std()


def _ts_rank(x: pd.DataFrame, t: int) -> pd.DataFrame:
    # "The ranking of x relative to its values over the past t days", scaled to
    # [0, 1] so the operator is dimensionless like the rest of the table.
    return _roll(x, t).rank(pct=True)


def _autocorr(x: pd.DataFrame, t: int, n: int) -> pd.DataFrame:
    # "The autocorrelation coefficient of x with a lag of n over the past t days."
    return _roll(x, t).corr(x.shift(n))


OPERATORS: dict[str, OpSpec] = {
    spec.name: spec
    for spec in [
        # Unary, no parameters.
        OpSpec("Neg", 1, 0, "The opposite value of x.", "plain", lambda x: -x),
        OpSpec("Abs", 1, 0, "The absolute value of x.", "plain", lambda x: x.abs()),
        OpSpec("Square", 1, 0, "The square value of x.", "plain", lambda x: x**2),
        OpSpec("Inv", 1, 0, "The inverse value of x.", "plain", lambda x: 1.0 / x),
        OpSpec("Sign", 1, 0, "The sign of x.", "plain", np.sign),
        OpSpec("Sin", 1, 0, "Sine of x.", "plain", np.sin),
        OpSpec("Cos", 1, 0, "Cosine of x.", "plain", np.cos),
        OpSpec("Tanh", 1, 0, "Hyperbolic tangent of x.", "plain", np.tanh),
        # log of a non-positive value is undefined; NaN rather than a complex
        # number or -inf, consistent with _finite() below.
        OpSpec("Log", 1, 0, "The natural logarithm of x.", "plain",
               lambda x: np.log(x.where(x > 0))),
        # Unary with one lookback parameter.
        OpSpec("Delay", 1, 1, "The value of x at t trading days prior.", "window",
               lambda x, t: x.shift(t)),
        OpSpec("Diff", 1, 1, "x minus its value t days prior.", "window",
               lambda x, t: x - x.shift(t)),
        OpSpec("Pct", 1, 1, "The rate of change of x relative to t days prior.", "window",
               lambda x, t: x / x.shift(t) - 1.0),
        OpSpec("Ma", 1, 1, "The mean value of x over the past t days.", "window",
               lambda x, t: _roll(x, t).mean()),
        OpSpec("Med", 1, 1, "The median value of x over the past t days.", "window",
               lambda x, t: _roll(x, t).median()),
        OpSpec("Sum", 1, 1, "The sum of x over the past t days.", "window",
               lambda x, t: _roll(x, t).sum()),
        OpSpec("Std", 1, 1, "The standard deviation of x over the past t days.", "window",
               lambda x, t: _roll(x, t).std()),
        OpSpec("Max", 1, 1, "The maximum value of x over the past t days.", "window",
               lambda x, t: _roll(x, t).max()),
        OpSpec("Min", 1, 1, "The minimum value of x over the past t days.", "window",
               lambda x, t: _roll(x, t).min()),
        OpSpec("Rank", 1, 1, "The ranking of x relative to its past t values.", "window",
               _ts_rank),
        OpSpec("Skew", 1, 1, "The skewness of x over the past t days.", "window",
               lambda x, t: _roll(x, t).skew()),
        OpSpec("Kurt", 1, 1, "The kurtosis of x over the past t days.", "window",
               lambda x, t: _roll(x, t).kurt()),
        OpSpec("Vari", 1, 1, "The variation of x over the past t days, Std/Ma.", "window",
               _vari),
        OpSpec("Zscore", 1, 1, "The z-score of x over the past t days.", "window",
               _zscore),
        OpSpec("Autocorr", 1, 2, "The autocorrelation of x with lag n over past t days.",
               "window_lag", _autocorr),
        # Binary.
        OpSpec("Add", 2, 0, "x plus y.", "plain", lambda x, y: x + y),
        OpSpec("Sub", 2, 0, "x minus y.", "plain", lambda x, y: x - y),
        OpSpec("Mul", 2, 0, "x times y.", "plain", lambda x, y: x * y),
        OpSpec("Div", 2, 0, "x divided by y.", "plain", lambda x, y: x / y),
        OpSpec("Greater", 2, 0, "1 if x > y else 0.", "plain",
               lambda x, y: (x > y).astype(float).where(x.notna() & y.notna())),
        OpSpec("Less", 2, 0, "1 if x < y else 0.", "plain",
               lambda x, y: (x < y).astype(float).where(x.notna() & y.notna())),
        OpSpec("Cov", 2, 1, "The covariance of x and y over the past t days.", "window",
               lambda x, y, t: _roll(x, t).cov(y)),
        OpSpec("Corr", 2, 1, "The Pearson correlation of x and y over the past t days.",
               "window", lambda x, y, t: _roll(x, t).corr(y)),
    ]
}


def operator_catalog() -> str:
    """The ``{available_operators}`` prompt slot (paper Appendix K)."""
    lines = []
    for spec in OPERATORS.values():
        sig = ", ".join(["x", "y"][: spec.n_inputs] + ["t", "n"][: spec.n_params])
        lines.append(f"- {spec.name}({sig}): {spec.description}")
    return "\n".join(lines)


# --- expression tree -------------------------------------------------------


@dataclass(frozen=True)
class Field:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Op:
    name: str
    inputs: tuple["Expr", ...]
    params: tuple[int, ...] = ()

    def __str__(self) -> str:
        args = [str(i) for i in self.inputs] + [str(p) for p in self.params]
        return f"{self.name}({','.join(args)})"


Expr = Field | Op


def walk(expr: Expr):
    """Yield every node in the tree, parents before children."""
    yield expr
    if isinstance(expr, Op):
        for child in expr.inputs:
            yield from walk(child)


def depth(expr: Expr) -> int:
    if isinstance(expr, Field):
        return 1
    return 1 + max(depth(c) for c in expr.inputs)


def operator_names(expr: Expr) -> list[str]:
    return [n.name for n in walk(expr) if isinstance(n, Op)]


# --- validation ------------------------------------------------------------


def validate(
    expr: Expr,
    window_range: tuple[int, int] = DEFAULT_WINDOW_RANGE,
    min_operators: int = 2,
) -> None:
    """Raise :class:`InvalidExpressionError` with LLM-facing feedback.

    Implements ``IsValid`` from Algorithm 1 line 15. The message is fed back to
    the LLM verbatim for iterative correction (line 16-17).
    """
    lo, hi = window_range
    for node in walk(expr):
        if isinstance(node, Field):
            if node.name not in FIELDS:
                raise InvalidExpressionError(
                    f"Unknown data field '{node.name}'. Available fields: "
                    f"{', '.join(FIELDS)}."
                )
            continue
        spec = OPERATORS.get(node.name)
        if spec is None:
            raise InvalidExpressionError(
                f"Unknown operator '{node.name}'. It must be one of the listed operators."
            )
        if len(node.inputs) != spec.n_inputs:
            raise InvalidExpressionError(
                f"Operator '{node.name}' takes {spec.n_inputs} input(s), got {len(node.inputs)}."
            )
        if len(node.params) != spec.n_params:
            raise InvalidExpressionError(
                f"Operator '{node.name}' takes {spec.n_params} parameter(s), "
                f"got {len(node.params)}."
            )
        for p in node.params:
            if int(p) != p or p < 1:
                raise InvalidExpressionError(
                    f"Parameter {p} of '{node.name}' must be a positive integer."
                )
        if spec.kind in ("window", "window_lag") and not (lo <= node.params[0] <= hi):
            raise InvalidExpressionError(
                f"Lookback window {node.params[0]} of '{node.name}' is outside the "
                f"allowed range [{lo}, {hi}]."
            )
        if spec.kind == "window_lag" and not (1 <= node.params[1] < node.params[0]):
            raise InvalidExpressionError(
                f"Lag {node.params[1]} of '{node.name}' must be at least 1 and smaller "
                f"than the window {node.params[0]}."
            )
    ops = operator_names(expr)
    # Appendix K: "at least two distinct operations ... Avoid creating overly
    # simplistic alphas."
    if len(set(ops)) < min_operators:
        raise InvalidExpressionError(
            f"The alpha uses only {len(set(ops))} distinct operator(s); it must use at "
            f"least {min_operators} to have sufficient complexity."
        )
    if not any(isinstance(n, Field) for n in walk(expr)):
        raise InvalidExpressionError("The alpha must reference at least one data field.")


# --- the LLM's operation-list form -----------------------------------------


def from_operations(operations: list[dict], arguments: dict[str, float]) -> Expr:
    """Compile the LLM's ``formula`` op-list into an :class:`Expr`.

    Appendix K (Figure 16) has the LLM emit a straight-line program: a list of
    ``{"name", "param", "input", "output"}`` dictionaries where ``input`` names
    raw fields or earlier outputs, and ``param`` names keys of ``arguments``.
    The final operation's output is the alpha. Parameters stay symbolic in the
    formula so one structure can be backtested under several argument sets
    (Appendix D).
    """
    if not operations:
        raise InvalidExpressionError("The 'formula' list is empty.")
    env: dict[str, Expr] = {name: Field(name) for name in FIELDS}
    last: Expr | None = None
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise InvalidExpressionError(f"Operation {i} is not a JSON object.")
        name = op.get("name")
        if not isinstance(name, str):
            raise InvalidExpressionError(f"Operation {i} is missing a string 'name'.")
        inputs = []
        for ref in op.get("input") or []:
            if not isinstance(ref, str):
                raise InvalidExpressionError(
                    f"Operation '{name}' has a non-string input {ref!r}; inputs must "
                    "name a data field or an earlier output, never a number."
                )
            if ref not in env:
                raise InvalidExpressionError(
                    f"Operation '{name}' refers to unknown input '{ref}'. It must be a "
                    f"data field ({', '.join(FIELDS)}) or the output of an earlier "
                    "operation."
                )
            inputs.append(env[ref])
        params = []
        for key in op.get("param") or []:
            if key not in arguments:
                raise InvalidExpressionError(
                    f"Operation '{name}' uses parameter '{key}', which has no value in "
                    "'arguments'."
                )
            value = arguments[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidExpressionError(
                    f"Parameter '{key}' must be a number, got {value!r}."
                )
            if float(value) != int(value):
                raise InvalidExpressionError(
                    f"Parameter '{key}' must be a whole number of days, got {value!r}."
                )
            params.append(int(value))
        node = Op(name, tuple(inputs), tuple(params))
        output = op.get("output")
        if not isinstance(output, str) or not output:
            raise InvalidExpressionError(f"Operation '{name}' is missing a string 'output'.")
        if output in FIELDS:
            raise InvalidExpressionError(
                f"Operation '{name}' writes to '{output}', which shadows a data field. "
                "Choose a different output name."
            )
        env[output] = node
        last = node
    assert last is not None
    return last


# --- string form -----------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:([A-Za-z_][A-Za-z_0-9]*)|(-?\d+)|([(),]))")


def parse(text: str, validate_expr: bool = False,
          window_range: tuple[int, int] = DEFAULT_WINDOW_RANGE) -> Expr:
    """Parse the functional string form back into an :class:`Expr`."""
    pos = 0

    def token() -> tuple[str, str] | None:
        nonlocal pos
        m = _TOKEN.match(text, pos)
        if m is None:
            return None
        pos = m.end()
        if m.group(1) is not None:
            return ("name", m.group(1))
        if m.group(2) is not None:
            return ("int", m.group(2))
        return ("punct", m.group(3))

    def expect(punct: str) -> None:
        t = token()
        if t != ("punct", punct):
            raise InvalidExpressionError(f"Expected '{punct}' at offset {pos} in {text!r}")

    def node() -> Expr:
        nonlocal pos
        t = token()
        if t is None:
            raise InvalidExpressionError(f"Unexpected end of expression in {text!r}")
        kind, value = t
        if kind == "int":
            raise InvalidExpressionError(
                f"Bare number {value!r} cannot be an operand; numbers are only "
                "operator parameters."
            )
        if kind != "name":
            raise InvalidExpressionError(f"Unexpected token {value!r} in {text!r}")
        save = pos
        nxt = token()
        if nxt != ("punct", "("):
            pos = save
            return Field(value)
        inputs: list[Expr] = []
        params: list[int] = []
        while True:
            save = pos
            t2 = token()
            if t2 is not None and t2[0] == "int":
                params.append(int(t2[1]))
            else:
                pos = save
                inputs.append(node())
            t3 = token()
            if t3 == ("punct", ")"):
                break
            if t3 != ("punct", ","):
                raise InvalidExpressionError(f"Expected ',' or ')' in {text!r}")
        return Op(value, tuple(inputs), tuple(params))

    expr = node()
    if _TOKEN.match(text, pos) is not None:
        raise InvalidExpressionError(f"Trailing characters after the expression in {text!r}")
    if validate_expr:
        validate(expr, window_range=window_range)
    return expr


# --- evaluation ------------------------------------------------------------


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    """Map +/-inf to NaN.

    The paper does not specify this. It is done because operators like ``Inv``
    and ``Div`` produce infinities on zero denominators, and an inf poisons
    every downstream aggregate (a single inf makes a whole day's mean and std
    inf, and silently zeroes a least-squares fit). NaN propagates the same
    "no value here" meaning that missing panel data already carries, and the
    metric layer drops NaN pairs. See METHOD.md.
    """
    return frame.replace([np.inf, -np.inf], np.nan)


def evaluate(expr: Expr, fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate to a (date x asset) frame given wide per-field frames."""
    if isinstance(expr, Field):
        try:
            return fields[expr.name]
        except KeyError as e:
            raise InvalidExpressionError(f"Unknown data field '{expr.name}'.") from e
    spec = OPERATORS.get(expr.name)
    if spec is None:
        raise InvalidExpressionError(f"Unknown operator '{expr.name}'.")
    args = [evaluate(i, fields) for i in expr.inputs]
    with np.errstate(all="ignore"):
        out = spec.fn(*args, *expr.params)
    if not isinstance(out, pd.DataFrame):  # numpy ufuncs on a frame keep the type
        out = pd.DataFrame(out, index=args[0].index, columns=args[0].columns)
    return _finite(out)


def wide_fields(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the canonical panel into one (date x asset) frame per raw field."""
    return {name: panel[name].unstack("asset").sort_index() for name in FIELDS}


def signal(expr: Expr, panel: pd.DataFrame) -> pd.Series:
    """Evaluate to a ``(date, asset)`` Series aligned to the panel index."""
    values = evaluate(expr, wide_fields(panel))
    out = values.stack(future_stack=True)
    out.index = out.index.set_names(["date", "asset"])
    return out.reindex(panel.index)


# --- root genes (Frequent Subtree Avoidance) -------------------------------


def abstract(expr: Expr) -> Expr:
    """The paper's ``Abs(.)``: drop concrete parameter values, keep structure.

    "For instance, Abs(Ma(vwap,20)) becomes Ma(vwap,t)." Parameters are
    replaced by a placeholder so ``Ma(vwap,20)`` and ``Ma(vwap,5)`` compare
    equal as structural motifs.
    """
    if isinstance(expr, Field):
        return expr
    return Op(expr.name, tuple(abstract(c) for c in expr.inputs), ())


def _param_placeholders(expr: Expr) -> str:
    """Render an abstracted tree with the paper's ``t`` placeholder."""
    if isinstance(expr, Field):
        return expr.name
    spec = OPERATORS.get(expr.name)
    n_params = spec.n_params if spec else 0
    args = [_param_placeholders(c) for c in expr.inputs] + ["t", "n"][:n_params]
    return f"{expr.name}({','.join(args)})"


def render_abstract(expr: Expr) -> str:
    return _param_placeholders(abstract(expr))


def root_genes(expr: Expr) -> set[Expr]:
    """Abstracted root genes of an alpha (the paper's ``G-bar(f)``).

    "We define a root gene as a subtree in an alpha's expression tree whose
    leaves are exclusively raw input features." Every subtree qualifies here
    because this grammar has no numeric leaves — constants only ever appear as
    operator parameters — so the root genes are all subtrees rooted at an
    operator, abstracted. Returned as trees (not strings) so FSA can test
    subtree containment structurally; use :func:`render_abstract` to display.
    """
    return {abstract(node) for node in walk(expr) if isinstance(node, Op)}


def contains_gene(haystack: Expr, needle: Expr) -> bool:
    """Is the abstracted ``needle`` a subtree of the abstracted ``haystack``?"""
    return needle in {abstract(n) for n in walk(haystack)}


def complexity(expr: Expr) -> int:
    """Node count; used only for reporting."""
    return sum(1 for _ in walk(expr))


def max_window(expr: Expr) -> int:
    """Longest lookback the expression needs, for warm-up accounting."""
    total = 0
    for node in walk(expr):
        if isinstance(node, Op) and node.params:
            total = max(total, int(node.params[0]))
    # Nested rolling windows compound; bound generously rather than exactly.
    return total * max(1, depth(expr) - 1) if total else 0


def is_constant(values: pd.DataFrame, tol: float = 1e-12) -> bool:
    """True if the signal has no cross-sectional dispersion on most days."""
    spread = values.std(axis=1, skipna=True)
    live = spread.notna()
    if not live.any():
        return True
    return bool((spread[live] <= tol).mean() > 0.5) or not math.isfinite(spread[live].mean())
