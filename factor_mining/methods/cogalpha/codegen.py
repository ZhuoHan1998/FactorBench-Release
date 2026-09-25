"""Parsing, static safety, and script assembly for LLM-written alpha code.

The generation prompts (Appendix C.1) require each factor to arrive wrapped in
``<<function N>> ... <</function N>>`` as a single Python function taking one
stock's time series and returning a Series named after the function. This
module recovers those functions, applies the deterministic half of the Code
Quality Agent plus the Static Safety leakage scan (Appendix A.3), and wraps a
factor in the ``__main__`` script contract that ``factor_bench.eval.executors``
already runs for code factors.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

FUNCTION_BLOCK = re.compile(
    r"<<\s*function\s*\d*\s*>>(.*?)<</\s*function\s*\d*\s*>>", re.S | re.I
)
CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

# Section 3.1: OHLCV only.
ALLOWED_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
# Appendix C.1 pre-imports, minus talib (unavailable here — see METHOD.md).
ALLOWED_NAMES: frozenset[str] = frozenset({"np", "pd", "stats", "math"})


class InvalidFactorCodeError(ValueError):
    """The generated code is unusable; the message is fed back to the LLM."""


@dataclass(frozen=True)
class FactorCode:
    """One generated factor function."""

    name: str
    source: str
    docstring: str

    def __hash__(self) -> int:  # dedupe by normalised source
        return hash(normalise(self.source))


def extract_functions(text: str) -> list[str]:
    """Pull each ``<<function N>>`` block out of a model response.

    Falls back to code fences and then to raw ``def`` scanning, because models
    drop the delimiters often enough that discarding the whole response would
    waste a generation.
    """
    blocks = [b.strip() for b in FUNCTION_BLOCK.findall(text)]
    if blocks:
        return [b for b in blocks if b]
    fenced = [f.strip() for f in CODE_FENCE.findall(text)]
    if fenced:
        return [f for f in fenced if "def " in f]
    if "def " in text:
        start = text.index("def ")
        return [text[start:].strip()]
    return []


def normalise(source: str) -> str:
    """Structural fingerprint used to deduplicate near-identical factors."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source.strip()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.name = "f"
            node.body = [
                n for n in node.body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
            ]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.dump(tree)


def _has_nested_loop(fn: ast.FunctionDef) -> bool:
    """Appendix C.1: "Nested loops are absolutely forbidden."""
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.While)):
            for inner in ast.walk(node):
                if inner is not node and isinstance(inner, (ast.For, ast.While)):
                    return True
    return False


def _negative_int(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)


def leakage_findings(fn: ast.FunctionDef) -> list[str]:
    """Static Safety scan (Appendix A.3, "Temporal Leakage Unit Test").

    Catches the forward-looking constructs the paper names — negative shifts,
    misaligned windows — plus the reversal trick that produces the same effect.
    An empirical check in ``execution.py`` backstops this; neither alone is
    sufficient.
    """
    findings: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name in ("shift", "diff", "pct_change", "tshift"):
            for arg in list(node.args) + [k.value for k in node.keywords
                                          if k.arg in ("periods", "lag")]:
                if _negative_int(arg):
                    findings.append(
                        f"`{name}()` is called with a negative period, which reads "
                        "future values."
                    )
        for keyword in node.keywords:
            if keyword.arg == "center" and getattr(keyword.value, "value", False) is True:
                findings.append(
                    "`center=True` centres a rolling window on the current row, so "
                    "it includes future observations."
                )
    # x[::-1] then a rolling op is a reversal-based lookahead.
    for node in ast.walk(fn):
        if isinstance(node, ast.Slice) and node.step is not None and _negative_int(node.step):
            findings.append(
                "A reversed slice (`[::-1]`) combined with a rolling operation reads "
                "future values."
            )
    return list(dict.fromkeys(findings))


def validate(source: str) -> FactorCode:
    """Deterministic Code Quality checks. Raises with LLM-facing feedback."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise InvalidFactorCodeError(f"The code does not parse: {e.msg} (line {e.lineno}).") from e

    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        raise InvalidFactorCodeError(
            f"Expected exactly one top-level function, found {len(functions)}."
        )
    fn = functions[0]
    if not fn.name.startswith("factor_"):
        raise InvalidFactorCodeError(
            f"The function name '{fn.name}' must start with 'factor_'."
        )
    if len(fn.args.args) != 1:
        raise InvalidFactorCodeError(
            f"The function must take exactly one argument (the DataFrame), "
            f"found {len(fn.args.args)}."
        )
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
        raise InvalidFactorCodeError(
            "Do not import anything; np, pd, stats and math are already available."
        )
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id in (
            "exec", "eval", "compile", "open", "__import__", "globals", "locals"
        ):
            raise InvalidFactorCodeError(f"Use of '{node.id}' is not allowed in a factor.")
        if isinstance(node, ast.While):
            test = node.test
            if isinstance(test, ast.Constant) and bool(test.value):
                raise InvalidFactorCodeError("`while True` (or any unbounded loop) is forbidden.")
    if _has_nested_loop(fn):
        raise InvalidFactorCodeError(
            "Nested loops are forbidden. Rewrite with vectorized pandas/numpy operations."
        )
    if (findings := leakage_findings(fn)):
        raise InvalidFactorCodeError(
            "The factor looks ahead in time: " + " ".join(findings)
        )
    docstring = ast.get_docstring(fn) or ""
    if not docstring:
        raise InvalidFactorCodeError("The function needs a docstring explaining its logic.")
    return FactorCode(name=fn.name, source=source.strip(), docstring=docstring)


# The RD-Agent-style ``__main__`` contract that factor_bench already executes:
# read daily_pv.h5 from the working directory, write result.h5.
SCRIPT_TEMPLATE = '''\
"""{name} — generated by CogAlpha.

{docstring}
"""

import math  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from scipy import stats  # noqa: F401

{source}


def main() -> None:
    df = pd.read_hdf("daily_pv.h5", key="data")
    df = df[[{fields}]].sort_index()
    # Appendix C.1: the function receives one stock's time series at a time.
    out = df.groupby(level=1, group_keys=False).apply({name})
    if isinstance(out, pd.DataFrame):
        out = out.iloc[:, 0]
    out = out.reindex(df.index)
    out.name = "{name}"
    out.to_hdf("result.h5", key="data")


if __name__ == "__main__":
    main()
'''


def build_script(factor: FactorCode) -> str:
    """Wrap a factor function in the executable ``__main__`` contract."""
    fields = ", ".join(f'"{f}"' for f in ALLOWED_FIELDS)
    return SCRIPT_TEMPLATE.format(
        name=factor.name,
        docstring=factor.docstring.replace('"""', "'''"),
        source=factor.source,
        fields=fields,
    )
