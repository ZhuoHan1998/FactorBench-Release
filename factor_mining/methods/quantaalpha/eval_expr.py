"""Evaluate one quantaalpha-v1 DSL expression against a daily_pv.h5 panel.

Runs inside .venv-quantaalpha (official pins + their extended function_lib)
and mirrors the official ``template.jinjia2`` execution exactly, so evaluation
semantics are the official ones. Invoked as a subprocess by factor_bench's
``quantaalpha-v1`` executor.

Usage:
    python -m factor_mining.methods.quantaalpha.eval_expr EXPR IN_H5 OUT_H5
"""

import sys

import numpy as np
import pandas as pd
from quantaalpha.factors.coder.expr_parser import (  # noqa: F401
    parse_expression,
    parse_symbol,
)
from quantaalpha.factors.coder.function_lib import *  # noqa: F401,F403


def main(expression: str, h5_path: str, out_path: str) -> None:
    df = pd.read_hdf(h5_path, key="data")
    # Identical steps to the official template.jinjia2:
    expr = parse_symbol(expression, df.columns)
    expr = parse_expression(expr)
    for col in df.columns:
        expr = expr.replace(col[1:], f"df['{col}']")
    result = eval(expr)  # noqa: S307 — official execution contract
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    pd.Series(result, index=df.index).astype(np.float64).to_hdf(out_path, key="data")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
