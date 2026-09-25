"""Evaluate one ``alphaforge-v1`` expression against the shared panel.

Run as a subprocess by ``factor_bench.eval.executors``. AlphaForge ships its
own fork of the ``alphagen`` package whose operators differ from upstream's
(``ts_mean`` vs ``Mean``, plus ``Inv`` / ``S_log1p`` / ``ts_div`` /
``ts_pctchange``), so the two forks cannot share a process — hence a
subprocess rather than an in-process executor.

Reading an expression back is done the way the official ``combine_AFF.py``
does it: AlphaForge's ``Expression.__str__`` emits a Python-evaluable form
(infix binary operators, bare feature names, bare constants), and upstream
recovers the tree with ``eval`` under ``from alphagen.data.expression import *``
plus the feature singletons. The only change here is that the namespace is
restricted to those names instead of the whole module globals, which also
removes upstream's need to rewrite ``open`` -> ``open_``.

Usage:
    python -m factor_mining.methods.alphaforge.eval_expr <expression> <market> <out.h5>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, load_panel
from factor_mining.methods.alphaforge.data import PanelStockData
from factor_mining.methods.alphaforge.vendored import use_alphaforge

use_alphaforge()

from alphagen.data import expression as _expression  # noqa: E402
from alphagen_qlib.stock_data import FeatureType  # noqa: E402


def _eval_namespace() -> dict[str, object]:
    """Expression classes plus the feature singletons, and nothing else."""
    names: dict[str, object] = {
        name: obj
        for name, obj in vars(_expression).items()
        if isinstance(obj, type) and issubclass(obj, _expression.Expression)
    }
    for feature in FeatureType:
        names[feature.name.lower()] = _expression.Feature(feature)
    return names


def parse(expression: str):
    """Recover an AlphaForge ``Expression`` from its ``__str__`` form."""
    return eval(expression, {"__builtins__": {}}, _eval_namespace())  # noqa: S307


def compute(expression: str, market: str) -> pd.Series:
    panel = load_panel(market)
    # One window spanning all segments, matching the alphagen-v1 executor.
    data = PanelStockData(panel, DEFAULT_SPLIT.train[0], DEFAULT_SPLIT.test[1], market=market)
    expr = parse(expression)
    frame = data.make_dataframe(expr.evaluate(data), columns=["signal"])
    signal = frame["signal"]
    signal.index = signal.index.rename(["date", "asset"])
    return signal


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    expression, market, out_path = argv
    signal = compute(expression, market)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    signal.to_hdf(out_path, key="data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
