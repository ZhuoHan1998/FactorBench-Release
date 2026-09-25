"""Evaluate one ``alphasage-v1`` expression against the shared panel.

Run as a subprocess by ``factor_bench.eval.executors``. AlphaSAGE ships its own
fork of ``alphagen`` whose operator names differ from every other fork in this
repo (``TsMean`` where upstream has ``Mean`` and AlphaForge has ``ts_mean``;
plus ``SLog1p``, ``Inv``, ``TsDiv``, ``TsPctChange``, ``TsIr``,
``TsMinMaxDiff``, ``TsMaxDiff``, ``TsMinDiff``), so the forks cannot share a
process — hence a subprocess, run under this method's own venv.

Reading an expression back mirrors the official
``run_adaptive_combination.py::load_alpha_pool``, which recovers the tree with
``eval(s.replace('open', 'open_').replace('$', ''))``. AlphaSAGE's
``Expression.__str__`` emits a Python-evaluable functional form (``$close``
features, ``TsMean($close,10)`` operators, bare numeric constants). The only
change here is that the namespace is restricted to the expression classes plus
the six feature singletons, which also removes upstream's need to rewrite
``open`` -> ``open_``.

Usage:
    python -m factor_mining.methods.alphasage.eval_expr <expression> <market> <out.h5>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from factor_bench.data import DEFAULT_SPLIT, load_panel
from factor_mining.methods.alphasage.data import PanelStockData
from factor_mining.methods.alphasage.vendored import use_alphasage

use_alphasage()

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
    """Recover an AlphaSAGE ``Expression`` from its ``__str__`` form."""
    # Features render as ``$close``; strip the sigil so the text is valid
    # Python, exactly as the official loader does.
    text = expression.replace("$", "")
    return eval(text, {"__builtins__": {}}, _eval_namespace())  # noqa: S307


def compute(expression: str, market: str) -> pd.Series:
    panel = load_panel(market)
    # One window spanning all segments, matching the other expression executors.
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
