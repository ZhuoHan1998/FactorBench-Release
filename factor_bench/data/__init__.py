"""Shared data contract: canonical panel, universes, split, and target.

Everything in FactorBench — every mining method and every evaluation —
consumes market data through this package, so that all methods see exactly
the same raw data, universe, chronological split, and prediction target.
"""

from factor_bench.data.loader import load_panel
from factor_bench.data.schema import FIELDS, MARKETS
from factor_bench.data.splits import DEFAULT_SPLIT, Split
from factor_bench.data.target import TARGET, TargetSpec

__all__ = [
    "load_panel",
    "FIELDS",
    "MARKETS",
    "DEFAULT_SPLIT",
    "Split",
    "TARGET",
    "TargetSpec",
]
