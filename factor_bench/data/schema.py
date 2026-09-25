"""Canonical panel schema shared by all mining methods and evaluation."""

from __future__ import annotations

MARKETS: tuple[str, ...] = ("sp100", "hsi", "csi300", "ftse100", "nikkei225")

# Input fields available to factors. Order matters for tensor-based methods
# (adapters may rely on a stable field ordering).
FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "return_1d",
    "adv",
)

# Auxiliary columns present in the cached panels but not exposed as factor
# inputs (forward returns are targets — exposing them would be leakage).
AUX_COLUMNS: tuple[str, ...] = (
    "forward_return_1d",
    "forward_return_5d",
    "forward_return_20d",
    "market_cap",
    "tradable",
)

INDEX_NAMES: tuple[str, str] = ("date", "asset")
