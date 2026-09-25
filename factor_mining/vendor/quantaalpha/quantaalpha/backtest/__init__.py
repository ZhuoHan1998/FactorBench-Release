"""
Backtest V2: Qlib official factor sets (alpha158/alpha360), custom factor JSON, LLM-driven factor computation.
"""

from .factor_loader import FactorLoader
from .factor_calculator import FactorCalculator
from .runner import BacktestRunner

__version__ = "2.0.0"
__all__ = ["FactorLoader", "FactorCalculator", "BacktestRunner"]

