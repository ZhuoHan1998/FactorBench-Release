"""
Factor Regulator Module.

Provides:
- FactorRegulator: Factor duplication and complexity checking
- FactorConsistencyChecker: Semantic consistency between hypothesis, description, expression
- FactorQualityGate: Integrated quality gate
"""

from quantaalpha.factors.regulator.factor_regulator import FactorRegulator

# Optional: consistency checker (optional dependency)
try:
    from quantaalpha.factors.regulator.consistency_checker import (
        FactorConsistencyChecker,
        ConsistencyCheckResult,
        ComplexityChecker,
        RedundancyChecker,
        FactorQualityGate
    )
    CONSISTENCY_CHECKER_AVAILABLE = True
except ImportError:
    CONSISTENCY_CHECKER_AVAILABLE = False
    FactorConsistencyChecker = None
    ConsistencyCheckResult = None
    ComplexityChecker = None
    RedundancyChecker = None
    FactorQualityGate = None


__all__ = [
    'FactorRegulator',
    'FactorConsistencyChecker',
    'ConsistencyCheckResult',
    'ComplexityChecker',
    'RedundancyChecker',
    'FactorQualityGate',
    'CONSISTENCY_CHECKER_AVAILABLE'
]
