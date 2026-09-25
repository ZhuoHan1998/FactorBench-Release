#!/usr/bin/env python3
"""
Factor calculator - uses expression parser or LLM to compute factor values.
Supports: 1) direct expression parser; 2) LLM-generated code for complex factors.
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class FactorCalculator:
    """Factor calculator."""
    
    OPERATIONS_DOC = """
Only the following operations are allowed in expressions: 
### **Cross-sectional Functions**
- **RANK(A)**: Ranking of each element in the cross-sectional dimension of A.
- **ZSCORE(A)**: Z-score of each element in the cross-sectional dimension of A.
- **MEAN(A)**: Mean value of each element in the cross-sectional dimension of A.
- **STD(A)**: Standard deviation in the cross-sectional dimension of A.
- **MAX(A)**: Maximum value in the cross-sectional dimension of A.
- **MIN(A)**: Minimum value in the cross-sectional dimension of A.
- **MEDIAN(A)**: Median value in the cross-sectional dimension of A

### **Time-Series Functions**
- **DELTA(A, n)**: Change in value of A over n periods.
- **DELAY(A, n)**: Value of A delayed by n periods.
- **TS_MEAN(A, n)**: Mean value of sequence A over the past n days.
- **TS_SUM(A, n)**: Sum of sequence A over the past n days.
- **TS_RANK(A, n)**: Time-series rank of the last value of A in the past n days.
- **TS_ZSCORE(A, n)**: Z-score for each sequence in A over the past n days.
- **TS_MEDIAN(A, n)**: Median value of sequence A over the past n days.
- **TS_PCTCHANGE(A, p)**: Percentage change in the value of sequence A over p periods.
- **TS_MIN(A, n)**: Minimum value of A in the past n days.
- **TS_MAX(A, n)**: Maximum value of A in the past n days.
- **TS_ARGMAX(A, n)**: The index of the maximum value of A over the past n days.
- **TS_ARGMIN(A, n)**: The index of the minimum value of A over the past n days.
- **TS_QUANTILE(A, p, q)**: Rolling quantile of sequence A over the past p periods.
- **TS_STD(A, n)**: Standard deviation of sequence A over the past n days.
- **TS_VAR(A, p)**: Rolling variance of sequence A over the past p periods.
- **TS_CORR(A, B, n)**: Correlation coefficient between A and B over the past n days.
- **TS_COVARIANCE(A, B, n)**: Covariance between A and B over the past n days.
- **TS_MAD(A, n)**: Rolling Median Absolute Deviation of A over the past n days.

### **Moving Averages and Smoothing Functions**
- **SMA(A, n, m)**: Simple moving average of A over n periods with modifier m.
- **WMA(A, n)**: Weighted moving average of A over n periods.
- **EMA(A, n)**: Exponential moving average of A over n periods.
- **DECAYLINEAR(A, d)**: Linearly weighted moving average of A over d periods.

### **Mathematical Operations**
- **PROD(A, n)**: Product of values in A over the past n days.
- **LOG(A)**: Natural logarithm of each element in A.
- **SQRT(A)**: Square root of each element in A.
- **POW(A, n)**: Raise each element in A to the power of n.
- **SIGN(A)**: Sign of each element in A.
- **EXP(A)**: Exponential of each element in A.
- **ABS(A)**: Absolute value of A.
- **MAX(A, B)**: Maximum value between A and B.
- **MIN(A, B)**: Minimum value between A and B.
- **INV(A)**: Reciprocal of each element in A.
- **FLOOR(A)**: Floor of each element in A.

### **Conditional and Logical Functions**
- **COUNT(C, n)**: Count of samples satisfying condition C in the past n periods.
- **SUMIF(A, n, C)**: Sum of A over the past n periods if condition C is met.
- **FILTER(A, C)**: Filtering sequence A based on condition C.
- **(C1)&&(C2)**: Logical AND operation.
- **(C1)||(C2)**: Logical OR operation.
- **(C1)?(A):(B)**: Conditional expression.

### **Regression and Residual Functions**
- **SEQUENCE(n)**: A sequence from 1 to n.
- **REGBETA(A, B, n)**: Regression coefficient of A on B over the past n samples.
- **REGRESI(A, B, n)**: Residual of regression of A on B over the past n samples.

### **Technical Indicators**
- **RSI(A, n)**: Relative Strength Index of A over n periods.
- **MACD(A, short, long)**: Moving Average Convergence Divergence.
- **BB_MIDDLE(A, n)**: Middle Bollinger Band.
- **BB_UPPER(A, n)**: Upper Bollinger Band.
- **BB_LOWER(A, n)**: Lower Bollinger Band.
"""
    
    def __init__(self, config: Dict, data_df: Optional[pd.DataFrame] = None):
        """Args: config; data_df optional (loaded from qlib if not provided)."""
        self.config = config
        self.data_df = data_df
        self.llm_config = config.get('llm', {})
        self.calc_config = config.get('factor_calculation', {})
        
        self.cache_dir = Path(self.llm_config.get('cache_dir', './factor_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(self.calc_config.get('output_dir', './computed_factors'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def set_data(self, data_df: pd.DataFrame):
        """Set stock data."""
        self.data_df = data_df
        
    def calculate_factors(self, factors: List[Dict]) -> pd.DataFrame:
        """Compute factor values. factors: list of dicts with factor_name, factor_expression, etc."""
        if self.data_df is None:
            raise ValueError("Data not set; call set_data() or provide data_df in __init__")
        
        results = {}
        success_count = 0
        fail_count = 0
        
        for factor_info in factors:
            factor_name = factor_info.get('factor_name', 'unknown')
            factor_expr = factor_info.get('factor_expression', '')
            
            logger.info(f"  Computing factor: {factor_name}")
            
            try:
                if self.llm_config.get('cache_results', True):
                    cached_result = self._load_from_cache(factor_expr)
                    if cached_result is not None:
                        results[factor_name] = cached_result
                        success_count += 1
                        valid_count = cached_result.notna().sum()
                        total_count = len(cached_result)
                        logger.info(f"    From cache (valid: {valid_count}/{total_count})")
                        continue
                
                factor_value = self._calculate_with_parser(factor_expr)
                
                if factor_value is not None:
                    results[factor_name] = factor_value
                    success_count += 1
                    valid_count = factor_value.notna().sum()
                    total_count = len(factor_value)
                    logger.info(f"    OK (valid: {valid_count}/{total_count})")
                    if self.llm_config.get('cache_results', True):
                        self._save_to_cache(factor_expr, factor_value)
                else:
                    if self.llm_config.get('enabled', True):
                        factor_value = self._calculate_with_llm(factor_info)
                        if factor_value is not None:
                            results[factor_name] = factor_value
                            success_count += 1
                            logger.info(f"    LLM OK")
                        else:
                            fail_count += 1
                            logger.warning(f"    Failed")
                    else:
                        fail_count += 1
                        logger.warning(f"    Expression incompatible and LLM disabled")
                        
            except Exception as e:
                fail_count += 1
                logger.error(f"    Error: {str(e)}")
        
        logger.info(f"  Factor computation done: success {success_count}, failed {fail_count}")
        
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    def _calculate_with_parser(self, expr: str) -> Optional[pd.Series]:
        """Compute factor using expression parser."""
        try:
            from quantaalpha.factors.coder.expr_parser import (
                parse_expression, parse_symbol
            )
            import quantaalpha.factors.coder.function_lib as func_lib
            
            df = self.data_df.copy()
            
            parsed_expr = parse_symbol(expr, df.columns)
            parsed_expr = parse_expression(parsed_expr)
            
            for col in df.columns:
                if col.startswith('$'):
                    parsed_expr = parsed_expr.replace(col[1:], f"df['{col}']")
            
            exec_globals = {
                'df': df,
                'np': np,
                'pd': pd,
            }
            for name in dir(func_lib):
                if not name.startswith('_'):
                    exec_globals[name] = getattr(func_lib, name)
            
            result = eval(parsed_expr, exec_globals)
            
            if isinstance(result, pd.Series):
                return result
            elif isinstance(result, pd.DataFrame):
                return result.iloc[:, 0]
            else:
                return pd.Series(result, index=df.index)
                
        except Exception as e:
            logger.debug(f"Expression parse failed: {str(e)}")
            return None
    
    def _calculate_with_llm(self, factor_info: Dict) -> Optional[pd.Series]:
        """
        Compute factor using LLM-generated code. Returns factor series or None.
        """
        factor_name = factor_info.get('factor_name', 'unknown')
        factor_expr = factor_info.get('factor_expression', '')
        
        if self.llm_config.get('cache_results', True):
            cached_result = self._load_from_cache(factor_expr)
            if cached_result is not None:
                logger.debug(f"    Load from cache: {factor_name}")
                return cached_result
        
        try:
            code = self._generate_factor_code(factor_info)
            
            if code is None:
                return None
            
            result = self._execute_factor_code(code, factor_name)
            
            if result is not None and self.llm_config.get('cache_results', True):
                self._save_to_cache(factor_expr, result)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM computation failed: {str(e)}")
            return None
    
    def _generate_factor_code(self, factor_info: Dict) -> Optional[str]:
        """Generate factor computation code via LLM. Returns Python code string or None."""
        try:
            from quantaalpha.llm.client import APIBackend
        except ImportError:
            logger.error("Cannot import LLM module; ensure quantaalpha is installed")
            return None
        
        factor_name = factor_info.get('factor_name', 'unknown')
        factor_expr = factor_info.get('factor_expression', '')
        factor_desc = factor_info.get('factor_description', '')
        variables = factor_info.get('variables', {})
        
        system_prompt = f"""You are an expert quantitative analyst. Your task is to convert factor expressions into executable Python code.

The code should:
1. Use pandas DataFrame operations
2. Handle the datetime and instrument multi-index properly
3. Use the function library provided

{self.OPERATIONS_DOC}

The input data is a pandas DataFrame with multi-index (datetime, instrument) and columns: $open, $high, $low, $close, $volume, $vwap.

Please output ONLY the factor expression string that can be directly used with the expression parser. 
The expression should use $variable format (e.g., $close, $open, $volume).
Do NOT include any Python code, just the expression string.
"""

        user_prompt = f"""Convert this factor into an expression:

Factor Name: {factor_name}
Factor Expression: {factor_expr}
Factor Description: {factor_desc}
Variables: {json.dumps(variables, ensure_ascii=False)}

Please provide the corrected expression that uses only the allowed operations.
Output format: Just the expression string, nothing else.
"""

        try:
            api = APIBackend()
            response = api.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            expr = response.strip().strip('"\'')
            
            if self._validate_expression(expr):
                return expr
            else:
                logger.warning(f"LLM expression invalid: {expr}")
                return None
                
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return None
    
    def _validate_expression(self, expr: str) -> bool:
        """Validate expression."""
        if not expr:
            return False
        if '$' not in expr:
            return False
        if expr.count('(') != expr.count(')'):
            return False
        return True
    
    def _execute_factor_code(self, expr: str, factor_name: str) -> Optional[pd.Series]:
        """Execute factor computation code. Returns result series or None."""
        try:
            return self._calculate_with_parser(expr)
        except Exception as e:
            logger.error(f"Execute factor code failed: {str(e)}")
            return None
    
    def _get_cache_key(self, expr: str) -> str:
        """Cache key from expression."""
        return hashlib.md5(expr.encode()).hexdigest()
    
    def _load_from_cache(self, expr: str) -> Optional[pd.Series]:
        """Load factor from cache."""
        cache_key = self._get_cache_key(expr)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                return pd.read_pickle(cache_file)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, expr: str, result: pd.Series):
        """Save factor to cache."""
        cache_key = self._get_cache_key(expr)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            result.to_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Save to cache failed: {str(e)}")


class QlibDataProvider:
    """Qlib data provider."""
    
    def __init__(self, config: Dict):
        """Args: config dict."""
        self.config = config
        self.data_config = config.get('data', {})
        self._initialized = False
        
    def _init_qlib(self):
        """Initialize Qlib."""
        if self._initialized:
            return
            
        import qlib
        from qlib.config import REG_CN, REG_US
        
        provider_uri = self.data_config.get('provider_uri', '~/.qlib/qlib_data/cn_data')
        region_str = self.data_config.get('region', 'cn')
        region = REG_US if region_str == 'us' else REG_CN
        
        qlib.init(provider_uri=provider_uri, region=region)
        self._initialized = True
        logger.info(f"Qlib initialized: {provider_uri} (region={region_str})")
        
    def get_stock_data(self, 
                      start_time: Optional[str] = None,
                      end_time: Optional[str] = None,
                      instruments: Optional[str] = None) -> pd.DataFrame:
        """Get stock data. Args: start_time, end_time, instruments (market)."""
        self._init_qlib()
        
        from qlib.data import D
        
        start_time = start_time or self.data_config.get('start_time', '2016-01-01')
        end_time = end_time or self.data_config.get('end_time', '2025-12-31')
        instruments = instruments or self.data_config.get('market', 'csi300')
        
        stock_list = D.instruments(instruments)
        
        fields = ['$open', '$high', '$low', '$close', '$volume', '$vwap']
        
        df = D.features(
            stock_list,
            fields,
            start_time=start_time,
            end_time=end_time,
            freq='day'
        )
        
        df.columns = fields
        
        df['$return'] = df['$close'] / df.groupby('instrument')['$close'].shift(1) - 1
        
        logger.info(f"Loaded stock data: {len(df)} rows")
        
        return df

