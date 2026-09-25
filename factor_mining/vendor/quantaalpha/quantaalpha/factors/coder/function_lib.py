import numpy as np
import pandas as pd
import operator
from joblib import Parallel, delayed


def datatype_adapter(func):
    def wrapper(*args):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            new_args = (pd.DataFrame(args[0]),)
            result = func(*new_args)
            return result
        if len(args) == 1 and isinstance(args[0], (float, int)):
            new_args = (pd.DataFrame([args[0]]),)
            result = func(*new_args)
            return float(result.iloc[0])
        if (len(args) == 2 and isinstance(args[0], np.ndarray) and not isinstance(args[1], np.ndarray)):
            new_args = (pd.DataFrame(args[0]), args[1])
            result = func(*new_args)
        elif (len(args) == 2 and isinstance(args[1], np.ndarray) and not isinstance(args[0], np.ndarray)):
            new_args = (args[0], pd.DataFrame(args[1]))
            result = func(*new_args)
        else:
            result = func(*args)
        return result

    return wrapper

@datatype_adapter
def DELTA(df:pd.DataFrame, p:int=1):
    return df.groupby('instrument').transform(lambda x: x.diff(periods=p))

@datatype_adapter
def RANK(df:pd.DataFrame):
    """Cross-sectional rank."""
    return df.groupby('datetime').rank(pct=True)

@datatype_adapter
def MEAN(df:pd.DataFrame):
    """Cross-sectional mean."""
    return df.groupby('datetime').mean()

@datatype_adapter
def STD(df:pd.DataFrame):
    """Cross-sectional std."""
    return df.groupby('datetime').std()

@datatype_adapter
def SKEW(df:pd.DataFrame):
    """Cross-sectional skewness."""
    from scipy.stats import skew as scipy_skew
    return df.groupby('datetime').transform(lambda x: scipy_skew(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 3 else np.nan)

@datatype_adapter
def KURT(df:pd.DataFrame):
    """Cross-sectional kurtosis."""
    from scipy.stats import kurtosis
    def calc_kurt(group):
        k = kurtosis(group.dropna(), fisher=True, nan_policy='omit')
        return pd.Series(k, index=group.index)
    return df.groupby('datetime').transform(lambda x: kurtosis(x.dropna(), fisher=True, nan_policy='omit') if len(x.dropna()) >= 4 else np.nan)

@datatype_adapter
def MAX(df:pd.DataFrame):
    """Cross-sectional max."""
    return df.groupby('datetime').max()

@datatype_adapter
def MIN(df:pd.DataFrame):
    """Cross-sectional min."""
    return df.groupby('datetime').min()

@datatype_adapter
def MEDIAN(df:pd.DataFrame):
    """Cross-sectional median."""
    return df.groupby('datetime').median()


@datatype_adapter
def TS_KURT(df:pd.DataFrame, p:int=5):
    """Rolling kurtosis."""
    from scipy.stats import kurtosis
    def rolling_kurt(x):
        return x.rolling(p, min_periods=min(4, p)).apply(
            lambda arr: kurtosis(arr, fisher=True, nan_policy='omit') if len(arr.dropna()) >= 4 else np.nan,
            raw=False
        )
    return df.groupby('instrument').transform(rolling_kurt)

@datatype_adapter
def TS_SKEW(df:pd.DataFrame, p:int=5):
    """Rolling skewness."""
    from scipy.stats import skew as scipy_skew
    def rolling_skew(x):
        return x.rolling(p, min_periods=min(3, p)).apply(
            lambda arr: scipy_skew(arr, nan_policy='omit') if len(arr.dropna()) >= 3 else np.nan,
            raw=False
        )
    return df.groupby('instrument').transform(rolling_skew)

@datatype_adapter
def TS_RANK(df:pd.DataFrame, p:int=5):
    """Time-series percentile rank."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).rank(pct=True))

@datatype_adapter
def TS_MAX(df:pd.DataFrame, p:int=5):
    """Time-series max."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).max())

@datatype_adapter
def TS_MIN(df:pd.DataFrame, p:int=5):
    """Time-series min."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).min())

@datatype_adapter
def TS_MEAN(df:pd.DataFrame, p:int=5):
    """Time-series mean."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).mean())

@datatype_adapter
def TS_MEDIAN(df:pd.DataFrame, p:int=5):
    """Time-series median."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).median())

@datatype_adapter
def PERCENTILE(df: pd.DataFrame, q: float, p: int = None):
    """
    Quantile of given data. q in [0,1]; if p given, rolling quantile.
    """
    assert 0 <= q <= 1, "Quantile q must be in [0, 1]"
    
    if p is not None:
        return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).quantile(q))
    else:
        return df.groupby('instrument').transform(lambda x: x.quantile(q))



@datatype_adapter
def TS_SUM(df:pd.DataFrame, p:int=5):
    """Time-series rolling sum."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())


@datatype_adapter
def TS_ARGMAX(df: pd.DataFrame, p: int = 5):
    """Days since max in past p days."""
    def rolling_argmax(window):
        return len(window) - window.argmax() - 1
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(rolling_argmax, raw=True))

@datatype_adapter
def TS_ARGMIN(df: pd.DataFrame, p: int = 5):
    """Days since min in past p days."""
    def rolling_argmin(window):
        return len(window) - window.argmin() - 1
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(rolling_argmin, raw=True))



def MAX(x:pd.DataFrame, y:pd.DataFrame, z:pd.DataFrame=None):
    """Element-wise max of DataFrames."""
    if z is None:
        return np.maximum(x, y)
    else:
        return np.maximum(np.maximum(x, y), z)




def MIN(x:pd.DataFrame, y:pd.DataFrame, z:pd.DataFrame=None):
    """Element-wise min of DataFrames.""" 
    if z is None:
        return np.minimum(x, y)
    else:
        return np.minimum(np.minimum(x, y), z)
    


@datatype_adapter
def ABS(df:pd.DataFrame):
    """Element-wise absolute value."""   
    return df.groupby('instrument').transform(lambda x: x.abs())    

@datatype_adapter
def DELAY(df:pd.DataFrame, p:int=1):
    """Delay data by p periods."""
    assert p >= 0, ValueError("DELAY period must be >= 0 (look-ahead bias)")
    return df.groupby('instrument').transform(lambda x: x.shift(p))


def TS_CORR(df1:pd.Series, df2: np.ndarray | pd.Series, p:int=5):
    """Rolling correlation of two series."""
    if isinstance(df2, np.ndarray):
        if p != len(df2):
            p = len(df2)
        def corr(window):
            x = window
            y = df2[:len(window)]
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            
            cov = np.sum((x - mean_x) * (y - mean_y))
            std_x = np.sqrt(np.sum((x - mean_x) ** 2))
            std_y = np.sqrt(np.sum((y - mean_y) ** 2))
            
            if std_x == 0 or std_y == 0:
                return 0.0
            return cov / (std_x * std_y)
        
        return df1.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=2).apply(corr, raw=True))
    elif isinstance(df2, (pd.Series, pd.DataFrame)):
        def rolling_corr(group, df2, p):
            instrument = group.name
            if isinstance(df2, pd.DataFrame) and 'instrument' in df2.index.names:
                df2_group = df2.xs(instrument, level='instrument')
            elif isinstance(df2, pd.Series) and 'instrument' in df2.index.names:
                df2_group = df2.xs(instrument, level='instrument')
            else:
                df2_group = df2
            return group.rolling(p, min_periods=2).corr(df2_group)

        result = df1.groupby('instrument').apply(lambda x: rolling_corr(x, df2, p))
        result = result.reset_index(level=0, drop=True).sort_index()
        return result
    else:
        raise TypeError(f"TS_CORR does not support df2 type: {type(df2)}")


def TS_COVARIANCE(df1:pd.DataFrame, df2:pd.DataFrame, p:int=5):  
    """Rolling covariance of two series."""
    if isinstance(df2, np.ndarray):
        if p != len(df2):
            p = len(df2)
        def cov(window):
            return np.cov(window, df2[:len(window)])[0, 1] if len(window) > 1 else 0.0
        return df1.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=2).apply(cov, raw=True))
    elif isinstance(df2, (pd.Series, pd.DataFrame)):
        def rolling_cov(group, df2, p):
            instrument = group.name
            if isinstance(df2, pd.DataFrame) and 'instrument' in df2.index.names:
                df2_group = df2.xs(instrument, level='instrument')
            elif isinstance(df2, pd.Series) and 'instrument' in df2.index.names:
                df2_group = df2.xs(instrument, level='instrument')
            else:
                df2_group = df2
            return group.rolling(p, min_periods=2).cov(df2_group)

        result = df1.groupby('instrument').apply(lambda x: rolling_cov(x, df2, p))
        result = result.reset_index(level=0, drop=True).sort_index()
        return result
    else:
        raise TypeError(f"TS_COVARIANCE does not support df2 type: {type(df2)}")

@datatype_adapter
def TS_STD(df:pd.DataFrame, p:int=20):
    """Rolling standard deviation."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).std())





@datatype_adapter
def TS_VAR(df: pd.DataFrame, p: int = 5, ddof: int = 1):
    """Rolling variance."""
    return df.groupby('instrument').transform(
        lambda x: x.rolling(p, min_periods=1).var(ddof=ddof)
    )

@datatype_adapter
def SIGN(df: pd.DataFrame):
    """Element-wise sign."""
    return np.sign(df)

@datatype_adapter
def SMA(df:pd.DataFrame, m:float=None, n:float=None):
    """Simple moving average. Y_{i+1} = m/n*X_i + (1 - m/n)*Y_i if n given."""
        
    if isinstance(m, int) and m >= 1 and n is None:
        return df.groupby('instrument').transform(lambda x: x.rolling(m, min_periods=1).mean())
    else:
        return df.groupby('instrument').transform(lambda x: x.ewm(alpha=n/m).mean())

@datatype_adapter
def EMA(df:pd.DataFrame, p):
    """Exponential moving average with period p."""
    return df.groupby('instrument').transform(lambda x: x.ewm(span=int(p), min_periods=1).mean())
    
@datatype_adapter
def WMA(df:pd.DataFrame, p:int=20):
    """
    Weighted moving average over p periods (recent has higher weight).
    """
    weights = [0.9**i for i in range(p)][::-1]
    def calculate_wma(window):
        return (window * weights[:len(window)]).sum() / sum(weights[:len(window)])

    return df.groupby('instrument').transform(lambda x: x.rolling(window=p, min_periods=1).apply(calculate_wma, raw=True))

@datatype_adapter
def COUNT(cond:pd.DataFrame, p:int=20):
    """
    Conditional count over rolling window p.
    """
    return cond.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())

@datatype_adapter
def SUMIF(df:pd.DataFrame, p:int, cond:pd.DataFrame):
    """
    Rolling sum of df where cond is true over window p.
    """
    return (df * cond).groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())

@datatype_adapter
def FILTER(df:pd.DataFrame, cond:pd.DataFrame):
    """
    Filter series by condition; where cond is false, set to 0.
    """
    return df.mul(cond)
    

@datatype_adapter
def PROD(df:pd.DataFrame, p:int=5):
    """
    Rolling product over window p.
    """

    if isinstance(p, int):
        return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(lambda x: x.prod(), raw=True))
    else:
        return df.mul(p)    

@datatype_adapter
def DECAYLINEAR(df:pd.DataFrame, p:int=5):
    """
    Linearly decay weighted average over p periods.
    """
    assert isinstance(p, int), ValueError(f"DECAYLINEAR expects positive int, got {type(p).__name__}")
    decay_weights = np.arange(1, p+1, 1)
    decay_weights = decay_weights / decay_weights.sum()
    
    def calculate_deycaylinear(window):
        return (window * decay_weights[:len(window)]).sum()
    
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(calculate_deycaylinear, raw=True))

@datatype_adapter
def HIGHDAY(df:pd.DataFrame, p:int=5):
    """
    Days since max in window p.
    """
    assert isinstance(p, int), ValueError(f"HIGHDAY expects positive int, got {type(p).__name__}")
    def highday(window):
        return len(window) - window.argmax(axis=0)
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(highday, raw=True))

@datatype_adapter
def LOWDAY(df:pd.DataFrame, p:int=5):
    """
    Days since min in window p.
    """
    assert isinstance(p, int), ValueError(f"LOWDAY expects positive int, got {type(p).__name__}")
    def lowday(window):
        return len(window) - window.argmin(axis=0)
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(lowday, raw=True))
    

def SEQUENCE(n):
    """
    Sequence 1 to n.
    """
    assert isinstance(n, int), ValueError(f"SEQUENCE(n) expects positive int, got {type(n).__name__}")
    return np.linspace(1, n, n, dtype=np.float32)

@datatype_adapter
def SUMAC(df:pd.DataFrame, p:int=10):
    """
    Rolling cumulative sum over window p.
    """
    assert isinstance(p, int), ValueError(f"SUMAC expects positive int, got {type(p).__name__}")
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())



def calculate_beta(y, x):
    """Regression coefficient (beta)."""
    X = np.vstack([x, np.ones(len(x))]).T
    beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

def rolling_beta(df1_group, df2_group, p):
    """Rolling beta of df1 on df2."""
    result = np.empty(len(df1_group))
    result[:] = np.nan

    for i in range(p - 1, len(df1_group)):
        window_y = df1_group.iloc[i - p + 1 : i + 1].values
        window_x = df2_group.iloc[:p].values if df1_group.shape != df2_group.shape else df2_group.iloc[i - p + 1 : i + 1].values
        result[i] = calculate_beta(window_y, window_x)

    return pd.Series(result, index=df1_group.index)


def REGBETA(df1: pd.DataFrame, df2: pd.DataFrame, p: int = 5, n_jobs: int = -1):
    """
    Rolling regression coefficient (beta) of df1 on df2.
    """
    assert not (isinstance(df2, np.ndarray) and isinstance(df1, np.ndarray)), "df1 and df2 cannot both be np.ndarray; at least one must be DataFrame (e.g. $close)."
    if isinstance(df2, np.ndarray) or isinstance(df1, np.ndarray):
        if isinstance(df1, np.ndarray):
            df3 = df1
            df1 = df2
            df2 = df3
            p = min(len(df2), p)
            df2 = pd.Series(df2)
        df1 = df1.fillna(0)
        
        df1_groups = list(df1.groupby('instrument'))
        df2 = pd.Series(df2[:p])
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(rolling_beta)(df1_group, df2, p)
            for _, df1_group in df1_groups
        )
        
        result = pd.concat(results)
        result = result.sort_index()
        return result
    
    else:
        assert df1.index.equals(df2.index), "df1 and df2 indices must align"
        
        df1 = df1.fillna(0)
        df2 = df2.fillna(0)
        
        df1_groups = list(df1.groupby('instrument'))
        df2_groups = list(df2.groupby('instrument'))
        
        if len(df1_groups) != len(df2_groups):
            raise ValueError("df1 and df2 group counts must match.")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(rolling_beta)(df1_group, df2_group, p)
            for (_, df1_group), (_, df2_group) in zip(df1_groups, df2_groups)
        )
        
        result = pd.concat(results)
        result = result.sort_index()
        return result



def calculate_residuals(y, x):
    """Residual (actual - predicted)."""
    X = np.vstack([x, np.ones(len(x))]).T
    beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = beta * x + intercept
    residuals = y - y_pred
    return residuals[-1]

def rolling_residuals(df1_group, df2_group, p):
    """Rolling residual of df1 on df2."""
    result = np.empty(len(df1_group))
    result[:] = np.nan

    for i in range(p - 1, len(df1_group)):
        window_y = df1_group.iloc[i - p + 1 : i + 1].values
        window_x = df2_group.iloc[:p].values if df1_group.shape != df2_group.shape else df2_group.iloc[i - p + 1 : i + 1].values
        result[i] = calculate_residuals(window_y, window_x)

    return pd.Series(result, index=df1_group.index)


def REGRESI(df1: pd.DataFrame, df2: pd.DataFrame, p: int = 5, n_jobs: int = -1):
    """
    Rolling residual of df1 on df2.
    """
    
    assert not (isinstance(df2, np.ndarray) and isinstance(df1, np.ndarray)), "df1 and df2 cannot both be np.ndarray; at least one must be DataFrame (e.g. $close)."
    if isinstance(df2, np.ndarray) or isinstance(df1, np.ndarray):
        if isinstance(df1, np.ndarray):
            df3 = df1
            df1 = df2
            df2 = df3
            p = min(len(df2), p)
        df1 = df1.fillna(0)
        df2 = pd.Series(df2[:p])
        
        df1_groups = list(df1.groupby('instrument'))
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(rolling_residuals)(df1_group, df2, p)
            for _, df1_group in df1_groups
        )
        
        result = pd.concat(results)
        result = result.sort_index()
        return result
    
    else:
        if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
            datetime_level = df1.index.get_level_values('datetime')
            df2_aligned = df2.reindex(datetime_level, method='ffill')
            df2_aligned.index = df1.index
            df2 = df2_aligned
        elif not df1.index.equals(df2.index):
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                try:
                    df2 = df2.reindex(df1.index)
                except Exception:
                    assert df1.index.equals(df2.index), "df1 and df2 indices must align"
            else:
                assert df1.index.equals(df2.index), "df1 and df2 indices must align"
        
        df1 = df1.fillna(0)
        df2 = df2.fillna(0)
        
        df1_groups = list(df1.groupby('instrument'))
        
        if isinstance(df2.index, pd.MultiIndex) and 'instrument' in df2.index.names:
            df2_groups = list(df2.groupby('instrument'))
            if len(df1_groups) != len(df2_groups):
                raise ValueError("df1 and df2 group counts must match.")
            results = Parallel(n_jobs=n_jobs)(
                delayed(rolling_residuals)(df1_group, df2_group, p)
                for (_, df1_group), (_, df2_group) in zip(df1_groups, df2_groups)
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(rolling_residuals)(df1_group, df2, p)
                for _, df1_group in df1_groups
            )
        
        result = pd.concat(results)
        result = result.sort_index()
        return result

        
# Math
@datatype_adapter
def EXP(df:pd.DataFrame):
    """
    Element-wise exp.
    """
    return df.apply(np.exp)

@datatype_adapter
def SQRT(df: pd.DataFrame):
    """Element-wise sqrt."""
    if isinstance(df, int):
        return np.sqrt(df)
    return df.apply(np.sqrt)

@datatype_adapter
def LOG(df:pd.DataFrame):
    """Natural logarithm."""
    if isinstance(df, int):
        return np.log(df)
    return (df+1).apply(np.log)

@datatype_adapter
def INV(df: pd.DataFrame):
    """Reciprocal (1/x)."""
    return 1 / df

@datatype_adapter
def POW(df:pd.DataFrame, n:int):
    """Element-wise power."""
    return np.power(df, n)

def FLOOR(df:pd.DataFrame):
    """Floor (round down)."""
    return df.apply(np.floor)

@datatype_adapter
def TS_ZSCORE(df: pd.DataFrame, p:int=5):
    assert isinstance(p, int), ValueError(f"TS_ZSCORE expects positive int, got {type(p).__name__}")
    # assert isinstance(df, pd.DataFrame), ValueError(f"TS_ZSCORE expects pd.DataFrame, got {type(df).__name__}")
    return (df - df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).mean())) / df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).std())

@datatype_adapter
def ZSCORE(df):
    mean = df.groupby('datetime').mean()
    std = df.groupby('datetime').std()
    zscore = (df - mean) / std
    return zscore

@datatype_adapter
def SCALE(df: pd.DataFrame, target_sum: float = 1.0):
    """Scale series so absolute sum equals target_sum."""
    abs_sum = ABS(df).groupby('datetime').sum()
    return df.multiply(target_sum).div(abs_sum, axis=0)


@datatype_adapter
def TS_MAD(df: pd.DataFrame, p: int = 5):
    """Rolling median absolute deviation (MAD = median(|X_i - median(X)|))."""
    def rolling_mad(window):
        median_val = np.median(window)
        abs_dev = np.abs(window - median_val)
        return np.median(abs_dev)
    
    return df.groupby('instrument').transform(
        lambda x: x.rolling(p, min_periods=1).apply(rolling_mad, raw=True)
    )


@datatype_adapter
def TS_QUANTILE(df: pd.DataFrame, p: int = 5, q: float = 0.5):
    """Rolling quantile. Auto-detects parameter order if swapped (q, p -> p, q)."""
    if isinstance(p, float) and 0 < p < 1 and isinstance(q, (int, float)) and q > 1:
        p, q = int(q), p
    p = int(p)
    q = float(q)
    assert 0 <= q <= 1, f"Quantile q must be in [0, 1], got {q}"
    assert p >= 1, f"Window p must >= 1, got {p}"
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).quantile(q))

@datatype_adapter
def TS_PCTCHANGE(df: pd.DataFrame, p: int = 1):
    """Percentage change over p periods (default 1)."""
    return df.groupby('instrument').transform(lambda x: x.pct_change(periods=p, fill_method=None).fillna(0))


def ADD(df1, df2):
    """Add with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.add)

def SUBTRACT(df1, df2):
    """Subtract with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.subtract)

def MULTIPLY(df1, df2):
    """Multiply with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.multiply)

def DIVIDE(df1, df2):
    """Divide with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.divide)

def _arithmetic_with_alignment(df1, df2, op_func):
    """Arithmetic op with index alignment."""
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        df2 = df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        df1 = df1_aligned
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index)
            else:
                df2 = df2.reindex(df1.index)
        except Exception:
            pass
    
    try:
        result = op_func(df1, df2)
    except (ValueError, TypeError) as e:
        if 'identically-labeled' in str(e) or 'Can only compare' in str(e) or 'index' in str(e).lower():
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index, fill_value=0)
            elif isinstance(df1.index, pd.MultiIndex):
                datetime_level = df1.index.get_level_values('datetime')
                df2 = df2.reindex(datetime_level, method='ffill')
                df2.index = df1.index
            result = op_func(df1, df2)
        else:
            raise
    
    return result
    
def AND(df1, df2):
    """Logical AND with index alignment."""
    df1_aligned, df2_aligned = _align_for_operation(df1, df2)
    return np.bitwise_and(df1_aligned.astype(np.bool_), df2_aligned.astype(np.bool_))

def OR(df1, df2):
    """Logical OR with index alignment."""
    df1_aligned, df2_aligned = _align_for_operation(df1, df2)
    return np.bitwise_or(df1_aligned.astype(np.bool_), df2_aligned.astype(np.bool_))

def WHERE(condition, true_value, false_value):
    """Conditional expression (WHERE) with index alignment."""
    
    if isinstance(condition, (pd.DataFrame, pd.Series)):
        target_index = condition.index
    elif isinstance(true_value, (pd.DataFrame, pd.Series)):
        target_index = true_value.index
    elif isinstance(false_value, (pd.DataFrame, pd.Series)):
        target_index = false_value.index
    else:
        return np.where(condition, true_value, false_value)
    
    if isinstance(true_value, (pd.DataFrame, pd.Series)) and not true_value.index.equals(target_index):
        if isinstance(target_index, pd.MultiIndex) and not isinstance(true_value.index, pd.MultiIndex):
            datetime_level = target_index.get_level_values('datetime')
            true_value = true_value.reindex(datetime_level, method='ffill')
            true_value.index = target_index
        else:
            true_value = true_value.reindex(target_index, fill_value=0)
    
    if isinstance(false_value, (pd.DataFrame, pd.Series)) and not false_value.index.equals(target_index):
        if isinstance(target_index, pd.MultiIndex) and not isinstance(false_value.index, pd.MultiIndex):
            datetime_level = target_index.get_level_values('datetime')
            false_value = false_value.reindex(datetime_level, method='ffill')
            false_value.index = target_index
        else:
            false_value = false_value.reindex(target_index, fill_value=0)
    
    if isinstance(condition, (pd.DataFrame, pd.Series)) and not condition.index.equals(target_index):
        condition = condition.reindex(target_index, fill_value=False)
    
    result = np.where(condition, true_value, false_value)
    
    if isinstance(result, np.ndarray) and isinstance(target_index, pd.MultiIndex):
        result = pd.Series(result, index=target_index)
    elif isinstance(result, np.ndarray) and isinstance(target_index, pd.Index):
        result = pd.Series(result, index=target_index)
    
    return result

def _align_for_operation(df1, df2):
    """Align two DataFrame/Series indices for binary ops."""
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return df1, df2
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return df1, df2
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return df1, df2
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        return df1, df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        return df1_aligned, df2
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2_aligned = df2.reindex(df1.index)
                return df1, df2_aligned
            else:
                df2_aligned = df2.reindex(df1.index)
                return df1, df2_aligned
        except Exception:
            return df1, df2
    
    return df1, df2

def GT(df1, df2):
    """Greater than with index alignment."""
    return _compare_with_alignment(df1, df2, operator.gt)

def LT(df1, df2):
    """Less than with index alignment."""
    return _compare_with_alignment(df1, df2, operator.lt)

def GE(df1, df2):
    """Greater or equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.ge)

def LE(df1, df2):
    """Less or equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.le)

def EQ(df1, df2):
    """Equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.eq)

def NE(df1, df2):
    """Not equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.ne)

def _compare_with_alignment(df1, df2, op_func):
    """Compare two DataFrame/Series with index alignment."""
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return op_func(df2, df1) if op_func in [operator.lt, operator.le] else op_func(df1, df2)
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        df2 = df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        df1 = df1_aligned
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index)
            else:
                df2 = df2.reindex(df1.index)
        except Exception:
            pass
    
    try:
        result = op_func(df1, df2)
    except (ValueError, TypeError) as e:
        if 'identically-labeled' in str(e) or 'Can only compare' in str(e):
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index, fill_value=0)
            elif isinstance(df1.index, pd.MultiIndex):
                datetime_level = df1.index.get_level_values('datetime')
                df2 = df2.reindex(datetime_level, method='ffill')
                df2.index = df1.index
            result = op_func(df1, df2)
        else:
            raise
    
    return result



def MACD(price_df, short_window=12, long_window=26):
    """MACD indicator (short EMA - long EMA)."""
    short_ema = EMA(price_df, short_window)
    long_ema = EMA(price_df, long_window)
    macd = short_ema - long_ema
    return macd


def RSI(price_df, window=14):
    """RSI (Relative Strength Index)."""
    price_change = DELTA(price_df, 1)
    up = (price_change > 0) * price_change
    down = (price_change < 0) * ABS(price_change)
    avg_up = EMA(up, window)
    avg_down = EMA(down, window)
    rsi = 100 - (100 / (1 + (avg_up / avg_down)))
    return rsi




def _calculate_rolling_mean(group_data):
    """Dynamic rolling mean for one group."""
    price_group, window_group, group_name = group_data
    result = pd.Series(index=price_group.index, dtype=float)
    
    for i in range(len(price_group)):
        curr_window = int(window_group.iloc[i].values)
        if curr_window < 1:
            curr_window = 1
        if i < curr_window:
            result.iloc[i] = price_group.iloc[:i+1].mean()
        else:
            result.iloc[i] = price_group.iloc[i-curr_window+1:i+1].mean()
    
    return group_name, result

def _calculate_rolling_std(group_data):
    """Dynamic rolling std for one group."""
    price_group, window_group, group_name = group_data
    result = pd.Series(index=price_group.index, dtype=float)
    
    for i in range(len(price_group)):
        curr_window = int(window_group.iloc[i].values)
        if curr_window < 1:
            curr_window = 1
        if i < curr_window:
            result.iloc[i] = price_group.iloc[:i+1].std()
        else:
            result.iloc[i] = price_group.iloc[i-curr_window+1:i+1].std()
    
    return group_name, result



@datatype_adapter
def BB_MIDDLE(price_df, window, n_jobs=-1):
    """Bollinger Band middle (supports dynamic window, parallel)."""
    if isinstance(window, (int, float)):
        return price_df.groupby('instrument').transform(lambda x: x.rolling(int(window), min_periods=1).mean())
    else:
        window.index = price_df.index
        groups_data = [
            (price_group, 
             window.xs(group_name, level='instrument'), 
             group_name)
            for group_name, price_group in price_df.groupby('instrument')
        ]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_rolling_mean)(group_data)
            for group_data in groups_data
        )
        
        final_result = pd.concat([result for _, result in sorted(results, key=lambda x: x[0])])
        return final_result

@datatype_adapter
def BB_UPPER(price_df, window, n_jobs=-1):
    """Bollinger Band upper (supports dynamic window, parallel)."""
    
    if isinstance(window, (int, float)):
        middle_band = BB_MIDDLE(price_df, window, n_jobs)
        std = price_df.groupby('instrument').transform(lambda x: x.rolling(int(window), min_periods=1).std())
    else:
        window.index = price_df.index
        middle_band = BB_MIDDLE(price_df, window, n_jobs)
        groups_data = [
            (price_group, 
             window.xs(group_name, level='instrument'), 
             group_name)
            for group_name, price_group in price_df.groupby('instrument')
        ]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_rolling_std)(group_data)
            for group_data in groups_data
        )
        
        std = pd.concat([result for _, result in sorted(results, key=lambda x: x[0])])
    
    return middle_band + std

@datatype_adapter
def BB_LOWER(price_df, window, n_jobs=-1):
    """Bollinger Band lower (supports dynamic window, parallel)."""
    
    if isinstance(window, (int, float)):
        middle_band = BB_MIDDLE(price_df, window, n_jobs)
        std = price_df.groupby('instrument').transform(lambda x: x.rolling(int(window), min_periods=1).std())
    else:
        window.index = price_df.index
        middle_band = BB_MIDDLE(price_df, window, n_jobs)
        groups_data = [
            (price_group, 
             window.xs(group_name, level='instrument'), 
             group_name)
            for group_name, price_group in price_df.groupby('instrument')
        ]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_rolling_std)(group_data)
            for group_data in groups_data
        )
        
        std = pd.concat([result for _, result in sorted(results, key=lambda x: x[0])])
    
    return middle_band - std
