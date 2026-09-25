"""
Time measurement utilities.
"""
import time
import functools
from quantaalpha.log import logger


def measure_time(func):
    """Decorator: measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper
