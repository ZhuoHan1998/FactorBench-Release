"""
AlphaAgent logging module - compatibility layer.

Maps alphaagent.log to rdagent.log so all alphaagent.log imports work.
Provides AlphaAgent-specific APIs: log_trace_path, set_trace_path.
"""

from pathlib import Path
from rdagent.log import rdagent_logger as _rdagent_logger
from rdagent.log.utils import LogColors


class _AlphaAgentLoggerWrapper:
    """
    Wraps rdagent_logger and adds log_trace_path / set_trace_path. Other attributes/methods delegate to rdagent_logger.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    # ---------- AlphaAgent extension ----------
    @property
    def log_trace_path(self) -> Path:
        """Return current log trace path."""
        return self._inner.storage.path

    def set_trace_path(self, path) -> None:
        """Set new log trace path."""
        from rdagent.log.storage import FileStorage
        self._inner.storage = FileStorage(Path(path))

    # ---------- Delegate to rdagent_logger ----------
    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in ("_inner",):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


logger = _AlphaAgentLoggerWrapper(_rdagent_logger)

__all__ = ["logger", "LogColors"]
