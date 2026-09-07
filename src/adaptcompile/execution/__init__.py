"""Framework-neutral selected-program execution."""

from .._execution import ExecutionOutcome, ExecutionRecord
from .core import AdaptationBackend, execute_selection

__all__ = [
    "AdaptationBackend",
    "ExecutionOutcome",
    "ExecutionRecord",
    "execute_selection",
]
