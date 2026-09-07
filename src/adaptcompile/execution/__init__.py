"""Framework-neutral selected-program execution."""

from .core import (
    AdaptationBackend,
    ExecutionOutcome,
    ExecutionRecord,
    execute_selection,
)

__all__ = [
    "AdaptationBackend",
    "ExecutionOutcome",
    "ExecutionRecord",
    "execute_selection",
]
