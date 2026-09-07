"""Framework-neutral post-execution evaluation and measurement."""

from .core import (
    AdaptationEvaluator,
    EvaluationOutcome,
    EvaluationRecord,
    evaluate_execution,
)

__all__ = [
    "AdaptationEvaluator",
    "EvaluationOutcome",
    "EvaluationRecord",
    "evaluate_execution",
]
