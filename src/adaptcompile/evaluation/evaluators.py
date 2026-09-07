"""Dependency-free reference adaptation evaluators."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from .._validation import nonempty_name
from ..episode import LearningEpisode
from ..errors import ValidationError
from ..geometry import AdaptationGeometry
from ..model import ModelContext
from ..program import ProgramSpec

ModelT = TypeVar("ModelT")
ModelT_contra = TypeVar("ModelT_contra", contravariant=True)


class _EvaluationHandler(Protocol[ModelT_contra]):
    def __call__(
        self,
        model: ModelT_contra,
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> AdaptationGeometry: ...


__all__ = ["CallableEvaluator"]


class CallableEvaluator(Generic[ModelT]):
    """Delegate post-execution measurement to one user callable."""

    __slots__ = ("_evaluator_id", "_handler")

    def __init__(
        self,
        handler: _EvaluationHandler[ModelT],
        *,
        evaluator_id: str = "callable",
    ) -> None:
        if not callable(handler):
            raise ValidationError("handler must be callable")
        self._handler = handler
        self._evaluator_id = nonempty_name(evaluator_id, field="evaluator_id")

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    def evaluate(
        self,
        model: ModelT,
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> AdaptationGeometry:
        measured = self._handler(
            model,
            model_context=model_context,
            episode=episode,
            program=program,
        )
        if not isinstance(measured, AdaptationGeometry):
            raise ValidationError("evaluator handler must return an AdaptationGeometry")
        return measured
