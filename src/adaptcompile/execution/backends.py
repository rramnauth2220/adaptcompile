"""Dependency-free reference execution backends."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from .._validation import nonempty_name
from ..episode import LearningEpisode
from ..errors import ValidationError
from ..model import ModelContext
from ..program import ProgramSpec

ModelT = TypeVar("ModelT")


class _ExecutionHandler(Protocol[ModelT]):
    def __call__(
        self,
        model: ModelT,
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> ModelT: ...


class _SupportPredicate(Protocol):
    def __call__(self, program: ProgramSpec) -> bool: ...


__all__ = ["CallableBackend"]


class CallableBackend(Generic[ModelT]):
    """Delegate execution to one user callable and an optional support predicate."""

    __slots__ = ("_backend_id", "_handler", "_support_predicate")

    def __init__(
        self,
        handler: _ExecutionHandler[ModelT],
        *,
        backend_id: str = "callable",
        supports: _SupportPredicate | None = None,
    ) -> None:
        if not callable(handler):
            raise ValidationError("handler must be callable")
        if supports is not None and not callable(supports):
            raise ValidationError("supports must be callable or None")
        self._handler = handler
        self._backend_id = nonempty_name(backend_id, field="backend_id")
        self._support_predicate = supports

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def supports(self, program: ProgramSpec) -> bool:
        if not isinstance(program, ProgramSpec):
            raise ValidationError("program must be a ProgramSpec")
        if self._support_predicate is None:
            return True
        supported = self._support_predicate(program)
        if not isinstance(supported, bool):
            raise ValidationError("supports predicate must return a boolean")
        return supported

    def execute(
        self,
        model: ModelT,
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> ModelT:
        return self._handler(
            model,
            model_context=model_context,
            episode=episode,
            program=program,
        )
