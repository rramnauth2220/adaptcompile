"""Framework-neutral execution contracts and orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from .._execution import ExecutionOutcome, ExecutionRecord
from .._validation import nonempty_name
from ..compiler.selection import ProgramSelection
from ..episode import LearningEpisode
from ..errors import ValidationError
from ..model import ModelContext
from ..program import ProgramSpec
from ..serialization import immutable_json_mapping

ModelT = TypeVar("ModelT")


@runtime_checkable
class AdaptationBackend(Protocol[ModelT]):
    """Structural interface for executing a concrete adaptation program."""

    @property
    def backend_id(self) -> str: ...

    def supports(self, program: ProgramSpec) -> bool: ...

    def execute(
        self,
        model: ModelT,
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> ModelT: ...


def execute_selection(
    selection: ProgramSelection,
    *,
    programs: Sequence[ProgramSpec],
    model: ModelT,
    model_context: ModelContext,
    episode: LearningEpisode,
    backend: AdaptationBackend[ModelT],
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionOutcome[ModelT]:
    """Resolve and execute the selected concrete program exactly once."""
    if not isinstance(selection, ProgramSelection):
        raise ValidationError("selection must be a ProgramSelection")
    selected = selection.selected
    if selected is None or not selected.feasible:
        raise ValidationError("selection has no feasible selected candidate")
    if not isinstance(model_context, ModelContext):
        raise ValidationError("model_context must be a ModelContext")
    if selection.model_key != model_context.identity_key:
        raise ValidationError("selection model_key does not match model_context")
    if not isinstance(episode, LearningEpisode):
        raise ValidationError("episode must be a LearningEpisode")
    if selection.episode_key != episode.identity_key:
        raise ValidationError("selection episode_key does not match episode")

    if not isinstance(programs, Sequence) or isinstance(programs, (str, bytes)):
        raise ValidationError("programs must be a sequence")
    candidates = tuple(programs)
    if not candidates:
        raise ValidationError("execution requires at least one ProgramSpec")
    if any(not isinstance(program, ProgramSpec) for program in candidates):
        raise ValidationError("programs must contain ProgramSpec instances")
    fingerprints = [program.fingerprint for program in candidates]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValidationError("supplied program fingerprints must be unique")

    selected_prediction = selected.prediction
    selected_fingerprint = selected_prediction.program_fingerprint
    matches = [
        program for program in candidates if program.fingerprint == selected_fingerprint
    ]
    if len(matches) != 1:
        raise ValidationError(
            "exactly one supplied ProgramSpec must match the selected fingerprint"
        )
    program = matches[0]
    if (
        program.fingerprint != selected_fingerprint
    ):  # pragma: no cover - explicit invariant
        raise ValidationError(
            "resolved ProgramSpec fingerprint does not match selection"
        )
    program_family = None if program.family is None else program.family.fingerprint
    if selected_prediction.family_fingerprint != program_family:
        raise ValidationError("resolved ProgramSpec family does not match selection")

    if not isinstance(backend, AdaptationBackend):
        raise ValidationError("backend must satisfy the AdaptationBackend protocol")
    backend_id = nonempty_name(backend.backend_id, field="backend_id")
    supported = backend.supports(program)
    if not isinstance(supported, bool):
        raise ValidationError("backend supports(program) must return a boolean")
    if not supported:
        raise ValidationError("backend does not support the selected ProgramSpec")
    validated_metadata = immutable_json_mapping(
        {} if metadata is None else metadata, location="execution metadata"
    )

    adapted_model = backend.execute(
        model,
        model_context=model_context,
        episode=episode,
        program=program,
    )
    record = ExecutionRecord(
        model_key=model_context.identity_key,
        episode_key=episode.identity_key,
        family_fingerprint=program_family,
        program_fingerprint=program.fingerprint,
        program=program,
        backend_id=backend_id,
        selected_utility=selected.utility,
        metadata=validated_metadata,
    )
    return ExecutionOutcome(model=adapted_model, record=record)
