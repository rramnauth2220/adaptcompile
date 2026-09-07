"""Framework-neutral evaluation contracts and orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from .._execution import ExecutionOutcome, ExecutionRecord
from .._identity import _validate_episode_key, _validate_model_key
from .._validation import nonempty_name
from ..dataset import ModelIdentityKey
from ..episode import EpisodeIdentityKey, LearningEpisode
from ..errors import ValidationError
from ..geometry import AdaptationGeometry
from ..model import ModelContext
from ..program import ProgramSpec
from ..result import AdaptationResult
from ..serialization import immutable_json_mapping, to_json_safe

ModelT = TypeVar("ModelT")
ModelT_contra = TypeVar("ModelT_contra", contravariant=True)


@runtime_checkable
class AdaptationEvaluator(Protocol[ModelT_contra]):
    """Structural interface for measuring an adapted runtime object."""

    @property
    def evaluator_id(self) -> str: ...

    def evaluate(
        self,
        model: ModelT_contra,
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> AdaptationGeometry: ...


@dataclass(frozen=True)
class EvaluationRecord:
    """Serializable provenance for one completed post-execution evaluation."""

    model_key: ModelIdentityKey
    episode_key: EpisodeIdentityKey
    family_fingerprint: str | None
    program_fingerprint: str
    program: ProgramSpec
    backend_id: str
    evaluator_id: str
    before: AdaptationGeometry
    after: AdaptationGeometry
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_key", _validate_model_key(self.model_key))
        object.__setattr__(self, "episode_key", _validate_episode_key(self.episode_key))
        if self.family_fingerprint is not None:
            object.__setattr__(
                self,
                "family_fingerprint",
                nonempty_name(self.family_fingerprint, field="family_fingerprint"),
            )
        object.__setattr__(
            self,
            "program_fingerprint",
            nonempty_name(self.program_fingerprint, field="program_fingerprint"),
        )
        if not isinstance(self.program, ProgramSpec):
            raise ValidationError("program must be a ProgramSpec")
        if self.program_fingerprint != self.program.fingerprint:
            raise ValidationError("program_fingerprint must equal program.fingerprint")
        expected_family = (
            None if self.program.family is None else self.program.family.fingerprint
        )
        if self.family_fingerprint != expected_family:
            raise ValidationError(
                "family_fingerprint must match the program's family fingerprint"
            )
        object.__setattr__(
            self,
            "backend_id",
            nonempty_name(self.backend_id, field="backend_id"),
        )
        object.__setattr__(
            self,
            "evaluator_id",
            nonempty_name(self.evaluator_id, field="evaluator_id"),
        )
        if not isinstance(self.before, AdaptationGeometry):
            raise ValidationError("before must be an AdaptationGeometry")
        if not isinstance(self.after, AdaptationGeometry):
            raise ValidationError("after must be an AdaptationGeometry")
        if set(self.before) != set(self.after):
            raise ValidationError(
                "before and after metrics must match exactly "
                f"(before={sorted(self.before)}, after={sorted(self.after)})"
            )
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="evaluation metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe evaluation record."""
        safe = to_json_safe(
            {
                "model_key": self.model_key,
                "episode_key": self.episode_key,
                "family_fingerprint": self.family_fingerprint,
                "program_fingerprint": self.program_fingerprint,
                "program": self.program.to_dict(),
                "backend_id": self.backend_id,
                "evaluator_id": self.evaluator_id,
                "before": self.before.to_dict(),
                "after": self.after.to_dict(),
                "metadata": self.metadata,
            }
        )
        if not isinstance(safe, dict):  # pragma: no cover - construction guarantees it
            raise RuntimeError("evaluation record serialization must produce an object")
        return cast(dict[str, Any], safe)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationRecord:
        """Restore an evaluation record from :meth:`to_dict` output."""
        try:
            return cls(
                model_key=tuple(data["model_key"]),
                episode_key=tuple(data["episode_key"]),
                family_fingerprint=data.get("family_fingerprint"),
                program_fingerprint=data["program_fingerprint"],
                program=ProgramSpec.from_dict(data["program"]),
                backend_id=data["backend_id"],
                evaluator_id=data["evaluator_id"],
                before=AdaptationGeometry.from_dict(data["before"]),
                after=AdaptationGeometry.from_dict(data["after"]),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized EvaluationRecord") from error


@dataclass(frozen=True)
class EvaluationOutcome:
    """Measured adaptation result paired with serializable evaluation provenance."""

    result: AdaptationResult
    record: EvaluationRecord

    def __post_init__(self) -> None:
        if not isinstance(self.result, AdaptationResult):
            raise ValidationError("result must be an AdaptationResult")
        if not isinstance(self.record, EvaluationRecord):
            raise ValidationError("record must be an EvaluationRecord")
        if self.result.model_context.identity_key != self.record.model_key:
            raise ValidationError(
                "result model_context identity must match record model_key"
            )
        if self.result.episode.identity_key != self.record.episode_key:
            raise ValidationError(
                "result episode identity must match record episode_key"
            )
        if self.result.program != self.record.program:
            raise ValidationError("result program must exactly match record program")
        if self.result.before_geometry != self.record.before:
            raise ValidationError("result before geometry must match record before")
        if self.result.after_geometry != self.record.after:
            raise ValidationError("result after geometry must match record after")


def _validate_execution_record(record: ExecutionRecord) -> ProgramSpec:
    if not isinstance(record, ExecutionRecord):
        raise ValidationError("execution record must be an ExecutionRecord")
    program = record.program
    if not isinstance(program, ProgramSpec):
        raise ValidationError("execution record program must be a ProgramSpec")
    if record.program_fingerprint != program.fingerprint:
        raise ValidationError(
            "execution program_fingerprint must equal execution program.fingerprint"
        )
    expected_family = None if program.family is None else program.family.fingerprint
    if record.family_fingerprint != expected_family:
        raise ValidationError(
            "execution family_fingerprint must match the program's family fingerprint"
        )
    nonempty_name(record.backend_id, field="backend_id")
    return program


def evaluate_execution(
    execution: ExecutionOutcome[ModelT],
    *,
    evaluator: AdaptationEvaluator[ModelT],
    model_context: ModelContext,
    episode: LearningEpisode,
    before: AdaptationGeometry,
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationOutcome:
    """Measure an executed runtime exactly once and create an observed result."""
    if not isinstance(execution, ExecutionOutcome):
        raise ValidationError("execution must be an ExecutionOutcome")
    if not isinstance(model_context, ModelContext):
        raise ValidationError("model_context must be a ModelContext")
    if execution.record.model_key != model_context.identity_key:
        raise ValidationError("execution model_key does not match model_context")
    if not isinstance(episode, LearningEpisode):
        raise ValidationError("episode must be a LearningEpisode")
    if execution.record.episode_key != episode.identity_key:
        raise ValidationError("execution episode_key does not match episode")

    program = _validate_execution_record(execution.record)
    if not isinstance(evaluator, AdaptationEvaluator):
        raise ValidationError("evaluator must satisfy the AdaptationEvaluator protocol")
    evaluator_id = nonempty_name(evaluator.evaluator_id, field="evaluator_id")
    if not isinstance(before, AdaptationGeometry):
        raise ValidationError("before must be an AdaptationGeometry")
    validated_metadata = immutable_json_mapping(
        {} if metadata is None else metadata, location="evaluation metadata"
    )

    measured_after = evaluator.evaluate(
        execution.model,
        model_context=model_context,
        episode=episode,
        program=program,
    )
    if not isinstance(measured_after, AdaptationGeometry):
        raise ValidationError("evaluator must return an AdaptationGeometry")

    result = AdaptationResult(
        model_context=model_context,
        episode=episode,
        program=program,
        before=before,
        after=measured_after,
    )
    record = EvaluationRecord(
        model_key=model_context.identity_key,
        episode_key=episode.identity_key,
        family_fingerprint=execution.record.family_fingerprint,
        program_fingerprint=execution.record.program_fingerprint,
        program=program,
        backend_id=execution.record.backend_id,
        evaluator_id=evaluator_id,
        before=before,
        after=measured_after,
        metadata=validated_metadata,
    )
    return EvaluationOutcome(result=result, record=record)
