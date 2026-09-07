"""Framework-neutral execution contracts and orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from .._validation import nonempty_name, score
from ..compiler.data import _validate_episode_key, _validate_model_key
from ..compiler.selection import ProgramSelection
from ..dataset import ModelIdentityKey
from ..episode import EpisodeIdentityKey, LearningEpisode
from ..errors import ValidationError
from ..model import ModelContext
from ..program import ProgramSpec
from ..serialization import immutable_json_mapping, to_json_safe

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


@dataclass(frozen=True)
class ExecutionRecord:
    """Serializable provenance for one selected-program execution."""

    model_key: ModelIdentityKey
    episode_key: EpisodeIdentityKey
    family_fingerprint: str | None
    program_fingerprint: str
    program: ProgramSpec
    backend_id: str
    selected_utility: float
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
            "selected_utility",
            score(self.selected_utility, field="selected_utility"),
        )
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="execution metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe execution record."""
        safe = to_json_safe(
            {
                "model_key": self.model_key,
                "episode_key": self.episode_key,
                "family_fingerprint": self.family_fingerprint,
                "program_fingerprint": self.program_fingerprint,
                "program": self.program.to_dict(),
                "backend_id": self.backend_id,
                "selected_utility": self.selected_utility,
                "metadata": self.metadata,
            }
        )
        if not isinstance(safe, dict):  # pragma: no cover - construction guarantees it
            raise RuntimeError("execution record serialization must produce an object")
        return cast(dict[str, Any], safe)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionRecord:
        """Restore an execution record from :meth:`to_dict` output."""
        try:
            return cls(
                model_key=tuple(data["model_key"]),
                episode_key=tuple(data["episode_key"]),
                family_fingerprint=data.get("family_fingerprint"),
                program_fingerprint=data["program_fingerprint"],
                program=ProgramSpec.from_dict(data["program"]),
                backend_id=data["backend_id"],
                selected_utility=data["selected_utility"],
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized ExecutionRecord") from error


@dataclass(frozen=True, eq=False)
class ExecutionOutcome(Generic[ModelT]):
    """In-memory backend result paired with serializable execution provenance."""

    model: ModelT
    record: ExecutionRecord

    def __post_init__(self) -> None:
        if not isinstance(self.record, ExecutionRecord):
            raise ValidationError("record must be an ExecutionRecord")


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
