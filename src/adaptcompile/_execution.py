"""Internal execution records shared across runtime phases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from ._identity import _validate_episode_key, _validate_model_key
from ._validation import nonempty_name, score
from .dataset import ModelIdentityKey
from .episode import EpisodeIdentityKey
from .errors import ValidationError
from .program import ProgramSpec
from .serialization import immutable_json_mapping, to_json_safe

ModelT = TypeVar("ModelT")


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
