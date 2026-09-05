"""Central adaptation experiment record."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

from ._validation import score_mapping
from .episode import LearningEpisode
from .errors import ValidationError
from .geometry import AdaptationGeometry
from .model import ModelContext
from .program import ProgramSpec
from .serialization import immutable_json_mapping, read_json, to_json_safe, write_json


@dataclass(frozen=True)
class AdaptationResult:
    """Measurements before and after applying a program to a learning episode.

    The record observes geometry for one ``(model_context, episode, program)``
    combination. ``geometry`` aliases ``after_geometry``; ``delta_geometry`` (and
    its ``delta`` alias) represents ``after - before``.
    """

    model_context: ModelContext
    episode: LearningEpisode
    program: ProgramSpec
    before: Mapping[str, float]
    after: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_context, ModelContext):
            raise ValidationError("model_context must be a ModelContext")
        if not isinstance(self.episode, LearningEpisode):
            raise ValidationError("episode must be a LearningEpisode")
        if not isinstance(self.program, ProgramSpec):
            raise ValidationError("program must be a ProgramSpec")
        before = score_mapping(self.before, field="before")
        after = score_mapping(self.after, field="after")
        if not before:
            raise ValidationError("before and after must contain at least one metric")
        if set(before) != set(after):
            raise ValidationError(
                "before and after metrics must match exactly "
                f"(before={sorted(before)}, after={sorted(after)})"
            )
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="metadata"),
        )

    @property
    def before_geometry(self) -> AdaptationGeometry:
        return AdaptationGeometry(self.before)

    @property
    def after_geometry(self) -> AdaptationGeometry:
        return AdaptationGeometry(self.after)

    @property
    def geometry(self) -> AdaptationGeometry:
        """The post-adaptation behavioral state (alias of ``after_geometry``)."""
        return self.after_geometry

    @property
    def delta_geometry(self) -> AdaptationGeometry:
        """Behavioral change computed as ``after - before``."""
        return self.after_geometry.delta(self.before_geometry)

    @property
    def delta(self) -> AdaptationGeometry:
        return self.delta_geometry

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_context": self.model_context.to_dict(),
            "episode": self.episode.to_dict(),
            "program": self.program.to_dict(),
            "before": dict(self.before),
            "after": dict(self.after),
            "metadata": to_json_safe(self.metadata, location="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdaptationResult:
        try:
            return cls(
                model_context=ModelContext.from_dict(data["model_context"]),
                episode=LearningEpisode.from_dict(data["episode"]),
                program=ProgramSpec.from_dict(data["program"]),
                before=data["before"],
                after=data["after"],
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized AdaptationResult") from error

    def to_json(
        self,
        destination: str | PathLike[str] | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        return write_json(self.to_dict(), destination, indent=indent)

    @classmethod
    def from_json(cls, source: str | PathLike[str]) -> AdaptationResult:
        return cls.from_dict(read_json(source))
