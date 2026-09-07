"""Learning episode representation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

from ._validation import nonempty_name, string_keyed_mapping
from .errors import ValidationError
from .serialization import (
    immutable_json_mapping,
    json_dumps,
    read_json,
    to_json_safe,
    write_json,
)

EpisodeIdentityKey = tuple[str, str]


@dataclass(frozen=True)
class LearningEpisode:
    """A model-independent learning problem and its evaluation sets.

    Dataset-like objects are held by reference and deliberately excluded from
    serialization. The evaluation mapping and metadata are detached and read-only.
    """

    name: str
    train: Any
    evaluations: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    episode_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", nonempty_name(self.name, field="name"))
        evaluations = string_keyed_mapping(self.evaluations, field="evaluations")
        if not evaluations:
            raise ValidationError(
                "evaluations must contain at least one evaluation set"
            )
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="metadata"),
        )
        if self.episode_id is not None:
            object.__setattr__(
                self,
                "episode_id",
                nonempty_name(self.episode_id, field="episode_id"),
            )

    @property
    def evaluation_names(self) -> tuple[str, ...]:
        """Names of the declared evaluation sets, in construction order."""
        return tuple(self.evaluations)

    @property
    def identity_key(self) -> EpisodeIdentityKey:
        """Dataset-independent identity used by corpus and compiler joins."""
        return _episode_identity_key(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration only; dataset contents are intentionally omitted."""
        return {
            "name": self.name,
            "episode_id": self.episode_id,
            "evaluation_names": list(self.evaluation_names),
            "metadata": to_json_safe(self.metadata, location="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LearningEpisode:
        """Restore configuration with ``None`` placeholders for dataset objects."""
        try:
            names = data["evaluation_names"]
            evaluations = {name: None for name in names}
            return cls(
                name=data["name"],
                train=None,
                evaluations=evaluations,
                metadata=data.get("metadata", {}),
                episode_id=data.get("episode_id"),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized LearningEpisode") from error

    def to_json(
        self,
        destination: str | PathLike[str] | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        return write_json(self.to_dict(), destination, indent=indent)

    @classmethod
    def from_json(cls, source: str | PathLike[str]) -> LearningEpisode:
        return cls.from_dict(read_json(source))

    def __repr__(self) -> str:
        return (
            f"LearningEpisode(name={self.name!r}, train={type(self.train).__name__}, "
            f"evaluations={list(self.evaluations)!r}, episode_id={self.episode_id!r})"
        )


def _episode_identity_key(episode: LearningEpisode) -> EpisodeIdentityKey:
    """Return the conservative, dataset-independent identity used by collections."""
    if episode.episode_id is not None:
        return ("episode_id", episode.episode_id)
    return ("configuration", json_dumps(episode.to_dict(), indent=None))
