"""Declarative model and base-state context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

from ._validation import nonempty_name
from .errors import ValidationError
from .serialization import immutable_json_mapping, read_json, to_json_safe, write_json


@dataclass(frozen=True)
class ModelContext:
    """A declarative description of the pre-adaptation model/base state.

    This record identifies a model state; it does not contain, load, or inspect
    model weights and has no dependency on a model framework.
    """

    model_id: str
    revision: str | None = None
    base_state_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            nonempty_name(self.model_id, field="model_id"),
        )
        if self.revision is not None:
            object.__setattr__(
                self,
                "revision",
                nonempty_name(self.revision, field="revision"),
            )
        if self.base_state_id is not None:
            object.__setattr__(
                self,
                "base_state_id",
                nonempty_name(self.base_state_id, field="base_state_id"),
            )
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="metadata"),
        )

    @property
    def identity_key(self) -> tuple[str, str | None, str | None]:
        """Identity used to hold model/base state fixed within a study."""
        return (self.model_id, self.revision, self.base_state_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "base_state_id": self.base_state_id,
            "metadata": to_json_safe(self.metadata, location="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelContext:
        try:
            return cls(
                model_id=data["model_id"],
                revision=data.get("revision"),
                base_state_id=data.get("base_state_id"),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized ModelContext") from error

    def to_json(
        self,
        destination: str | PathLike[str] | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        return write_json(self.to_dict(), destination, indent=indent)

    @classmethod
    def from_json(cls, source: str | PathLike[str]) -> ModelContext:
        return cls.from_dict(read_json(source))

    def __repr__(self) -> str:
        return (
            f"ModelContext(model_id={self.model_id!r}, revision={self.revision!r}, "
            f"base_state_id={self.base_state_id!r})"
        )
