"""Conceptual adaptation program families."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

from ._validation import nonempty_name
from .errors import ValidationError
from .serialization import (
    immutable_json_mapping,
    read_json,
    to_json_safe,
    write_json,
)


@dataclass(frozen=True)
class ProgramFamily:
    """A conceptual adaptation choice, before model-specific realization.

    Structural equality includes every field. ``fingerprint`` identifies only the
    conceptual semantics expressed by ``method`` and ``parameters``.
    """

    name: str
    method: str
    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", nonempty_name(self.name, field="name"))
        object.__setattr__(self, "method", nonempty_name(self.method, field="method"))
        object.__setattr__(
            self,
            "parameters",
            immutable_json_mapping(self.parameters, location="parameters"),
        )
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="metadata"),
        )

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 identity derived from method and parameters only."""
        conceptual = {
            "method": self.method,
            "parameters": to_json_safe(self.parameters, location="parameters"),
        }
        canonical = json.dumps(
            conceptual,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "parameters": to_json_safe(self.parameters, location="parameters"),
            "metadata": to_json_safe(self.metadata, location="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProgramFamily:
        try:
            return cls(
                name=data["name"],
                method=data["method"],
                parameters=data["parameters"],
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized ProgramFamily") from error

    def to_json(
        self,
        destination: str | PathLike[str] | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        return write_json(self.to_dict(), destination, indent=indent)

    @classmethod
    def from_json(cls, source: str | PathLike[str]) -> ProgramFamily:
        return cls.from_dict(read_json(source))

    def __hash__(self) -> int:
        canonical = json.dumps(
            to_json_safe(self.to_dict()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hash(canonical)
