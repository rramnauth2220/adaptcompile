"""Declarative adaptation program specifications."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

from ._validation import nonempty_name
from .errors import ValidationError
from .family import ProgramFamily
from .serialization import (
    immutable_json_mapping,
    read_json,
    to_json_safe,
    write_json,
)


@dataclass(frozen=True)
class ProgramSpec:
    """A backend-independent declaration of an adaptation procedure.

    Structural equality includes every field. ``fingerprint`` is narrower: it
    identifies only executable semantics (``method`` and ``parameters``).
    """

    name: str
    method: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    family: ProgramFamily | None = None

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
        if self.family is not None:
            if not isinstance(self.family, ProgramFamily):
                raise ValidationError("family must be a ProgramFamily or None")
            if self.method != self.family.method:
                raise ValidationError(
                    "program method must match its ProgramFamily method "
                    f"({self.method!r} != {self.family.method!r})"
                )

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 identity derived from method and parameters only."""
        executable = {
            "method": self.method,
            "parameters": to_json_safe(self.parameters, location="parameters"),
        }
        canonical = json.dumps(
            executable,
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
            "family": self.family.to_dict() if self.family is not None else None,
            "metadata": to_json_safe(self.metadata, location="metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProgramSpec:
        try:
            family_data = data.get("family")
            family = (
                ProgramFamily.from_dict(family_data)
                if family_data is not None
                else None
            )
            return cls(
                name=data["name"],
                method=data["method"],
                parameters=data.get("parameters", {}),
                metadata=data.get("metadata", {}),
                family=family,
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized ProgramSpec") from error

    def to_json(
        self,
        destination: str | PathLike[str] | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        return write_json(self.to_dict(), destination, indent=indent)

    @classmethod
    def from_json(cls, source: str | PathLike[str]) -> ProgramSpec:
        return cls.from_dict(read_json(source))

    def __hash__(self) -> int:
        canonical = json.dumps(
            to_json_safe(self.to_dict()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hash(canonical)
