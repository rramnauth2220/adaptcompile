"""Small JSON serialization helpers used by public records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from .errors import SerializationError

JSONScalar = None | bool | int | float | str
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def to_json_safe(value: Any, *, location: str = "value") -> JSONValue:
    """Return a detached JSON-safe copy or raise :class:`SerializationError`."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError(f"{location} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationError(
                    f"{location} has non-string mapping key {key!r}"
                )
            converted[key] = to_json_safe(item, location=f"{location}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            to_json_safe(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise SerializationError(
        f"{location} contains unsupported value of type {type(value).__name__}"
    )


def freeze_json(value: JSONValue) -> Any:
    """Recursively make a JSON-safe value immutable."""
    if isinstance(value, dict):
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    return value


def immutable_json_mapping(
    value: Mapping[str, Any], *, location: str
) -> Mapping[str, Any]:
    """Validate, detach, and recursively freeze a string-keyed mapping."""
    safe = to_json_safe(value, location=location)
    # Defensive: the annotation alone is not runtime proof.
    if not isinstance(safe, dict):
        raise SerializationError(f"{location} must be a mapping")
    return cast(Mapping[str, Any], freeze_json(safe))


def json_dumps(data: Mapping[str, Any], *, indent: int | None = 2) -> str:
    """Serialize a mapping after enforcing strict, portable JSON values."""
    safe = to_json_safe(data)
    return json.dumps(safe, indent=indent, sort_keys=True, allow_nan=False)


def write_json(
    data: Mapping[str, Any],
    destination: str | PathLike[str] | None = None,
    *,
    indent: int | None = 2,
) -> str:
    """Return JSON text and optionally write it to *destination*."""
    text = json_dumps(data, indent=indent)
    if destination is not None:
        Path(destination).write_text(text + "\n", encoding="utf-8")
    return text


def read_json(source: str | PathLike[str]) -> dict[str, Any]:
    """Read a JSON object from text or a filesystem path.

    Strings beginning with ``{`` are treated as JSON; other strings are paths.
    ``PathLike`` values are always paths.
    """
    if isinstance(source, str) and source.lstrip().startswith("{"):
        text = source
    else:
        text = Path(source).read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as error:
        raise SerializationError(f"invalid JSON: {error.msg}") from error
    if not isinstance(loaded, dict):
        raise SerializationError("serialized adaptcompile record must be a JSON object")
    return loaded
