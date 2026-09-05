"""Internal validation helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType
from typing import Any

from .errors import ValidationError


def nonempty_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    return value


def string_keyed_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be a mapping")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        nonempty_name(key, field=f"{field} key")
        copied[key] = item
    return MappingProxyType(copied)


def score(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{field} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValidationError(f"{field} must be finite")
    return converted


def score_mapping(value: Any, *, field: str) -> Mapping[str, float]:
    mapping = string_keyed_mapping(value, field=field)
    return MappingProxyType(
        {
            name: score(item, field=f"{field}[{name!r}]")
            for name, item in mapping.items()
        }
    )
