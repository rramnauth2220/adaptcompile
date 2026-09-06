"""Multidimensional behavioral geometry."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from os import PathLike
from typing import TYPE_CHECKING, Any, cast

from ._validation import score_mapping
from .errors import ValidationError
from .serialization import read_json, write_json

if TYPE_CHECKING:
    from pandas import DataFrame


class AdaptationGeometry(Mapping[str, float]):
    """An immutable, arbitrary-dimensional mapping of behavioral measurements."""

    __slots__ = ("_metrics",)

    def __init__(self, metrics: Mapping[str, float]) -> None:
        normalized = score_mapping(metrics, field="metrics")
        if not normalized:
            raise ValidationError("geometry must contain at least one metric")
        self._metrics = normalized

    @property
    def metrics(self) -> Mapping[str, float]:
        """A read-only metric mapping."""
        return self._metrics

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(self._metrics)

    def __getitem__(self, name: str) -> float:
        try:
            return self._metrics[name]
        except KeyError as error:
            raise KeyError(f"unknown geometry metric {name!r}") from error

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"AdaptationGeometry({dict(self._metrics)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdaptationGeometry):
            return NotImplemented
        return dict(self._metrics) == dict(other._metrics)

    def __hash__(self) -> int:
        return hash(frozenset(self._metrics.items()))

    def to_dict(self) -> dict[str, float]:
        return dict(self._metrics)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdaptationGeometry:
        return cls(data)

    def to_json(
        self,
        destination: str | PathLike[str] | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        return write_json(self.to_dict(), destination, indent=indent)

    @classmethod
    def from_json(cls, source: str | PathLike[str]) -> AdaptationGeometry:
        return cls.from_dict(read_json(source))

    def compare(self, other: AdaptationGeometry) -> AdaptationGeometry:
        """Return ``self - other`` after requiring identical metric dimensions."""
        self._require_comparable(other)
        return AdaptationGeometry(
            {name: value - other[name] for name, value in self._metrics.items()}
        )

    def delta(self, baseline: AdaptationGeometry) -> AdaptationGeometry:
        """Alias for :meth:`compare`, naming *baseline* explicitly."""
        return self.compare(baseline)

    def to_dataframe(self) -> DataFrame:
        """Return a one-row pandas DataFrame (requires the ``dataframe`` extra)."""
        try:
            from pandas import DataFrame
        except ImportError as error:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() requires pandas; install adaptcompile[dataframe]"
            ) from error
        return cast("DataFrame", DataFrame([self.to_dict()]))

    def _require_comparable(self, other: AdaptationGeometry) -> None:
        if not isinstance(other, AdaptationGeometry):
            raise TypeError("other must be an AdaptationGeometry")
        own = set(self._metrics)
        theirs = set(other._metrics)
        if own != theirs:
            missing = sorted(own - theirs)
            extra = sorted(theirs - own)
            raise ValidationError(
                "geometry metrics do not match "
                f"(missing from other={missing}, extra in other={extra})"
            )
