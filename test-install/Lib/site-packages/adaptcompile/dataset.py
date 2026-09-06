"""Corpus-level collections of observed adaptations."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from .episode import EpisodeIdentityKey, _episode_identity_key
from .errors import ValidationError
from .result import AdaptationResult

if TYPE_CHECKING:
    from pandas import DataFrame

ModelIdentityKey = tuple[str, str | None, str | None]
GroupKey = TypeVar("GroupKey", bound=Hashable)


class AdaptationDataset(Sequence[AdaptationResult]):
    """An immutable corpus of observations spanning models, episodes, and programs.

    Repeated ``(model, episode, program)`` observations are retained because seeds,
    experiment runs, and measured outcomes may legitimately differ.
    """

    __slots__ = ("_results",)

    def __init__(self, results: Iterable[AdaptationResult]) -> None:
        collected = tuple(results)
        if not collected:
            raise ValidationError("an AdaptationDataset requires at least one result")
        if any(not isinstance(result, AdaptationResult) for result in collected):
            raise ValidationError(
                "all dataset entries must be AdaptationResult instances"
            )
        self._results = collected

    @property
    def results(self) -> tuple[AdaptationResult, ...]:
        return self._results

    @overload
    def __getitem__(self, index: int) -> AdaptationResult: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[AdaptationResult, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> AdaptationResult | tuple[AdaptationResult, ...]:
        return self._results[index]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[AdaptationResult]:
        return iter(self._results)

    def __repr__(self) -> str:
        return f"AdaptationDataset(n_results={len(self)})"

    def by_model(self) -> dict[ModelIdentityKey, AdaptationDataset]:
        """Group by ``ModelContext.identity_key``."""
        return self._grouped(lambda result: result.model_context.identity_key)

    def by_episode(self) -> dict[EpisodeIdentityKey, AdaptationDataset]:
        """Group by explicit ID or conservative serialized episode configuration."""
        return self._grouped(lambda result: _episode_identity_key(result.episode))

    def by_family(self) -> dict[str | None, AdaptationDataset]:
        """Group by conceptual fingerprint; ``None`` retains familyless programs."""
        return self._grouped(
            lambda result: (
                result.program.family.fingerprint
                if result.program.family is not None
                else None
            )
        )

    def by_program(self) -> dict[str, AdaptationDataset]:
        """Group by concrete executable program fingerprint."""
        return self._grouped(lambda result: result.program.fingerprint)

    def filter(
        self, predicate: Callable[[AdaptationResult], bool]
    ) -> AdaptationDataset:
        """Return observations matching *predicate*.

        Like direct construction, an empty selection raises :class:`ValidationError`.
        """
        return AdaptationDataset(result for result in self if predicate(result))

    def to_records(self) -> list[dict[str, Any]]:
        """Return unambiguous flat observation records with nested run metadata."""
        records: list[dict[str, Any]] = []
        for result in self:
            family = result.program.family
            record: dict[str, Any] = {
                "model_id": result.model_context.model_id,
                "model_revision": result.model_context.revision,
                "base_state_id": result.model_context.base_state_id,
                "episode": result.episode.name,
                "episode_id": result.episode.episode_id,
                "program_family_fingerprint": (
                    family.fingerprint if family is not None else None
                ),
                "program_family_name": family.name if family is not None else None,
                "program_family_method": (
                    family.method if family is not None else None
                ),
                "program_family_parameters": (
                    family.to_dict()["parameters"] if family is not None else None
                ),
                "program_fingerprint": result.program.fingerprint,
                "program_name": result.program.name,
                "method": result.program.method,
                "program_parameters": result.program.to_dict()["parameters"],
                "metadata": result.to_dict()["metadata"],
            }
            record.update(
                {
                    f"before_{name}": value
                    for name, value in result.before_geometry.items()
                }
            )
            record.update(
                {
                    f"after_{name}": value
                    for name, value in result.after_geometry.items()
                }
            )
            record.update(
                {
                    f"delta_{name}": value
                    for name, value in result.delta_geometry.items()
                }
            )
            records.append(record)
        return records

    def to_dataframe(self) -> DataFrame:
        """Return a pandas DataFrame (requires the ``dataframe`` extra)."""
        try:
            from pandas import DataFrame
        except ImportError as error:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() requires pandas; install adaptcompile[dataframe]"
            ) from error
        return cast("DataFrame", DataFrame(self.to_records()))

    def _grouped(
        self, key: Callable[[AdaptationResult], GroupKey]
    ) -> dict[GroupKey, AdaptationDataset]:
        grouped: dict[GroupKey, list[AdaptationResult]] = {}
        for result in self:
            grouped.setdefault(key(result), []).append(result)
        return {
            identity: AdaptationDataset(results)
            for identity, results in grouped.items()
        }
