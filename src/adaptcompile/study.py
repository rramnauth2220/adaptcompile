"""Comparison utilities for candidate adaptation programs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, cast, overload

from ._validation import nonempty_name
from .episode import LearningEpisode, _episode_identity_key
from .errors import ValidationError
from .model import ModelContext
from .result import AdaptationResult

if TYPE_CHECKING:
    from pandas import DataFrame


class AdaptationStudy(Sequence[AdaptationResult]):
    """Candidate programs evaluated with one fixed model context and episode."""

    __slots__ = ("_results",)

    def __init__(self, results: Iterable[AdaptationResult]) -> None:
        collected = tuple(results)
        if not collected:
            raise ValidationError("an AdaptationStudy requires at least one result")
        if any(not isinstance(result, AdaptationResult) for result in collected):
            raise ValidationError(
                "all study entries must be AdaptationResult instances"
            )
        reference_episode = collected[0].episode
        if any(
            not self._same_episode(reference_episode, result.episode)
            for result in collected[1:]
        ):
            raise ValidationError(
                "episode mismatch: all study results must share an episode "
                "identity; explicit "
                "episode_id values or serialized episode configurations differ"
            )
        reference_context = collected[0].model_context
        if any(
            result.model_context.identity_key != reference_context.identity_key
            for result in collected[1:]
        ):
            raise ValidationError(
                "model-context mismatch: all study results must share model_id, "
                "revision, and base_state_id"
            )
        self._results = collected

    @property
    def results(self) -> tuple[AdaptationResult, ...]:
        return self._results

    @property
    def episode(self) -> LearningEpisode:
        return self._results[0].episode

    @property
    def model_context(self) -> ModelContext:
        return self._results[0].model_context

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
        return (
            f"AdaptationStudy(n_results={len(self)}, "
            f"model_id={self.model_context.model_id!r}, "
            f"episode={self.episode.name!r})"
        )

    def best(self, metric: str, *, maximize: bool = True) -> AdaptationResult:
        """Return the best result for one post-adaptation metric."""
        metric = nonempty_name(metric, field="metric")
        self._require_metrics((metric,))
        choose = max if maximize else min
        return choose(self._results, key=lambda result: result.after[metric])

    def rank(
        self, metric: str, *, maximize: bool = True
    ) -> tuple[AdaptationResult, ...]:
        """Return results ordered best-to-worst for one post-adaptation metric."""
        metric = nonempty_name(metric, field="metric")
        self._require_metrics((metric,))
        return tuple(
            sorted(
                self._results,
                key=lambda result: result.after[metric],
                reverse=maximize,
            )
        )

    def pareto_front(
        self,
        *,
        maximize: Iterable[str] = (),
        minimize: Iterable[str] = (),
    ) -> tuple[AdaptationResult, ...]:
        """Return non-dominated results in their original study order."""
        maximize_names = tuple(maximize)
        minimize_names = tuple(minimize)
        all_names = maximize_names + minimize_names
        if not all_names:
            raise ValidationError("pareto_front requires at least one metric")
        if len(set(all_names)) != len(all_names):
            raise ValidationError(
                "Pareto metrics must be unique and cannot be both maximized "
                "and minimized"
            )
        for name in all_names:
            nonempty_name(name, field="Pareto metric")
        self._require_metrics(all_names)

        front: list[AdaptationResult] = []
        for candidate in self._results:
            dominated = any(
                other is not candidate
                and self._dominates(other, candidate, maximize_names, minimize_names)
                for other in self._results
            )
            if not dominated:
                front.append(candidate)
        return tuple(front)

    def to_dataframe(self) -> DataFrame:
        """Return one row per result with post-adaptation metrics as columns."""
        try:
            from pandas import DataFrame
        except ImportError as error:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() requires pandas; install adaptcompile[dataframe]"
            ) from error
        metric_names = self._common_metric_names()
        rows = []
        for result in self._results:
            row: dict[str, str | float | None] = {
                "model_id": result.model_context.model_id,
                "model_revision": result.model_context.revision,
                "base_state_id": result.model_context.base_state_id,
                "episode": result.episode.name,
                "episode_id": result.episode.episode_id,
                "program": result.program.name,
                "method": result.program.method,
                "program_fingerprint": result.program.fingerprint,
            }
            row.update({name: result.after[name] for name in metric_names})
            rows.append(row)
        return cast("DataFrame", DataFrame(rows))

    def _common_metric_names(self) -> tuple[str, ...]:
        expected = tuple(self._results[0].after)
        expected_set = set(expected)
        for index, result in enumerate(self._results[1:], start=1):
            if set(result.after) != expected_set:
                raise ValidationError(
                    "to_dataframe requires identical metric dimensions; "
                    f"result {index} has {sorted(result.after)}, "
                    f"expected {sorted(expected)}"
                )
        return expected

    def _require_metrics(self, names: Iterable[str]) -> None:
        for index, result in enumerate(self._results):
            missing = [name for name in names if name not in result.after]
            if missing:
                raise ValidationError(
                    f"result {index} ({result.program.name!r}) is missing "
                    f"metrics {missing}"
                )

    @staticmethod
    def _same_episode(left: LearningEpisode, right: LearningEpisode) -> bool:
        # Result construction guarantees these types. Avoid equality because episode
        # dataset objects may define unsafe or array-valued equality operations.
        return _episode_identity_key(left) == _episode_identity_key(right)

    @staticmethod
    def _dominates(
        left: AdaptationResult,
        right: AdaptationResult,
        maximize: tuple[str, ...],
        minimize: tuple[str, ...],
    ) -> bool:
        no_worse = all(left.after[name] >= right.after[name] for name in maximize)
        no_worse = no_worse and all(
            left.after[name] <= right.after[name] for name in minimize
        )
        strictly_better = any(
            left.after[name] > right.after[name] for name in maximize
        ) or any(left.after[name] < right.after[name] for name in minimize)
        return no_worse and strictly_better
