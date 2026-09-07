"""Dependency-free program-selection values and interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, cast

from .._validation import nonempty_name, score
from ..dataset import ModelIdentityKey
from ..episode import EpisodeIdentityKey
from ..errors import ValidationError
from ..serialization import immutable_json_mapping, to_json_safe
from .data import _validate_episode_key, _validate_model_key
from .prediction import GeometryPrediction


def _positive_weights(value: Any, *, field_name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be a mapping")
    normalized: dict[str, float] = {}
    for raw_metric, raw_weight in value.items():
        metric = nonempty_name(raw_metric, field=f"{field_name} metric")
        weight = score(raw_weight, field=f"{field_name}[{metric!r}]")
        if weight <= 0:
            raise ValidationError(f"{field_name}[{metric!r}] must be greater than zero")
        normalized[metric] = weight
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class GeometryConstraint:
    """Inclusive acceptable bounds for one post-adaptation geometry metric."""

    metric: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric", nonempty_name(self.metric, field="constraint metric")
        )
        if self.minimum is None and self.maximum is None:
            raise ValidationError("a geometry constraint requires at least one bound")
        if self.minimum is not None:
            object.__setattr__(
                self,
                "minimum",
                score(self.minimum, field=f"minimum for {self.metric!r}"),
            )
        if self.maximum is not None:
            object.__setattr__(
                self,
                "maximum",
                score(self.maximum, field=f"maximum for {self.metric!r}"),
            )
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValidationError("constraint minimum must not exceed maximum")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeometryConstraint:
        try:
            return cls(
                metric=data["metric"],
                minimum=data.get("minimum"),
                maximum=data.get("maximum"),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized GeometryConstraint") from error


@dataclass(frozen=True)
class SelectionObjective:
    """Explicit weighted utility terms and hard geometry constraints."""

    maximize: Mapping[str, float] = field(default_factory=dict)
    minimize: Mapping[str, float] = field(default_factory=dict)
    constraints: tuple[GeometryConstraint, ...] = ()

    def __post_init__(self) -> None:
        maximize = _positive_weights(self.maximize, field_name="maximize")
        minimize = _positive_weights(self.minimize, field_name="minimize")
        if not maximize and not minimize:
            raise ValidationError("a selection objective requires at least one metric")
        overlap = set(maximize).intersection(minimize)
        if overlap:
            raise ValidationError(
                "objective metrics cannot be both maximized and minimized: "
                f"{sorted(overlap)}"
            )
        try:
            constraints = tuple(self.constraints)
        except TypeError as error:
            raise ValidationError("constraints must be a sequence") from error
        if any(not isinstance(item, GeometryConstraint) for item in constraints):
            raise ValidationError(
                "all objective constraints must be GeometryConstraint instances"
            )
        constrained_metrics = [item.metric for item in constraints]
        if len(set(constrained_metrics)) != len(constrained_metrics):
            raise ValidationError("constraint metrics must be unique")
        object.__setattr__(self, "maximize", maximize)
        object.__setattr__(self, "minimize", minimize)
        object.__setattr__(self, "constraints", constraints)

    @property
    def required_metrics(self) -> frozenset[str]:
        return frozenset(
            (
                *self.maximize,
                *self.minimize,
                *(item.metric for item in self.constraints),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        safe = to_json_safe(
            {
                "maximize": self.maximize,
                "minimize": self.minimize,
                "constraints": [item.to_dict() for item in self.constraints],
            }
        )
        if not isinstance(safe, dict):  # pragma: no cover - construction guarantees it
            raise RuntimeError("objective serialization must produce an object")
        return cast(dict[str, Any], safe)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionObjective:
        try:
            raw_constraints = data.get("constraints", ())
            return cls(
                maximize=data.get("maximize", {}),
                minimize=data.get("minimize", {}),
                constraints=tuple(
                    GeometryConstraint.from_dict(item) for item in raw_constraints
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized SelectionObjective") from error


@dataclass(frozen=True)
class CandidateScore:
    """Utility and hard-constraint evaluation for one prediction."""

    prediction: GeometryPrediction
    utility: float
    feasible: bool
    violated_constraints: tuple[GeometryConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, GeometryPrediction):
            raise ValidationError("prediction must be a GeometryPrediction")
        object.__setattr__(self, "utility", score(self.utility, field="utility"))
        if not isinstance(self.feasible, bool):
            raise ValidationError("feasible must be a boolean")
        try:
            violations = tuple(self.violated_constraints)
        except TypeError as error:
            raise ValidationError("violated_constraints must be a sequence") from error
        if any(not isinstance(item, GeometryConstraint) for item in violations):
            raise ValidationError(
                "violated_constraints must contain GeometryConstraint instances"
            )
        if self.feasible != (not violations):
            raise ValidationError(
                "feasible must be true exactly when violated_constraints is empty"
            )
        object.__setattr__(self, "violated_constraints", violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction.to_dict(),
            "utility": self.utility,
            "feasible": self.feasible,
            "violated_constraints": [
                item.to_dict() for item in self.violated_constraints
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateScore:
        try:
            return cls(
                prediction=GeometryPrediction.from_dict(data["prediction"]),
                utility=data["utility"],
                feasible=data["feasible"],
                violated_constraints=tuple(
                    GeometryConstraint.from_dict(item)
                    for item in data.get("violated_constraints", ())
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized CandidateScore") from error


@dataclass(frozen=True)
class ProgramSelection:
    """Complete ranked evaluation for one model-and-episode decision context."""

    model_key: ModelIdentityKey
    episode_key: EpisodeIdentityKey
    objective: SelectionObjective
    selected: CandidateScore | None
    candidates: tuple[CandidateScore, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_key", _validate_model_key(self.model_key))
        object.__setattr__(self, "episode_key", _validate_episode_key(self.episode_key))
        if not isinstance(self.objective, SelectionObjective):
            raise ValidationError("objective must be a SelectionObjective")
        try:
            candidates = tuple(self.candidates)
        except TypeError as error:
            raise ValidationError("candidates must be a sequence") from error
        if not candidates:
            raise ValidationError("a program selection requires at least one candidate")
        if any(not isinstance(item, CandidateScore) for item in candidates):
            raise ValidationError("candidates must contain CandidateScore instances")
        if any(
            item.prediction.model_key != self.model_key
            or item.prediction.episode_key != self.episode_key
            for item in candidates
        ):
            raise ValidationError("candidate identities do not match the selection")
        expected = tuple(
            sorted(candidates, key=lambda item: (not item.feasible, -item.utility))
        )
        if candidates != expected:
            raise ValidationError("candidates must use the required ranking order")
        if self.selected is None:
            if any(item.feasible for item in candidates):
                raise ValidationError(
                    "selected cannot be None when a candidate is feasible"
                )
        elif not isinstance(self.selected, CandidateScore):
            raise ValidationError("selected must be a CandidateScore or None")
        elif not self.selected.feasible or self.selected != candidates[0]:
            raise ValidationError(
                "selected must be the highest-ranked feasible candidate"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="selection metadata"),
        )

    @property
    def has_feasible_candidate(self) -> bool:
        return self.selected is not None

    @property
    def selected_prediction(self) -> GeometryPrediction | None:
        return None if self.selected is None else self.selected.prediction

    def to_dict(self) -> dict[str, Any]:
        safe = to_json_safe(
            {
                "model_key": self.model_key,
                "episode_key": self.episode_key,
                "objective": self.objective.to_dict(),
                "selected": None if self.selected is None else self.selected.to_dict(),
                "candidates": [item.to_dict() for item in self.candidates],
                "metadata": self.metadata,
            }
        )
        if not isinstance(safe, dict):  # pragma: no cover - construction guarantees it
            raise RuntimeError("selection serialization must produce an object")
        return cast(dict[str, Any], safe)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProgramSelection:
        try:
            raw_selected = data.get("selected")
            return cls(
                model_key=tuple(data["model_key"]),
                episode_key=tuple(data["episode_key"]),
                objective=SelectionObjective.from_dict(data["objective"]),
                selected=(
                    None
                    if raw_selected is None
                    else CandidateScore.from_dict(raw_selected)
                ),
                candidates=tuple(
                    CandidateScore.from_dict(item) for item in data["candidates"]
                ),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized ProgramSelection") from error


class ProgramSelector(Protocol):
    """Structural interface for selection over predicted candidate geometries."""

    def select(
        self,
        predictions: Sequence[GeometryPrediction],
        objective: SelectionObjective,
    ) -> ProgramSelection: ...
