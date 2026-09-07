"""Dependency-free prediction values and interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from .._validation import nonempty_name
from ..dataset import ModelIdentityKey
from ..episode import EpisodeIdentityKey
from ..errors import ValidationError
from ..geometry import AdaptationGeometry
from ..serialization import immutable_json_mapping, to_json_safe
from .data import (
    CompilerDataset,
    TargetKind,
    _validate_episode_key,
    _validate_model_key,
)


@dataclass(frozen=True)
class GeometryPrediction:
    """Immutable predicted target and reconstructed post-adaptation geometry."""

    model_key: ModelIdentityKey
    episode_key: EpisodeIdentityKey
    family_fingerprint: str | None
    program_fingerprint: str
    predicted_target: AdaptationGeometry
    predicted_geometry: AdaptationGeometry
    target_kind: TargetKind
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_key", _validate_model_key(self.model_key))
        object.__setattr__(self, "episode_key", _validate_episode_key(self.episode_key))
        if self.family_fingerprint is not None:
            object.__setattr__(
                self,
                "family_fingerprint",
                nonempty_name(self.family_fingerprint, field="family_fingerprint"),
            )
        object.__setattr__(
            self,
            "program_fingerprint",
            nonempty_name(self.program_fingerprint, field="program_fingerprint"),
        )
        if not isinstance(self.predicted_target, AdaptationGeometry):
            raise ValidationError("predicted_target must be an AdaptationGeometry")
        if not isinstance(self.predicted_geometry, AdaptationGeometry):
            raise ValidationError("predicted_geometry must be an AdaptationGeometry")
        if self.target_kind not in {"after", "delta"}:
            raise ValidationError("target_kind must be 'after' or 'delta'")
        if set(self.predicted_target) != set(self.predicted_geometry):
            raise ValidationError(
                "predicted target and geometry metrics must match exactly"
            )
        if (
            self.target_kind == "after"
            and self.predicted_target != self.predicted_geometry
        ):
            raise ValidationError(
                "an 'after' predicted target must equal predicted_geometry"
            )
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe prediction record."""
        safe = to_json_safe(
            {
                "model_key": self.model_key,
                "episode_key": self.episode_key,
                "family_fingerprint": self.family_fingerprint,
                "program_fingerprint": self.program_fingerprint,
                "predicted_target": self.predicted_target.to_dict(),
                "predicted_geometry": self.predicted_geometry.to_dict(),
                "target_kind": self.target_kind,
                "metadata": self.metadata,
            }
        )
        if not isinstance(safe, dict):  # pragma: no cover - construction guarantees it
            raise RuntimeError("prediction serialization must produce an object")
        return cast(dict[str, Any], safe)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeometryPrediction:
        """Restore a prediction from :meth:`to_dict` output."""
        try:
            return cls(
                model_key=tuple(data["model_key"]),
                episode_key=tuple(data["episode_key"]),
                family_fingerprint=data.get("family_fingerprint"),
                program_fingerprint=data["program_fingerprint"],
                predicted_target=AdaptationGeometry.from_dict(data["predicted_target"]),
                predicted_geometry=AdaptationGeometry.from_dict(
                    data["predicted_geometry"]
                ),
                target_kind=data["target_kind"],
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError) as error:
            raise ValidationError("malformed serialized GeometryPrediction") from error


class GeometryPredictor(Protocol):
    """Structural interface for fitted compiler-geometry predictors."""

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def target_names(self) -> tuple[str, ...]: ...

    @property
    def target_kind(self) -> TargetKind: ...

    @property
    def is_fitted(self) -> bool: ...

    def fit(self, dataset: CompilerDataset) -> GeometryPredictor: ...

    def predict(
        self,
        *,
        features: Mapping[str, float],
        model_key: ModelIdentityKey,
        episode_key: EpisodeIdentityKey,
        family_fingerprint: str | None,
        program_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> GeometryPrediction: ...

    def predict_dataset(
        self, dataset: CompilerDataset
    ) -> tuple[GeometryPrediction, ...]: ...
