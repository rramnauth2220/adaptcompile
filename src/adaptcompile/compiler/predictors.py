"""Optional reference predictor implementations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol, cast

from .._validation import score
from ..dataset import ModelIdentityKey
from ..episode import EpisodeIdentityKey
from ..errors import ValidationError
from ..geometry import AdaptationGeometry
from .data import CompilerDataset, TargetKind, _validated_feature_mapping
from .prediction import GeometryPrediction

if TYPE_CHECKING:
    from .prediction import GeometryPredictor

__all__ = ["RidgeGeometryPredictor"]


class _RidgeEstimator(Protocol):
    def fit(
        self, features: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]
    ) -> object: ...

    def predict(self, features: Sequence[Sequence[float]]) -> object: ...


class RidgeGeometryPredictor:
    """Small multi-output ridge reference implementation.

    Scikit-learn is imported only when :meth:`fit` constructs the estimator.
    """

    __slots__ = (
        "_alpha",
        "_estimator",
        "_feature_names",
        "_fit_intercept",
        "_target_kind",
        "_target_names",
    )

    def __init__(self, *, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        normalized_alpha = score(alpha, field="alpha")
        if normalized_alpha < 0:
            raise ValidationError("alpha must be non-negative")
        if not isinstance(fit_intercept, bool):
            raise ValidationError("fit_intercept must be a boolean")
        self._alpha = normalized_alpha
        self._fit_intercept = fit_intercept
        self._estimator: _RidgeEstimator | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._target_names: tuple[str, ...] | None = None
        self._target_kind: TargetKind | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def fit_intercept(self) -> bool:
        return self._fit_intercept

    @property
    def is_fitted(self) -> bool:
        return self._estimator is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        if self._feature_names is None:  # pragma: no cover - guarded state invariant
            raise RuntimeError("fitted predictor has no feature schema")
        return self._feature_names

    @property
    def target_names(self) -> tuple[str, ...]:
        self._require_fitted()
        if self._target_names is None:  # pragma: no cover - guarded state invariant
            raise RuntimeError("fitted predictor has no target schema")
        return self._target_names

    @property
    def target_kind(self) -> TargetKind:
        self._require_fitted()
        if self._target_kind is None:  # pragma: no cover - guarded state invariant
            raise RuntimeError("fitted predictor has no target kind")
        return self._target_kind

    def fit(self, dataset: CompilerDataset) -> RidgeGeometryPredictor:
        """Fit one multi-output ridge model using the dataset's exact schemas."""
        if not isinstance(dataset, CompilerDataset):
            raise ValidationError("dataset must be a CompilerDataset")
        feature_names = dataset.feature_names
        target_names = dataset.target_names
        target_kind = dataset.target_kind
        if target_kind == "delta":
            missing_baselines = [
                f"baseline.{name}"
                for name in target_names
                if f"baseline.{name}" not in feature_names
            ]
            if missing_baselines:
                raise ValidationError(
                    f"delta prediction requires baseline features {missing_baselines}"
                )

        feature_rows = [
            [record.features[name] for name in feature_names] for record in dataset
        ]
        target_rows = [
            [record.target[name] for name in target_names] for record in dataset
        ]
        estimator = self._new_estimator()
        estimator.fit(feature_rows, target_rows)

        self._estimator = estimator
        self._feature_names = feature_names
        self._target_names = target_names
        self._target_kind = target_kind
        return self

    def predict(
        self,
        *,
        features: Mapping[str, float],
        model_key: ModelIdentityKey,
        episode_key: EpisodeIdentityKey,
        family_fingerprint: str | None,
        program_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> GeometryPrediction:
        """Predict from features and candidate identity, without an observed target."""
        self._require_fitted()
        normalized = _validated_feature_mapping(features)
        expected = set(self.feature_names)
        actual = set(normalized)
        if actual != expected:
            raise ValidationError(
                "prediction feature schema mismatch "
                f"(missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)})"
            )
        feature_row = [normalized[name] for name in self.feature_names]
        estimator = self._estimator
        if estimator is None:  # pragma: no cover - guarded by _require_fitted
            raise RuntimeError("fitted predictor has no estimator")
        raw_rows = cast(Sequence[object], estimator.predict([feature_row]))
        try:
            raw_target = raw_rows[0]
        except (IndexError, TypeError) as error:
            raise ValidationError(
                "ridge predictor returned no prediction row"
            ) from error
        raw_values: tuple[object, ...]
        if len(self.target_names) == 1 and isinstance(raw_target, Real):
            raw_values = (raw_target,)
        else:
            try:
                raw_values = tuple(cast(Iterable[object], raw_target))
            except TypeError as error:
                raise ValidationError(
                    "ridge predictor returned a malformed target row"
                ) from error
        if len(raw_values) != len(self.target_names):
            raise ValidationError(
                "ridge predictor returned an incompatible target dimension"
            )
        predicted_target = AdaptationGeometry(
            {
                name: score(value, field=f"predicted target {name!r}")
                for name, value in zip(self.target_names, raw_values, strict=True)
            }
        )
        if self.target_kind == "after":
            predicted_geometry = predicted_target
        else:
            missing_baselines = [
                f"baseline.{name}"
                for name in self.target_names
                if f"baseline.{name}" not in normalized
            ]
            if missing_baselines:  # pragma: no cover - fitted schema normally guards it
                raise ValidationError(
                    f"delta prediction requires baseline features {missing_baselines}"
                )
            predicted_geometry = AdaptationGeometry(
                {
                    name: normalized[f"baseline.{name}"] + predicted_target[name]
                    for name in self.target_names
                }
            )
        return GeometryPrediction(
            model_key=model_key,
            episode_key=episode_key,
            family_fingerprint=family_fingerprint,
            program_fingerprint=program_fingerprint,
            predicted_target=predicted_target,
            predicted_geometry=predicted_geometry,
            target_kind=self.target_kind,
            metadata=metadata or {},
        )

    def predict_dataset(
        self, dataset: CompilerDataset
    ) -> tuple[GeometryPrediction, ...]:
        """Predict each record in order without reading its observed target."""
        self._require_fitted()
        if not isinstance(dataset, CompilerDataset):
            raise ValidationError("dataset must be a CompilerDataset")
        if dataset.feature_names != self.feature_names:
            raise ValidationError(
                "prediction dataset feature schema does not match fitted schema"
            )
        return tuple(
            self.predict(
                features=record.features,
                model_key=record.model_key,
                episode_key=record.episode_key,
                family_fingerprint=record.family_fingerprint,
                program_fingerprint=record.program_fingerprint,
                metadata=record.metadata,
            )
            for record in dataset
        )

    def _new_estimator(self) -> _RidgeEstimator:
        try:
            module = import_module("sklearn.linear_model")
        except ImportError as error:
            raise ImportError(
                "RidgeGeometryPredictor requires scikit-learn; "
                'pip install "adaptcompile[predict]"'
            ) from error
        ridge_type = cast(Callable[..., _RidgeEstimator], module.Ridge)
        return ridge_type(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValidationError(
                "RidgeGeometryPredictor must be fitted before prediction"
            )


if TYPE_CHECKING:
    _protocol_check: GeometryPredictor = RidgeGeometryPredictor()
