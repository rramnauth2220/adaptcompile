from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

import adaptcompile
import adaptcompile.compiler as compiler
import adaptcompile.compiler.predictors as predictor_module
from adaptcompile import AdaptationGeometry, ValidationError
from adaptcompile.compiler import (
    CompilerDataset,
    CompilerRecord,
    GeometryPrediction,
    GeometryPredictor,
)
from adaptcompile.compiler.predictors import RidgeGeometryPredictor


def _record(*, target_kind: str = "delta") -> CompilerRecord:
    return CompilerRecord(
        model_key=("model", "revision", "base"),
        episode_key=("episode_id", "episode"),
        family_fingerprint=None,
        program_fingerprint="program",
        features={"program.value": 1.0, "baseline.gain": 0.25},
        target=AdaptationGeometry({"gain": 0.5}),
        target_kind=target_kind,  # type: ignore[arg-type]
        metadata={"seed": 1},
    )


def test_prediction_public_api_is_explicit_and_not_root_reexported() -> None:
    assert set(compiler.__all__) == {
        "CompilerDataset",
        "CompilerRecord",
        "GeometryPrediction",
        "GeometryPredictor",
        "build_compiler_dataset",
    }
    assert predictor_module.__all__ == ["RidgeGeometryPredictor"]
    assert not hasattr(adaptcompile, "GeometryPrediction")
    assert not hasattr(adaptcompile, "GeometryPredictor")
    assert not hasattr(adaptcompile, "RidgeGeometryPredictor")
    assert GeometryPredictor.__name__ == "GeometryPredictor"
    assert RidgeGeometryPredictor.__name__ == "RidgeGeometryPredictor"


def test_after_prediction_is_immutable_and_roundtrips() -> None:
    geometry = AdaptationGeometry({"gain": 1.4, "cost": -0.2})
    prediction = GeometryPrediction(
        model_key=("model", "revision", "base"),
        episode_key=("episode_id", "episode"),
        family_fingerprint="family",
        program_fingerprint="program",
        predicted_target=geometry,
        predicted_geometry=AdaptationGeometry({"cost": -0.2, "gain": 1.4}),
        target_kind="after",
        metadata={"source": "synthetic", "nested": {"fold": 1}},
    )

    assert prediction.predicted_target == prediction.predicted_geometry
    assert prediction.predicted_geometry["gain"] == 1.4
    assert prediction.predicted_geometry["cost"] == -0.2
    assert GeometryPrediction.from_dict(prediction.to_dict()) == prediction
    json.dumps(prediction.to_dict())
    with pytest.raises(FrozenInstanceError):
        prediction.target_kind = "delta"  # type: ignore[misc]
    with pytest.raises(TypeError):
        prediction.metadata["source"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        prediction.metadata["nested"]["fold"] = 2  # type: ignore[index]


def test_delta_prediction_preserves_negative_and_unbounded_values() -> None:
    prediction = GeometryPrediction(
        model_key=("model", None, None),
        episode_key=("configuration", "{}"),
        family_fingerprint=None,
        program_fingerprint="program",
        predicted_target=AdaptationGeometry({"gain": 1.5, "retention": -0.4}),
        predicted_geometry=AdaptationGeometry({"gain": 2.1, "retention": -0.1}),
        target_kind="delta",
    )

    assert prediction.predicted_target["retention"] == -0.4
    assert prediction.predicted_geometry["gain"] == 2.1


def test_prediction_validation_rejects_inconsistent_values() -> None:
    common = {
        "model_key": ("model", None, None),
        "episode_key": ("episode_id", "episode"),
        "family_fingerprint": None,
        "program_fingerprint": "program",
        "metadata": {},
    }
    with pytest.raises(ValidationError, match="must equal"):
        GeometryPrediction(
            **common,
            predicted_target=AdaptationGeometry({"gain": 0.2}),
            predicted_geometry=AdaptationGeometry({"gain": 0.3}),
            target_kind="after",
        )
    with pytest.raises(ValidationError, match="metrics must match"):
        GeometryPrediction(
            **common,
            predicted_target=AdaptationGeometry({"gain": 0.2}),
            predicted_geometry=AdaptationGeometry({"other": 0.3}),
            target_kind="delta",
        )
    with pytest.raises(ValidationError, match="target_kind"):
        GeometryPrediction(
            **common,
            predicted_target=AdaptationGeometry({"gain": 0.2}),
            predicted_geometry=AdaptationGeometry({"gain": 0.3}),
            target_kind="utility",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="malformed"):
        GeometryPrediction.from_dict({"target_kind": "after"})


@pytest.mark.parametrize("alpha", [True, -1.0, float("nan"), float("inf")])
def test_ridge_configuration_validation(alpha: object) -> None:
    with pytest.raises(ValidationError, match="alpha"):
        RidgeGeometryPredictor(alpha=alpha)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="fit_intercept"):
        RidgeGeometryPredictor(fit_intercept=1)  # type: ignore[arg-type]


def test_unfitted_predictor_fails_before_optional_dependency_is_needed() -> None:
    predictor = RidgeGeometryPredictor()
    record = _record()
    dataset = CompilerDataset([record])

    assert predictor.is_fitted is False
    with pytest.raises(ValidationError, match="must be fitted"):
        predictor.predict(
            features=record.features,
            model_key=record.model_key,
            episode_key=record.episode_key,
            family_fingerprint=record.family_fingerprint,
            program_fingerprint=record.program_fingerprint,
        )
    with pytest.raises(ValidationError, match="must be fitted"):
        predictor.predict_dataset(dataset)
    with pytest.raises(ValidationError, match="must be fitted"):
        _ = predictor.feature_names


def test_missing_sklearn_failure_is_lazy_and_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> object:
        if name == "sklearn.linear_model":
            raise ImportError("synthetically unavailable")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(predictor_module, "import_module", unavailable)
    predictor = RidgeGeometryPredictor()
    with pytest.raises(ImportError, match=r'pip install "adaptcompile\[predict\]"'):
        predictor.fit(CompilerDataset([_record()]))
