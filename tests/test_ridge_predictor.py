from __future__ import annotations

from collections.abc import Mapping

import pytest

pytest.importorskip("sklearn")

from adaptcompile import AdaptationGeometry, ValidationError
from adaptcompile.compiler import CompilerDataset, CompilerRecord
from adaptcompile.compiler.predictors import RidgeGeometryPredictor


def _dataset(
    rows: list[tuple[float, Mapping[str, float], float]],
    *,
    target_kind: str,
    metric: str = "gain",
) -> CompilerDataset:
    return CompilerDataset(
        [
            CompilerRecord(
                model_key=("model", "revision", "base"),
                episode_key=("episode_id", f"episode-{index}"),
                family_fingerprint=None,
                program_fingerprint=f"program-{index}",
                features={"program.x": x, **features},
                target=AdaptationGeometry({metric: target}),
                target_kind=target_kind,  # type: ignore[arg-type]
                metadata={"row": index},
            )
            for index, (x, features, target) in enumerate(rows)
        ]
    )


def _predict_first(predictor: RidgeGeometryPredictor, dataset: CompilerDataset):
    record = dataset[0]
    return predictor.predict(
        features=record.features,
        model_key=record.model_key,
        episode_key=record.episode_key,
        family_fingerprint=record.family_fingerprint,
        program_fingerprint=record.program_fingerprint,
        metadata=record.metadata,
    )


def test_fit_after_multioutput_and_preserve_identity() -> None:
    records = []
    for index, x in enumerate((-1.0, 0.0, 1.0, 2.0)):
        records.append(
            CompilerRecord(
                model_key=("model", "revision", "base"),
                episode_key=("episode_id", f"episode-{index}"),
                family_fingerprint="family",
                program_fingerprint=f"program-{index}",
                features={"program.x": x},
                target=AdaptationGeometry({"zeta": -x + 3.0, "alpha": 2.0 * x + 1.0}),
                target_kind="after",
            )
        )
    dataset = CompilerDataset(records)
    predictor = RidgeGeometryPredictor(alpha=0.0).fit(dataset)
    prediction = predictor.predict(
        features={"program.x": 3.0},
        model_key=("new-model", "revision", "base"),
        episode_key=("episode_id", "new-episode"),
        family_fingerprint="new-family",
        program_fingerprint="new-program",
    )

    assert predictor.is_fitted
    assert predictor.feature_names == dataset.feature_names
    assert predictor.target_names == ("alpha", "zeta")
    assert predictor.target_kind == "after"
    assert prediction.predicted_target == prediction.predicted_geometry
    assert prediction.predicted_geometry["alpha"] == pytest.approx(7.0)
    assert prediction.predicted_geometry["zeta"] == pytest.approx(0.0)
    assert prediction.model_key[0] == "new-model"
    assert prediction.episode_key == ("episode_id", "new-episode")
    assert prediction.family_fingerprint == "new-family"
    assert prediction.program_fingerprint == "new-program"


def test_delta_reconstruction_uses_baseline_and_preserves_negative() -> None:
    records = []
    for index, x in enumerate((0.0, 1.0, 2.0, 3.0)):
        records.append(
            CompilerRecord(
                model_key=("model", None, None),
                episode_key=("episode_id", f"episode-{index}"),
                family_fingerprint=None,
                program_fingerprint=f"program-{index}",
                features={
                    "program.x": x,
                    "baseline.gain": 0.4 + x,
                    "baseline.retention": 0.9 - 0.1 * x,
                },
                target=AdaptationGeometry({"retention": -0.1 * x, "gain": 0.2 * x}),
                target_kind="delta",
            )
        )
    dataset = CompilerDataset(records)
    predictor = RidgeGeometryPredictor(alpha=1e-12).fit(dataset)
    prediction = predictor.predict(
        features={
            "baseline.retention": 0.9,
            "program.x": 3.0,
            "baseline.gain": 10.0,
        },
        model_key=("model", None, None),
        episode_key=("episode_id", "inference"),
        family_fingerprint=None,
        program_fingerprint="inference-program",
    )

    assert predictor.target_names == ("gain", "retention")
    assert prediction.target_kind == "delta"
    assert prediction.predicted_target["retention"] < 0
    assert prediction.predicted_geometry["gain"] == pytest.approx(
        10.0 + prediction.predicted_target["gain"]
    )
    assert prediction.predicted_geometry["retention"] == pytest.approx(
        0.9 + prediction.predicted_target["retention"]
    )


@pytest.mark.parametrize(
    "bad_value",
    [True, float("nan"), float("inf"), "1", [1.0], {"nested": 1.0}],
)
def test_prediction_rejects_invalid_feature_values(bad_value: object) -> None:
    dataset = _dataset(
        [(0.0, {"baseline.gain": 0.1}, 0.0), (1.0, {"baseline.gain": 0.1}, 1.0)],
        target_kind="delta",
    )
    predictor = RidgeGeometryPredictor().fit(dataset)
    features = dict(dataset[0].features)
    features["program.x"] = bad_value  # type: ignore[assignment]

    with pytest.raises(ValidationError, match="real number|finite"):
        predictor.predict(
            features=features,  # type: ignore[arg-type]
            model_key=dataset[0].model_key,
            episode_key=dataset[0].episode_key,
            family_fingerprint=None,
            program_fingerprint="program",
        )


def test_prediction_rejects_missing_and_extra_features() -> None:
    dataset = _dataset(
        [(0.0, {"baseline.gain": 0.1}, 0.0), (1.0, {"baseline.gain": 0.1}, 1.0)],
        target_kind="delta",
    )
    predictor = RidgeGeometryPredictor().fit(dataset)
    common = {
        "model_key": dataset[0].model_key,
        "episode_key": dataset[0].episode_key,
        "family_fingerprint": None,
        "program_fingerprint": "program",
    }
    with pytest.raises(ValidationError, match="missing=.*program.x"):
        predictor.predict(features={"baseline.gain": 0.1}, **common)
    with pytest.raises(ValidationError, match="missing=.*baseline.gain"):
        predictor.predict(features={"program.x": 0.0}, **common)
    with pytest.raises(ValidationError, match="extra=.*model.extra"):
        predictor.predict(
            features={**dataset[0].features, "model.extra": 1.0}, **common
        )


def test_feature_mapping_order_does_not_change_prediction() -> None:
    dataset = _dataset(
        [(0.0, {"baseline.gain": 0.2}, 0.0), (1.0, {"baseline.gain": 0.3}, 1.0)],
        target_kind="delta",
    )
    predictor = RidgeGeometryPredictor().fit(dataset)
    common = {
        "model_key": dataset[0].model_key,
        "episode_key": dataset[0].episode_key,
        "family_fingerprint": None,
        "program_fingerprint": "program",
    }
    first = predictor.predict(
        features={"program.x": 0.5, "baseline.gain": 0.25}, **common
    )
    second = predictor.predict(
        features={"baseline.gain": 0.25, "program.x": 0.5}, **common
    )
    assert first == second


def test_fit_rejects_delta_schema_without_required_baseline() -> None:
    dataset = _dataset([(0.0, {}, 0.1), (1.0, {}, 0.2)], target_kind="delta")
    with pytest.raises(ValidationError, match="baseline.gain"):
        RidgeGeometryPredictor().fit(dataset)


def test_repeated_fit_replaces_schema_and_learned_state() -> None:
    after = _dataset(
        [(0.0, {}, 1.0), (1.0, {}, 2.0)], target_kind="after", metric="gain"
    )
    delta = _dataset(
        [(0.0, {"baseline.cost": 2.0}, -1.0), (1.0, {"baseline.cost": 3.0}, -2.0)],
        target_kind="delta",
        metric="cost",
    )
    predictor = RidgeGeometryPredictor().fit(after)
    assert predictor.target_names == ("gain",)

    returned = predictor.fit(delta)
    assert returned is predictor
    assert predictor.feature_names == delta.feature_names
    assert predictor.target_names == ("cost",)
    assert predictor.target_kind == "delta"


def test_replicates_are_separate_supervised_rows() -> None:
    dataset = _dataset(
        [
            (1.0, {"baseline.gain": 0.0}, 0.0),
            (1.0, {"baseline.gain": 0.0}, 9.0),
            (1.0, {"baseline.gain": 0.0}, 9.0),
        ],
        target_kind="delta",
    )
    prediction = _predict_first(RidgeGeometryPredictor().fit(dataset), dataset)
    assert prediction.predicted_target["gain"] == pytest.approx(6.0)


def test_predict_dataset_does_not_read_evaluation_target_values() -> None:
    training = _dataset(
        [(0.0, {}, 0.0), (1.0, {}, 1.0), (2.0, {}, 2.0)],
        target_kind="after",
    )
    predictor = RidgeGeometryPredictor(alpha=0.0).fit(training)
    shared = {
        "model_key": ("model", None, None),
        "episode_key": ("episode_id", "evaluation"),
        "family_fingerprint": None,
        "program_fingerprint": "candidate",
        "features": {"program.x": 1.5},
        "target_kind": "after",
        "metadata": {"split": "evaluation"},
    }
    ordinary = CompilerDataset(
        [CompilerRecord(**shared, target=AdaptationGeometry({"gain": 1.5}))]
    )
    changed = CompilerDataset(
        [CompilerRecord(**shared, target=AdaptationGeometry({"gain": -9999.0}))]
    )

    assert predictor.predict_dataset(ordinary) == predictor.predict_dataset(changed)


def test_predict_dataset_validates_only_feature_schema_and_preserves_order() -> None:
    training = _dataset([(0.0, {}, 0.0), (1.0, {}, 1.0)], target_kind="after")
    predictor = RidgeGeometryPredictor().fit(training)
    predictions = predictor.predict_dataset(training)

    assert len(predictions) == len(training)
    assert [item.program_fingerprint for item in predictions] == [
        record.program_fingerprint for record in training
    ]
    incompatible = _dataset(
        [(0.0, {"model.extra": 1.0}, 0.0), (1.0, {"model.extra": 1.0}, 1.0)],
        target_kind="after",
    )
    with pytest.raises(ValidationError, match="does not match fitted schema"):
        predictor.predict_dataset(incompatible)


def test_fit_rejects_non_compiler_dataset() -> None:
    with pytest.raises(ValidationError, match="CompilerDataset"):
        RidgeGeometryPredictor().fit(object())  # type: ignore[arg-type]
