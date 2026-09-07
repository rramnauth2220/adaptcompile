"""Synthetic end-to-end prediction across multiple candidate identities."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)
from adaptcompile.compiler import build_compiler_dataset
from adaptcompile.compiler.predictors import RidgeGeometryPredictor


def test_multi_model_episode_program_prediction_for_both_target_modes() -> None:
    models = (ModelContext("model-a"), ModelContext("model-b"))
    episodes = tuple(
        LearningEpisode("episode", [], {"gain": [], "cost": []}, episode_id=name)
        for name in ("episode-a", "episode-b")
    )
    programs = tuple(
        ProgramSpec(name, "adapter", {"strength": strength})
        for name, strength in (("small", 0.25), ("large", 0.75))
    )
    results = []
    for model_index, model in enumerate(models):
        for episode_index, episode in enumerate(episodes):
            for program_index, program in enumerate(programs):
                baseline_gain = 0.1 + 0.05 * model_index
                gain_delta = 0.2 + 0.1 * episode_index + 0.2 * program_index
                cost_delta = -0.05 - 0.03 * program_index
                results.append(
                    AdaptationResult(
                        model,
                        episode,
                        program,
                        before={"gain": baseline_gain, "cost": 0.9},
                        after={
                            "gain": baseline_gain + gain_delta,
                            "cost": 0.9 + cost_delta,
                        },
                        metadata={"row": len(results)},
                    )
                )
    results.append(
        AdaptationResult(
            results[0].model_context,
            results[0].episode,
            results[0].program,
            results[0].before,
            {"gain": 0.32, "cost": 0.84},
            metadata={"row": len(results), "replicate": True},
        )
    )
    observations = AdaptationDataset(results)
    model_descriptors = {
        model.identity_key: {"scale": 1.0 + index} for index, model in enumerate(models)
    }
    episode_descriptors = {
        episode.identity_key: {"difficulty": 0.25 + 0.5 * index}
        for index, episode in enumerate(episodes)
    }
    program_descriptors = {
        program.fingerprint: {"relative_size": 0.25 + 0.5 * index}
        for index, program in enumerate(programs)
    }
    arguments = {
        "model_descriptors": model_descriptors,
        "episode_descriptors": episode_descriptors,
        "program_descriptors": program_descriptors,
    }

    delta_data = build_compiler_dataset(observations, **arguments, target="delta")
    after_data = build_compiler_dataset(observations, **arguments, target="after")
    delta_predictor = RidgeGeometryPredictor().fit(delta_data)
    after_predictor = RidgeGeometryPredictor().fit(after_data)
    delta_predictions = delta_predictor.predict_dataset(delta_data)
    after_predictions = after_predictor.predict_dataset(after_data)

    assert len(delta_predictions) == len(after_predictions) == 9
    for record, prediction in zip(delta_data, delta_predictions, strict=True):
        assert prediction.predicted_geometry["gain"] == pytest.approx(
            record.features["baseline.gain"] + prediction.predicted_target["gain"]
        )
        assert prediction.predicted_geometry["cost"] == pytest.approx(
            record.features["baseline.cost"] + prediction.predicted_target["cost"]
        )
    assert all(
        prediction.predicted_target == prediction.predicted_geometry
        for prediction in after_predictions
    )
    assert delta_predictions[-1].program_fingerprint == results[-1].program.fingerprint
