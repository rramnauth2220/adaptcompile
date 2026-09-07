"""Synthetic end-to-end prediction and selection across decision contexts."""

from __future__ import annotations

from collections import defaultdict

import pytest

pytest.importorskip("sklearn")

from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramFamily,
    ProgramSpec,
)
from adaptcompile.compiler import (
    GeometryConstraint,
    GeometryPrediction,
    SelectionObjective,
    build_compiler_dataset,
)
from adaptcompile.compiler.predictors import RidgeGeometryPredictor
from adaptcompile.compiler.selectors import LinearUtilitySelector


def test_prediction_to_selection_across_model_episode_program_grid() -> None:
    models = (ModelContext("model-a"), ModelContext("model-b"))
    episodes = tuple(
        LearningEpisode("episode", [], {"score": []}, episode_id=name)
        for name in ("episode-a", "episode-b")
    )
    family = ProgramFamily("generic family", "generic", {"kind": "finite"})
    programs = tuple(
        ProgramSpec(name, "generic", {"level": level}, family=family)
        for name, level in (("low", 0), ("middle", 1), ("high", 2))
    )
    observations = []
    for model_index, model in enumerate(models):
        for episode_index, episode in enumerate(episodes):
            for program_index, program in enumerate(programs):
                baseline_gain = 0.1 + 0.05 * model_index
                observations.append(
                    AdaptationResult(
                        model,
                        episode,
                        program,
                        before={"gain": baseline_gain, "retention": 0.95, "cost": 0.5},
                        after={
                            "gain": baseline_gain
                            + 0.2
                            + 0.1 * episode_index
                            + 0.2 * program_index,
                            "retention": 0.95 - 0.12 * program_index,
                            "cost": 0.5 + 0.1 * program_index,
                        },
                    )
                )
    data = build_compiler_dataset(
        AdaptationDataset(observations),
        model_descriptors={
            model.identity_key: {"index": index} for index, model in enumerate(models)
        },
        episode_descriptors={
            episode.identity_key: {"index": index}
            for index, episode in enumerate(episodes)
        },
        program_descriptors={
            program.fingerprint: {"level": index}
            for index, program in enumerate(programs)
        },
        target="delta",
    )
    predictions = RidgeGeometryPredictor(alpha=0).fit(data).predict_dataset(data)
    by_decision: dict[tuple[object, object], list[GeometryPrediction]] = defaultdict(
        list
    )
    for prediction in predictions:
        by_decision[(prediction.model_key, prediction.episode_key)].append(prediction)

    objective = SelectionObjective(
        maximize={"gain": 1},
        minimize={"cost": 0.1},
        constraints=(GeometryConstraint("retention", minimum=0.75),),
    )
    selector = LinearUtilitySelector()

    assert len(by_decision) == len(models) * len(episodes)
    for (model_key, episode_key), candidates in by_decision.items():
        selection = selector.select(candidates, objective)
        selected = selection.selected
        assert selected is not None
        assert selection.model_key == selected.prediction.model_key == model_key
        assert selection.episode_key == selected.prediction.episode_key == episode_key
        assert selected.prediction.family_fingerprint == family.fingerprint
        assert selected.prediction.program_fingerprint == programs[1].fingerprint

        high = next(
            score
            for score in selection.candidates
            if score.prediction.program_fingerprint == programs[2].fingerprint
        )
        assert high.utility > selected.utility
        assert high.feasible is False
        assert high.violated_constraints == objective.constraints
