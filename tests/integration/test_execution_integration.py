"""Synthetic full-pipeline selected-program execution."""

from __future__ import annotations

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
    SelectionObjective,
    build_compiler_dataset,
)
from adaptcompile.compiler.predictors import RidgeGeometryPredictor
from adaptcompile.compiler.selectors import LinearUtilitySelector
from adaptcompile.execution import ExecutionOutcome, execute_selection
from adaptcompile.execution.backends import CallableBackend


def test_full_prediction_selection_execution_pipeline_runs_one_program() -> None:
    context = ModelContext("synthetic/model", "v1", "base")
    episode = LearningEpisode(
        "synthetic task", [], {"evaluation": []}, episode_id="task-v1"
    )
    family = ProgramFamily("scaling", "scale", {"domain": "toy"})
    programs = tuple(
        ProgramSpec(name, "scale", {"amount": amount}, family=family)
        for name, amount in (("small", 0.0), ("middle", 1.0), ("large", 2.0))
    )
    observations = AdaptationDataset(
        [
            AdaptationResult(
                context,
                episode,
                program,
                before={"gain": 0.2, "retention": 0.95},
                after={
                    "gain": 0.2 + 0.3 * index,
                    "retention": 0.95 - 0.12 * index,
                },
            )
            for index, program in enumerate(programs)
        ]
    )
    compiler_data = build_compiler_dataset(
        observations,
        program_descriptors={
            program.fingerprint: {"amount": index}
            for index, program in enumerate(programs)
        },
        target="delta",
    )
    predictions = (
        RidgeGeometryPredictor(alpha=0)
        .fit(compiler_data)
        .predict_dataset(compiler_data)
    )
    selection = LinearUtilitySelector().select(
        predictions,
        SelectionObjective(
            maximize={"gain": 1},
            constraints=(GeometryConstraint("retention", minimum=0.75),),
        ),
    )
    assert selection.selected_prediction is not None
    assert selection.selected_prediction.program_fingerprint == programs[1].fingerprint

    executed: list[ProgramSpec] = []

    def scale(
        model: dict[str, float],
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> dict[str, float]:
        executed.append(program)
        return {"strength": model["strength"] + program.parameters["amount"]}

    outcome = execute_selection(
        selection,
        programs=programs,
        model={"strength": 1.0},
        model_context=context,
        episode=episode,
        backend=CallableBackend(
            scale,
            backend_id="toy-scale",
            supports=lambda program: program.method == "scale",
        ),
    )

    assert isinstance(outcome, ExecutionOutcome)
    assert executed == [programs[1]]
    assert outcome.model == {"strength": 2.0}
    assert outcome.record.program is programs[1]
    assert outcome.record.program_fingerprint == programs[1].fingerprint
    assert outcome.record.family_fingerprint == family.fingerprint
    assert outcome.record.model_key == context.identity_key
    assert outcome.record.episode_key == episode.identity_key
    assert outcome.record.backend_id == "toy-scale"
    assert not isinstance(outcome, AdaptationResult)
    assert not hasattr(outcome, "adaptation_result")
