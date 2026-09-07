"""Synthetic full-loop prediction, execution, and measured evaluation."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from adaptcompile import (
    AdaptationDataset,
    AdaptationGeometry,
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
from adaptcompile.evaluation import EvaluationOutcome, evaluate_execution
from adaptcompile.evaluation.evaluators import CallableEvaluator
from adaptcompile.execution import execute_selection
from adaptcompile.execution.backends import CallableBackend


def test_full_pipeline_converts_runtime_measurement_into_observation() -> None:
    context = ModelContext("synthetic/model", "v1", "base")
    episode = LearningEpisode(
        "synthetic task", [], {"heldout": []}, episode_id="task-v1"
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

    backend_calls: list[ProgramSpec] = []

    def execute(
        model: dict[str, float],
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> dict[str, float]:
        backend_calls.append(program)
        amount = float(program.parameters["amount"])
        model["gain"] = 0.2 + 0.5 * amount
        model["retention"] = 0.95 - 0.14 * amount
        return model

    execution = execute_selection(
        selection,
        programs=programs,
        model={"gain": 0.2, "retention": 0.95},
        model_context=context,
        episode=episode,
        backend=CallableBackend(execute, backend_id="toy-scale"),
    )

    evaluator_calls: list[ProgramSpec] = []

    def measure(
        model: dict[str, float],
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> AdaptationGeometry:
        evaluator_calls.append(program)
        return AdaptationGeometry(
            {"gain": model["gain"], "retention": model["retention"]}
        )

    evaluation = evaluate_execution(
        execution,
        evaluator=CallableEvaluator(measure, evaluator_id="toy-measurement"),
        model_context=context,
        episode=episode,
        before=AdaptationGeometry({"gain": 0.2, "retention": 0.95}),
    )

    assert isinstance(evaluation, EvaluationOutcome)
    assert backend_calls == [programs[1]]
    assert evaluator_calls == [programs[1]]
    assert evaluation.result.after["gain"] == pytest.approx(0.7)
    assert evaluation.result.after["retention"] == pytest.approx(0.81)
    assert (
        evaluation.result.after_geometry
        != selection.selected_prediction.predicted_geometry
    )
    assert evaluation.result.program is execution.record.program
    assert evaluation.result.program is programs[1]
    assert evaluation.result.model_context is context
    assert evaluation.result.episode is episode
    assert evaluation.record.backend_id == "toy-scale"
    assert evaluation.record.evaluator_id == "toy-measurement"
