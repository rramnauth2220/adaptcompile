"""Evaluate an adapted toy runtime and create a measured observation."""

from adaptcompile import AdaptationGeometry, LearningEpisode, ModelContext, ProgramSpec
from adaptcompile.compiler import GeometryPrediction, SelectionObjective
from adaptcompile.compiler.selectors import LinearUtilitySelector
from adaptcompile.evaluation import evaluate_execution
from adaptcompile.evaluation.evaluators import CallableEvaluator
from adaptcompile.execution import execute_selection
from adaptcompile.execution.backends import CallableBackend

context = ModelContext("synthetic/model", revision="v1", base_state_id="base")
episode = LearningEpisode("synthetic task", [], {"heldout": []}, episode_id="task-v1")
programs = (
    ProgramSpec("gentle", "scale", {"amount": 0.4}),
    ProgramSpec("strong", "scale", {"amount": 0.8}),
)
predicted_geometries = (
    AdaptationGeometry({"score": 1.4}),
    AdaptationGeometry({"score": 1.8}),
)
predictions = tuple(
    GeometryPrediction(
        model_key=context.identity_key,
        episode_key=episode.identity_key,
        family_fingerprint=None,
        program_fingerprint=program.fingerprint,
        predicted_target=geometry,
        predicted_geometry=geometry,
        target_kind="after",
    )
    for program, geometry in zip(programs, predicted_geometries, strict=True)
)
selection = LinearUtilitySelector().select(
    predictions, SelectionObjective(maximize={"score": 1.0})
)


def adapt(
    model: dict[str, float],
    *,
    model_context: ModelContext,
    episode: LearningEpisode,
    program: ProgramSpec,
) -> dict[str, float]:
    model["score"] += 0.5 * float(program.parameters["amount"])
    return model


execution = execute_selection(
    selection,
    programs=programs,
    model={"score": 1.0},
    model_context=context,
    episode=episode,
    backend=CallableBackend(adapt, backend_id="toy-scale"),
)


def measure(
    model: dict[str, float],
    *,
    model_context: ModelContext,
    episode: LearningEpisode,
    program: ProgramSpec,
) -> AdaptationGeometry:
    return AdaptationGeometry({"score": model["score"]})


evaluation = evaluate_execution(
    execution,
    evaluator=CallableEvaluator(measure, evaluator_id="toy-measurement"),
    model_context=context,
    episode=episode,
    before=AdaptationGeometry({"score": 1.0}),
)

print("Before:", evaluation.result.before)
print("Measured after:", evaluation.result.after)
print("Backend:", evaluation.record.backend_id)
print("Evaluator:", evaluation.record.evaluator_id)
