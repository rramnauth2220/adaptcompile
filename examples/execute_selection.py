"""Execute one synthetically selected program through a callable backend."""

from adaptcompile import AdaptationGeometry, LearningEpisode, ModelContext, ProgramSpec
from adaptcompile.compiler import GeometryPrediction, SelectionObjective
from adaptcompile.compiler.selectors import LinearUtilitySelector
from adaptcompile.execution import execute_selection
from adaptcompile.execution.backends import CallableBackend

context = ModelContext("synthetic/model", revision="v1", base_state_id="base")
episode = LearningEpisode(
    "synthetic task", [], {"evaluation": []}, episode_id="task-v1"
)
programs = tuple(
    ProgramSpec(name, "scale", {"amount": amount})
    for name, amount in (("small", 0.25), ("medium", 0.5), ("large", 1.0))
)
geometries = (
    {"gain": 0.4, "cost": 0.2},
    {"gain": 0.7, "cost": 0.4},
    {"gain": 0.9, "cost": 1.5},
)
predictions = tuple(
    GeometryPrediction(
        model_key=context.identity_key,
        episode_key=episode.identity_key,
        family_fingerprint=None,
        program_fingerprint=program.fingerprint,
        predicted_target=AdaptationGeometry(geometry),
        predicted_geometry=AdaptationGeometry(geometry),
        target_kind="after",
    )
    for program, geometry in zip(programs, geometries, strict=True)
)
selection = LinearUtilitySelector().select(
    predictions,
    SelectionObjective(maximize={"gain": 1}, minimize={"cost": 0.25}),
)
print("Selected score:", selection.selected)


def scale(
    model: dict[str, float],
    *,
    model_context: ModelContext,
    episode: LearningEpisode,
    program: ProgramSpec,
) -> dict[str, float]:
    return {"strength": model["strength"] * program.parameters["amount"]}


execution = execute_selection(
    selection,
    programs=programs,
    model={"strength": 2.0},
    model_context=context,
    episode=episode,
    backend=CallableBackend(
        scale,
        backend_id="toy-scale",
        supports=lambda program: program.method == "scale",
    ),
)
print("Executed program:", execution.record.program_fingerprint)
print("Backend:", execution.record.backend_id)
print("Runtime model:", execution.model)
