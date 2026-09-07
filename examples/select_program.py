"""Predict and select among synthetic candidate adaptation programs."""

from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)
from adaptcompile.compiler import (
    GeometryConstraint,
    SelectionObjective,
    build_compiler_dataset,
)
from adaptcompile.compiler.predictors import RidgeGeometryPredictor
from adaptcompile.compiler.selectors import LinearUtilitySelector

model = ModelContext("synthetic/model", revision="v1", base_state_id="base")
episode = LearningEpisode(
    "synthetic task", [], {"evaluation": []}, episode_id="task-v1"
)
programs = tuple(
    ProgramSpec(name, "generic", {"level": level})
    for name, level in (("small", 0), ("medium", 1), ("large", 2), ("largest", 3))
)
observations = AdaptationDataset(
    [
        AdaptationResult(
            model,
            episode,
            program,
            before={"gain": 0.2, "retention": 0.95, "cost": 0.5},
            after={
                "gain": 0.2 + 0.3 * index,
                "retention": 0.95 - 0.1 * index,
                "cost": 0.5 + 0.2 * index,
            },
        )
        for index, program in enumerate(programs)
    ]
)
compiler_data = build_compiler_dataset(
    observations,
    program_descriptors={
        program.fingerprint: {"level": index} for index, program in enumerate(programs)
    },
    target="delta",
)

predictions = (
    RidgeGeometryPredictor(alpha=0).fit(compiler_data).predict_dataset(compiler_data)
)
objective = SelectionObjective(
    maximize={"gain": 1.0},
    minimize={"cost": 0.2},
    constraints=(GeometryConstraint("retention", minimum=0.74),),
)
selection = LinearUtilitySelector().select(predictions, objective)

for candidate in selection.candidates:
    print(
        candidate.prediction.program_fingerprint,
        f"utility={candidate.utility:.3f}",
        f"feasible={candidate.feasible}",
    )
if selection.selected is not None:
    print("Selected:", selection.selected.prediction.program_fingerprint)
