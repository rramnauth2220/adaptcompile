"""Fit the optional ridge reference predictor on fully synthetic data."""

from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)
from adaptcompile.compiler import build_compiler_dataset
from adaptcompile.compiler.predictors import RidgeGeometryPredictor

model = ModelContext("synthetic/model", revision="v1", base_state_id="base")
episode = LearningEpisode(
    "synthetic task", [], {"gain": [], "retention": []}, episode_id="task-v1"
)
programs = tuple(
    ProgramSpec(f"candidate-{index}", "generic", {"strength": strength})
    for index, strength in enumerate((0.0, 0.3, 0.6, 0.9))
)
observations = AdaptationDataset(
    [
        AdaptationResult(
            model,
            episode,
            program,
            before={"gain": 0.2, "retention": 0.9},
            after={
                "gain": 0.2 + 0.5 * index,
                "retention": 0.9 - 0.1 * index,
            },
        )
        for index, program in enumerate(programs)
    ]
)
compiler_data = build_compiler_dataset(
    observations,
    program_descriptors={
        program.fingerprint: {"relative_size": index / (len(programs) - 1)}
        for index, program in enumerate(programs)
    },
    target="delta",
)

predictor = RidgeGeometryPredictor(alpha=1.0).fit(compiler_data)
candidate = compiler_data[-1]
prediction = predictor.predict(
    features=candidate.features,
    model_key=candidate.model_key,
    episode_key=candidate.episode_key,
    family_fingerprint=candidate.family_fingerprint,
    program_fingerprint=candidate.program_fingerprint,
)

print("Predicted delta:", prediction.predicted_target)
print("Predicted post-adaptation geometry:", prediction.predicted_geometry)
