"""Assemble a compiler-ready dataset from fully synthetic observations."""

from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)
from adaptcompile.compiler import build_compiler_dataset

model = ModelContext("synthetic/model", revision="v1", base_state_id="base")
episode = LearningEpisode("synthetic task", [], {"accuracy": []}, episode_id="task-v1")
programs = (
    ProgramSpec("small", "adapter", {"rank": 4}),
    ProgramSpec("large", "adapter", {"rank": 8}),
)
observations = AdaptationDataset(
    [
        AdaptationResult(
            model,
            episode,
            program,
            before={"accuracy": 0.25},
            after={"accuracy": outcome},
            metadata={"seed": index},
        )
        for index, (program, outcome) in enumerate(
            zip(programs, (0.55, 0.70), strict=True), start=1
        )
    ]
)

compiler_data = build_compiler_dataset(
    observations,
    model_descriptors={model.identity_key: {"scale": 1.0}},
    episode_descriptors={episode.identity_key: {"n_examples": 8}},
    program_descriptors={
        program.fingerprint: {"relative_size": index / len(programs)}
        for index, program in enumerate(programs, start=1)
    },
    target="delta",
)

print(compiler_data.feature_names)
print(compiler_data[0].target)
