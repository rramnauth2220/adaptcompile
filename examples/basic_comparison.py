"""Compare two fake adaptation programs without a model or ML dependencies."""

from adaptcompile import (
    AdaptationResult,
    AdaptationStudy,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)

model_context = ModelContext(
    model_id="example/model",
    revision="synthetic-revision-1",
    base_state_id="synthetic-base-state-1",
)
episode = LearningEpisode(
    name="toy-task",
    episode_id="toy-task-v1",
    train=["example"],
    evaluations={
        "acquisition": ["a"],
        "transfer": ["b"],
        "preservation": ["c"],
    },
    metadata={"learning_family": "toy", "synthetic": True},
)

programs = [
    ProgramSpec(
        name="early",
        method="localized_lora",
        parameters={"layers": [2, 3, 4], "rank": 16},
    ),
    ProgramSpec(
        name="middle",
        method="localized_lora",
        parameters={"layers": [12, 13, 14], "rank": 16},
    ),
]

results = [
    AdaptationResult(
        model_context=model_context,
        episode=episode,
        program=programs[0],
        before={"acquisition": 0.10, "transfer": 0.10, "preservation": 0.98},
        after={"acquisition": 0.81, "transfer": 0.77, "preservation": 0.99},
    ),
    AdaptationResult(
        model_context=model_context,
        episode=episode,
        program=programs[1],
        before={"acquisition": 0.10, "transfer": 0.10, "preservation": 0.98},
        after={"acquisition": 0.93, "transfer": 0.84, "preservation": 0.95},
    ),
]

study = AdaptationStudy(results)
print("\nBest transfer:", study.best("transfer").program.name)
print("Best transfer delta:", study.best("transfer").delta.to_dict())
print(
    "Pareto front:",
    [
        result.program.name
        for result in study.pareto_front(
            maximize=["acquisition", "transfer", "preservation"]
        )
    ],
)
