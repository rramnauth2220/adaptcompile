"""Synthetic end-to-end pressure test for compiler data assembly."""

from __future__ import annotations

from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramFamily,
    ProgramSpec,
)
from adaptcompile.compiler import build_compiler_dataset


def test_two_model_three_episode_three_program_compiler_dataset() -> None:
    models = (
        ModelContext("synthetic/model", "revision-a", "base-a"),
        ModelContext("synthetic/model", "revision-b", "base-b"),
    )
    episodes = tuple(
        LearningEpisode(
            "synthetic episode",
            [],
            {"acquisition": [], "preservation": []},
            episode_id=f"episode-{index}",
        )
        for index in range(3)
    )
    families = tuple(
        ProgramFamily(f"family-{index}", "adapter", {"conceptual_region": index})
        for index in range(3)
    )
    programs = tuple(
        ProgramSpec(
            f"program-{index}",
            "adapter",
            {"layers": [index, index + 1]},
            family=family,
        )
        for index, family in enumerate(families)
    )

    results = []
    for model_index, model in enumerate(models):
        for episode_index, episode in enumerate(episodes):
            for program_index, program in enumerate(programs):
                baseline = 0.1 + 0.01 * model_index + 0.01 * episode_index
                results.append(
                    AdaptationResult(
                        model,
                        episode,
                        program,
                        before={
                            "acquisition": baseline,
                            "preservation": 0.95,
                        },
                        after={
                            "acquisition": baseline + 0.2 + 0.01 * program_index,
                            "preservation": 0.94 - 0.01 * program_index,
                        },
                        metadata={"seed": len(results)},
                    )
                )
    original = results[0]
    results.append(
        AdaptationResult(
            original.model_context,
            original.episode,
            original.program,
            original.before,
            {"acquisition": 0.35, "preservation": 0.93},
            metadata={"seed": 999, "replicate": True},
        )
    )
    adaptations = AdaptationDataset(results)

    model_descriptors = {
        model.identity_key: {
            "layer_count": 24 + 8 * index,
            "scale": 1.0 + index,
        }
        for index, model in enumerate(models)
    }
    episode_descriptors = {
        episode.identity_key: {
            "n_examples": 8 + index,
            "difficulty": 0.2 + 0.1 * index,
        }
        for index, episode in enumerate(episodes)
    }
    family_descriptors = {
        family.fingerprint: {"locality": 0.25 + 0.25 * index}
        for index, family in enumerate(families)
    }
    program_descriptors = {
        program.fingerprint: {
            "relative_depth": 0.2 + 0.3 * index,
            "parameter_fraction": 0.01 + 0.01 * index,
        }
        for index, program in enumerate(programs)
    }
    interaction_descriptors = {
        (
            result.model_context.identity_key,
            result.episode.identity_key,
            result.program.fingerprint,
        ): {"sensitivity": 0.01 * (model_index + episode_index + program_index + 1)}
        for model_index, model in enumerate(models)
        for episode_index, episode in enumerate(episodes)
        for program_index, program in enumerate(programs)
        for result in [
            next(
                item
                for item in results
                if item.model_context.identity_key == model.identity_key
                and item.episode.identity_key == episode.identity_key
                and item.program.fingerprint == program.fingerprint
            )
        ]
    }

    arguments = {
        "model_descriptors": model_descriptors,
        "episode_descriptors": episode_descriptors,
        "family_descriptors": family_descriptors,
        "program_descriptors": program_descriptors,
        "interaction_descriptors": interaction_descriptors,
    }
    delta = build_compiler_dataset(
        adaptations,
        **arguments,
        target="delta",
        descriptor_metadata={"schema_name": "synthetic", "schema_version": "1"},
    )
    after = build_compiler_dataset(adaptations, **arguments, target="after")

    assert len(adaptations) == 19
    assert len(delta) == len(after) == 19
    assert delta.feature_names == (
        "model.layer_count",
        "model.scale",
        "episode.difficulty",
        "episode.n_examples",
        "family.locality",
        "program.parameter_fraction",
        "program.relative_depth",
        "interaction.sensitivity",
        "baseline.acquisition",
        "baseline.preservation",
    )
    assert delta.target_names == ("acquisition", "preservation")
    assert delta[0].target == results[0].delta_geometry
    assert after[0].target == results[0].after_geometry
    assert delta[-1].features == delta[0].features
    assert delta[-1].metadata["replicate"] is True
    assert delta[-1].target != delta[0].target
    assert delta.metadata["schema_version"] == "1"
