"""Pressure test the representation layer with explicitly synthetic results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from adaptcompile import (
    AdaptationResult,
    AdaptationStudy,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)

METRICS = ("acquisition", "transfer", "boundedness", "preservation")
MODEL_CONTEXT = ModelContext(
    model_id="synthetic-research/base-model-8b",
    revision="synthetic-revision-001",
    base_state_id="synthetic-pre-adaptation-state-v1",
    metadata={"synthetic": True},
)
PROGRAM_PARAMETERS = {
    "early": {"layers": [2, 3, 4, 5], "rank": 16, "learning_rate": 2e-4},
    "middle": {"layers": [12, 13, 14, 15], "rank": 16, "learning_rate": 2e-4},
    "late": {"layers": [24, 25, 26, 27], "rank": 16, "learning_rate": 2e-4},
}
SYNTHETIC_AFTER = {
    "relation-001": {
        "early": (0.78, 0.66, 0.94, 0.97),
        "middle": (0.93, 0.85, 0.92, 0.95),
        "late": (0.87, 0.74, 0.97, 0.97),
    },
    "relation-002": {
        "early": (0.74, 0.62, 0.95, 0.98),
        "middle": (0.89, 0.80, 0.93, 0.96),
        "late": (0.84, 0.70, 0.97, 0.98),
    },
    "relation-003": {
        "early": (0.80, 0.69, 0.93, 0.96),
        "middle": (0.94, 0.87, 0.91, 0.94),
        "late": (0.88, 0.76, 0.96, 0.96),
    },
}


@dataclass(frozen=True)
class SyntheticExperimentSlice:
    """Test-only container; deliberately not part of the package API."""

    model_context: ModelContext
    studies: Mapping[str, AdaptationStudy]


def _scores(values: tuple[float, float, float, float]) -> dict[str, float]:
    return dict(zip(METRICS, values, strict=True))


@pytest.fixture
def experiment_slice() -> SyntheticExperimentSlice:
    studies: dict[str, AdaptationStudy] = {}
    for episode_index, (episode_id, program_scores) in enumerate(
        SYNTHETIC_AFTER.items(), start=1
    ):
        episode = LearningEpisode(
            name=f"synthetic relation episode {episode_index}",
            episode_id=episode_id,
            train={"dataset_ref": f"memory://{episode_id}/train"},
            evaluations={
                metric: {"dataset_ref": f"memory://{episode_id}/{metric}"}
                for metric in METRICS
            },
            metadata={
                "learning_family": "synthetic-relational-learning",
                "family_version": "v1",
                "synthetic": True,
            },
        )
        before = _scores((0.08, 0.06, 0.98, 0.98))
        results = []
        for program_name, after_values in program_scores.items():
            program = ProgramSpec(
                name=program_name,
                method="localized_lora",
                parameters=PROGRAM_PARAMETERS[program_name],
                metadata={"region": program_name},
            )
            results.append(
                AdaptationResult(
                    model_context=MODEL_CONTEXT,
                    episode=episode,
                    program=program,
                    before=before,
                    after=_scores(after_values),
                    metadata={
                        "experiment_id": "synthetic-localization-pressure-test",
                        "seed": 100 + episode_index,
                        "synthetic": True,
                    },
                )
            )
        studies[episode_id] = AdaptationStudy(results)
    return SyntheticExperimentSlice(MODEL_CONTEXT, studies)


def test_realistic_slice_maps_cleanly_to_existing_abstractions(
    experiment_slice: SyntheticExperimentSlice,
) -> None:
    assert len(experiment_slice.studies) == 3
    for episode_id, study in experiment_slice.studies.items():
        assert len(study) == 3
        assert study.model_context is experiment_slice.model_context
        assert study.model_context.identity_key == (
            "synthetic-research/base-model-8b",
            "synthetic-revision-001",
            "synthetic-pre-adaptation-state-v1",
        )
        assert all(
            result.model_context is experiment_slice.model_context for result in study
        )
        assert all(result.episode.episode_id == episode_id for result in study)
        assert all(
            result.episode.metadata["learning_family"]
            == "synthetic-relational-learning"
            for result in study
        )
        assert all(result.metadata["synthetic"] is True for result in study)
        assert all("model" not in result.metadata for result in study)
        assert all("model_revision" not in result.metadata for result in study)
        assert all("base_state_id" not in result.metadata for result in study)
        assert all(tuple(result.geometry) == METRICS for result in study)
        assert all(result.geometry == result.after_geometry for result in study)
        assert all(result.delta_geometry["acquisition"] > 0 for result in study)


def test_ranking_best_and_pareto_match_synthetic_tradeoffs(
    experiment_slice: SyntheticExperimentSlice,
) -> None:
    for study in experiment_slice.studies.values():
        assert [result.program.name for result in study.rank("acquisition")] == [
            "middle",
            "late",
            "early",
        ]
        assert study.best("transfer").program.name == "middle"
        assert {
            result.program.name for result in study.pareto_front(maximize=METRICS)
        } == {"middle", "late"}


def test_executable_fingerprints_repeat_across_episodes(
    experiment_slice: SyntheticExperimentSlice,
) -> None:
    by_program: dict[str, list[ProgramSpec]] = {}
    for study in experiment_slice.studies.values():
        for result in study:
            by_program.setdefault(result.program.name, []).append(result.program)

    for repeated_specs in by_program.values():
        assert len({program.fingerprint for program in repeated_specs}) == 1
        assert repeated_specs[0] == repeated_specs[1]
        assert repeated_specs[0] is not repeated_specs[1]
