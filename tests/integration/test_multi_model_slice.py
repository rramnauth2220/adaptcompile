"""Pressure test the ontology across two explicitly synthetic model contexts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    AdaptationStudy,
    LearningEpisode,
    ModelContext,
    ProgramFamily,
    ProgramSpec,
    ValidationError,
)

METRICS = ("acquisition", "transfer", "boundedness", "preservation")
PROGRAM_FAMILIES = {
    region: ProgramFamily(
        name=f"{region}-region-lora",
        method="localized_lora",
        parameters={
            "region": region,
            "rank": 16,
            "learning_rate": 2e-4,
        },
        metadata={"synthetic": True},
    )
    for region in ("early", "middle", "late")
}
REALIZATION_LAYERS = {
    "model-a": {
        "early": [2, 3, 4, 5],
        "middle": [14, 15, 16, 17],
        "late": [26, 27, 28, 29],
    },
    "model-b": {
        "early": [1, 2, 3, 4],
        "middle": [11, 12, 13, 14],
        "late": [21, 22, 23, 24],
    },
}
SYNTHETIC_AFTER = {
    "model-a": {
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
    },
    "model-b": {
        "relation-001": {
            "early": (0.69, 0.60, 0.97, 0.98),
            "middle": (0.83, 0.71, 0.95, 0.96),
            "late": (0.96, 0.73, 0.90, 0.92),
        },
        "relation-002": {
            "early": (0.66, 0.58, 0.96, 0.97),
            "middle": (0.80, 0.69, 0.94, 0.95),
            "late": (0.93, 0.71, 0.89, 0.91),
        },
        "relation-003": {
            "early": (0.72, 0.63, 0.95, 0.96),
            "middle": (0.85, 0.73, 0.93, 0.94),
            "late": (0.97, 0.75, 0.88, 0.90),
        },
    },
}


@dataclass(frozen=True)
class MultiModelSlice:
    """Test-only organization for two models and their per-episode studies."""

    contexts: Mapping[str, ModelContext]
    episodes: tuple[LearningEpisode, ...]
    studies: Mapping[str, Mapping[str, AdaptationStudy]]
    dataset: AdaptationDataset


def _scores(values: tuple[float, float, float, float]) -> dict[str, float]:
    return dict(zip(METRICS, values, strict=True))


def _context_variant(canonical: ModelContext, *, program_name: str) -> ModelContext:
    """Vary annotations while preserving controlled experimental identity."""
    if program_name == "early":
        return canonical
    return ModelContext(
        model_id=canonical.model_id,
        revision=canonical.revision,
        base_state_id=canonical.base_state_id,
        metadata={
            "display_name": canonical.metadata["display_name"],
            "record_annotation": program_name,
            "synthetic": True,
        },
    )


@pytest.fixture
def multi_model_slice() -> MultiModelSlice:
    contexts = {
        "model-a": ModelContext(
            model_id="synthetic-research/model-a-8b",
            revision="synthetic-a-revision-001",
            base_state_id="synthetic-a-base-v1",
            metadata={"display_name": "Synthetic Model A", "synthetic": True},
        ),
        "model-b": ModelContext(
            model_id="synthetic-research/model-b-7b",
            revision="synthetic-b-revision-001",
            base_state_id="synthetic-b-base-v1",
            metadata={"display_name": "Synthetic Model B", "synthetic": True},
        ),
    }
    episodes = tuple(
        LearningEpisode(
            name=f"synthetic relation episode {index}",
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
        for index, episode_id in enumerate(SYNTHETIC_AFTER["model-a"], start=1)
    )

    studies: dict[str, dict[str, AdaptationStudy]] = {}
    for model_key, canonical_context in contexts.items():
        model_studies: dict[str, AdaptationStudy] = {}
        before = (
            _scores((0.08, 0.06, 0.98, 0.98))
            if model_key == "model-a"
            else _scores((0.12, 0.05, 0.97, 0.96))
        )
        for episode_index, episode in enumerate(episodes, start=1):
            episode_id = episode.episode_id
            if episode_id is None:  # Defensive; fixture episodes always have IDs.
                raise RuntimeError("synthetic episode must have an episode_id")
            results = []
            for program_name, after_values in SYNTHETIC_AFTER[model_key][
                episode_id
            ].items():
                family = ProgramFamily.from_dict(
                    PROGRAM_FAMILIES[program_name].to_dict()
                )
                program = ProgramSpec(
                    name=f"{model_key}-{program_name}",
                    method="localized_lora",
                    parameters={
                        "layers": REALIZATION_LAYERS[model_key][program_name],
                        "rank": 16,
                        "learning_rate": 2e-4,
                    },
                    metadata={"synthetic": True},
                    family=family,
                )
                results.append(
                    AdaptationResult(
                        model_context=_context_variant(
                            canonical_context, program_name=program_name
                        ),
                        episode=episode,
                        program=program,
                        before=before,
                        after=_scores(after_values),
                        metadata={
                            "experiment_id": "synthetic-multi-model-pressure-test",
                            "seed": 1000 * (1 if model_key == "model-a" else 2)
                            + episode_index,
                            "synthetic": True,
                        },
                    )
                )
            model_studies[episode_id] = AdaptationStudy(results)
        studies[model_key] = model_studies
    dataset = AdaptationDataset(
        result
        for model_studies in studies.values()
        for study in model_studies.values()
        for result in study
    )
    return MultiModelSlice(contexts, episodes, studies, dataset)


def test_six_studies_reuse_exact_episode_objects(
    multi_model_slice: MultiModelSlice,
) -> None:
    all_studies = [
        study
        for model_studies in multi_model_slice.studies.values()
        for study in model_studies.values()
    ]
    assert len(all_studies) == 6
    assert len(multi_model_slice.dataset) == 18
    assert len(multi_model_slice.dataset.by_model()) == 2
    assert len(multi_model_slice.dataset.by_episode()) == 3
    assert len(multi_model_slice.dataset.by_family()) == 3
    assert len(multi_model_slice.dataset.by_program()) == 6
    for episode in multi_model_slice.episodes:
        episode_id = episode.episode_id
        assert episode_id is not None
        assert multi_model_slice.studies["model-a"][episode_id].episode is episode
        assert multi_model_slice.studies["model-b"][episode_id].episode is episode


def test_model_boundaries_and_context_metadata_are_harmless(
    multi_model_slice: MultiModelSlice,
) -> None:
    episode_id = "relation-001"
    model_a_study = multi_model_slice.studies["model-a"][episode_id]
    model_b_study = multi_model_slice.studies["model-b"][episode_id]

    with pytest.raises(ValidationError, match="model-context mismatch"):
        AdaptationStudy([model_a_study[0], model_b_study[0]])

    for model_key, model_studies in multi_model_slice.studies.items():
        expected_identity = multi_model_slice.contexts[model_key].identity_key
        for study in model_studies.values():
            assert study.model_context.identity_key == expected_identity
            assert (
                len({repr(dict(result.model_context.metadata)) for result in study}) > 1
            )
            assert (
                study.best("acquisition").model_context.identity_key
                == expected_identity
            )


def test_family_identity_is_shared_but_realizations_are_model_specific(
    multi_model_slice: MultiModelSlice,
) -> None:
    family_fingerprints: dict[str, set[str]] = {}
    realization_fingerprints: dict[str, dict[str, set[str]]] = {}
    program_objects: dict[str, list[ProgramSpec]] = {}
    for model_key, model_studies in multi_model_slice.studies.items():
        for study in model_studies.values():
            for result in study:
                family = result.program.family
                assert family is not None
                region = family.parameters["region"]
                assert isinstance(region, str)
                family_fingerprints.setdefault(region, set()).add(family.fingerprint)
                realization_fingerprints.setdefault(region, {}).setdefault(
                    model_key, set()
                ).add(result.program.fingerprint)
                program_objects.setdefault(f"{model_key}:{region}", []).append(
                    result.program
                )

    assert all(len(values) == 1 for values in family_fingerprints.values())
    for by_model in realization_fingerprints.values():
        assert all(len(values) == 1 for values in by_model.values())
        assert len(set().union(*by_model.values())) == 2
    assert all(items[0] is not items[-1] for items in program_objects.values())


def test_shared_program_has_different_geometry_by_model(
    multi_model_slice: MultiModelSlice,
) -> None:
    model_a = multi_model_slice.studies["model-a"]["relation-001"]
    model_b = multi_model_slice.studies["model-b"]["relation-001"]
    model_a_late = next(
        result
        for result in model_a
        if result.program.family is not None
        and result.program.family.parameters["region"] == "late"
    )
    model_b_late = next(
        result
        for result in model_b
        if result.program.family is not None
        and result.program.family.parameters["region"] == "late"
    )

    assert model_a_late.after_geometry != model_b_late.after_geometry
    assert (
        model_b_late.after_geometry["acquisition"]
        > model_a_late.after_geometry["acquisition"]
    )
    assert (
        model_b_late.after_geometry["preservation"]
        < model_a_late.after_geometry["preservation"]
    )


def test_all_observations_flatten_externally(
    multi_model_slice: MultiModelSlice,
) -> None:
    rows = multi_model_slice.dataset.to_records()
    required_fields = {
        "model_id",
        "model_revision",
        "base_state_id",
        "episode",
        "episode_id",
        "program_family_fingerprint",
        "program_family_name",
        "program_family_method",
        "program_family_parameters",
        "program_fingerprint",
        "program_name",
        "method",
        "program_parameters",
        "before_acquisition",
        "before_transfer",
        "before_boundedness",
        "before_preservation",
        "after_acquisition",
        "after_transfer",
        "after_boundedness",
        "after_preservation",
        "delta_acquisition",
        "delta_transfer",
        "delta_boundedness",
        "delta_preservation",
        "metadata",
    }

    assert len(rows) == 18
    assert all(required_fields <= row.keys() for row in rows)
    assert (
        len(
            {
                (row["model_id"], row["episode_id"], row["program_fingerprint"])
                for row in rows
            }
        )
        == 18
    )
    assert all("model_metadata" not in row for row in rows)


def test_replicate_observation_is_retained(
    multi_model_slice: MultiModelSlice,
) -> None:
    original = multi_model_slice.dataset[0]
    replicate = AdaptationResult(
        model_context=original.model_context,
        episode=original.episode,
        program=original.program,
        before=original.before,
        after=original.after,
        metadata={
            "experiment_id": "synthetic-multi-model-replicate",
            "seed": 9999,
            "synthetic": True,
        },
    )
    with_replicate = AdaptationDataset([*multi_model_slice.dataset, replicate])
    program_group = with_replicate.by_program()[original.program.fingerprint]
    exact_replicates = program_group.filter(
        lambda result: (
            result.model_context.identity_key == original.model_context.identity_key
            and result.episode.episode_id == original.episode.episode_id
        )
    )

    assert len(with_replicate) == 19
    assert len(exact_replicates) == 2
    assert [
        record["metadata"]["experiment_id"] for record in exact_replicates.to_records()
    ] == [
        "synthetic-multi-model-pressure-test",
        "synthetic-multi-model-replicate",
    ]
