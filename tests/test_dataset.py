from __future__ import annotations

from collections.abc import Mapping

import pytest

from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramFamily,
    ProgramSpec,
    ValidationError,
)


def _result(
    context: ModelContext,
    episode: LearningEpisode,
    program: ProgramSpec,
    *,
    before: Mapping[str, float],
    after: Mapping[str, float],
    seed: int,
) -> AdaptationResult:
    return AdaptationResult(
        model_context=context,
        episode=episode,
        program=program,
        before=before,
        after=after,
        metadata={"experiment_id": "dataset-test", "seed": seed},
    )


@pytest.fixture
def corpus() -> AdaptationDataset:
    family = ProgramFamily("middle", "localized_lora", {"region": "middle", "rank": 16})
    repeated_program = ProgramSpec(
        "middle-model-a",
        "localized_lora",
        {"layers": [12, 13], "rank": 16},
        family=family,
    )
    familyless_program = ProgramSpec("prompt", "prompt_adaptation", {"shots": 4})
    model_a = ModelContext("model-a", "revision-a", "state-a")
    model_b = ModelContext("model-b", "revision-b", "state-b")
    episode_a = LearningEpisode("episode-a", [], {"gain": []}, episode_id="e-a")
    episode_b = LearningEpisode("episode-b", [], {"accuracy": []}, episode_id="e-b")
    return AdaptationDataset(
        [
            _result(
                model_a,
                episode_a,
                repeated_program,
                before={"gain": 0.1},
                after={"gain": 0.7},
                seed=1,
            ),
            _result(
                model_a,
                episode_a,
                repeated_program,
                before={"gain": 0.1},
                after={"gain": 0.72},
                seed=2,
            ),
            _result(
                model_b,
                episode_b,
                familyless_program,
                before={"accuracy": 0.4},
                after={"accuracy": 0.8},
                seed=3,
            ),
        ]
    )


def test_construction_iteration_indexing_and_repr(corpus: AdaptationDataset) -> None:
    assert len(corpus) == 3
    assert corpus.results == tuple(corpus)
    assert corpus[0] is corpus.results[0]
    assert corpus[:2] == corpus.results[:2]
    assert repr(corpus) == "AdaptationDataset(n_results=3)"

    with pytest.raises(ValidationError, match="at least one"):
        AdaptationDataset([])
    with pytest.raises(ValidationError, match="AdaptationResult"):
        AdaptationDataset([object()])  # type: ignore[list-item]


def test_grouping_uses_semantic_identities_and_retains_familyless(
    corpus: AdaptationDataset,
) -> None:
    family = corpus[0].program.family
    assert family is not None
    assert set(corpus.by_model()) == {
        ("model-a", "revision-a", "state-a"),
        ("model-b", "revision-b", "state-b"),
    }
    assert set(corpus.by_episode()) == {
        ("episode_id", "e-a"),
        ("episode_id", "e-b"),
    }
    assert set(corpus.by_program()) == {
        corpus[0].program.fingerprint,
        corpus[2].program.fingerprint,
    }
    assert set(corpus.by_family()) == {
        family.fingerprint,
        None,
    }
    assert len(corpus.by_program()[corpus[0].program.fingerprint]) == 2


def test_episode_fallback_grouping_uses_full_configuration() -> None:
    context = ModelContext("model")
    program = ProgramSpec("program", "method")
    first = LearningEpisode("same", object(), {"gain": object()}, {"version": 1})
    matching = LearningEpisode("same", object(), {"gain": object()}, {"version": 1})
    different = LearningEpisode("same", object(), {"gain": object()}, {"version": 2})
    dataset = AdaptationDataset(
        [
            _result(
                context,
                episode,
                program,
                before={"gain": 0.1},
                after={"gain": 0.2},
                seed=index,
            )
            for index, episode in enumerate((first, matching, different), start=1)
        ]
    )

    groups = dataset.by_episode()
    assert len(groups) == 2
    assert sorted(len(group) for group in groups.values()) == [1, 2]
    assert all(identity[0] == "configuration" for identity in groups)


def test_replicates_are_not_deduplicated_and_preserve_provenance(
    corpus: AdaptationDataset,
) -> None:
    replicates = corpus.by_program()[corpus[0].program.fingerprint]
    records = replicates.to_records()

    assert len(replicates) == 2
    assert [record["metadata"]["seed"] for record in records] == [1, 2]
    assert records[0]["model_id"] == records[1]["model_id"]
    assert records[0]["episode_id"] == records[1]["episode_id"]
    assert records[0]["program_fingerprint"] == records[1]["program_fingerprint"]


def test_filter_accepts_a_predicate_and_rejects_empty_selection(
    corpus: AdaptationDataset,
) -> None:
    model_a = corpus.filter(lambda result: result.model_context.model_id == "model-a")
    assert len(model_a) == 2

    with pytest.raises(ValidationError, match="at least one"):
        corpus.filter(lambda result: result.model_context.model_id == "missing")


def test_to_records_has_explicit_geometry_prefixes_and_nested_metadata(
    corpus: AdaptationDataset,
) -> None:
    family_record, _, familyless_record = corpus.to_records()

    assert family_record["program_family_fingerprint"] is not None
    assert family_record["program_family_name"] == "middle"
    assert family_record["program_family_method"] == "localized_lora"
    assert family_record["program_family_parameters"] == {
        "region": "middle",
        "rank": 16,
    }
    assert family_record["before_gain"] == 0.1
    assert family_record["after_gain"] == 0.7
    assert family_record["delta_gain"] == pytest.approx(0.6)
    assert "gain" not in family_record
    assert family_record["metadata"] == {"experiment_id": "dataset-test", "seed": 1}
    assert familyless_record["program_family_fingerprint"] is None
    assert familyless_record["program_family_parameters"] is None


def test_dataframe_supports_mixed_geometry_dimensions(
    corpus: AdaptationDataset,
) -> None:
    pd = pytest.importorskip("pandas")
    frame = corpus.to_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3
    assert "after_gain" in frame
    assert "after_accuracy" in frame
    assert pd.isna(frame.loc[0, "after_accuracy"])
    assert pd.isna(frame.loc[2, "after_gain"])
