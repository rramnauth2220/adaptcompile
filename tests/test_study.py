from __future__ import annotations

import pytest
from conftest import make_result

from adaptcompile import (
    AdaptationResult,
    AdaptationStudy,
    LearningEpisode,
    ModelContext,
    ProgramFamily,
    ProgramSpec,
    ValidationError,
)


def test_creation_best_ranking_and_sequence_behavior(episode) -> None:
    low = make_result(episode, "low", gain=0.5, retention=0.99, cost=1.0)
    high = make_result(episode, "high", gain=0.9, retention=0.90, cost=3.0)
    study = AdaptationStudy([low, high])

    assert len(study) == 2
    assert study.results == (low, high)
    assert study.episode is episode
    assert study.model_context.identity_key == (
        "example/model",
        "revision-1",
        "base-state-1",
    )
    assert study.best("gain") is high
    assert study.best("cost", maximize=False) is low
    assert study.rank("gain") == (high, low)


def test_pareto_front_supports_maximize_and_minimize(episode) -> None:
    cheap = make_result(episode, "cheap", gain=0.7, retention=0.95, cost=1.0)
    strong = make_result(episode, "strong", gain=0.9, retention=0.95, cost=3.0)
    dominated = make_result(episode, "dominated", gain=0.6, retention=0.90, cost=2.0)
    study = AdaptationStudy([cheap, strong, dominated])

    assert study.pareto_front(maximize=["gain", "retention"], minimize=["cost"]) == (
        cheap,
        strong,
    )


def test_dataframe_uses_after_metrics(episode) -> None:
    pd = pytest.importorskip("pandas")
    result = make_result(episode, "candidate", gain=0.8, retention=0.9, cost=2.0)
    frame = AdaptationStudy([result]).to_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "program"] == "candidate"
    assert frame.loc[0, "gain"] == 0.8
    assert frame.loc[0, "model_id"] == "example/model"
    assert frame.loc[0, "model_revision"] == "revision-1"
    assert frame.loc[0, "base_state_id"] == "base-state-1"


def test_invalid_metric_and_pareto_configuration(episode) -> None:
    study = AdaptationStudy(
        [make_result(episode, "candidate", gain=0.8, retention=0.9, cost=2.0)]
    )
    with pytest.raises(ValidationError, match="missing metrics"):
        study.best("unknown")
    with pytest.raises(ValidationError, match="at least one metric"):
        study.pareto_front()
    with pytest.raises(ValidationError, match="must be unique"):
        study.pareto_front(maximize=["gain"], minimize=["gain"])


def test_mismatched_dimensions_are_explicit(episode) -> None:
    full = make_result(episode, "full", gain=0.8, retention=0.9, cost=2.0)
    partial = type(full)(
        model_context=full.model_context,
        episode=episode,
        program=full.program,
        before={"gain": 0.1},
        after={"gain": 0.8},
    )
    study = AdaptationStudy([full, partial])

    with pytest.raises(ValidationError, match="identical metric dimensions"):
        study.to_dataframe()
    with pytest.raises(ValidationError, match="missing metrics"):
        study.pareto_front(maximize=["gain", "retention"])


def test_different_fallback_configurations_cannot_be_mixed(episode) -> None:
    other = LearningEpisode("other", [], {"gain": []})
    left = make_result(episode, "left", gain=0.8, retention=0.9, cost=2.0)
    right = type(left)(
        model_context=left.model_context,
        episode=other,
        program=left.program,
        before={"gain": 0.1},
        after={"gain": 0.8},
    )
    with pytest.raises(ValidationError, match="episode identity"):
        AdaptationStudy([left, right])


def test_same_name_with_distinct_explicit_ids_cannot_be_mixed(model_context) -> None:
    left_episode = LearningEpisode("same", [], {"gain": []}, episode_id="dataset-a")
    right_episode = LearningEpisode("same", [], {"gain": []}, episode_id="dataset-b")
    program = ProgramSpec("candidate", "method")
    left = AdaptationResult(
        model_context, left_episode, program, {"gain": 0.0}, {"gain": 0.5}
    )
    right = AdaptationResult(
        model_context, right_episode, program, {"gain": 0.0}, {"gain": 0.6}
    )

    with pytest.raises(ValidationError, match="episode identity"):
        AdaptationStudy([left, right])


def test_matching_explicit_id_is_authoritative_over_labels(model_context) -> None:
    left_episode = LearningEpisode("old-label", [], {"gain": []}, episode_id="run-1")
    right_episode = LearningEpisode("new-label", [], {"other": []}, episode_id="run-1")
    program = ProgramSpec("candidate", "method")
    left = AdaptationResult(
        model_context, left_episode, program, {"gain": 0.0}, {"gain": 0.5}
    )
    right = AdaptationResult(
        model_context, right_episode, program, {"gain": 0.0}, {"gain": 0.6}
    )

    assert len(AdaptationStudy([left, right])) == 2


def test_no_id_fallback_requires_complete_serialized_configuration(
    model_context,
) -> None:
    first_episode = LearningEpisode("same", object(), {"gain": object()}, {"seed": 1})
    matching_episode = LearningEpisode(
        "same", object(), {"gain": object()}, {"seed": 1}
    )
    different_episode = LearningEpisode(
        "same", object(), {"gain": object()}, {"seed": 2}
    )
    program = ProgramSpec("candidate", "method")

    def result_for(source: LearningEpisode) -> AdaptationResult:
        return AdaptationResult(
            model_context, source, program, {"gain": 0.0}, {"gain": 0.5}
        )

    matching_study = AdaptationStudy(
        [result_for(first_episode), result_for(matching_episode)]
    )
    assert len(matching_study) == 2
    with pytest.raises(ValidationError, match="episode identity"):
        AdaptationStudy([result_for(first_episode), result_for(different_episode)])


def test_separately_constructed_equivalent_contexts_can_share_study(episode) -> None:
    first = ModelContext("model", "revision", "base-state")
    second = ModelContext("model", "revision", "base-state")

    study = AdaptationStudy(
        [
            make_result(
                episode,
                "first",
                gain=0.5,
                retention=0.9,
                cost=1.0,
                model_context=first,
            ),
            make_result(
                episode,
                "second",
                gain=0.6,
                retention=0.9,
                cost=1.0,
                model_context=second,
            ),
        ]
    )

    assert study.model_context is first


@pytest.mark.parametrize(
    "other_context",
    [
        ModelContext("other-model", "revision", "base-state"),
        ModelContext("model", "other-revision", "base-state"),
        ModelContext("model", "revision", "other-base-state"),
    ],
    ids=["different-model", "different-revision", "different-base-state"],
)
def test_model_context_identity_mismatch_is_rejected(
    episode, other_context: ModelContext
) -> None:
    reference = ModelContext("model", "revision", "base-state")
    left = make_result(
        episode,
        "left",
        gain=0.5,
        retention=0.9,
        cost=1.0,
        model_context=reference,
    )
    right = make_result(
        episode,
        "right",
        gain=0.6,
        retention=0.9,
        cost=1.0,
        model_context=other_context,
    )

    with pytest.raises(ValidationError, match="model-context mismatch"):
        AdaptationStudy([left, right])


def test_model_context_metadata_does_not_change_study_identity(episode) -> None:
    first = ModelContext("model", "revision", "base-state", {"note": "first"})
    second = ModelContext("model", "revision", "base-state", {"note": "second"})
    left = make_result(
        episode,
        "left",
        gain=0.5,
        retention=0.9,
        cost=1.0,
        model_context=first,
    )
    right = make_result(
        episode,
        "right",
        gain=0.6,
        retention=0.9,
        cost=1.0,
        model_context=second,
    )

    assert first != second
    assert len(AdaptationStudy([left, right])) == 2


def test_study_allows_multiple_realizations_of_one_family(
    episode, model_context
) -> None:
    family = ProgramFamily("middle", "localized_lora", {"region": "middle"})
    programs = [
        ProgramSpec(
            "middle-a",
            "localized_lora",
            {"layers": [10, 11]},
            family=family,
        ),
        ProgramSpec(
            "middle-b",
            "localized_lora",
            {"layers": [12, 13]},
            family=family,
        ),
    ]
    results = [
        AdaptationResult(
            model_context=model_context,
            episode=episode,
            program=program,
            before={"gain": 0.1},
            after={"gain": score},
        )
        for program, score in zip(programs, (0.7, 0.8), strict=True)
    ]

    study = AdaptationStudy(results)
    assert len(study) == 2
    assert study.best("gain").program is programs[1]
