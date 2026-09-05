from __future__ import annotations

import pytest

from adaptcompile import LearningEpisode, SerializationError, ValidationError


def test_construction_supports_arbitrary_evaluations_and_metadata() -> None:
    train = object()
    held_out = object()
    episode = LearningEpisode(
        "custom",
        train,
        {"counterfactual_robustness": held_out},
        {"seed": 7},
    )

    assert episode.train is train
    assert episode.evaluations["counterfactual_robustness"] is held_out
    assert episode.evaluation_names == ("counterfactual_robustness",)
    assert episode.metadata["seed"] == 7
    assert "evaluations=['counterfactual_robustness']" in repr(episode)


@pytest.mark.parametrize(
    ("name", "evaluations"),
    [("", {"valid": []}), ("valid", {}), ("valid", {"": []})],
)
def test_validation(name: str, evaluations: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LearningEpisode(name, [], evaluations)


def test_dataset_objects_are_omitted_from_serialization() -> None:
    episode = LearningEpisode(
        "custom",
        object(),
        {"freeform": object()},
        episode_id="custom-v2",
    )
    restored = LearningEpisode.from_dict(episode.to_dict())

    assert episode.to_dict() == restored.to_dict()
    assert restored.train is None
    assert restored.evaluations["freeform"] is None
    assert restored.episode_id == "custom-v2"


def test_episode_id_is_optional_but_must_be_nonempty() -> None:
    assert LearningEpisode("custom", [], {"metric": []}).episode_id is None
    with pytest.raises(ValidationError, match="episode_id"):
        LearningEpisode("custom", [], {"metric": []}, episode_id=" ")


def test_unsupported_metadata_fails_clearly() -> None:
    with pytest.raises(SerializationError, match="unsupported value"):
        LearningEpisode("custom", [], {"metric": []}, {"bad": object()})
