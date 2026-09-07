from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

import adaptcompile
import adaptcompile.compiler as compiler
from adaptcompile import (
    AdaptationDataset,
    AdaptationGeometry,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramFamily,
    ProgramSpec,
    ValidationError,
)
from adaptcompile.compiler import (
    CompilerDataset,
    CompilerRecord,
    build_compiler_dataset,
)


def _result(
    *,
    model: ModelContext | None = None,
    episode: LearningEpisode | None = None,
    program: ProgramSpec | None = None,
    before: Mapping[str, float] | None = None,
    after: Mapping[str, float] | None = None,
    seed: int = 1,
) -> AdaptationResult:
    family = ProgramFamily("family", "adapter", {"region": "middle"})
    return AdaptationResult(
        model_context=model or ModelContext("model", "rev", "base"),
        episode=episode
        or LearningEpisode("episode", [], {"metric": []}, episode_id="episode-1"),
        program=program
        or ProgramSpec("program", "adapter", {"rank": 4}, family=family),
        before=before or {"zeta": 0.2, "alpha": 0.1},
        after=after or {"zeta": 0.7, "alpha": 0.4},
        metadata={"seed": seed, "experiment_id": "synthetic"},
    )


def _all_descriptors(result: AdaptationResult) -> dict[str, object]:
    family = result.program.family
    assert family is not None
    interaction_key = (
        result.model_context.identity_key,
        result.episode.identity_key,
        result.program.fingerprint,
    )
    return {
        "model_descriptors": {
            result.model_context.identity_key: {"width": 8, "depth": 4.0}
        },
        "episode_descriptors": {result.episode.identity_key: {"examples": 12}},
        "family_descriptors": {family.fingerprint: {"locality": 0.5}},
        "program_descriptors": {
            result.program.fingerprint: {"parameter_fraction": 0.02}
        },
        "interaction_descriptors": {interaction_key: {"sensitivity": 0.3}},
    }


def test_compiler_public_api_is_small_and_not_root_reexported() -> None:
    assert set(compiler.__all__) == {
        "CompilerDataset",
        "CompilerRecord",
        "build_compiler_dataset",
    }
    assert compiler.CompilerRecord is CompilerRecord
    assert not hasattr(adaptcompile, "CompilerRecord")
    assert not hasattr(adaptcompile, "CompilerDataset")
    assert not hasattr(adaptcompile, "build_compiler_dataset")


def test_episode_identity_property_reuses_collection_semantics() -> None:
    explicit = LearningEpisode("same", object(), {"metric": object()}, episode_id="a")
    fallback = LearningEpisode("same", object(), {"metric": object()}, {"v": 1})
    matching = LearningEpisode("same", object(), {"metric": object()}, {"v": 1})

    assert explicit.identity_key == ("episode_id", "a")
    assert fallback.identity_key == matching.identity_key
    assert fallback.identity_key[0] == "configuration"


def test_builds_all_namespaces_with_deterministic_schema_and_provenance() -> None:
    result = _result()
    descriptor_arguments = _all_descriptors(result)
    data = build_compiler_dataset(
        AdaptationDataset([result]),
        **descriptor_arguments,  # type: ignore[arg-type]
        target="delta",
        descriptor_metadata={"schema_name": "synthetic", "schema_version": "1"},
    )
    record = data[0]

    assert data.feature_names == (
        "model.depth",
        "model.width",
        "episode.examples",
        "family.locality",
        "program.parameter_fraction",
        "interaction.sensitivity",
        "baseline.alpha",
        "baseline.zeta",
    )
    assert tuple(record.features) == data.feature_names
    assert data.target_names == ("alpha", "zeta")
    assert data.target_kind == "delta"
    assert record.target == result.delta_geometry
    assert record.model_key == result.model_context.identity_key
    assert record.episode_key == result.episode.identity_key
    assert record.family_fingerprint == result.program.family.fingerprint  # type: ignore[union-attr]
    assert record.program_fingerprint == result.program.fingerprint
    assert record.metadata == result.metadata
    assert data.metadata == {"schema_name": "synthetic", "schema_version": "1"}
    assert "model_id" not in record.features
    assert "seed" not in record.features
    assert not hasattr(record, "model_context")
    assert not hasattr(record, "episode")
    assert not hasattr(record, "program")
    exported = data.to_records()
    assert exported[0]["model_key"] == ["model", "rev", "base"]
    assert exported[0]["features"]["baseline.alpha"] == 0.1
    assert exported[0]["target"] == pytest.approx({"zeta": 0.5, "alpha": 0.3})
    json.dumps(exported)

    with pytest.raises(TypeError):
        record.features["model.width"] = 9.0  # type: ignore[index]
    with pytest.raises(TypeError):
        record.metadata["seed"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        data.metadata["schema_version"] = "2"  # type: ignore[index]


def test_after_and_delta_target_semantics() -> None:
    result = _result()
    adaptations = AdaptationDataset([result])

    after = build_compiler_dataset(adaptations, target="after")
    delta = build_compiler_dataset(adaptations, target="delta")

    assert after[0].target == result.after_geometry
    assert after.target_kind == "after"
    assert delta[0].target == result.delta_geometry
    assert delta[0].target["alpha"] == pytest.approx(0.3)
    assert delta.target_kind == "delta"
    with pytest.raises(ValidationError, match="target must be"):
        build_compiler_dataset(adaptations, target="utility")  # type: ignore[arg-type]


def test_baseline_is_per_observation_and_cannot_be_overwritten() -> None:
    first = _result(before={"alpha": 0.1}, after={"alpha": 0.4}, seed=1)
    second = _result(before={"alpha": 0.25}, after={"alpha": 0.5}, seed=2)
    data = build_compiler_dataset(AdaptationDataset([first, second]))

    assert data.feature_names == ("baseline.alpha",)
    assert [record.features["baseline.alpha"] for record in data] == [0.1, 0.25]
    assert [record.target["alpha"] for record in data] == pytest.approx([0.3, 0.25])

    key = first.model_context.identity_key
    for name in ("baseline.alpha", "after.alpha", "delta.alpha", "target.alpha"):
        with pytest.raises(ValidationError, match="reserved namespace"):
            build_compiler_dataset(
                AdaptationDataset([first]),
                model_descriptors={key: {name: 1.0}},
            )


@pytest.mark.parametrize(
    "bad_value",
    [True, float("nan"), float("inf"), -float("inf"), "1", [1], {"nested": 1}],
)
def test_descriptor_values_follow_finite_real_numeric_policy(bad_value: object) -> None:
    result = _result(before={"metric": 0.1}, after={"metric": 0.2})
    with pytest.raises(ValidationError, match="real number|finite"):
        build_compiler_dataset(
            AdaptationDataset([result]),
            model_descriptors={result.model_context.identity_key: {"value": bad_value}},  # type: ignore[dict-item]
        )


def test_integer_descriptor_values_are_accepted_and_normalized() -> None:
    result = _result(before={"metric": 0.1}, after={"metric": 0.2})
    data = build_compiler_dataset(
        AdaptationDataset([result]),
        model_descriptors={result.model_context.identity_key: {"count": 7}},
    )
    assert data[0].features["model.count"] == 7.0
    assert isinstance(data[0].features["model.count"], float)


@pytest.mark.parametrize("name", ["", " "])
def test_empty_descriptor_names_are_rejected(name: str) -> None:
    result = _result()
    with pytest.raises(ValidationError, match="non-empty"):
        build_compiler_dataset(
            AdaptationDataset([result]),
            model_descriptors={result.model_context.identity_key: {name: 1.0}},
        )


@pytest.mark.parametrize(
    "namespace", ["model", "episode.x", "family.x", "program.x", "interaction.x"]
)
def test_prequalified_descriptor_names_are_rejected(namespace: str) -> None:
    result = _result()
    with pytest.raises(ValidationError, match="reserved namespace"):
        build_compiler_dataset(
            AdaptationDataset([result]),
            model_descriptors={result.model_context.identity_key: {namespace: 1.0}},
        )


@pytest.mark.parametrize(
    ("argument", "identity"),
    [
        ("model_descriptors", ("missing", None, None)),
        ("episode_descriptors", ("episode_id", "missing")),
        ("family_descriptors", "missing"),
        ("program_descriptors", "missing"),
        (
            "interaction_descriptors",
            (("missing", None, None), ("episode_id", "missing"), "missing"),
        ),
    ],
)
def test_missing_descriptor_keys_fail_clearly(argument: str, identity: object) -> None:
    result = _result()
    with pytest.raises(ValidationError, match="missing .* descriptors"):
        build_compiler_dataset(
            AdaptationDataset([result]),
            **{argument: {identity: {"value": 1.0}}},
        )


def test_familyless_program_policy_is_explicit() -> None:
    program = ProgramSpec("familyless", "prompt", {"shots": 2})
    result = _result(program=program)
    valid = build_compiler_dataset(AdaptationDataset([result]))
    assert valid[0].family_fingerprint is None
    assert not any(name.startswith("family.") for name in valid.feature_names)

    with pytest.raises(ValidationError, match="has no ProgramFamily"):
        build_compiler_dataset(
            AdaptationDataset([result]),
            family_descriptors={"unused": {"value": 1.0}},
        )


def test_heterogeneous_feature_and_target_schemas_fail_clearly() -> None:
    first = _result(
        model=ModelContext("model-a"),
        before={"metric": 0.1},
        after={"metric": 0.2},
    )
    second = _result(
        model=ModelContext("model-b"),
        before={"metric": 0.1},
        after={"metric": 0.3},
    )
    with pytest.raises(ValidationError, match="feature schema mismatch"):
        build_compiler_dataset(
            AdaptationDataset([first, second]),
            model_descriptors={
                first.model_context.identity_key: {"a": 1.0},
                second.model_context.identity_key: {"b": 2.0},
            },
        )

    mixed = _result(before={"other": 0.1}, after={"other": 0.2})
    with pytest.raises(ValidationError, match="feature schema mismatch"):
        build_compiler_dataset(AdaptationDataset([first, mixed]))


def test_compiler_dataset_validates_target_schema_and_kind_directly() -> None:
    common = {
        "model_key": ("model", None, None),
        "episode_key": ("episode_id", "episode"),
        "family_fingerprint": None,
        "program_fingerprint": "program",
        "features": {"baseline.metric": 0.1},
    }
    after = CompilerRecord(
        **common,
        target=AdaptationGeometry({"metric": 0.2}),
        target_kind="after",
    )
    different_target = CompilerRecord(
        **common,
        target=AdaptationGeometry({"other": 0.2}),
        target_kind="after",
    )
    delta = CompilerRecord(
        **common,
        target=AdaptationGeometry({"metric": 0.1}),
        target_kind="delta",
    )

    with pytest.raises(ValidationError, match="target schema mismatch"):
        CompilerDataset([after, different_target])
    with pytest.raises(ValidationError, match="target kind mismatch"):
        CompilerDataset([after, delta])
    with pytest.raises(ValidationError, match="at least one"):
        CompilerDataset([])
    with pytest.raises(ValidationError, match="CompilerRecord"):
        CompilerDataset([object()])  # type: ignore[list-item]


def test_semantic_identity_joins_distinguish_labels_and_share_fingerprints() -> None:
    model_a = ModelContext("same-model", "revision-a")
    model_b = ModelContext("same-model", "revision-b")
    episode_a = LearningEpisode("same episode", [], {"m": []}, episode_id="a")
    episode_b = LearningEpisode("same episode", [], {"m": []}, episode_id="b")
    family_a = ProgramFamily("label-a", "adapter", {"region": 1})
    family_b = ProgramFamily("label-b", "adapter", {"region": 1})
    program_a = ProgramSpec("label-a", "adapter", {"rank": 4}, family=family_a)
    program_b = ProgramSpec("label-b", "adapter", {"rank": 4}, family=family_b)
    assert family_a.fingerprint == family_b.fingerprint
    assert program_a.fingerprint == program_b.fingerprint
    results = [
        _result(
            model=model_a,
            episode=episode_a,
            program=program_a,
            before={"m": 0.1},
            after={"m": 0.2},
        ),
        _result(
            model=model_b,
            episode=episode_b,
            program=program_b,
            before={"m": 0.1},
            after={"m": 0.3},
        ),
    ]
    data = build_compiler_dataset(
        AdaptationDataset(results),
        model_descriptors={
            model_a.identity_key: {"v": 1},
            model_b.identity_key: {"v": 2},
        },
        episode_descriptors={
            episode_a.identity_key: {"v": 3},
            episode_b.identity_key: {"v": 4},
        },
        family_descriptors={family_a.fingerprint: {"v": 5}},
        program_descriptors={program_a.fingerprint: {"v": 6}},
    )

    assert [record.features["model.v"] for record in data] == [1.0, 2.0]
    assert [record.features["episode.v"] for record in data] == [3.0, 4.0]
    assert [record.features["family.v"] for record in data] == [5.0, 5.0]
    assert [record.features["program.v"] for record in data] == [6.0, 6.0]


def test_interaction_identity_and_replicates_are_preserved() -> None:
    first = _result(seed=1)
    replicate = _result(seed=2, after={"zeta": 0.8, "alpha": 0.5})
    other_episode = LearningEpisode(
        "episode", [], {"metric": []}, episode_id="episode-2"
    )
    other = _result(episode=other_episode, seed=3)

    def interaction_key(result: AdaptationResult) -> tuple[object, object, str]:
        return (
            result.model_context.identity_key,
            result.episode.identity_key,
            result.program.fingerprint,
        )

    data = build_compiler_dataset(
        AdaptationDataset([first, replicate, other]),
        interaction_descriptors={
            interaction_key(first): {"value": 1.0},
            interaction_key(other): {"value": 2.0},
        },  # type: ignore[arg-type]
    )

    assert len(data) == 3
    assert [record.features["interaction.value"] for record in data] == [1.0, 1.0, 2.0]
    assert [record.metadata["seed"] for record in data] == [1, 2, 3]
    assert data[0].features == data[1].features
    assert data[0].target != data[1].target


def test_dict_order_does_not_affect_feature_order() -> None:
    result = _result()
    key = result.model_context.identity_key
    first = build_compiler_dataset(
        AdaptationDataset([result]),
        model_descriptors={key: {"z": 1, "a": 2}},
    )
    second = build_compiler_dataset(
        AdaptationDataset([result]),
        model_descriptors={key: {"a": 2, "z": 1}},
    )
    assert first.feature_names == second.feature_names
    assert first.to_records() == second.to_records()


def test_dataframe_export_has_unambiguous_columns() -> None:
    pd = pytest.importorskip("pandas")
    result = _result()
    data = build_compiler_dataset(AdaptationDataset([result]))
    frame = data.to_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert "feature.baseline.alpha" in frame
    assert "target.alpha" in frame
    assert "metadata" in frame
