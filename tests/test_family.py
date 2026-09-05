from __future__ import annotations

import pytest

from adaptcompile import ProgramFamily, ProgramSpec, SerializationError, ValidationError


def test_family_fingerprint_is_conceptual_but_equality_is_structural() -> None:
    baseline = ProgramFamily(
        "middle",
        "localized_lora",
        {"region": "middle", "rank": 16},
    )
    renamed = ProgramFamily(
        "candidate-middle",
        "localized_lora",
        {"rank": 16, "region": "middle"},
    )
    annotated = ProgramFamily(
        "middle",
        "localized_lora",
        {"region": "middle", "rank": 16},
        {"annotation": "synthetic"},
    )

    assert baseline.fingerprint == renamed.fingerprint == annotated.fingerprint
    assert baseline != renamed
    assert baseline != annotated


def test_conceptual_parameter_change_alters_family_fingerprint() -> None:
    middle = ProgramFamily("middle", "localized_lora", {"region": "middle"})
    late = ProgramFamily("late", "localized_lora", {"region": "late"})

    assert middle.fingerprint != late.fingerprint


def test_same_family_can_have_different_executable_realizations() -> None:
    family = ProgramFamily(
        "middle",
        "localized_lora",
        {"region": "middle", "rank": 16},
    )
    model_a = ProgramSpec(
        "model-a-middle",
        "localized_lora",
        {"layers": [14, 15, 16, 17], "rank": 16},
        family=family,
    )
    model_b = ProgramSpec(
        "model-b-middle",
        "localized_lora",
        {"layers": [11, 12, 13, 14], "rank": 16},
        family=family,
    )

    assert model_a.family is family
    assert model_b.family is family
    assert model_a.family.fingerprint == model_b.family.fingerprint
    assert model_a.fingerprint != model_b.fingerprint


def test_spec_fingerprint_excludes_name_metadata_and_family() -> None:
    family = ProgramFamily("middle", "localized_lora", {"region": "middle"})
    alternate_family = ProgramFamily(
        "broad-middle",
        "localized_lora",
        {"region": "middle-half"},
    )
    baseline = ProgramSpec(
        "first",
        "localized_lora",
        {"layers": [12, 13], "rank": 16},
        family=family,
    )
    reconstructed = ProgramSpec(
        "renamed",
        "localized_lora",
        {"rank": 16, "layers": [12, 13]},
        metadata={"note": "different"},
        family=alternate_family,
    )

    assert baseline.fingerprint == reconstructed.fingerprint
    assert baseline != reconstructed


def test_realization_change_alters_spec_fingerprint() -> None:
    family = ProgramFamily("middle", "localized_lora", {"region": "middle"})
    first = ProgramSpec("first", "localized_lora", {"layers": [12, 13]}, family=family)
    second = ProgramSpec(
        "second", "localized_lora", {"layers": [14, 15]}, family=family
    )

    assert first.fingerprint != second.fingerprint


def test_family_and_spec_methods_must_match() -> None:
    family = ProgramFamily("middle", "localized_lora", {"region": "middle"})

    with pytest.raises(ValidationError, match="method must match"):
        ProgramSpec("candidate", "full_finetune", {}, family=family)


def test_family_serialization_and_nested_spec_roundtrip() -> None:
    family = ProgramFamily(
        "middle",
        "localized_lora",
        {"region": "middle", "tags": ["relative"]},
        {"synthetic": True},
    )
    program = ProgramSpec(
        "model-a-middle",
        "localized_lora",
        {"layers": [14, 15, 16, 17]},
        family=family,
    )

    assert ProgramFamily.from_dict(family.to_dict()) == family
    restored_family = ProgramFamily.from_json(family.to_json())
    restored_program = ProgramSpec.from_json(program.to_json())
    assert restored_family == family
    assert restored_family.fingerprint == family.fingerprint
    assert ProgramSpec.from_dict(program.to_dict()) == program
    assert restored_program == program
    assert restored_program.fingerprint == program.fingerprint
    assert restored_program.family is not None
    assert restored_program.family.fingerprint == family.fingerprint
    assert program.to_dict()["family"] == family.to_dict()


def test_family_validation_and_immutable_parameters() -> None:
    parameters = {"region": "middle", "tags": ["relative"]}
    family = ProgramFamily("middle", "localized_lora", parameters)
    parameters["region"] = "changed"

    assert family.parameters["region"] == "middle"
    assert family.parameters["tags"] == ("relative",)
    with pytest.raises(ValidationError):
        ProgramFamily("", "localized_lora", {})
    with pytest.raises(SerializationError):
        ProgramFamily("middle", "localized_lora", {"bad": object()})
