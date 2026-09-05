from __future__ import annotations

import pytest

from adaptcompile import ProgramSpec, SerializationError, ValidationError


def test_parameters_are_preserved_but_immutable() -> None:
    original = {"layers": [2, 3], "nested": {"rank": 16}}
    program = ProgramSpec("late", "lora", original)
    original["layers"].append(4)

    assert program.to_dict()["parameters"] == {
        "layers": [2, 3],
        "nested": {"rank": 16},
    }
    assert program.parameters["layers"] == (2, 3)


def test_equality_hash_and_fingerprint_are_stable_across_key_order() -> None:
    left = ProgramSpec("p", "method", {"rank": 8, "alpha": 16})
    right = ProgramSpec("p", "method", {"alpha": 16, "rank": 8})

    assert left == right
    assert hash(left) == hash(right)
    assert left.fingerprint == right.fingerprint
    assert len(left.fingerprint) == 64


def test_fingerprint_is_executable_identity_but_equality_is_structural() -> None:
    baseline = ProgramSpec("A", "lora", {"rank": 8})
    renamed = ProgramSpec("B", "lora", {"rank": 8})
    annotated = ProgramSpec("A", "lora", {"rank": 8}, {"note": "candidate"})

    assert baseline.fingerprint == renamed.fingerprint == annotated.fingerprint
    assert baseline != renamed
    assert baseline != annotated


def test_executable_changes_alter_fingerprint() -> None:
    baseline = ProgramSpec("A", "lora", {"rank": 8})

    assert baseline.fingerprint != ProgramSpec("A", "steering", {"rank": 8}).fingerprint
    assert baseline.fingerprint != ProgramSpec("A", "lora", {"rank": 16}).fingerprint


def test_serialization_roundtrip() -> None:
    program = ProgramSpec("p", "method", {"layers": [1, 2]}, {"owner": "lab"})
    assert ProgramSpec.from_dict(program.to_dict()) == program
    assert ProgramSpec.from_json(program.to_json()) == program


def test_malformed_specs_fail() -> None:
    with pytest.raises(ValidationError):
        ProgramSpec("", "method")
    with pytest.raises(SerializationError):
        ProgramSpec("p", "method", {"bad": object()})
