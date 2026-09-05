from __future__ import annotations

import json
from importlib.metadata import version

import pytest

import adaptcompile
from adaptcompile import ProgramSpec, SerializationError


def test_json_file_roundtrip(tmp_path) -> None:
    target = tmp_path / "program.json"
    original = ProgramSpec("candidate", "method", {"rank": 8})

    returned_text = original.to_json(target)
    restored = ProgramSpec.from_json(target)

    assert restored == original
    assert json.loads(returned_text) == json.loads(target.read_text(encoding="utf-8"))


def test_non_string_keys_are_rejected() -> None:
    with pytest.raises(SerializationError, match="non-string mapping key"):
        ProgramSpec("candidate", "method", {1: "bad"})  # type: ignore[dict-item]


def test_invalid_json_fails_clearly() -> None:
    with pytest.raises(SerializationError, match="invalid JSON"):
        ProgramSpec.from_json("{not valid}")


def test_public_version_comes_from_package_metadata() -> None:
    assert adaptcompile.__version__ == version("adaptcompile")


def test_public_api_is_explicit_and_stable() -> None:
    expected = {
        "AdaptCompileError",
        "AdaptationDataset",
        "AdaptationGeometry",
        "AdaptationResult",
        "AdaptationStudy",
        "LearningEpisode",
        "ModelContext",
        "ProgramFamily",
        "ProgramSpec",
        "SerializationError",
        "ValidationError",
        "__version__",
    }

    assert set(adaptcompile.__all__) == expected
    assert not hasattr(adaptcompile, "MetricSpec")
    assert not hasattr(adaptcompile, "EvaluationRecord")
    assert not hasattr(adaptcompile, "Compiler")
    assert not hasattr(adaptcompile, "version")
