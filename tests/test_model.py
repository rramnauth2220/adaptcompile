from __future__ import annotations

import pytest

from adaptcompile import ModelContext, SerializationError, ValidationError


def test_construction_identity_and_immutable_metadata() -> None:
    metadata = {"note": "primary", "tags": ["synthetic"]}
    context = ModelContext(
        model_id="example/model",
        revision="revision-1",
        base_state_id="base-state-1",
        metadata=metadata,
    )
    metadata["note"] = "changed"

    assert context.identity_key == (
        "example/model",
        "revision-1",
        "base-state-1",
    )
    assert context.metadata["note"] == "primary"
    assert context.metadata["tags"] == ("synthetic",)
    assert repr(context) == (
        "ModelContext(model_id='example/model', revision='revision-1', "
        "base_state_id='base-state-1')"
    )


def test_identity_ignores_metadata_but_structural_equality_does_not() -> None:
    first = ModelContext("model", "revision", "state", {"note": "first"})
    second = ModelContext("model", "revision", "state", {"note": "second"})

    assert first.identity_key == second.identity_key
    assert first != second


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_id", ""), ("revision", " "), ("base_state_id", "")],
)
def test_identity_fields_must_be_nonempty(field: str, value: str) -> None:
    values = {
        "model_id": "model",
        "revision": None,
        "base_state_id": None,
    }
    values[field] = value
    with pytest.raises(ValidationError, match=field):
        ModelContext(**values)  # type: ignore[arg-type]


def test_optional_identity_fields() -> None:
    context = ModelContext("model")
    assert context.identity_key == ("model", None, None)


def test_serialization_roundtrip(tmp_path) -> None:
    target = tmp_path / "context.json"
    original = ModelContext("model", "revision", "state", {"owner": "lab"})

    original.to_json(target)

    restored = ModelContext.from_json(target)
    assert ModelContext.from_dict(original.to_dict()) == original
    assert restored == original
    assert restored.identity_key == original.identity_key


def test_metadata_must_be_json_safe() -> None:
    with pytest.raises(SerializationError):
        ModelContext("model", metadata={"bad": object()})
