from __future__ import annotations

import pytest

from adaptcompile import AdaptationResult, ProgramFamily, ProgramSpec, ValidationError


def test_before_after_geometry_and_delta(episode, model_context) -> None:
    result = AdaptationResult(
        model_context=model_context,
        episode=episode,
        program=ProgramSpec("candidate", "steering", {"strength": 0.5}),
        before={"gain": 0.1, "retention": 0.9},
        after={"gain": 0.8, "retention": 0.85},
    )

    assert result.geometry == result.after_geometry
    assert result.geometry.to_dict() == {"gain": 0.8, "retention": 0.85}
    assert result.delta.to_dict() == {
        "gain": pytest.approx(0.7),
        "retention": pytest.approx(-0.05),
    }


def test_before_and_after_must_match(episode, model_context) -> None:
    with pytest.raises(ValidationError, match="must match exactly"):
        AdaptationResult(
            model_context=model_context,
            episode=episode,
            program=ProgramSpec("candidate", "method"),
            before={"a": 1},
            after={"b": 1},
        )


def test_serialization_roundtrip_preserves_record_configuration(
    episode, model_context
) -> None:
    original = AdaptationResult(
        model_context=model_context,
        episode=episode,
        program=ProgramSpec(
            "candidate",
            "method",
            {"rank": 4},
            family=ProgramFamily("family", "method", {"scope": "local"}),
        ),
        before={"gain": 0.1},
        after={"gain": 0.8},
        metadata={"run_id": "abc"},
    )
    restored = AdaptationResult.from_json(original.to_json())

    assert restored.to_dict() == original.to_dict()
    assert restored.model_context == original.model_context
    assert restored.episode.train is None
    assert restored.program == original.program
    assert restored.program.family == original.program.family
    assert restored.program.fingerprint == original.program.fingerprint
    assert restored.program.family is not None
    assert original.program.family is not None
    assert restored.program.family.fingerprint == original.program.family.fingerprint
    assert restored.delta == original.delta


def test_model_context_is_required_and_validated(episode) -> None:
    with pytest.raises(ValidationError, match="model_context"):
        AdaptationResult(
            model_context=object(),  # type: ignore[arg-type]
            episode=episode,
            program=ProgramSpec("candidate", "method"),
            before={"gain": 0.1},
            after={"gain": 0.8},
        )
