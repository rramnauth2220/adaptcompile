from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

import adaptcompile
import adaptcompile.compiler as compiler
import adaptcompile.execution as execution
import adaptcompile.execution.backends as backend_module
from adaptcompile import (
    AdaptationGeometry,
    LearningEpisode,
    ModelContext,
    ProgramFamily,
    ProgramSpec,
    SerializationError,
    ValidationError,
)
from adaptcompile.compiler import (
    GeometryConstraint,
    GeometryPrediction,
    ProgramSelection,
    SelectionObjective,
)
from adaptcompile.compiler.selectors import LinearUtilitySelector
from adaptcompile.execution import (
    AdaptationBackend,
    ExecutionOutcome,
    ExecutionRecord,
    execute_selection,
)
from adaptcompile.execution.backends import CallableBackend


def _context() -> ModelContext:
    return ModelContext("model", "revision", "base")


def _episode(identifier: str = "episode") -> LearningEpisode:
    return LearningEpisode("task", [], {"score": []}, episode_id=identifier)


def _program(
    name: str,
    amount: float,
    *,
    family: ProgramFamily | None = None,
) -> ProgramSpec:
    return ProgramSpec(name, "scale", {"amount": amount}, family=family)


def _prediction(
    program: ProgramSpec,
    score: float,
    *,
    model_context: ModelContext | None = None,
    episode: LearningEpisode | None = None,
    family_fingerprint: str | None | object = ...,  # type: ignore[assignment]
) -> GeometryPrediction:
    context = model_context or _context()
    learning_episode = episode or _episode()
    geometry = AdaptationGeometry({"score": score})
    if family_fingerprint is ...:
        family_fingerprint = (
            None if program.family is None else program.family.fingerprint
        )
    return GeometryPrediction(
        model_key=context.identity_key,
        episode_key=learning_episode.identity_key,
        family_fingerprint=family_fingerprint,  # type: ignore[arg-type]
        program_fingerprint=program.fingerprint,
        predicted_target=geometry,
        predicted_geometry=geometry,
        target_kind="after",
    )


def _selection(
    programs: list[ProgramSpec],
    *,
    scores: list[float] | None = None,
    constraint: GeometryConstraint | None = None,
) -> ProgramSelection:
    values = scores or [float(index + 1) for index in range(len(programs))]
    objective = SelectionObjective(
        maximize={"score": 1},
        constraints=() if constraint is None else (constraint,),
    )
    return LinearUtilitySelector().select(
        [
            _prediction(program, value)
            for program, value in zip(programs, values, strict=True)
        ],
        objective,
    )


def _record(program: ProgramSpec, **overrides: Any) -> ExecutionRecord:
    context = _context()
    episode = _episode()
    values: dict[str, Any] = {
        "model_key": context.identity_key,
        "episode_key": episode.identity_key,
        "family_fingerprint": (
            None if program.family is None else program.family.fingerprint
        ),
        "program_fingerprint": program.fingerprint,
        "program": program,
        "backend_id": "toy",
        "selected_utility": 1.25,
        "metadata": {"request": "synthetic", "nested": {"attempt": 1}},
    }
    values.update(overrides)
    return ExecutionRecord(**values)


def test_execution_public_api_is_separate_and_explicit() -> None:
    assert execution.__all__ == [
        "AdaptationBackend",
        "ExecutionOutcome",
        "ExecutionRecord",
        "execute_selection",
    ]
    assert backend_module.__all__ == ["CallableBackend"]
    for name in execution.__all__:
        assert not hasattr(adaptcompile, name)
        assert not hasattr(compiler, name)
    assert not hasattr(adaptcompile, "CallableBackend")
    assert not hasattr(compiler, "CallableBackend")


def test_execution_record_familyless_is_immutable_and_roundtrips() -> None:
    program = _program("small", 0.5)
    record = _record(program)

    assert record.family_fingerprint is None
    assert record.selected_utility == 1.25
    assert ExecutionRecord.from_dict(record.to_dict()) == record
    json.dumps(record.to_dict())
    with pytest.raises(FrozenInstanceError):
        record.backend_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.metadata["request"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.metadata["nested"]["attempt"] = 2  # type: ignore[index]


def test_execution_record_preserves_program_family() -> None:
    family = ProgramFamily("scaling", "scale", {"domain": "toy"})
    program = _program("small", 0.5, family=family)
    record = _record(program)

    assert record.program is program
    assert record.family_fingerprint == family.fingerprint
    assert ExecutionRecord.from_dict(record.to_dict()) == record


def test_execution_record_rejects_identity_mismatches() -> None:
    program = _program("small", 0.5)
    with pytest.raises(ValidationError, match="program_fingerprint"):
        _record(program, program_fingerprint="wrong")

    family = ProgramFamily("scaling", "scale", {"domain": "toy"})
    family_program = _program("family", 0.5, family=family)
    with pytest.raises(ValidationError, match="family_fingerprint"):
        _record(family_program, family_fingerprint="wrong")
    with pytest.raises(ValidationError, match="family_fingerprint"):
        _record(program, family_fingerprint="unexpected")


@pytest.mark.parametrize("backend_id", ["", " ", None, 1])
def test_execution_record_rejects_invalid_backend_id(backend_id: object) -> None:
    with pytest.raises(ValidationError, match="backend_id"):
        _record(_program("small", 0.5), backend_id=backend_id)


@pytest.mark.parametrize("utility", [True, float("nan"), float("inf"), -float("inf")])
def test_execution_record_rejects_invalid_utility(utility: object) -> None:
    with pytest.raises(ValidationError, match="selected_utility"):
        _record(_program("small", 0.5), selected_utility=utility)


def test_execution_record_validates_metadata() -> None:
    with pytest.raises(SerializationError, match="execution metadata"):
        _record(_program("small", 0.5), metadata={"runtime": object()})


def test_execution_outcome_keeps_opaque_model_and_has_no_serialization() -> None:
    model = {"strength": 1.0}
    record = _record(_program("small", 0.5))
    outcome = ExecutionOutcome(model=model, record=record)

    assert outcome.model is model
    assert outcome.record is record
    assert not hasattr(outcome, "to_dict")
    with pytest.raises(ValidationError, match="ExecutionRecord"):
        ExecutionOutcome(model=model, record=object())  # type: ignore[arg-type]


def test_callable_backend_passes_exact_execution_inputs() -> None:
    context = _context()
    episode = _episode()
    program = _program("small", 0.5)
    model = {"strength": 1.0}
    seen: dict[str, object] = {}

    def handler(
        runtime: dict[str, float],
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> dict[str, float]:
        seen.update(
            runtime=runtime,
            model_context=model_context,
            episode=episode,
            program=program,
        )
        return {"strength": runtime["strength"] + program.parameters["amount"]}

    backend = CallableBackend(handler, backend_id="toy")
    adapted = backend.execute(
        model, model_context=context, episode=episode, program=program
    )

    assert backend.backend_id == "toy"
    assert backend.supports(program) is True
    assert seen == {
        "runtime": model,
        "model_context": context,
        "episode": episode,
        "program": program,
    }
    assert adapted == {"strength": 1.5}


def test_callable_backend_custom_support_is_strict() -> None:
    def handler(model: object, **_: object) -> object:
        return model

    supported = CallableBackend(
        handler, supports=lambda program: program.method == "scale"
    )
    rejected = CallableBackend(handler, supports=lambda program: False)
    malformed = CallableBackend(handler, supports=lambda program: 1)  # type: ignore[arg-type,return-value]
    program = _program("small", 0.5)

    assert supported.supports(program)
    assert rejected.supports(program) is False
    with pytest.raises(ValidationError, match="return a boolean"):
        malformed.supports(program)


def test_callable_backend_configuration_validation() -> None:
    with pytest.raises(ValidationError, match="handler"):
        CallableBackend(object())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="backend_id"):
        CallableBackend(lambda model, **_: model, backend_id=" ")
    with pytest.raises(ValidationError, match="supports"):
        CallableBackend(lambda model, **_: model, supports=object())  # type: ignore[arg-type]


def test_callable_backend_handler_exception_propagates() -> None:
    failure = RuntimeError("backend failed")

    def fail(model: object, **_: object) -> object:
        raise failure

    backend = CallableBackend(fail)
    with pytest.raises(RuntimeError) as captured:
        backend.execute(
            object(),
            model_context=_context(),
            episode=_episode(),
            program=_program("small", 0.5),
        )
    assert captured.value is failure


def test_user_backend_satisfies_protocol_structurally() -> None:
    class ToyBackend:
        backend_id = "toy"

        def supports(self, program: ProgramSpec) -> bool:
            return True

        def execute(
            self,
            model: dict[str, float],
            *,
            model_context: ModelContext,
            episode: LearningEpisode,
            program: ProgramSpec,
        ) -> dict[str, float]:
            return model

    backend: AdaptationBackend[dict[str, float]] = ToyBackend()
    assert isinstance(backend, AdaptationBackend)


def test_execute_selection_resolves_and_executes_one_exact_program() -> None:
    context = _context()
    episode = _episode()
    programs = [_program("small", 0.5), _program("large", 1.5)]
    selection = _selection(programs)
    calls: list[ProgramSpec] = []

    def handler(
        model: dict[str, float],
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> dict[str, float]:
        calls.append(program)
        return {"strength": model["strength"] * program.parameters["amount"]}

    outcome = execute_selection(
        selection,
        programs=programs,
        model={"strength": 2.0},
        model_context=context,
        episode=episode,
        backend=CallableBackend(handler, backend_id="toy-scale"),
        metadata={"request": "unit"},
    )

    assert calls == [programs[1]]
    assert outcome.model == {"strength": 3.0}
    assert outcome.record.model_key == context.identity_key
    assert outcome.record.episode_key == episode.identity_key
    assert outcome.record.program is programs[1]
    assert outcome.record.program_fingerprint == programs[1].fingerprint
    assert outcome.record.backend_id == "toy-scale"
    assert outcome.record.selected_utility == selection.selected.utility  # type: ignore[union-attr]
    assert outcome.record.metadata == {"request": "unit"}


def _counting_backend(calls: list[ProgramSpec], *, supported: bool = True):
    def handler(
        model: object,
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> object:
        calls.append(program)
        return model

    return CallableBackend(handler, supports=lambda program: supported)


def test_no_feasible_selection_prevents_execution() -> None:
    program = _program("small", 0.5)
    selection = _selection(
        [program], constraint=GeometryConstraint("score", minimum=10)
    )
    calls: list[ProgramSpec] = []

    with pytest.raises(ValidationError, match="no feasible"):
        execute_selection(
            selection,
            programs=[program],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=_counting_backend(calls),
        )
    assert calls == []


@pytest.mark.parametrize("mismatch", ["model", "episode"])
def test_decision_identity_mismatch_prevents_execution(mismatch: str) -> None:
    program = _program("small", 0.5)
    selection = _selection([program])
    calls: list[ProgramSpec] = []
    context = ModelContext("other") if mismatch == "model" else _context()
    episode = _episode("other") if mismatch == "episode" else _episode()

    with pytest.raises(ValidationError, match=f"{mismatch}_key"):
        execute_selection(
            selection,
            programs=[program],
            model=object(),
            model_context=context,
            episode=episode,
            backend=_counting_backend(calls),
        )
    assert calls == []


def test_missing_selected_program_prevents_execution() -> None:
    selected = _program("selected", 1)
    other = _program("other", 2)
    calls: list[ProgramSpec] = []
    with pytest.raises(ValidationError, match="exactly one"):
        execute_selection(
            _selection([selected]),
            programs=[other],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=_counting_backend(calls),
        )
    assert calls == []


def test_empty_program_candidates_prevent_execution() -> None:
    selected = _program("selected", 1)
    calls: list[ProgramSpec] = []
    with pytest.raises(ValidationError, match="at least one ProgramSpec"):
        execute_selection(
            _selection([selected]),
            programs=[],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=_counting_backend(calls),
        )
    assert calls == []


def test_duplicate_program_fingerprints_prevent_execution() -> None:
    selected = _program("selected", 1)
    duplicate = ProgramSpec(
        "different label",
        selected.method,
        selected.parameters,
        metadata={"other": True},
    )
    calls: list[ProgramSpec] = []
    with pytest.raises(ValidationError, match="unique"):
        execute_selection(
            _selection([selected]),
            programs=[selected, duplicate],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=_counting_backend(calls),
        )
    assert calls == []


def test_family_mismatch_prevents_execution() -> None:
    selected_family = ProgramFamily("selected", "scale", {"kind": "a"})
    supplied_family = ProgramFamily("supplied", "scale", {"kind": "b"})
    selected = _program("selected", 1, family=selected_family)
    supplied = _program("supplied", 1, family=supplied_family)
    calls: list[ProgramSpec] = []

    with pytest.raises(ValidationError, match="family"):
        execute_selection(
            _selection([selected]),
            programs=[supplied],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=_counting_backend(calls),
        )
    assert calls == []


def test_unsupported_program_prevents_execution() -> None:
    program = _program("selected", 1)
    calls: list[ProgramSpec] = []
    with pytest.raises(ValidationError, match="does not support"):
        execute_selection(
            _selection([program]),
            programs=[program],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=_counting_backend(calls, supported=False),
        )
    assert calls == []


def test_invalid_backend_id_prevents_execution() -> None:
    program = _program("selected", 1)
    calls: list[ProgramSpec] = []

    class InvalidBackend:
        backend_id = " "

        def supports(self, candidate: ProgramSpec) -> bool:
            return True

        def execute(
            self,
            model: object,
            *,
            model_context: ModelContext,
            episode: LearningEpisode,
            program: ProgramSpec,
        ) -> object:
            calls.append(program)
            return model

    with pytest.raises(ValidationError, match="backend_id"):
        execute_selection(
            _selection([program]),
            programs=[program],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=InvalidBackend(),
        )
    assert calls == []


def test_execute_selection_propagates_exact_backend_exception() -> None:
    program = _program("selected", 1)
    failure = RuntimeError("backend failed")

    def fail(model: object, **_: object) -> object:
        raise failure

    with pytest.raises(RuntimeError) as captured:
        execute_selection(
            _selection([program]),
            programs=[program],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=CallableBackend(fail),
        )
    assert captured.value is failure


@pytest.mark.parametrize("metadata", [[], "", 0, False])
def test_nonmapping_execution_metadata_prevents_execution(metadata: object) -> None:
    program = _program("selected", 1)
    calls: list[ProgramSpec] = []
    with pytest.raises(SerializationError, match="execution metadata"):
        execute_selection(
            _selection([program]),
            programs=[program],
            model=object(),
            model_context=_context(),
            episode=_episode(),
            backend=_counting_backend(calls),
            metadata=metadata,  # type: ignore[arg-type]
        )
    assert calls == []


@pytest.mark.parametrize(
    ("metadata", "expected"), [(None, {}), ({"run": "toy"}, {"run": "toy"})]
)
def test_none_and_mapping_execution_metadata_are_preserved(
    metadata: dict[str, str] | None, expected: dict[str, str]
) -> None:
    program = _program("selected", 1)
    calls: list[ProgramSpec] = []

    outcome = execute_selection(
        _selection([program]),
        programs=[program],
        model=object(),
        model_context=_context(),
        episode=_episode(),
        backend=_counting_backend(calls),
        metadata=metadata,
    )

    assert calls == [program]
    assert outcome.record.metadata == expected
