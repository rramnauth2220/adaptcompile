"""Unit tests for framework-neutral post-execution evaluation."""

from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

import adaptcompile
import adaptcompile.compiler as compiler
import adaptcompile.evaluation as evaluation
import adaptcompile.evaluation.core as evaluation_core
import adaptcompile.evaluation.evaluators as evaluator_module
import adaptcompile.execution as execution_module
from adaptcompile import (
    AdaptationGeometry,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramFamily,
    ProgramSpec,
)
from adaptcompile.errors import SerializationError, ValidationError
from adaptcompile.evaluation import (
    AdaptationEvaluator,
    EvaluationOutcome,
    EvaluationRecord,
    evaluate_execution,
)
from adaptcompile.evaluation.evaluators import CallableEvaluator
from adaptcompile.execution import ExecutionOutcome, ExecutionRecord


def _context(model_id: str = "synthetic/model") -> ModelContext:
    return ModelContext(model_id, revision="v1", base_state_id="base")


def _episode(episode_id: str = "task-v1") -> LearningEpisode:
    return LearningEpisode("synthetic task", [], {"heldout": []}, episode_id=episode_id)


def _program(*, with_family: bool = False) -> ProgramSpec:
    family = ProgramFamily("scaling", "scale", {}) if with_family else None
    return ProgramSpec("selected", "scale", {"amount": 1.0}, family=family)


def _execution(
    *,
    model: dict[str, float] | None = None,
    context: ModelContext | None = None,
    episode: LearningEpisode | None = None,
    program: ProgramSpec | None = None,
) -> ExecutionOutcome[dict[str, float]]:
    context = _context() if context is None else context
    episode = _episode() if episode is None else episode
    program = _program() if program is None else program
    return ExecutionOutcome(
        model={"gain": 0.7, "retention": 0.85} if model is None else model,
        record=ExecutionRecord(
            model_key=context.identity_key,
            episode_key=episode.identity_key,
            family_fingerprint=(
                None if program.family is None else program.family.fingerprint
            ),
            program_fingerprint=program.fingerprint,
            program=program,
            backend_id="toy-backend",
            selected_utility=0.9,
        ),
    )


def _before() -> AdaptationGeometry:
    return AdaptationGeometry({"gain": 0.2, "retention": 0.95})


def _measuring_evaluator(
    calls: list[ProgramSpec],
) -> CallableEvaluator[dict[str, float]]:
    def measure(
        model: dict[str, float],
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> AdaptationGeometry:
        calls.append(program)
        return AdaptationGeometry(
            {"gain": model["gain"], "retention": model["retention"]}
        )

    return CallableEvaluator(measure, evaluator_id="toy-evaluator")


def _record(*, program: ProgramSpec | None = None) -> EvaluationRecord:
    program = _program() if program is None else program
    return EvaluationRecord(
        model_key=_context().identity_key,
        episode_key=_episode().identity_key,
        family_fingerprint=(
            None if program.family is None else program.family.fingerprint
        ),
        program_fingerprint=program.fingerprint,
        program=program,
        backend_id="toy-backend",
        evaluator_id="toy-evaluator",
        before=_before(),
        after=AdaptationGeometry({"gain": 0.7, "retention": 0.85}),
        metadata={"split": "heldout", "nested": [1, {"ok": True}]},
    )


def test_evaluation_public_api_is_separate_and_explicit() -> None:
    assert evaluation.__all__ == [
        "AdaptationEvaluator",
        "EvaluationOutcome",
        "EvaluationRecord",
        "evaluate_execution",
    ]
    assert evaluator_module.__all__ == ["CallableEvaluator"]
    for name in evaluation.__all__:
        assert getattr(evaluation, name) is globals()[name]
        assert not hasattr(adaptcompile, name)
        assert not hasattr(compiler, name)
        assert not hasattr(execution_module, name)


def test_evaluation_has_no_prediction_to_observation_dependency() -> None:
    source = inspect.getsource(evaluation_core)
    assert "compiler.prediction" not in source
    assert "predicted_geometry" not in source
    assert "predicted_target" not in source


def test_evaluation_record_is_immutable_serializable_and_roundtrips() -> None:
    record = _record()

    restored = EvaluationRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.program == record.program
    assert restored.before == record.before
    assert restored.after == record.after
    assert restored.metadata == record.metadata
    assert isinstance(json.dumps(record.to_dict()), str)
    with pytest.raises(FrozenInstanceError):
        record.backend_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.metadata["split"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.metadata["nested"][1]["ok"] = False  # type: ignore[index]


def test_evaluation_record_preserves_program_family() -> None:
    program = _program(with_family=True)
    record = _record(program=program)

    assert record.program is program
    assert record.program_fingerprint == program.fingerprint
    assert record.family_fingerprint == program.family.fingerprint  # type: ignore[union-attr]
    assert EvaluationRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"program_fingerprint": "wrong"}, "program_fingerprint"),
        ({"family_fingerprint": "wrong"}, "family_fingerprint"),
        ({"backend_id": ""}, "backend_id"),
        ({"evaluator_id": ""}, "evaluator_id"),
        ({"before": {}}, "before must be"),
        ({"after": {}}, "after must be"),
    ],
)
def test_evaluation_record_rejects_invalid_contract_fields(
    overrides: dict[str, Any], message: str
) -> None:
    program = _program()
    values: dict[str, Any] = {
        "model_key": _context().identity_key,
        "episode_key": _episode().identity_key,
        "family_fingerprint": None,
        "program_fingerprint": program.fingerprint,
        "program": program,
        "backend_id": "backend",
        "evaluator_id": "evaluator",
        "before": _before(),
        "after": AdaptationGeometry({"gain": 0.7, "retention": 0.85}),
    }
    values.update(overrides)

    with pytest.raises(ValidationError, match=message):
        EvaluationRecord(**values)


def test_evaluation_record_preserves_adaptation_result_geometry_compatibility() -> None:
    program = _program()
    with pytest.raises(ValidationError, match="metrics must match exactly"):
        EvaluationRecord(
            model_key=_context().identity_key,
            episode_key=_episode().identity_key,
            family_fingerprint=None,
            program_fingerprint=program.fingerprint,
            program=program,
            backend_id="backend",
            evaluator_id="evaluator",
            before=AdaptationGeometry({"gain": 0.2}),
            after=AdaptationGeometry({"retention": 0.8}),
        )


def test_evaluation_record_rejects_non_json_metadata() -> None:
    program = _program()
    with pytest.raises(SerializationError, match="evaluation metadata"):
        EvaluationRecord(
            model_key=_context().identity_key,
            episode_key=_episode().identity_key,
            family_fingerprint=None,
            program_fingerprint=program.fingerprint,
            program=program,
            backend_id="backend",
            evaluator_id="evaluator",
            before=_before(),
            after=AdaptationGeometry({"gain": 0.7, "retention": 0.85}),
            metadata={"bad": object()},
        )


def test_consistent_evaluation_outcome_is_no_runtime_model_or_serialization() -> None:
    record = _record()
    result = AdaptationResult(
        _context(), _episode(), record.program, before=record.before, after=record.after
    )
    outcome = EvaluationOutcome(result=result, record=record)

    assert outcome.result is result
    assert outcome.record is record
    assert not hasattr(outcome, "model")
    assert not hasattr(outcome, "to_dict")


def test_evaluation_outcome_rejects_model_identity_mismatch() -> None:
    record = _record()
    result = AdaptationResult(
        _context("different/model"),
        _episode(),
        record.program,
        before=record.before,
        after=record.after,
    )

    with pytest.raises(ValidationError, match="model_context identity"):
        EvaluationOutcome(result=result, record=record)


def test_evaluation_outcome_rejects_episode_identity_mismatch() -> None:
    record = _record()
    result = AdaptationResult(
        _context(),
        _episode("different-task"),
        record.program,
        before=record.before,
        after=record.after,
    )

    with pytest.raises(ValidationError, match="episode identity"):
        EvaluationOutcome(result=result, record=record)


def test_evaluation_outcome_requires_exact_declarative_program() -> None:
    record = _record()
    different_program = ProgramSpec(
        "renamed",
        record.program.method,
        record.program.parameters,
        metadata={"source": "different"},
    )
    assert different_program.fingerprint == record.program_fingerprint
    assert different_program != record.program
    result = AdaptationResult(
        _context(),
        _episode(),
        different_program,
        before=record.before,
        after=record.after,
    )

    with pytest.raises(ValidationError, match="exactly match record program"):
        EvaluationOutcome(result=result, record=record)


def test_evaluation_outcome_rejects_before_geometry_mismatch() -> None:
    record = _record()
    result = AdaptationResult(
        _context(),
        _episode(),
        record.program,
        before=AdaptationGeometry({"gain": 0.3, "retention": 0.95}),
        after=record.after,
    )

    with pytest.raises(ValidationError, match="before geometry"):
        EvaluationOutcome(result=result, record=record)


def test_evaluation_outcome_rejects_after_geometry_mismatch() -> None:
    record = _record()
    result = AdaptationResult(
        _context(),
        _episode(),
        record.program,
        before=record.before,
        after=AdaptationGeometry({"gain": 0.8, "retention": 0.85}),
    )

    with pytest.raises(ValidationError, match="after geometry"):
        EvaluationOutcome(result=result, record=record)


@pytest.mark.parametrize("field", ["result", "record"])
def test_evaluation_outcome_validates_members(field: str) -> None:
    record = _record()
    result = AdaptationResult(
        _context(), _episode(), record.program, before=record.before, after=record.after
    )
    values: dict[str, object] = {"result": result, "record": record}
    values[field] = object()

    with pytest.raises(ValidationError, match=field):
        EvaluationOutcome(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("handler", "evaluator_id", "message"),
    [
        (None, "valid", "handler"),
        (lambda model, **kwargs: _before(), "", "evaluator_id"),
    ],
)
def test_callable_evaluator_validates_construction(
    handler: object, evaluator_id: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        CallableEvaluator(handler, evaluator_id=evaluator_id)  # type: ignore[arg-type]


def test_callable_evaluator_receives_exact_inputs_and_measures_runtime() -> None:
    context = _context()
    episode = _episode()
    program = _program()
    model = {"gain": 0.73, "retention": 0.84}
    received: list[tuple[object, ModelContext, LearningEpisode, ProgramSpec]] = []

    def handler(
        runtime: dict[str, float],
        *,
        model_context: ModelContext,
        episode: LearningEpisode,
        program: ProgramSpec,
    ) -> AdaptationGeometry:
        received.append((runtime, model_context, episode, program))
        return AdaptationGeometry(runtime)

    evaluator = CallableEvaluator(handler, evaluator_id="behavior-suite-v2")
    measured = evaluator.evaluate(
        model, model_context=context, episode=episode, program=program
    )

    assert evaluator.evaluator_id == "behavior-suite-v2"
    assert received == [(model, context, episode, program)]
    assert received[0][0] is model
    assert measured == AdaptationGeometry(model)


def test_external_class_satisfies_structural_evaluator_protocol() -> None:
    class ToyEvaluator:
        evaluator_id = "external-toy"

        def evaluate(
            self,
            model: dict[str, float],
            *,
            model_context: ModelContext,
            episode: LearningEpisode,
            program: ProgramSpec,
        ) -> AdaptationGeometry:
            return AdaptationGeometry(model)

    evaluator = ToyEvaluator()
    assert isinstance(evaluator, AdaptationEvaluator)
    outcome = evaluate_execution(
        _execution(),
        evaluator=evaluator,
        model_context=_context(),
        episode=_episode(),
        before=_before(),
    )
    assert outcome.record.evaluator_id == "external-toy"


def test_evaluate_execution_creates_measured_result_and_provenance() -> None:
    context = _context()
    episode = _episode()
    program = _program(with_family=True)
    runtime = {"gain": 0.71, "retention": 0.86}
    execution = _execution(
        model=runtime, context=context, episode=episode, program=program
    )
    calls: list[ProgramSpec] = []

    outcome = evaluate_execution(
        execution,
        evaluator=_measuring_evaluator(calls),
        model_context=context,
        episode=episode,
        before=_before(),
        metadata={"split": "heldout"},
    )

    assert calls == [program]
    assert isinstance(outcome, EvaluationOutcome)
    assert isinstance(outcome.result, AdaptationResult)
    assert outcome.result.model_context is context
    assert outcome.result.episode is episode
    assert outcome.result.program is program
    assert outcome.result.before_geometry == _before()
    assert outcome.result.after_geometry == AdaptationGeometry(runtime)
    assert outcome.record.model_key == context.identity_key
    assert outcome.record.episode_key == episode.identity_key
    assert outcome.record.program is program
    assert outcome.record.program_fingerprint == program.fingerprint
    assert outcome.record.family_fingerprint == program.family.fingerprint  # type: ignore[union-attr]
    assert outcome.record.backend_id == "toy-backend"
    assert outcome.record.evaluator_id == "toy-evaluator"
    assert outcome.record.before == _before()
    assert outcome.record.after == AdaptationGeometry(runtime)
    assert outcome.record.metadata == {"split": "heldout"}
    assert not hasattr(outcome.result, "predicted_after")


@pytest.mark.parametrize("mismatch", ["model", "episode"])
def test_identity_mismatch_prevents_evaluation(mismatch: str) -> None:
    calls: list[ProgramSpec] = []
    context = _context("other/model") if mismatch == "model" else _context()
    episode = _episode("other-task") if mismatch == "episode" else _episode()

    with pytest.raises(ValidationError, match=f"execution {mismatch}_key"):
        evaluate_execution(
            _execution(),
            evaluator=_measuring_evaluator(calls),
            model_context=context,
            episode=episode,
            before=_before(),
        )
    assert calls == []


@pytest.mark.parametrize("field", ["program_fingerprint", "family_fingerprint"])
def test_execution_program_inconsistency_prevents_evaluation(field: str) -> None:
    program = _program(with_family=True)
    execution = _execution(program=program)
    object.__setattr__(execution.record, field, "tampered")
    calls: list[ProgramSpec] = []

    with pytest.raises(ValidationError, match=field):
        evaluate_execution(
            execution,
            evaluator=_measuring_evaluator(calls),
            model_context=_context(),
            episode=_episode(),
            before=_before(),
        )
    assert calls == []


def test_invalid_evaluator_id_prevents_evaluation() -> None:
    calls: list[ProgramSpec] = []

    class InvalidIdEvaluator:
        evaluator_id = ""

        def evaluate(
            self,
            model: dict[str, float],
            *,
            model_context: ModelContext,
            episode: LearningEpisode,
            program: ProgramSpec,
        ) -> AdaptationGeometry:
            calls.append(program)
            return AdaptationGeometry(model)

    with pytest.raises(ValidationError, match="evaluator_id"):
        evaluate_execution(
            _execution(),
            evaluator=InvalidIdEvaluator(),
            model_context=_context(),
            episode=_episode(),
            before=_before(),
        )
    assert calls == []


def test_invalid_before_geometry_prevents_evaluation() -> None:
    calls: list[ProgramSpec] = []
    with pytest.raises(ValidationError, match="before must be"):
        evaluate_execution(
            _execution(),
            evaluator=_measuring_evaluator(calls),
            model_context=_context(),
            episode=_episode(),
            before={"gain": 0.2},  # type: ignore[arg-type]
        )
    assert calls == []


@pytest.mark.parametrize("metadata", [[], "", 0, False, {"bad": object()}])
def test_invalid_evaluation_metadata_prevents_evaluation(metadata: object) -> None:
    calls: list[ProgramSpec] = []
    with pytest.raises(SerializationError, match="evaluation metadata"):
        evaluate_execution(
            _execution(),
            evaluator=_measuring_evaluator(calls),
            model_context=_context(),
            episode=_episode(),
            before=_before(),
            metadata=metadata,  # type: ignore[arg-type]
        )
    assert calls == []


@pytest.mark.parametrize(
    ("metadata", "expected"), [(None, {}), ({"split": "heldout"}, {"split": "heldout"})]
)
def test_none_and_mapping_evaluation_metadata_are_preserved(
    metadata: dict[str, str] | None, expected: dict[str, str]
) -> None:
    outcome = evaluate_execution(
        _execution(),
        evaluator=_measuring_evaluator([]),
        model_context=_context(),
        episode=_episode(),
        before=_before(),
        metadata=metadata,
    )
    assert outcome.record.metadata == expected


@pytest.mark.parametrize("invalid", [{}, [], None, 1.0, "geometry"])
def test_callable_evaluator_rejects_non_geometry_returns(invalid: object) -> None:
    calls: list[object] = []

    def invalid_handler(model: object, **kwargs: object) -> Any:
        calls.append(model)
        return invalid

    evaluator = CallableEvaluator(invalid_handler)
    with pytest.raises(ValidationError, match="return an AdaptationGeometry"):
        evaluate_execution(
            _execution(),
            evaluator=evaluator,
            model_context=_context(),
            episode=_episode(),
            before=_before(),
        )
    assert len(calls) == 1


def test_external_evaluator_invalid_return_is_rejected_after_one_call() -> None:
    calls: list[object] = []

    class InvalidEvaluator:
        evaluator_id = "invalid-return"

        def evaluate(self, model: object, **kwargs: object) -> Any:
            calls.append(model)
            return {}

    with pytest.raises(ValidationError, match="return an AdaptationGeometry"):
        evaluate_execution(
            _execution(),
            evaluator=InvalidEvaluator(),
            model_context=_context(),
            episode=_episode(),
            before=_before(),
        )
    assert len(calls) == 1


def test_evaluator_exception_propagates_unchanged_without_retry() -> None:
    failure = RuntimeError("evaluation failed")
    calls: list[object] = []

    def fail(model: object, **kwargs: object) -> AdaptationGeometry:
        calls.append(model)
        raise failure

    with pytest.raises(RuntimeError, match="evaluation failed") as captured:
        evaluate_execution(
            _execution(),
            evaluator=CallableEvaluator(fail),
            model_context=_context(),
            episode=_episode(),
            before=_before(),
        )
    assert calls == [_execution().model]
    assert captured.value is failure
