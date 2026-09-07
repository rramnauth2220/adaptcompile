from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

import adaptcompile
import adaptcompile.compiler as compiler
import adaptcompile.compiler.selectors as selector_module
from adaptcompile import AdaptationGeometry, ValidationError
from adaptcompile.compiler import (
    CandidateScore,
    GeometryConstraint,
    GeometryPrediction,
    ProgramSelection,
    ProgramSelector,
    SelectionObjective,
)
from adaptcompile.compiler.selectors import LinearUtilitySelector


def _prediction(
    program: str,
    geometry: dict[str, float],
    *,
    model: str = "model",
    episode: str = "episode",
    target_kind: str = "after",
    target: dict[str, float] | None = None,
    family: str | None = None,
) -> GeometryPrediction:
    predicted_geometry = AdaptationGeometry(geometry)
    return GeometryPrediction(
        model_key=(model, None, None),
        episode_key=("episode_id", episode),
        family_fingerprint=family,
        program_fingerprint=program,
        predicted_target=(
            predicted_geometry
            if target_kind == "after"
            else AdaptationGeometry(target or {name: 0.0 for name in geometry})
        ),
        predicted_geometry=predicted_geometry,
        target_kind=target_kind,  # type: ignore[arg-type]
    )


def test_selection_public_api_is_explicit_and_not_root_reexported() -> None:
    assert set(compiler.__all__) == {
        "CandidateScore",
        "CompilerDataset",
        "CompilerRecord",
        "GeometryConstraint",
        "GeometryPrediction",
        "GeometryPredictor",
        "ProgramSelection",
        "ProgramSelector",
        "SelectionObjective",
        "build_compiler_dataset",
    }
    assert selector_module.__all__ == ["LinearUtilitySelector"]
    for name in (
        "CandidateScore",
        "GeometryConstraint",
        "ProgramSelection",
        "ProgramSelector",
        "SelectionObjective",
        "LinearUtilitySelector",
    ):
        assert not hasattr(adaptcompile, name)
    assert ProgramSelector.__name__ == "ProgramSelector"


def test_constraint_supports_minimum_maximum_and_inclusive_range() -> None:
    minimum = GeometryConstraint("foo", minimum=-2)
    maximum = GeometryConstraint("bar", maximum=4)
    bounded = GeometryConstraint("baz", minimum=-1, maximum=3)
    objective = SelectionObjective(
        maximize={"score": 1}, constraints=(minimum, maximum, bounded)
    )
    prediction = _prediction("program", {"score": 2, "foo": -2, "bar": 4, "baz": 3})

    selection = LinearUtilitySelector().select([prediction], objective)

    assert selection.selected is not None
    assert selection.selected.feasible
    assert minimum.minimum == -2.0
    assert maximum.maximum == 4.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "at least one bound"),
        ({"minimum": 2, "maximum": 1}, "must not exceed"),
        ({"minimum": True}, "real number"),
        ({"minimum": float("nan")}, "finite"),
        ({"maximum": float("inf")}, "finite"),
    ],
)
def test_constraint_rejects_invalid_bounds(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        GeometryConstraint("metric", **arguments)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="non-empty"):
        GeometryConstraint(" ", minimum=0)


def test_objective_supports_maximize_minimize_and_constraints() -> None:
    constraint = GeometryConstraint("limit", maximum=10)
    objective = SelectionObjective(
        maximize={"gain": 2, "retention": 0.5},
        minimize={"cost": 0.25},
        constraints=[constraint],  # type: ignore[arg-type]
    )

    assert objective.maximize == {"gain": 2.0, "retention": 0.5}
    assert objective.minimize == {"cost": 0.25}
    assert objective.constraints == (constraint,)
    assert objective.required_metrics == {"gain", "retention", "cost", "limit"}
    with pytest.raises(TypeError):
        objective.maximize["gain"] = 4  # type: ignore[index]


@pytest.mark.parametrize("weight", [0, -1, True, float("nan"), float("inf")])
def test_objective_rejects_invalid_weights(weight: object) -> None:
    with pytest.raises(ValidationError, match="greater than zero|real number|finite"):
        SelectionObjective(maximize={"metric": weight})  # type: ignore[dict-item]


def test_objective_rejects_empty_overlap_and_duplicate_constraints() -> None:
    with pytest.raises(ValidationError, match="at least one metric"):
        SelectionObjective()
    with pytest.raises(ValidationError, match="both maximized and minimized"):
        SelectionObjective(maximize={"same": 1}, minimize={"same": 2})
    with pytest.raises(ValidationError, match="must be unique"):
        SelectionObjective(
            maximize={"score": 1},
            constraints=(
                GeometryConstraint("limit", minimum=0),
                GeometryConstraint("limit", maximum=1),
            ),
        )


def test_utility_formula_is_exact_and_does_not_clip_values() -> None:
    prediction = _prediction(
        "program", {"gain": 0.8, "retention": 0.9, "cost": 2.0, "negative": -4}
    )
    objective = SelectionObjective(
        maximize={"gain": 1, "retention": 0.5, "negative": 2},
        minimize={"cost": 0.25},
    )

    score = LinearUtilitySelector().select([prediction], objective).candidates[0]

    assert score.utility == pytest.approx(-7.25)
    assert score.prediction.predicted_geometry["negative"] == -4
    assert score.feasible


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        (SelectionObjective(maximize={"foo": 2}), 6.0),
        (SelectionObjective(minimize={"bar": 0.5}), -2.0),
        (
            SelectionObjective(maximize={"foo": 2}, minimize={"bar": 0.5}),
            4.0,
        ),
    ],
)
def test_maximize_minimize_and_mixed_utility(
    objective: SelectionObjective, expected: float
) -> None:
    prediction = _prediction("program", {"foo": 3, "bar": 4})
    result = LinearUtilitySelector().select([prediction], objective)
    assert result.candidates[0].utility == expected


def test_constraints_are_hard_and_retain_exact_violations() -> None:
    retention = GeometryConstraint("retention", minimum=0.8)
    cost = GeometryConstraint("cost", maximum=2)
    objective = SelectionObjective(maximize={"gain": 1}, constraints=(retention, cost))
    infeasible = _prediction("high", {"gain": 100, "retention": 0.7, "cost": 3})
    feasible = _prediction("safe", {"gain": 1, "retention": 0.8, "cost": 2})

    selection = LinearUtilitySelector().select([infeasible, feasible], objective)

    assert selection.selected_prediction is feasible
    assert [item.prediction.program_fingerprint for item in selection.candidates] == [
        "safe",
        "high",
    ]
    high_score = selection.candidates[1]
    assert high_score.utility == 100
    assert high_score.feasible is False
    assert high_score.violated_constraints[0] is retention
    assert high_score.violated_constraints[1] is cost


def test_all_infeasible_returns_no_selection_but_keeps_ranked_scores() -> None:
    objective = SelectionObjective(
        maximize={"score": 1},
        constraints=(GeometryConstraint("limit", minimum=5),),
    )
    predictions = [
        _prediction("lower", {"score": 1, "limit": 0}),
        _prediction("higher", {"score": 2, "limit": 1}),
    ]

    selection = LinearUtilitySelector().select(predictions, objective)

    assert selection.selected is None
    assert selection.selected_prediction is None
    assert selection.has_feasible_candidate is False
    assert [item.prediction.program_fingerprint for item in selection.candidates] == [
        "higher",
        "lower",
    ]


def test_exact_utility_ties_preserve_input_order() -> None:
    objective = SelectionObjective(maximize={"score": 1})
    predictions = [
        _prediction("third", {"score": 2}),
        _prediction("first", {"score": 2}),
        _prediction("second", {"score": 2}),
    ]

    selection = LinearUtilitySelector().select(predictions, objective)

    assert [item.prediction.program_fingerprint for item in selection.candidates] == [
        "third",
        "first",
        "second",
    ]
    assert selection.selected_prediction is predictions[0]


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        ([], "at least one"),
        (
            [_prediction("a", {"x": 1}), _prediction("b", {"x": 2}, model="b")],
            "model_key",
        ),
        (
            [
                _prediction("a", {"x": 1}),
                _prediction("b", {"x": 2}, episode="b"),
            ],
            "episode_key",
        ),
        (
            [_prediction("same", {"x": 1}), _prediction("same", {"x": 2})],
            "unique",
        ),
    ],
)
def test_candidate_set_validation(
    predictions: list[GeometryPrediction], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        LinearUtilitySelector().select(
            predictions, SelectionObjective(maximize={"x": 1})
        )


def test_missing_required_metrics_rejected_but_extra_metrics_ignored() -> None:
    objective = SelectionObjective(
        maximize={"score": 1},
        constraints=(GeometryConstraint("limit", maximum=2),),
    )
    selector = LinearUtilitySelector()
    with pytest.raises(ValidationError, match="score"):
        selector.select([_prediction("a", {"limit": 1})], objective)
    with pytest.raises(ValidationError, match="limit"):
        selector.select([_prediction("a", {"score": 1})], objective)

    result = selector.select(
        [_prediction("a", {"score": 1, "limit": 1, "unused": 999})], objective
    )
    assert result.selected is not None
    assert result.selected.utility == 1


def test_target_kind_and_predicted_target_never_affect_selection() -> None:
    objective = SelectionObjective(
        maximize={"gain": 1},
        constraints=(GeometryConstraint("retention", minimum=0.8),),
    )
    after = [
        _prediction("a", {"gain": 1, "retention": 0.9}),
        _prediction("b", {"gain": 2, "retention": 0.7}),
    ]
    delta = [
        _prediction(
            "a",
            {"gain": 1, "retention": 0.9},
            target_kind="delta",
            target={"gain": -100, "retention": 50},
        ),
        _prediction(
            "b",
            {"gain": 2, "retention": 0.7},
            target_kind="delta",
            target={"gain": 999, "retention": -200},
        ),
    ]

    after_selection = LinearUtilitySelector().select(after, objective)
    delta_selection = LinearUtilitySelector().select(delta, objective)

    assert [item.utility for item in after_selection.candidates] == [
        item.utility for item in delta_selection.candidates
    ]
    assert [item.feasible for item in after_selection.candidates] == [
        item.feasible for item in delta_selection.candidates
    ]
    assert [
        item.prediction.program_fingerprint for item in after_selection.candidates
    ] == [item.prediction.program_fingerprint for item in delta_selection.candidates]
    assert after_selection.selected_prediction is not None
    assert delta_selection.selected_prediction is not None
    assert (
        after_selection.selected_prediction.program_fingerprint
        == delta_selection.selected_prediction.program_fingerprint
        == "a"
    )


def test_new_value_objects_are_immutable_and_roundtrip() -> None:
    constraint = GeometryConstraint("limit", minimum=-1, maximum=4)
    objective = SelectionObjective(
        maximize={"score": 2}, minimize={"cost": 0.5}, constraints=(constraint,)
    )
    prediction = _prediction("program", {"score": 3, "cost": 2, "limit": 0})
    selection = LinearUtilitySelector().select([prediction], objective)
    restored = ProgramSelection.from_dict(selection.to_dict())

    assert restored == selection
    assert CandidateScore.from_dict(selection.selected.to_dict()) == selection.selected  # type: ignore[union-attr]
    assert SelectionObjective.from_dict(objective.to_dict()) == objective
    assert GeometryConstraint.from_dict(constraint.to_dict()) == constraint
    json.dumps(selection.to_dict())
    with pytest.raises(FrozenInstanceError):
        constraint.minimum = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        selection.selected = None  # type: ignore[misc]


def test_nonfinite_utility_is_rejected() -> None:
    objective = SelectionObjective(maximize={"score": 1e308})
    with pytest.raises(ValidationError, match="utility must be finite"):
        LinearUtilitySelector().select(
            [_prediction("program", {"score": 1e308})], objective
        )
