from __future__ import annotations

import pytest

from adaptcompile import AdaptationGeometry, ValidationError


def test_arbitrary_dimensions_indexing_and_conversion() -> None:
    geometry = AdaptationGeometry({"novel_dimension": 0.5, "risk": 0.2})

    assert geometry["novel_dimension"] == 0.5
    assert geometry.metric_names == ("novel_dimension", "risk")
    assert geometry.to_dict() == {"novel_dimension": 0.5, "risk": 0.2}
    assert list(geometry) == ["novel_dimension", "risk"]


def test_empty_geometry_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one metric"):
        AdaptationGeometry({})


def test_compare_and_delta_are_self_minus_baseline() -> None:
    after = AdaptationGeometry({"gain": 0.8, "retention": 0.9})
    before = AdaptationGeometry({"gain": 0.2, "retention": 0.95})

    assert after.compare(before).to_dict() == {
        "gain": pytest.approx(0.6),
        "retention": pytest.approx(-0.05),
    }
    assert after.delta(before) == after.compare(before)


def test_mismatched_metrics_raise_explicit_error() -> None:
    with pytest.raises(ValidationError, match="metrics do not match"):
        AdaptationGeometry({"a": 1}).compare(AdaptationGeometry({"b": 1}))


def test_dataframe_has_one_row() -> None:
    pd = pytest.importorskip("pandas")
    frame = AdaptationGeometry({"a": 1.0, "b": 2.0}).to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert frame.to_dict(orient="records") == [{"a": 1.0, "b": 2.0}]


def test_serialization_roundtrip() -> None:
    geometry = AdaptationGeometry({"gain": 0.75, "retention": 0.98})

    restored = AdaptationGeometry.from_json(geometry.to_json())

    assert restored == geometry
    assert restored.to_dict() == geometry.to_dict()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_scores_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        AdaptationGeometry({"bad": value})
