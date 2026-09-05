from __future__ import annotations

import pytest

from adaptcompile import AdaptationResult, LearningEpisode, ModelContext, ProgramSpec


@pytest.fixture
def model_context() -> ModelContext:
    return ModelContext(
        model_id="example/model",
        revision="revision-1",
        base_state_id="base-state-1",
    )


@pytest.fixture
def episode() -> LearningEpisode:
    return LearningEpisode(
        name="toy",
        train=["train"],
        evaluations={"gain": ["g"], "retention": ["r"], "cost": ["c"]},
        metadata={"learning_family": "toy"},
    )


def make_result(
    episode: LearningEpisode,
    name: str,
    *,
    gain: float,
    retention: float,
    cost: float,
    model_context: ModelContext | None = None,
) -> AdaptationResult:
    return AdaptationResult(
        model_context=model_context
        or ModelContext("example/model", "revision-1", "base-state-1"),
        episode=episode,
        program=ProgramSpec(name=name, method="adapter", parameters={"rank": 4}),
        before={"gain": 0.1, "retention": 0.95, "cost": 0.0},
        after={"gain": gain, "retention": retention, "cost": cost},
    )
