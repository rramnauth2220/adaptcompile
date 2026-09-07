"""Dependency-free reference selector implementations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..errors import ValidationError
from .prediction import GeometryPrediction
from .selection import (
    CandidateScore,
    ProgramSelection,
    SelectionObjective,
)

if TYPE_CHECKING:
    from .selection import ProgramSelector

__all__ = ["LinearUtilitySelector"]


class LinearUtilitySelector:
    """Transparent weighted-utility selector with hard constraints."""

    def select(
        self,
        predictions: Sequence[GeometryPrediction],
        objective: SelectionObjective,
    ) -> ProgramSelection:
        """Rank candidates and select the highest-utility feasible prediction."""
        if not isinstance(objective, SelectionObjective):
            raise ValidationError("objective must be a SelectionObjective")
        if not isinstance(predictions, Sequence) or isinstance(
            predictions, (str, bytes)
        ):
            raise ValidationError("predictions must be a sequence")
        candidates = tuple(predictions)
        if not candidates:
            raise ValidationError("selection requires at least one prediction")
        if any(not isinstance(item, GeometryPrediction) for item in candidates):
            raise ValidationError(
                "predictions must contain GeometryPrediction instances"
            )

        model_key = candidates[0].model_key
        episode_key = candidates[0].episode_key
        if any(item.model_key != model_key for item in candidates[1:]):
            raise ValidationError("selection predictions must share one model_key")
        if any(item.episode_key != episode_key for item in candidates[1:]):
            raise ValidationError("selection predictions must share one episode_key")
        program_fingerprints = [item.program_fingerprint for item in candidates]
        if len(set(program_fingerprints)) != len(program_fingerprints):
            raise ValidationError("selection program_fingerprints must be unique")

        required_metrics = objective.required_metrics
        for index, prediction in enumerate(candidates):
            missing = required_metrics.difference(prediction.predicted_geometry)
            if missing:
                raise ValidationError(
                    f"candidate {index} predicted_geometry is missing required metrics "
                    f"{sorted(missing)}"
                )

        scores = tuple(self._score(prediction, objective) for prediction in candidates)
        ranked = tuple(
            sorted(scores, key=lambda item: (not item.feasible, -item.utility))
        )
        selected = ranked[0] if ranked[0].feasible else None
        return ProgramSelection(
            model_key=model_key,
            episode_key=episode_key,
            objective=objective,
            selected=selected,
            candidates=ranked,
        )

    @staticmethod
    def _score(
        prediction: GeometryPrediction, objective: SelectionObjective
    ) -> CandidateScore:
        geometry = prediction.predicted_geometry
        terms = [
            weight * geometry[metric] for metric, weight in objective.maximize.items()
        ]
        terms.extend(
            -weight * geometry[metric] for metric, weight in objective.minimize.items()
        )
        try:
            utility = math.fsum(terms)
        except (OverflowError, ValueError) as error:
            raise ValidationError("selection utility must be finite") from error
        if not math.isfinite(utility):
            raise ValidationError("selection utility must be finite")
        violated = tuple(
            constraint
            for constraint in objective.constraints
            if (
                constraint.minimum is not None
                and geometry[constraint.metric] < constraint.minimum
            )
            or (
                constraint.maximum is not None
                and geometry[constraint.metric] > constraint.maximum
            )
        )
        return CandidateScore(
            prediction=prediction,
            utility=utility,
            feasible=not violated,
            violated_constraints=violated,
        )


if TYPE_CHECKING:
    _protocol_check: ProgramSelector = LinearUtilitySelector()
