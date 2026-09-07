# Selecting a predicted program

Version 0.4 adds explicit selection over a finite set of predicted candidate
adaptations. Selection consumes `GeometryPrediction` objects and returns a ranked
`ProgramSelection`; it does not execute or synthesize programs.

## Why selection uses predicted geometry

Selectors evaluate `prediction.predicted_geometry`, the predicted post-adaptation
state. An `after` predictor produces that geometry directly, while a `delta`
predictor reconstructs it from the supplied baseline. The selection layer therefore
does not inspect `target_kind`, repeat delta reconstruction, or use
`predicted_target`.

## Objectives and utility

`SelectionObjective` requires the caller to state which metrics to maximize and/or
minimize and to assign every term a positive weight:

```text
utility = sum(maximize_weight[m] * geometry[m])
        - sum(minimize_weight[m] * geometry[m])
```

Weights are not normalized. Geometry dimensions are not normalized, standardized,
rounded, or clipped. `adaptcompile` does not automatically normalize geometry
dimensions before computing utility. Objective weights therefore operate in the
units supplied by the caller.

## Hard constraints

`GeometryConstraint` defines inclusive minimum and/or maximum bounds for one
post-adaptation metric. Constraints determine feasibility; they are never converted
to penalties. A feasible candidate always ranks before an infeasible candidate even
when the infeasible candidate has greater utility.

If every candidate violates a constraint, `selection.selected` is `None`. The full
ranked candidate list and each candidate's exact violated constraint objects remain
available for inspection.

```python
from adaptcompile.compiler import GeometryConstraint, SelectionObjective
from adaptcompile.compiler.selectors import LinearUtilitySelector

objective = SelectionObjective(
    maximize={"gain": 1.0, "retention": 0.5},
    minimize={"cost": 0.2},
    constraints=(GeometryConstraint("retention", minimum=0.8),),
)
selection = LinearUtilitySelector().select(predictions, objective)
```

Candidates must share one model and episode identity and have distinct program
fingerprints. Every required objective and constraint metric must be present. Exact
utility ties preserve input order.

`LinearUtilitySelector` is a transparent reference implementation of the
`ProgramSelector` interface, not a prescribed decision rule. Objective inference,
Pareto optimization, learned or uncertainty-aware selection, feature extraction,
program synthesis, and adaptation execution are outside version 0.4.
