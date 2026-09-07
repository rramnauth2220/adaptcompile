# Evaluating an executed adaptation

Version 0.6 closes the observation loop explicitly. Execution changes an opaque
runtime object through a user-supplied backend. Evaluation measures that adapted
runtime through a user-supplied `AdaptationEvaluator` and creates a genuine
`AdaptationResult` from explicit before-geometry and measured after-geometry.

## Prediction and observation are different

`GeometryPrediction.predicted_geometry` is forecasted behavior used for program
selection. `AdaptationResult.after` is behavior actually measured after execution.
They may disagree. The evaluation layer never receives a prediction or selection and
never copies predicted geometry into an observation.

## Before geometry is explicit

The caller supplies `before` as an `AdaptationGeometry`. It is the actual measured
pre-adaptation state and is never inferred from compiler features, prediction
baselines, metadata, or execution provenance.

```python
from adaptcompile import AdaptationGeometry
from adaptcompile.evaluation import evaluate_execution
from adaptcompile.evaluation.evaluators import CallableEvaluator

evaluation = evaluate_execution(
    execution,
    evaluator=CallableEvaluator(measure, evaluator_id="heldout-suite"),
    model_context=model_context,
    episode=episode,
    before=AdaptationGeometry({"gain": 0.2, "retention": 0.95}),
    metadata={"split": "heldout"},
)
```

## Evaluator boundary

An evaluator receives only the adapted runtime object, `ModelContext`,
`LearningEpisode`, and exact executed `ProgramSpec`. It returns an
`AdaptationGeometry`. The evaluator owns the measurement procedure;
`adaptcompile` treats the runtime object as opaque and owns contract validation and
provenance construction.

`CallableEvaluator` is the dependency-free reference implementation. Custom classes
can satisfy the structural `AdaptationEvaluator` protocol without subclassing.
Evaluator exceptions propagate unchanged and are not retried. Non-geometry return
values are rejected rather than coerced.

## Execution and evaluation provenance

`EvaluationRecord` retains model and episode identity, the exact `ProgramSpec` and
its program/family fingerprints, the execution `backend_id`, the measurement
`evaluator_id`, before/after geometry, and immutable JSON-safe metadata. It retains
no runtime model, prediction, selection, tensors, weights, or callable.

`EvaluationOutcome.result` is the measured `AdaptationResult`.
`EvaluationOutcome.record` is the serializable provenance record. The outcome does
not retain another reference to the live runtime object.

## Dataset reinsertion is explicit

Users may place `evaluation.result` into a new or existing `AdaptationDataset`.
`evaluate_execution()` does not mutate a dataset, rebuild compiler data, refit a
predictor, or automate a continual-learning lifecycle.
