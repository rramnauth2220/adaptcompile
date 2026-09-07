# adaptcompile

`adaptcompile` is a model-agnostic Python library for representing, measuring,
predicting, selecting, executing, and evaluating model adaptations.

`adaptcompile` treats model adaptation as a behavioral transformation that can be
represented, predicted, selected, executed, and measured independently of any
training or evaluation framework.

See the [architecture overview](https://github.com/rramnauth2220/adaptcompile/blob/main/docs/architecture.md) for how the objects fit
together.

## Installation

```bash
pip install adaptcompile
```

DataFrame exports are optional:

```bash
pip install "adaptcompile[dataframe]"
```

The reference prediction implementation is also optional:

```bash
pip install "adaptcompile[predict]"
```

For contributor setup:

```bash
python -m pip install -e ".[dev]"
```

## Minimal example

```python
from adaptcompile import (
    AdaptationResult,
    AdaptationStudy,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)

model = ModelContext("example/model", revision="v1", base_state_id="base-1")
episode = LearningEpisode(
    "toy-task",
    train=["example"],
    evaluations={"accuracy": ["held-out example"]},
    episode_id="toy-task-v1",
)

results = [
    AdaptationResult(
        model_context=model,
        episode=episode,
        program=ProgramSpec(name, "adapter", {"strength": strength}),
        before={"accuracy": 0.40},
        after={"accuracy": score},
    )
    for name, strength, score in [
        ("gentle", 0.25, 0.68),
        ("strong", 0.75, 0.81),
    ]
]

study = AdaptationStudy(results)
print(study.best("accuracy").program.name)
print(study.best("accuracy").delta.to_dict())
```

More runnable examples are available in [examples](https://github.com/rramnauth2220/adaptcompile/tree/main/examples).

## Core objects

- `ModelContext` identifies a model revision and pre-adaptation base state.
- `LearningEpisode` names a learning problem and references its datasets.
- `ProgramFamily` describes a conceptual adaptation family.
- `ProgramSpec` declares one concrete, backend-independent adaptation program.
- `AdaptationGeometry` stores arbitrary-dimensional behavioral measurements.
- `AdaptationResult` records before/after geometry for one observed adaptation.
- `AdaptationStudy` compares programs for one fixed model context and episode.
- `AdaptationDataset` stores observations across models, episodes, and programs.

Public records support JSON-safe serialization. Dataset-like objects attached to a
`LearningEpisode` are intentionally held by reference and are not serialized.
DataFrame export is available through the optional `dataframe` extra.

## Compiler-ready data

`adaptcompile.compiler` combines an `AdaptationDataset` with externally supplied
numeric decision-time descriptors to produce a validated supervised dataset.
Feature extraction remains external. The prediction layer can fit a predictor
over the assembled numeric features.

```python
from adaptcompile.compiler import build_compiler_dataset

compiler_data = build_compiler_dataset(
    observations,
    episode_descriptors=episode_features,
    program_descriptors=program_features,
    target="delta",
)
print(compiler_data.feature_names)
```

See the [compiler data contract](https://github.com/rramnauth2220/adaptcompile/blob/main/docs/compiler-data.md)
for descriptor identities, namespaces, baseline features, and validation rules.

## Predicting geometry

The dependency-free `GeometryPredictor` protocol defines target-free prediction from
compiler features and candidate identity. `RidgeGeometryPredictor` is one optional,
deliberately simple reference implementation:

```python
from adaptcompile.compiler.predictors import RidgeGeometryPredictor

predictor = RidgeGeometryPredictor().fit(compiler_data)
prediction = predictor.predict(
    features=compiler_data[0].features,
    model_key=compiler_data[0].model_key,
    episode_key=compiler_data[0].episode_key,
    family_fingerprint=compiler_data[0].family_fingerprint,
    program_fingerprint=compiler_data[0].program_fingerprint,
)
print(prediction.predicted_geometry)
```

See [prediction](https://github.com/rramnauth2220/adaptcompile/blob/main/docs/prediction.md)
for target semantics and delta reconstruction.

## Selecting a program

Selection uses predicted post-adaptation geometry, an explicit weighted objective,
and optional hard constraints:

```python
from adaptcompile.compiler import GeometryConstraint, SelectionObjective
from adaptcompile.compiler.selectors import LinearUtilitySelector

objective = SelectionObjective(
    maximize={"gain": 1.0},
    minimize={"cost": 0.2},
    constraints=(GeometryConstraint("retention", minimum=0.8),),
)
selection = LinearUtilitySelector().select(predictions, objective)
```

See [selection](https://github.com/rramnauth2220/adaptcompile/blob/main/docs/selection.md)
for ranking, feasibility, and tie semantics.

## Executing a selection

Execution resolves the selected fingerprint against explicit concrete programs and
delegates the matching `ProgramSpec` to a user-supplied backend:

```python
from adaptcompile.execution import execute_selection
from adaptcompile.execution.backends import CallableBackend

outcome = execute_selection(
    selection,
    programs=programs,
    model=model,
    model_context=model_context,
    episode=episode,
    backend=CallableBackend(handler, backend_id="custom"),
)
```

See [execution](https://github.com/rramnauth2220/adaptcompile/blob/main/docs/execution.md)
for program resolution, provenance, mutation, and framework-boundary semantics.

## Evaluating an execution

Evaluation measures the adapted runtime through a user-supplied evaluator and creates
an observed `AdaptationResult`. The caller supplies measured before-geometry
explicitly; after-geometry comes only from the evaluator.

```python
from adaptcompile.evaluation import evaluate_execution
from adaptcompile.evaluation.evaluators import CallableEvaluator

evaluation = evaluate_execution(
    outcome,
    evaluator=CallableEvaluator(measure, evaluator_id="custom"),
    model_context=model_context,
    episode=episode,
    before=before_geometry,
)
```

See [evaluation](https://github.com/rramnauth2220/adaptcompile/blob/main/docs/evaluation.md)
for prediction-versus-observation semantics and evaluation provenance.

## What adaptcompile does not do

- It does not implement model training or model loading.
- It does not implement LoRA or replace PEFT, TRL, or another training framework.
- It does not infer objectives, synthesize programs, infer metrics, or interpret
  program parameters in the core package.
- It has no mandatory ML-framework or numerical-stack dependencies; pandas and the
  scikit-learn reference predictor are optional extras.

## Status

`0.6.0` adds pluggable post-execution evaluation and explicit conversion of measured
behavior into observations. The API is usable but may evolve during the `0.x` series.
Execution and measurement remain delegated to user-supplied plugins.
