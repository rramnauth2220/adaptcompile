# adaptcompile

`adaptcompile` is a model-agnostic Python library for representing, measuring,
comparing, and predicting model adaptation.

`adaptcompile` treats model adaptation as a behavioral transformation that can be
represented, measured, compared, and predicted independently of the training
framework used to produce it.

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
Feature extraction remains external. Version 0.3 can fit a predictor over the
assembled numeric features.

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

## What adaptcompile does not do

- It does not train or load adapted models.
- It does not implement LoRA or replace PEFT, TRL, or another training framework.
- It does not select adaptation programs or execute adaptations.
- It has no mandatory ML-framework or numerical-stack dependencies; pandas and the
  scikit-learn reference predictor are optional extras.

## Status

`0.3.0` adds target-free prediction of adaptation geometry from compiler-ready
features. The API is usable but may evolve during the `0.x` series. The package does
not extract features, execute adaptations, or select programs.
