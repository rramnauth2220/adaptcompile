# adaptcompile

`adaptcompile` is a model-agnostic Python library for representing and comparing
observed model-adaptation experiments.

`adaptcompile` treats model adaptation as a behavioral transformation that can be
represented, measured, and compared independently of the training framework used to
produce it. It provides the representation and analysis foundation on which future
adaptation-compilation functionality can be built.

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
Feature extraction and prediction remain external in 0.2.

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

## What adaptcompile does not do

- It does not train or load models.
- It does not implement LoRA or replace PEFT, TRL, or another training framework.
- It does not currently predict outcomes or select adaptation programs.
- It has no mandatory ML-framework or numerical-stack dependencies; pandas is optional for DataFrame export.

## Status

`0.2.0` adds compiler-ready supervised data assembly. The API is usable but may
evolve during the `0.x` series. The package does not extract features, fit models,
predict outcomes, or select programs.
