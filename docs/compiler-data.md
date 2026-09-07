# Compiler data contract

`adaptcompile.compiler` turns observed `AdaptationResult` objects into a validated
supervised corpus. It joins an `AdaptationDataset` with numeric descriptors computed
by the caller. The package does not compute descriptors or fit a predictor.

## Data stages

`AdaptationDataset` is the flexible observation corpus. It preserves mixed metric
dimensions and repeated runs. `CompilerDataset` is the stricter fitting boundary:
every record has exactly the same feature names, target dimensions, and target mode.
Missing values are errors; they are never filled with zero or NaN.

## Descriptor joins

Descriptor mappings are optional and use semantic identities:

| Namespace | Mapping key |
| --- | --- |
| `model` | `ModelContext.identity_key` |
| `episode` | `LearningEpisode.identity_key` |
| `family` | `ProgramFamily.fingerprint` |
| `program` | `ProgramSpec.fingerprint` |
| `interaction` | `(model identity, episode identity, program fingerprint)` |

Keys are never human-readable labels. When a namespace mapping is supplied, every
applicable observation must resolve in it. A familyless program is valid when family
descriptors are omitted and is rejected when they are supplied.

Each descriptor mapping contains local names, such as `{"layer_count": 32}`. The
builder adds the namespace, producing `model.layer_count`. Values must be finite real
numbers; integers are converted to floats. Consistent with geometry validation,
booleans are rejected even though Python treats them as integers. Strings, lists,
nested mappings, NaN, and infinity are rejected. Categorical encoding is the
caller's responsibility.

External names cannot claim the `model`, `episode`, `family`, `program`,
`interaction`, `baseline`, `after`, `delta`, or `target` namespaces. This prevents
prequalified names and obvious output leakage. The package does not scan arbitrary
keywords inside otherwise valid local names.

## Records and ordering

Each `CompilerRecord` contains model, episode, optional family, and program identity;
a flat immutable feature mapping; an `AdaptationGeometry` target; the target mode;
and immutable run metadata. It does not retain a `ModelContext`, `LearningEpisode`,
`ProgramSpec`, model object, or dataset object. Result metadata remains provenance
and is not promoted into features. IDs and fingerprints are likewise never added as
numeric features.

Features are ordered by namespace:

```text
model, episode, family, program, interaction, baseline
```

Names are lexical within each namespace. Target names are lexical. Caller dictionary
insertion order therefore cannot change the schema.

## Baseline and targets

The builder always adds the real pre-adaptation geometry as `baseline.<metric>`.
Callers do not supply baseline descriptors and cannot overwrite that namespace.
Baseline values remain observation-specific, including across replicates.

`target="after"` uses `AdaptationResult.after_geometry`. `target="delta"` uses
`AdaptationResult.delta_geometry`, or `after - before`. A dataset cannot mix modes.

## Provenance and export

Repeated model/episode/program observations produce repeated compiler records; they
are not deduplicated. Their features may match while targets and run metadata differ.
Optional `descriptor_metadata` on the builder records schema-level provenance such
as a schema name or version on `CompilerDataset.metadata`.

`CompilerDataset.to_records()` returns nested JSON-safe dictionaries separating
identity, features, target, target kind, and metadata. `to_dataframe()` is available
with the `dataframe` extra and uses `feature.<name>` and `target.<metric>` columns.

## Example

```python
from adaptcompile.compiler import build_compiler_dataset

compiler_data = build_compiler_dataset(
    observations,
    model_descriptors=model_descriptors,
    episode_descriptors=episode_descriptors,
    program_descriptors=program_descriptors,
    target="delta",
    descriptor_metadata={"schema_name": "example", "schema_version": "1"},
)
```

`adaptcompile 0.2.0` assembles compiler-ready supervised data. It does not extract
features, train predictors, split datasets, predict adaptation outcomes, or select
programs.
