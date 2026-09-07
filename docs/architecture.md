# Architecture

`adaptcompile` separates experiment identity, executable declarations, predictions,
runtime execution, and measured observations. Runtime mechanics and measurement
procedures remain delegated to pluggable components.

```text
LearningEpisode + ModelContext + ProgramSpec
                    |
                    v
            AdaptationResult
             /             \
            v               v
   AdaptationStudy   AdaptationDataset
                             |
             external numeric descriptors
                             |
                             v
                    CompilerDataset
                             |
                    GeometryPredictor
                             |
                             v
                  GeometryPrediction(s)
                             |
                   SelectionObjective
                             |
                      ProgramSelector
                             |
                             v
                     ProgramSelection
                             |
                  resolve ProgramSpec
                             |
                   AdaptationBackend
                             |
                             v
                    ExecutionOutcome
                             |
                  AdaptationEvaluator
                             |
                             v
                    EvaluationOutcome
                             |
                             v
       AdaptationResult (new measured observation)
                             |
                             v
                    AdaptationDataset
```

- `ModelContext` identifies the pre-adaptation model, revision, and base state.
- `LearningEpisode` identifies a learning problem and references its training and
  evaluation datasets without serializing those datasets.
- `ProgramFamily` describes a conceptual adaptation family independently of a
  model-specific realization.
- `ProgramSpec` describes a concrete, backend-independent executable declaration and
  may reference a `ProgramFamily`.
- `AdaptationGeometry` is an immutable mapping of behavioral metric names to values.
- `AdaptationResult` records before and after geometry for one observed combination
  of model context, episode, and program.
- `AdaptationStudy` compares programs while holding model context and episode fixed.
- `AdaptationDataset` stores a corpus whose models, episodes, programs, metric
  dimensions, and replicate provenance may vary.
- `CompilerDataset` stores immutable supervised records assembled from an
  `AdaptationDataset`, external decision-time descriptors, and baseline geometry.
  It requires one consistent, deterministic feature and target schema.
- `GeometryPredictor` is a dependency-free structural interface for fitting compiler
  data and predicting from target-free feature mappings.
- `GeometryPrediction` retains candidate identity, raw target prediction, and
  reconstructed post-adaptation geometry.
- `SelectionObjective` states explicit weighted maximize/minimize terms and hard
  constraints over predicted post-adaptation geometry.
- `ProgramSelector` evaluates a finite candidate set for one model and episode.
- `ProgramSelection` stores all ranked candidate scores and the best feasible choice,
  if one exists.
- Execution resolves the selected fingerprint against an explicit concrete
  `ProgramSpec`; fingerprints alone are not executable specifications.
- `AdaptationBackend` is a framework-neutral structural interface that receives the
  runtime object, model context, episode, and exact program.
- `ExecutionOutcome` pairs the opaque backend-owned runtime result with a compact,
  serializable `ExecutionRecord`.
- `AdaptationEvaluator` measures the opaque adapted runtime and returns actual
  post-adaptation geometry.
- `EvaluationOutcome` pairs a genuine measured `AdaptationResult` with a compact,
  serializable `EvaluationRecord` carrying backend and evaluator provenance.

Names and metadata remain descriptive annotations. Stable identity is explicit:
model studies use `(model_id, revision, base_state_id)`, episodes may supply an
`episode_id`, and program/family fingerprints derive from their semantic parameters.

Descriptor extraction is external: the package validates and joins numeric mappings
but does not inspect models or datasets. Execution and evaluation are separate
pluggable boundaries.

`ExecutionOutcome` does not automatically become an `AdaptationResult`.
`evaluate_execution()` requires explicit measured before-geometry and obtains
after-geometry only from an evaluator. It never copies a `GeometryPrediction` into an
observation. Dataset insertion remains an explicit caller action.
