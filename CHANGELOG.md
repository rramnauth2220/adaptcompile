# Changelog

All notable public changes to this project are documented here.

## 0.6.0 - 2026-09-07

### Added

- Dependency-free generic `AdaptationEvaluator` structural protocol.
- Immutable, serializable `EvaluationRecord` provenance.
- Runtime-free `EvaluationOutcome` pairing a measured result with its record.
- `evaluate_execution()` with explicit before-geometry and pre-evaluation contract
  validation.
- `CallableEvaluator` reference implementation.
- Explicit conversion from measured post-execution geometry to `AdaptationResult`.

## 0.5.0 - 2026-09-07

### Added

- Dependency-free generic `AdaptationBackend` structural protocol.
- Immutable, serializable `ExecutionRecord` provenance.
- In-memory generic `ExecutionOutcome` runtime wrapper.
- `execute_selection()` with explicit selected-`ProgramSpec` resolution and identity
  validation.
- `CallableBackend` reference implementation.

## 0.4.0 - 2026-09-07

### Added

- Immutable `GeometryConstraint`, `SelectionObjective`, `CandidateScore`, and
  `ProgramSelection` values.
- Dependency-free `ProgramSelector` structural protocol.
- `LinearUtilitySelector` reference implementation.
- Explicit weighted maximize/minimize objectives and hard feasibility constraints.
- Deterministic candidate ranking with stable input-order tie behavior.

## 0.3.0 - 2026-09-07

### Added

- Immutable `GeometryPrediction` values.
- Dependency-free `GeometryPredictor` structural protocol.
- Target-free prediction from validated compiler features and candidate identity.
- Post-adaptation geometry reconstruction for delta targets using baseline features.
- Optional `RidgeGeometryPredictor` multi-output reference implementation.
- `predict` dependency extra for scikit-learn-backed prediction.

## 0.2.0 - 2026-09-06

### Added

- Compiler-facing supervised data contract under `adaptcompile.compiler`.
- Namespaced assembly of externally computed decision-time descriptors.
- Immutable compiler records with preserved identity and run provenance.
- Explicit after-geometry and delta-geometry targets.
- Deterministic, validated compiler feature and target schemas.
- Optional compiler DataFrame export and descriptor-schema metadata.

## 0.1.0 - 2026-09-05

### Added

- Declarative model and base-state contexts.
- Learning episodes with non-serialized dataset references.
- Conceptual program families and concrete program specifications.
- Arbitrary-dimensional adaptation geometry and before/after adaptation results.
- Controlled adaptation studies with ranking and Pareto analysis.
- Multi-model, multi-episode adaptation datasets with replicate-preserving grouping.
- JSON-safe serialization and stable conceptual/executable fingerprints.
- Optional pandas DataFrame exports.
