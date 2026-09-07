# Executing a selected adaptation

Version 0.5 adds a narrow execution boundary. `execute_selection()` resolves the
selected program fingerprint against explicit caller-supplied `ProgramSpec` objects,
hands the matching specification to an `AdaptationBackend`, and returns an
`ExecutionOutcome` with compact execution provenance.

## Backend boundary

`adaptcompile` owns declarative programs, selection, identity checks, resolution,
and provenance. A backend owns the runtime model, framework APIs, device placement,
mutation or copying behavior, training mechanics, and interpretation of
`ProgramSpec.method` and `ProgramSpec.parameters`.

Backends implement a dependency-free structural interface:

```python
class MyBackend:
    backend_id = "my-backend"

    def supports(self, program):
        return True

    def execute(self, model, *, model_context, episode, program):
        return adapted_model
```

Backends may mutate the supplied runtime object and return it or return a different
object. `adaptcompile` does not clone the model or assume either behavior. Backend
exceptions propagate unchanged.

## Fingerprints are not programs

A selected fingerprint is an identity, not an executable specification.
`execute_selection()` requires a nonempty sequence of concrete `ProgramSpec`
candidates, requires unique fingerprints, and resolves exactly one match. It never
reconstructs, modifies, realizes, or synthesizes a program.

The selection, `ModelContext`, and `LearningEpisode` identity keys must agree. The
resolved program and selected prediction must also agree on program and family
fingerprints. A selection with no feasible candidate cannot be executed.

## Callable reference backend

`CallableBackend` delegates execution to one user handler and optionally to one
support predicate. Without a predicate it reports support for every `ProgramSpec`.
It is a small protocol demonstration, not a training framework.

```python
from adaptcompile.execution import execute_selection
from adaptcompile.execution.backends import CallableBackend

backend = CallableBackend(handler, backend_id="toy", supports=can_handle)
outcome = execute_selection(
    selection,
    programs=programs,
    model=model,
    model_context=model_context,
    episode=episode,
    backend=backend,
)
```

`ExecutionRecord` serializes the identities, exact declarative program, backend ID,
selected utility, and JSON-safe metadata. It never retains the live model, full
selection, predictions, tensors, weights, checkpoints, or episode training data.

`ExecutionOutcome.model` is an opaque backend-owned runtime object. `adaptcompile`
does not inspect or serialize it.

## Execution is not prediction or evaluation

Executing a program does not establish measured behavioral improvement.
`ExecutionOutcome` is not an `AdaptationResult`, and predicted geometry is never
copied into observed after-geometry. A caller must separately evaluate and measure
the adapted runtime before constructing a new observation. Version 0.6 provides that
explicit boundary through [`evaluate_execution()`](evaluation.md).

The execution layer itself includes no framework backend, model loading, measurement,
retries, rollback, persistence, remote execution, or experiment orchestration.
