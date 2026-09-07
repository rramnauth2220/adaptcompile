# Predicting adaptation geometry

Version 0.3 adds a narrow prediction layer over the compiler data contract. A fitted
`GeometryPredictor` consumes externally supplied numeric compiler features plus a
candidate's semantic identity and returns a `GeometryPrediction`.

`adaptcompile` consumes compiler descriptors; it does not extract model, episode,
program, or interaction features itself. Identity keys and metadata remain separate
from the numeric prediction matrix.

## Prediction interface

`GeometryPredictor` is a dependency-free `typing.Protocol`. External predictors can
implement it structurally without inheriting from an adaptcompile base class. A
fitted predictor exposes its exact `feature_names`, `target_names`, `target_kind`,
and fitted state.

The primary `predict()` operation is target-free. It accepts a feature mapping and
candidate model, episode, optional family, and program identities. Missing or extra
features are rejected, and values use the same finite numeric policy as compiler
data. Input mapping order cannot alter the fitted feature order.

`predict_dataset()` is an evaluation convenience. It preserves record order and uses
only each record's features, identities, and metadata during inference; observed
targets are never prediction inputs.

## After and delta targets

For an `after` model, the raw predicted target already is the post-adaptation
geometry. `predicted_target` and `predicted_geometry` are therefore equal.

For a `delta` model, the raw target is the predicted change. Post-adaptation geometry
is reconstructed independently for every metric:

```text
predicted_geometry.metric = baseline.metric + predicted_target.metric
```

The baseline must be present in the prediction features. Missing baseline metrics
are errors, and a zero baseline is never assumed. Values are not clipped or otherwise
bounded because geometry dimensions are application-defined.

## Ridge reference implementation

Install the optional implementation with:

```bash
pip install "adaptcompile[predict]"
```

```python
from adaptcompile.compiler.predictors import RidgeGeometryPredictor

predictor = RidgeGeometryPredictor(alpha=1.0, fit_intercept=True)
predictor.fit(compiler_data)
prediction = predictor.predict(
    features=candidate_features,
    model_key=model_key,
    episode_key=episode_key,
    family_fingerprint=family_fingerprint,
    program_fingerprint=program_fingerprint,
)
```

`RidgeGeometryPredictor` fits one multi-output scikit-learn ridge model. It is a
lightweight demonstration of the public interface, not a prescribed scientific model
architecture. It performs no scaling, feature selection, splitting, tuning, or
uncertainty estimation.

Program selection and adaptation execution remain outside version 0.3.
