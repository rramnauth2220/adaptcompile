"""Minimal workflow exercised against an installed distribution."""

from __future__ import annotations

import argparse
import importlib.util
import math
from importlib.metadata import version

import adaptcompile
from adaptcompile import (
    AdaptationDataset,
    AdaptationGeometry,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)
from adaptcompile.compiler import (
    CandidateScore,
    CompilerDataset,
    GeometryConstraint,
    GeometryPrediction,
    GeometryPredictor,
    ProgramSelection,
    ProgramSelector,
    SelectionObjective,
    build_compiler_dataset,
)
from adaptcompile.compiler.predictors import RidgeGeometryPredictor
from adaptcompile.compiler.selectors import LinearUtilitySelector
from adaptcompile.evaluation import (
    AdaptationEvaluator,
    EvaluationOutcome,
    EvaluationRecord,
    evaluate_execution,
)
from adaptcompile.evaluation.evaluators import CallableEvaluator
from adaptcompile.execution import (
    AdaptationBackend,
    ExecutionOutcome,
    ExecutionRecord,
    execute_selection,
)
from adaptcompile.execution.backends import CallableBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataframe", action="store_true")
    parser.add_argument("--assert-no-pandas", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--execution", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--assert-no-sklearn", action="store_true")
    parser.add_argument("--assert-no-frameworks", action="store_true")
    arguments = parser.parse_args()

    if version("adaptcompile") != "0.6.0" or adaptcompile.__version__ != "0.6.0":
        raise SystemExit("unexpected installed adaptcompile version")
    if hasattr(adaptcompile, "CompilerDataset"):
        raise SystemExit("compiler API must not be re-exported from the package root")
    if arguments.assert_no_pandas and importlib.util.find_spec("pandas") is not None:
        raise SystemExit("core smoke environment unexpectedly contains pandas")
    if arguments.assert_no_sklearn and importlib.util.find_spec("sklearn") is not None:
        raise SystemExit("core smoke environment unexpectedly contains scikit-learn")
    if arguments.assert_no_frameworks:
        for module_name in ("numpy", "torch", "transformers", "peft"):
            if importlib.util.find_spec(module_name) is not None:
                raise SystemExit(
                    f"core smoke environment unexpectedly contains {module_name}"
                )
    if GeometryPredictor.__name__ != "GeometryPredictor":
        raise SystemExit("unexpected prediction protocol")
    if ProgramSelector.__name__ != "ProgramSelector":
        raise SystemExit("unexpected selection protocol")
    if AdaptationBackend.__name__ != "AdaptationBackend":
        raise SystemExit("unexpected execution protocol")
    if AdaptationEvaluator.__name__ != "AdaptationEvaluator":
        raise SystemExit("unexpected evaluation protocol")

    model = ModelContext("synthetic/model", "v1", "base")
    episode = LearningEpisode("synthetic", [], {"accuracy": []}, episode_id="e1")
    program = ProgramSpec("small", "adapter", {"rank": 4})
    observations = AdaptationDataset(
        [
            AdaptationResult(
                model,
                episode,
                program,
                before={"accuracy": 0.2},
                after={"accuracy": 0.6},
                metadata={"seed": 1},
            )
        ]
    )
    compiler_data = build_compiler_dataset(
        observations,
        model_descriptors={model.identity_key: {"scale": 1}},
        episode_descriptors={episode.identity_key: {"n_examples": 8}},
        program_descriptors={program.fingerprint: {"relative_size": 0.1}},
        target="delta",
    )
    if not isinstance(compiler_data, CompilerDataset):
        raise SystemExit("compiler builder returned the wrong type")
    if compiler_data.feature_names != (
        "model.scale",
        "episode.n_examples",
        "program.relative_size",
        "baseline.accuracy",
    ):
        raise SystemExit("unexpected compiler feature schema")
    if not math.isclose(compiler_data[0].target["accuracy"], 0.4):
        raise SystemExit("unexpected compiler target")
    if arguments.dataframe:
        frame = compiler_data.to_dataframe()
        if len(frame) != 1 or "feature.baseline.accuracy" not in frame:
            raise SystemExit("unexpected compiler DataFrame")
    if arguments.predict:
        prediction = (
            RidgeGeometryPredictor()
            .fit(compiler_data)
            .predict(
                features=compiler_data[0].features,
                model_key=compiler_data[0].model_key,
                episode_key=compiler_data[0].episode_key,
                family_fingerprint=compiler_data[0].family_fingerprint,
                program_fingerprint=compiler_data[0].program_fingerprint,
            )
        )
        if not isinstance(prediction, GeometryPrediction):
            raise SystemExit("ridge predictor returned the wrong prediction type")
        if not math.isclose(
            prediction.predicted_geometry["accuracy"],
            compiler_data[0].features["baseline.accuracy"]
            + prediction.predicted_target["accuracy"],
        ):
            raise SystemExit("delta prediction did not reconstruct from baseline")
    if arguments.selection or arguments.execution or arguments.evaluation:
        first_geometry = AdaptationGeometry({"accuracy": 0.6, "cost": 1.0})
        second_geometry = AdaptationGeometry({"accuracy": 0.9, "cost": 3.0})
        alternative = ProgramSpec("large", "adapter", {"rank": 8})
        predictions = (
            GeometryPrediction(
                model_key=model.identity_key,
                episode_key=episode.identity_key,
                family_fingerprint=None,
                program_fingerprint=program.fingerprint,
                predicted_target=first_geometry,
                predicted_geometry=first_geometry,
                target_kind="after",
            ),
            GeometryPrediction(
                model_key=model.identity_key,
                episode_key=episode.identity_key,
                family_fingerprint=None,
                program_fingerprint=alternative.fingerprint,
                predicted_target=second_geometry,
                predicted_geometry=second_geometry,
                target_kind="after",
            ),
        )
        selection = LinearUtilitySelector().select(
            predictions,
            SelectionObjective(
                maximize={"accuracy": 1.0},
                constraints=(GeometryConstraint("cost", maximum=2.0),),
            ),
        )
        if not isinstance(selection, ProgramSelection):
            raise SystemExit("selector returned the wrong selection type")
        if not isinstance(selection.selected, CandidateScore):
            raise SystemExit("selector did not return a feasible candidate")
        if selection.selected.prediction is not predictions[0]:
            raise SystemExit("selector ignored a hard constraint")
        if arguments.execution or arguments.evaluation:
            calls: list[ProgramSpec] = []

            def execute(
                runtime: dict[str, float],
                *,
                model_context: ModelContext,
                episode: LearningEpisode,
                program: ProgramSpec,
            ) -> dict[str, float]:
                calls.append(program)
                return {"value": runtime["value"] + program.parameters["rank"]}

            outcome = execute_selection(
                selection,
                programs=(program, alternative),
                model={"value": 1.0},
                model_context=model,
                episode=episode,
                backend=CallableBackend(
                    execute,
                    backend_id="synthetic-callable",
                    supports=lambda candidate: candidate.method == "adapter",
                ),
            )
            if not isinstance(outcome, ExecutionOutcome):
                raise SystemExit("execution returned the wrong outcome type")
            if not isinstance(outcome.record, ExecutionRecord):
                raise SystemExit("execution returned the wrong record type")
            if calls != [program] or outcome.record.program is not program:
                raise SystemExit(
                    "execution did not resolve exactly one selected program"
                )
            if outcome.model != {"value": 5.0}:
                raise SystemExit("callable backend returned an unexpected runtime")
            if arguments.evaluation:
                evaluation_calls: list[ProgramSpec] = []

                def evaluate(
                    runtime: dict[str, float],
                    *,
                    model_context: ModelContext,
                    episode: LearningEpisode,
                    program: ProgramSpec,
                ) -> AdaptationGeometry:
                    evaluation_calls.append(program)
                    return AdaptationGeometry({"accuracy": runtime["value"]})

                measured = evaluate_execution(
                    outcome,
                    evaluator=CallableEvaluator(
                        evaluate, evaluator_id="synthetic-measurement"
                    ),
                    model_context=model,
                    episode=episode,
                    before=AdaptationGeometry({"accuracy": 1.0}),
                )
                if not isinstance(measured, EvaluationOutcome):
                    raise SystemExit("evaluation returned the wrong outcome type")
                if not isinstance(measured.record, EvaluationRecord):
                    raise SystemExit("evaluation returned the wrong record type")
                if evaluation_calls != [program]:
                    raise SystemExit("evaluator did not run exactly once")
                if measured.result.after["accuracy"] != 5.0:
                    raise SystemExit("evaluation did not preserve measured geometry")
                if measured.record.backend_id != "synthetic-callable":
                    raise SystemExit("evaluation lost execution backend provenance")


if __name__ == "__main__":
    main()
