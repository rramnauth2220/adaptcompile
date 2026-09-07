"""Minimal workflow exercised against an installed distribution."""

from __future__ import annotations

import argparse
import importlib.util
import math
from importlib.metadata import version

import adaptcompile
from adaptcompile import (
    AdaptationDataset,
    AdaptationResult,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)
from adaptcompile.compiler import CompilerDataset, build_compiler_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataframe", action="store_true")
    parser.add_argument("--assert-no-pandas", action="store_true")
    arguments = parser.parse_args()

    if version("adaptcompile") != "0.2.0" or adaptcompile.__version__ != "0.2.0":
        raise SystemExit("unexpected installed adaptcompile version")
    if hasattr(adaptcompile, "CompilerDataset"):
        raise SystemExit("compiler API must not be re-exported from the package root")
    if arguments.assert_no_pandas and importlib.util.find_spec("pandas") is not None:
        raise SystemExit("core smoke environment unexpectedly contains pandas")

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


if __name__ == "__main__":
    main()
