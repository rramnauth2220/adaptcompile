"""Adapt already-computed synthetic experiment records into adaptcompile.

The values below are plausible test data, not published experimental results. This
example performs no training, model loading, downloads, or backend execution.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any

from adaptcompile import (
    AdaptationResult,
    AdaptationStudy,
    LearningEpisode,
    ModelContext,
    ProgramSpec,
)

METRICS = ("acquisition", "transfer", "boundedness", "preservation")
MODEL_CONTEXT = {
    "model": "synthetic-research/base-model-8b",
    "model_revision": "synthetic-revision-001",
    "base_state_id": "synthetic-pre-adaptation-state-v1",
}
PROGRAMS = {
    "early": {
        "name": "early",
        "method": "localized_lora",
        "parameters": {
            "layers": [2, 3, 4, 5],
            "rank": 16,
            "learning_rate": 2e-4,
        },
    },
    "middle": {
        "name": "middle",
        "method": "localized_lora",
        "parameters": {
            "layers": [12, 13, 14, 15],
            "rank": 16,
            "learning_rate": 2e-4,
        },
    },
    "late": {
        "name": "late",
        "method": "localized_lora",
        "parameters": {
            "layers": [24, 25, 26, 27],
            "rank": 16,
            "learning_rate": 2e-4,
        },
    },
}
SYNTHETIC_OUTPUTS = {
    "relation-001": {
        "early": (0.78, 0.66, 0.94, 0.97),
        "middle": (0.93, 0.85, 0.92, 0.95),
        "late": (0.87, 0.74, 0.97, 0.97),
    },
    "relation-002": {
        "early": (0.74, 0.62, 0.95, 0.98),
        "middle": (0.89, 0.80, 0.93, 0.96),
        "late": (0.84, 0.70, 0.97, 0.98),
    },
}


def scores(values: tuple[float, float, float, float]) -> dict[str, float]:
    return dict(zip(METRICS, values, strict=True))


def external_records() -> Iterator[dict[str, Any]]:
    """Stand in for records emitted by an external research pipeline."""
    for episode_index, (episode_id, outputs) in enumerate(
        SYNTHETIC_OUTPUTS.items(), start=1
    ):
        for program_name, after in outputs.items():
            yield {
                "episode_id": episode_id,
                "episode_name": f"synthetic relation episode {episode_index}",
                "learning_family": "synthetic-relational-learning",
                "program": PROGRAMS[program_name],
                "before": scores((0.08, 0.06, 0.98, 0.98)),
                "after": scores(after),
                **MODEL_CONTEXT,
                "experiment_id": "synthetic-adapter-example",
                "seed": 100 + episode_index,
                "synthetic": True,
            }


def adapt_record(record: Mapping[str, Any]) -> AdaptationResult:
    """Convert one external result record without adding a core adapter API."""
    before = record["before"]
    after = record["after"]
    program_data = record["program"]
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError("external before/after values must be mappings")
    if not isinstance(program_data, Mapping):
        raise TypeError("external program value must be a mapping")

    episode = LearningEpisode(
        name=record["episode_name"],
        episode_id=record["episode_id"],
        train={"dataset_ref": f"memory://{record['episode_id']}/train"},
        evaluations={
            metric: {"dataset_ref": f"memory://{record['episode_id']}/{metric}"}
            for metric in before
        },
        metadata={
            "learning_family": record["learning_family"],
            "synthetic": record["synthetic"],
        },
    )
    program = ProgramSpec(
        name=program_data["name"],
        method=program_data["method"],
        parameters=program_data["parameters"],
    )
    model_context = ModelContext(
        model_id=record["model"],
        revision=record["model_revision"],
        base_state_id=record["base_state_id"],
    )
    return AdaptationResult(
        model_context=model_context,
        episode=episode,
        program=program,
        before=before,
        after=after,
        metadata={
            "experiment_id": record["experiment_id"],
            "seed": record["seed"],
            "synthetic": record["synthetic"],
        },
    )


results = [adapt_record(record) for record in external_records()]
grouped: dict[str, list[AdaptationResult]] = defaultdict(list)
for result in results:
    episode_id = result.episode.episode_id
    if episode_id is None:
        raise RuntimeError("experiment records require stable episode IDs")
    grouped[episode_id].append(result)

studies = {
    episode_id: AdaptationStudy(episode_results)
    for episode_id, episode_results in grouped.items()
}

print(f"Converted {len(results)} synthetic records into {len(studies)} studies.")
for episode_id, study in studies.items():
    ranking = [result.program.name for result in study.rank("acquisition")]
    best_transfer = study.best("transfer").program.name
    pareto = [result.program.name for result in study.pareto_front(maximize=METRICS)]
    print(
        f"{episode_id}: acquisition={ranking}, "
        f"best_transfer={best_transfer}, pareto={pareto}"
    )
