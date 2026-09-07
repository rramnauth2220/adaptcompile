"""Validated assembly of observed adaptations into supervised records."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, overload

from .._validation import nonempty_name, score
from ..dataset import AdaptationDataset, ModelIdentityKey
from ..episode import EpisodeIdentityKey
from ..errors import ValidationError
from ..geometry import AdaptationGeometry
from ..serialization import immutable_json_mapping, to_json_safe

if TYPE_CHECKING:
    from pandas import DataFrame

TargetKind: TypeAlias = Literal["after", "delta"]
InteractionIdentityKey: TypeAlias = tuple[ModelIdentityKey, EpisodeIdentityKey, str]
DescriptorValues: TypeAlias = Mapping[str, Real]

_FEATURE_NAMESPACE_ORDER = (
    "model",
    "episode",
    "family",
    "program",
    "interaction",
    "baseline",
)
_FEATURE_NAMESPACE_INDEX = {
    namespace: index for index, namespace in enumerate(_FEATURE_NAMESPACE_ORDER)
}
_RESERVED_NAMESPACES = frozenset(
    (*_FEATURE_NAMESPACE_ORDER, "after", "delta", "target")
)


def _feature_sort_key(name: str) -> tuple[int, str]:
    namespace = name.partition(".")[0]
    return (
        _FEATURE_NAMESPACE_INDEX.get(namespace, len(_FEATURE_NAMESPACE_ORDER)),
        name,
    )


def _validated_feature_mapping(value: Any) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValidationError("features must be a mapping")
    normalized: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        name = nonempty_name(raw_name, field="feature name")
        namespace, separator, local_name = name.partition(".")
        if not separator or not local_name or namespace not in _FEATURE_NAMESPACE_INDEX:
            raise ValidationError(
                f"feature name {name!r} must use a supported namespace"
            )
        normalized[name] = score(raw_value, field=f"features[{name!r}]")
    if not normalized:
        raise ValidationError("features must contain at least one value")
    return MappingProxyType(
        dict(sorted(normalized.items(), key=lambda item: _feature_sort_key(item[0])))
    )


def _validate_model_key(value: Any) -> ModelIdentityKey:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValidationError("model_key must be a ModelContext identity tuple")
    model_id, revision, base_state_id = value
    nonempty_name(model_id, field="model_key model_id")
    for name, item in (("revision", revision), ("base_state_id", base_state_id)):
        if item is not None:
            nonempty_name(item, field=f"model_key {name}")
    return cast(ModelIdentityKey, value)


def _validate_episode_key(value: Any) -> EpisodeIdentityKey:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValidationError("episode_key must be a LearningEpisode identity tuple")
    kind, identity = value
    if kind not in {"episode_id", "configuration"}:
        raise ValidationError("episode_key has an unknown identity kind")
    nonempty_name(identity, field="episode_key identity")
    return cast(EpisodeIdentityKey, value)


@dataclass(frozen=True)
class CompilerRecord:
    """One immutable compiler-ready supervised observation."""

    model_key: ModelIdentityKey
    episode_key: EpisodeIdentityKey
    family_fingerprint: str | None
    program_fingerprint: str
    features: Mapping[str, float]
    target: AdaptationGeometry
    target_kind: TargetKind
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_key", _validate_model_key(self.model_key))
        object.__setattr__(self, "episode_key", _validate_episode_key(self.episode_key))
        if self.family_fingerprint is not None:
            object.__setattr__(
                self,
                "family_fingerprint",
                nonempty_name(self.family_fingerprint, field="family_fingerprint"),
            )
        object.__setattr__(
            self,
            "program_fingerprint",
            nonempty_name(self.program_fingerprint, field="program_fingerprint"),
        )
        object.__setattr__(self, "features", _validated_feature_mapping(self.features))
        if not isinstance(self.target, AdaptationGeometry):
            raise ValidationError("target must be an AdaptationGeometry")
        if self.target_kind not in {"after", "delta"}:
            raise ValidationError("target_kind must be 'after' or 'delta'")
        object.__setattr__(
            self,
            "metadata",
            immutable_json_mapping(self.metadata, location="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe nested record."""
        safe = to_json_safe(
            {
                "model_key": self.model_key,
                "episode_key": self.episode_key,
                "family_fingerprint": self.family_fingerprint,
                "program_fingerprint": self.program_fingerprint,
                "features": self.features,
                "target": self.target.to_dict(),
                "target_kind": self.target_kind,
                "metadata": self.metadata,
            }
        )
        if not isinstance(safe, dict):  # pragma: no cover - construction guarantees it
            raise RuntimeError("compiler record serialization must produce an object")
        return cast(dict[str, Any], safe)


class CompilerDataset(Sequence[CompilerRecord]):
    """Immutable supervised records with one deterministic feature/target schema."""

    __slots__ = ("_feature_names", "_metadata", "_records", "_target_names")

    def __init__(
        self,
        records: Iterable[CompilerRecord],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        collected = tuple(records)
        if not collected:
            raise ValidationError("a CompilerDataset requires at least one record")
        if any(not isinstance(record, CompilerRecord) for record in collected):
            raise ValidationError(
                "all compiler dataset entries must be CompilerRecord instances"
            )

        first = collected[0]
        feature_names = tuple(first.features)
        target_names = tuple(sorted(first.target.metric_names))
        expected_features = set(feature_names)
        expected_targets = set(target_names)
        for index, record in enumerate(collected[1:], start=1):
            actual_features = set(record.features)
            if actual_features != expected_features:
                raise ValidationError(
                    f"compiler feature schema mismatch at record {index} "
                    f"(missing={sorted(expected_features - actual_features)}, "
                    f"extra={sorted(actual_features - expected_features)})"
                )
            actual_targets = set(record.target.metric_names)
            if actual_targets != expected_targets:
                raise ValidationError(
                    f"compiler target schema mismatch at record {index} "
                    f"(missing={sorted(expected_targets - actual_targets)}, "
                    f"extra={sorted(actual_targets - expected_targets)})"
                )
            if record.target_kind != first.target_kind:
                raise ValidationError(
                    f"compiler target kind mismatch at record {index} "
                    f"({record.target_kind!r} != {first.target_kind!r})"
                )

        self._records = collected
        self._feature_names = feature_names
        self._target_names = target_names
        self._metadata = immutable_json_mapping(
            metadata or {}, location="compiler dataset metadata"
        )

    @property
    def records(self) -> tuple[CompilerRecord, ...]:
        return self._records

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def target_names(self) -> tuple[str, ...]:
        return self._target_names

    @property
    def target_kind(self) -> TargetKind:
        return self._records[0].target_kind

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only descriptor-schema provenance for this assembled dataset."""
        return self._metadata

    @overload
    def __getitem__(self, index: int) -> CompilerRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CompilerRecord, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> CompilerRecord | tuple[CompilerRecord, ...]:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[CompilerRecord]:
        return iter(self._records)

    def __repr__(self) -> str:
        return (
            f"CompilerDataset(n_records={len(self)}, "
            f"n_features={len(self.feature_names)}, target_kind={self.target_kind!r})"
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return JSON-safe records with identity, features, target, and metadata."""
        return [record.to_dict() for record in self]

    def to_dataframe(self) -> DataFrame:
        """Return a flattened pandas DataFrame (requires the dataframe extra)."""
        try:
            from pandas import DataFrame
        except ImportError as error:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() requires pandas; install adaptcompile[dataframe]"
            ) from error

        rows: list[dict[str, Any]] = []
        for record in self:
            row: dict[str, Any] = {
                "model_key": record.model_key,
                "episode_key": record.episode_key,
                "family_fingerprint": record.family_fingerprint,
                "program_fingerprint": record.program_fingerprint,
                "target_kind": record.target_kind,
                "metadata": record.to_dict()["metadata"],
            }
            row.update(
                {
                    f"feature.{name}": record.features[name]
                    for name in self.feature_names
                }
            )
            row.update(
                {f"target.{name}": record.target[name] for name in self.target_names}
            )
            rows.append(row)
        return cast("DataFrame", DataFrame(rows))


def _descriptor_features(
    namespace: str,
    descriptors: Mapping[Any, DescriptorValues] | None,
    identity: Any,
) -> dict[str, float]:
    if descriptors is None:
        return {}
    if not isinstance(descriptors, Mapping):
        raise ValidationError(f"{namespace}_descriptors must be a mapping")
    try:
        values = descriptors[identity]
    except KeyError as error:
        raise ValidationError(
            f"missing {namespace} descriptors for identity {identity!r}"
        ) from error
    if not isinstance(values, Mapping):
        raise ValidationError(
            f"{namespace} descriptors for {identity!r} must be a mapping"
        )

    features: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = nonempty_name(raw_name, field=f"{namespace} descriptor name")
        root = name.partition(".")[0]
        if root in _RESERVED_NAMESPACES:
            raise ValidationError(
                f"{namespace} descriptor name {name!r} uses reserved namespace "
                f"{root!r}; supply local descriptor names"
            )
        full_name = f"{namespace}.{name}"
        features[full_name] = score(
            raw_value, field=f"{namespace}_descriptors[{identity!r}][{name!r}]"
        )
    return features


def build_compiler_dataset(
    adaptations: AdaptationDataset,
    *,
    model_descriptors: Mapping[ModelIdentityKey, DescriptorValues] | None = None,
    episode_descriptors: Mapping[EpisodeIdentityKey, DescriptorValues] | None = None,
    family_descriptors: Mapping[str, DescriptorValues] | None = None,
    program_descriptors: Mapping[str, DescriptorValues] | None = None,
    interaction_descriptors: Mapping[InteractionIdentityKey, DescriptorValues]
    | None = None,
    target: TargetKind = "delta",
    descriptor_metadata: Mapping[str, Any] | None = None,
) -> CompilerDataset:
    """Assemble observed adaptations and external descriptors into supervised data.

    Descriptor mappings use semantic identities and local names. The library adds
    namespace prefixes and includes each observation's before geometry as
    ``baseline.*`` features.
    """
    if not isinstance(adaptations, AdaptationDataset):
        raise ValidationError("adaptations must be an AdaptationDataset")
    if target not in {"after", "delta"}:
        raise ValidationError("target must be 'after' or 'delta'")

    records: list[CompilerRecord] = []
    for result in adaptations:
        model_key = result.model_context.identity_key
        episode_key = result.episode.identity_key
        family = result.program.family
        family_fingerprint = family.fingerprint if family is not None else None
        program_fingerprint = result.program.fingerprint
        interaction_key = (model_key, episode_key, program_fingerprint)

        features: dict[str, float] = {}
        features.update(_descriptor_features("model", model_descriptors, model_key))
        features.update(
            _descriptor_features("episode", episode_descriptors, episode_key)
        )
        if family_descriptors is not None:
            if family_fingerprint is None:
                raise ValidationError(
                    "family descriptors were supplied but an adaptation program "
                    "has no ProgramFamily"
                )
            features.update(
                _descriptor_features("family", family_descriptors, family_fingerprint)
            )
        features.update(
            _descriptor_features("program", program_descriptors, program_fingerprint)
        )
        features.update(
            _descriptor_features(
                "interaction", interaction_descriptors, interaction_key
            )
        )
        features.update(
            {
                f"baseline.{name}": value
                for name, value in result.before_geometry.items()
            }
        )

        records.append(
            CompilerRecord(
                model_key=model_key,
                episode_key=episode_key,
                family_fingerprint=family_fingerprint,
                program_fingerprint=program_fingerprint,
                features=features,
                target=(
                    result.after_geometry
                    if target == "after"
                    else result.delta_geometry
                ),
                target_kind=target,
                metadata=result.metadata,
            )
        )
    return CompilerDataset(records, metadata=descriptor_metadata)
